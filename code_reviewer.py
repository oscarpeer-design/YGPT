# This file is responsible for reviewing code with the local model one piece at a time, and for reporting what the review found

# Import External Dependencies
# dataclass, field: used to create struct-like classes
from dataclasses import dataclass, field
# Path: library used to find file paths on computer
from pathlib import Path
# time: used to measure how long the model takes
import time
# tqdm: draws the progress bar (it is installed along with sentence-transformers)
from tqdm import tqdm

# Import Internal Depenencies
from chunk_risk import ReviewUnit, rank_chunks_for_review
from code_chunker import RepositoryData
from code_parser import ChunkedRepository
from config import (
    REVIEW_N_PREDICT, REVIEW_REPEAT_PENALTY, REVIEW_TIME_BUDGET_SECONDS, REVIEW_REPOSITORY_TIME_BUDGET_SECONDS, REVIEW_SUMMARY_MAX_FILE_ROWS, REVIEW_MIN_CALL_TIMEOUT_SECONDS,
    REVIEW_MAX_CONSECUTIVE_FAILURES, REVIEW_RELATED_CANDIDATES, REVIEW_RELATED_SIMILARITY_THRESHOLD,
    REVIEW_PROGRESS_NAME_CHARS, MODEL_TIMEOUT_SECONDS
)
from context_builder import (
    build_chunk_review_prompt, build_review_grammar, describe_chunk_location, get_parent_chunks,
    get_review_cache_path, load_cached_review, save_cached_review
)
from embedder_and_retriever import Embedder, Retriever
from model import LocalModel, ReviewFinding, parse_review_findings

# Describe a number of seconds in plain words (e.g. "5 minute" or "90 second")
def describe_duration(seconds: float) -> str:
    # from two minutes up, minutes are easier to read
    if seconds >= 120:
        return f"{seconds / 60:.0f} minute"
    return f"{seconds:.0f} second"

# Describe a finding on one line
def describe_finding(finding: ReviewFinding) -> str:
    return f"line {finding.line}: {finding.problem} (Fix: {finding.fix})"

# Write the findings of one review as a markdown section for the report
def build_findings_section(unit: ReviewUnit, findings: list[ReviewFinding]) -> str:
    # the heading says which kind of review it was and where the code is
    heading = f"### {unit.kind.capitalize()}: {describe_chunk_location(unit.chunk)}\n\n"
    # say why this code was looked at, if we know
    reasons = f"Looked at because: {'; '.join(unit.reasons)}.\n\n" if unit.reasons else ""
    # say anything else we know about where the code is used
    notes = "".join(f"{note}\n\n" for note in unit.notes)
    # one entry for each finding, quoting the code
    entries = [f"- **line {finding.line}**: `{finding.code}`\n  - Problem: {finding.problem}\n  - Fix: {finding.fix}" for finding in findings]
    return heading + reasons + notes + "\n".join(entries) + "\n"

# ReviewResult: everything learned while reviewing a file piece by piece
@dataclass
class ReviewResult:
    unit_count: int = 0 # how many reviews were queued (a piece of code can have a security review and a performance review)
    reviewed: int = 0 # how many were done by the model in this run
    cached: int = 0 # how many were loaded from an earlier run
    unflagged: int = 0 # how many chunks had no flagged lines, so there was nothing to ask the model
    failed: int = 0 # how many the model failed to do
    with_findings: int = 0 # how many reviews ended with at least one finding that checked out
    finding_count: int = 0 # how many findings checked out
    unverified: int = 0 # how many claims were thrown away: they named a line that was not flagged, argued against themselves, or only repeated the flag
    model_seconds: float = 0.0 # time spent waiting for the model (saved reviews cost nothing)
    stop_reason: str = "" # why the review stopped early (empty if it did not)
    findings: list[str] = field(default_factory=list) # one markdown section for each review that has findings
    failed_notes: list[str] = field(default_factory=list) # the reviews the model failed to do
    skipped_notes: list[str] = field(default_factory=list) # the pieces of code that could not be reviewed

    # add the result of another file to this one (used to total a repository)
    def add(self, other: "ReviewResult") -> None:
        self.unit_count += other.unit_count
        self.reviewed += other.reviewed
        self.cached += other.cached
        self.unflagged += other.unflagged
        self.failed += other.failed
        self.with_findings += other.with_findings
        self.finding_count += other.finding_count
        self.unverified += other.unverified
        self.model_seconds += other.model_seconds
        self.findings.extend(other.findings)
        self.failed_notes.extend(other.failed_notes)
        self.skipped_notes.extend(other.skipped_notes)

    # how many reviews were never reached (they are less worthwhile, so they are left for another run)
    def not_reached(self) -> int:
        # every review is either done, failed, or not reached
        done = self.reviewed + self.cached + self.failed
        return max(0, self.unit_count - done)

    # describe how the review went in a few sentences
    def describe(self) -> str:
        summary = (
            f"Did {self.reviewed + self.cached} of {self.unit_count} reviews, most worthwhile first ({self.cached} from an earlier run). "
            f"{self.finding_count} findings (in {self.with_findings} reviews); {self.unverified} claims were thrown away (they named a line that was not flagged, argued against themselves, or only repeated the flag). "
            f"Not done: {self.not_reached()} not reached, {self.unflagged} chunks had no flagged lines, {len(self.skipped_notes)} too large to review as a whole, {self.failed} failed. "
            f"Time spent on the model: {self.model_seconds / 60:.1f} minutes."
        )
        # the average only makes sense if the model did a review in this run
        if self.reviewed > 0:
            summary += f" Average {self.model_seconds / self.reviewed:.1f}s per review."
        if self.stop_reason:
            summary += f" The review stopped because {self.stop_reason}."
        if self.not_reached() > 0:
            summary += " Run the same review again to continue with the remaining reviews."
        return summary

# Reviews code one piece at a time, file by file, most worthwhile first within each file, until everything is done or the time guideline is used up.
# Each review gets the code itself, the suspicious lines found by a pattern search, and its most similar chunks as background (from any file), so every prompt is small enough for the small model.
class CodeReviewer:
    def __init__(self, model: LocalModel, retriever: Retriever, repository_data: RepositoryData):
        # the model that does the reviews, and the retriever that finds related chunks
        self.model = model
        self.retriever = retriever
        # which chunk each chunk is inside
        self.parents = get_parent_chunks(repository_data)
        # the reviews to do, most worthwhile first (only flagged lines are reviewed)
        self.plan = rank_chunks_for_review(repository_data)
        # the files, in a stable order (the order of the reviews inside each file stays most worthwhile first)
        self.files = sorted({chunk.file_path for chunk in repository_data.chunks})
        # the reviews of each file
        self.units_by_file: dict[Path, list[ReviewUnit]] = {file_path: [] for file_path in self.files}
        for unit in self.plan.units:
            self.units_by_file[unit.chunk.file_path].append(unit)
        # the result of the file being reviewed (a new one is started for each file)
        self.result = ReviewResult()
        # the time (in seconds) the file being reviewed may spend waiting for the model, and how to describe that limit
        self.time_budget = float(REVIEW_TIME_BUDGET_SECONDS)
        self.time_budget_description = f"the {describe_duration(REVIEW_TIME_BUDGET_SECONDS)} time guideline"
        # how many model failures there have been in a row, and why the whole review had to stop (empty if it did not)
        self.consecutive_failures = 0
        self.abort_reason = ""

    # Show a message without breaking the progress bar
    def say(self, message: str) -> None:
        tqdm.write(message)

    # start the result of one file: what was queued, what had nothing flagged, and what is too large to review as a whole
    def start_result(self, file_path: Path) -> ReviewResult:
        result = ReviewResult(unit_count=len(self.units_by_file[file_path]), unflagged=self.plan.unflagged_by_file.get(file_path, 0))
        # chunks too large for one prompt that contain other chunks are reviewed through those other chunks
        for chunk in self.plan.not_reviewable:
            if chunk.file_path == file_path:
                result.skipped_notes.append(f"- {describe_chunk_location(chunk)}")
        return result

    # review every file, one after another, and return the result of each. A single file gets the single-file guideline, a repository the repository guideline.
    def run(self, is_repository: bool) -> list[tuple[Path, ReviewResult]]:
        total_budget = float(REVIEW_REPOSITORY_TIME_BUDGET_SECONDS if is_repository else REVIEW_TIME_BUDGET_SECONDS)
        file_results: list[tuple[Path, ReviewResult]] = []
        # the files that have something to review, and how long the model has been waited for so far
        files_to_review = [file_path for file_path in self.files if self.units_by_file[file_path]]
        if is_repository:
            print(f"Reviewing {len(files_to_review)} of {len(self.files)} files one after another, with a guideline of {describe_duration(total_budget)} for the model in total.")
        else:
            print(f"Doing {len(self.plan.units)} reviews of flagged lines, most worthwhile first, with a guideline of {describe_duration(total_budget)} for the model.")
        print("Press Ctrl+C to stop early: reviews already done are saved and will not be repeated next time.")
        spent = 0.0
        for file_path in self.files:
            self.result = self.start_result(file_path)
            units = self.units_by_file[file_path]
            if units:
                # the file gets the time that is left divided by the files still to do (time that earlier files did not use goes to the later ones)
                files_left = len(files_to_review) - files_to_review.index(file_path)
                self.time_budget = max(0.0, total_budget - spent) / files_left
                self.time_budget_description = f"the {describe_duration(total_budget)} time guideline"
                # a review that had to stop everything leaves the rest of the files unreviewed
                if self.abort_reason:
                    self.result.stop_reason = self.abort_reason
                else:
                    if is_repository:
                        print(f"File {files_to_review.index(file_path) + 1} of {len(files_to_review)}: {file_path} ({len(units)} reviews)")
                    self.review_units(units, file_path if is_repository else None)
            spent += self.result.model_seconds
            file_results.append((file_path, self.result))
        return file_results

    # do the reviews of one file
    def review_units(self, units: list[ReviewUnit], file_path: Path | None) -> None:
        # the progress bar counts reviews done (the file is named in a repository review)
        description = f"Reviewing {file_path.name}" if file_path is not None else "Reviewing"
        progress = tqdm(units, desc=description, unit="review")
        try:
            for unit in progress:
                # show what is being reviewed and how it is going
                progress.set_postfix_str(self.describe_progress(unit))
                self.review_unit(unit)
                # stop as soon as a review says we should
                if self.result.stop_reason:
                    break
        except KeyboardInterrupt:
            self.say("\nReview stopped early by the user.")
            self.abort_reason = "it was stopped by the user"
            self.result.stop_reason = self.abort_reason
        finally:
            # stop the progress bar
            progress.close()

    # describe the review being done and how the whole review is going, for the progress bar
    def describe_progress(self, unit: ReviewUnit) -> str:
        name = unit.chunk.chunk_name[:REVIEW_PROGRESS_NAME_CHARS]
        return f"{unit.kind.lower()}: {name} | findings {self.result.finding_count} | model {self.result.model_seconds:.0f}s"

    # do one review: build its prompt, get the model's answer, and keep the findings
    def review_unit(self, unit: ReviewUnit) -> None:
        # build the prompt (the units were chosen because they fit, so this should always work)
        prompt = self.build_prompt(unit)
        if prompt is None:
            self.result.skipped_notes.append(f"- {describe_chunk_location(unit.chunk)}")
            return
        # the grammar forces the answer to name one of the flagged lines
        grammar = build_review_grammar(unit.hints)
        # get the model's answer (None means there is nothing to read: a failure or a stop has been recorded)
        response = self.get_response(unit, prompt, grammar)
        if response is None:
            return
        self.record_findings(unit, response)

    # build the prompt for a review, with its most similar chunks as background. Returns None if the code does not fit.
    def build_prompt(self, unit: ReviewUnit) -> str | None:
        chunk = unit.chunk
        # find similar chunks to use as background
        candidates = self.retriever.retrieve_related_to_chunk(unit.chunk_index, REVIEW_RELATED_CANDIDATES, REVIEW_RELATED_SIMILARITY_THRESHOLD)
        built = build_chunk_review_prompt(chunk, candidates, self.parents.get(chunk.chunk_id), unit.kind, unit.hints, unit.context_lines)
        # the chunk does not fit
        if built is None:
            return None
        prompt, related_chunks = built
        return prompt

    # get the answer for a prompt: from an earlier run if there is one, otherwise from the model. Returns None if there is no answer.
    def get_response(self, unit: ReviewUnit, prompt: str, grammar: str) -> str | None:
        # reuse an earlier review of exactly this prompt if there is one (this costs no time)
        cache_path = get_review_cache_path(self.model.model_name, prompt, grammar)
        saved_response = load_cached_review(cache_path)
        # a saved review we can not read is only a warning: we review again
        if isinstance(saved_response, Exception):
            self.say(f"Warning: {saved_response}")
            saved_response = None
        if saved_response is not None:
            self.result.cached += 1
            return saved_response
        # the time guideline is checked before each model call: everything from here on is less worthwhile and is left for another run
        if self.result.model_seconds >= self.time_budget:
            self.result.stop_reason = f"{self.time_budget_description} was reached"
            return None
        response = self.ask_model(prompt, grammar)
        # a failure (or an empty answer) is recorded
        if isinstance(response, Exception) or not response.strip():
            self.record_failure(unit, response)
            return None
        # the model worked, so the run of failures is over
        self.consecutive_failures = 0
        # save the review so that the next run does not repeat it (a problem saving is only a warning)
        saved = save_cached_review(cache_path, response)
        if isinstance(saved, Exception):
            self.say(f"Warning: {saved}")
        self.result.reviewed += 1
        return response

    # ask the model, timing the call. Returns the answer, or an Exception if the model failed.
    def ask_model(self, prompt: str, grammar: str) -> str | Exception:
        # a call may not run far past the time guideline, but always gets a minimum amount of time
        remaining = self.time_budget - self.result.model_seconds
        call_timeout = min(MODEL_TIMEOUT_SECONDS, max(REVIEW_MIN_CALL_TIMEOUT_SECONDS, remaining))
        # ask the model (its answer is forced into the shape of the grammar)
        started = time.time()
        response = self.model.query_model(prompt, REVIEW_N_PREDICT, call_timeout, grammar, REVIEW_REPEAT_PENALTY)
        # the time counts whether the call worked or not
        self.result.model_seconds += time.time() - started
        return response

    # record that the model failed to do a review, and stop everything if it keeps failing
    def record_failure(self, unit: ReviewUnit, response: str | Exception) -> None:
        # show why (an empty answer has no exception to show)
        reason = str(response) if isinstance(response, Exception) else "The model gave an empty answer"
        self.say(f"{reason} ({describe_chunk_location(unit.chunk)})")
        self.result.failed += 1
        self.result.failed_notes.append(f"- {unit.kind.lower()} review of {describe_chunk_location(unit.chunk)}")
        # too many failures in a row means something is wrong (for example the model can not be started), so no file is worth continuing
        self.consecutive_failures += 1
        if self.consecutive_failures >= REVIEW_MAX_CONSECUTIVE_FAILURES:
            self.abort_reason = f"the model failed {self.consecutive_failures} times in a row"
            self.result.stop_reason = self.abort_reason

    # read the model's answer and keep the findings that are about a flagged line and say something
    def record_findings(self, unit: ReviewUnit, response: str) -> None:
        findings, thrown_away = parse_review_findings(response, unit.chunk, unit.hints)
        self.result.unverified += thrown_away
        # nothing to keep
        if not findings:
            return
        self.result.with_findings += 1
        self.result.finding_count += len(findings)
        # show the findings now, and keep them for the report
        for finding in findings:
            self.say(f"[{unit.kind}] {unit.chunk.chunk_name}, {describe_finding(finding)}")
        self.result.findings.append(build_findings_section(unit, findings))

# Add up the results of all the files
def total_file_results(file_results: list[tuple[Path, ReviewResult]]) -> ReviewResult:
    total = ReviewResult()
    # the files' results are added one by one
    for file_path, result in file_results:
        total.add(result)
    return total

# Write a brief summary of a repository review: the files, the reviews, the findings, the time, and the files with the most to look at.
# max_rows limits how many files are listed (None lists them all).
def describe_repository_review(chunked: ChunkedRepository, file_results: list[tuple[Path, ReviewResult]], max_rows: int | None) -> str:
    total = total_file_results(file_results)
    # files that had reviews queued, and files where nothing was flagged
    reviewed_files = [item for item in file_results if item[1].unit_count > 0]
    quiet_files = chunked.file_count - len(chunked.skipped_files) - len(reviewed_files)
    lines = [
        f"Files: {chunked.file_count} found, {len(reviewed_files)} reviewed, {quiet_files} with nothing to review, {len(chunked.skipped_files)} could not be read.",
        f"Reviews: {total.reviewed + total.cached} done ({total.cached} from an earlier run), {total.not_reached()} not reached, {total.failed} failed.",
        f"Findings: {total.finding_count} kept (in {total.with_findings} reviews), {total.unverified} claims thrown away.",
        f"Time spent on the model: {total.model_seconds / 60:.1f} minutes (guideline {describe_duration(REVIEW_REPOSITORY_TIME_BUDGET_SECONDS)})."
    ]
    # why the review stopped early, if it did
    stop_reasons = [result.stop_reason for file_path, result in file_results if result.stop_reason]
    if stop_reasons:
        # each reason is said once, with the number of files it affected
        said = [reason if stop_reasons.count(reason) == 1 else f"{reason} ({stop_reasons.count(reason)} files)" for reason in sorted(set(stop_reasons))]
        lines.append("Stopped early: " + "; ".join(said) + ".")
    if total.not_reached() > 0:
        lines.append("Run the same review again to continue with the reviews that were not reached.")
    # a table of the files with reviews: most findings first, then most reviews not reached
    if reviewed_files:
        reviewed_files.sort(key=lambda item: (-item[1].finding_count, -item[1].not_reached(), str(item[0])))
        shown = reviewed_files if max_rows is None else reviewed_files[:max_rows]
        lines.extend(["", "| File | Reviews done | Findings | Not reached |", "| --- | --- | --- | --- |"])
        for file_path, result in shown:
            lines.append(f"| {file_path.name} | {result.reviewed + result.cached} of {result.unit_count} | {result.finding_count} | {result.not_reached()} |")
        if len(shown) < len(reviewed_files):
            lines.append(f"| ...and {len(reviewed_files) - len(shown)} more files (all of them are in the report) | | | |")
    return "\n".join(lines)

# The parts of a report about one file's findings and what could not be reviewed. heading is the markdown heading marker for the parts (for example "##").
def build_result_sections(result: ReviewResult, heading: str) -> list[str]:
    # the findings, or a note that there are none
    lines = [f"{heading} Findings", ""]
    lines.extend(result.findings if result.findings else ["No findings were reported.", ""])
    # the code that could not be reviewed
    if result.skipped_notes:
        lines.extend([f"{heading} Not reviewed as a whole: too large for one prompt (functions inside them are reviewed on their own)", ""] + result.skipped_notes + [""])
    if result.failed_notes:
        lines.extend([f"{heading} Not reviewed: the model failed", ""] + result.failed_notes + [""])
    return lines

# A reminder of how far the findings can be trusted
REPORT_REMINDER = "_Reviewed by a small local model. Each finding is about a line that a pattern search flagged and the model judged, but whether it is truly a defect still has to be checked._"

# Write text to a markdown file. Returns None, or an Exception if it could not be written.
def write_markdown(markdown_path: Path, lines: list[str]) -> None | Exception:
    try:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        return Exception(f"Unable to write the review to {markdown_path}. Error: {e}")

# Write the review of a single file to a markdown file. Returns None, or an Exception if it could not be written.
def write_review_report(target_path: Path, result: ReviewResult, markdown_path: Path) -> None | Exception:
    # the title and the summary, then the findings and what could not be reviewed
    lines = [f"# Code review: {target_path}", "", result.describe(), ""] + build_result_sections(result, "##")
    lines.append(REPORT_REMINDER)
    return write_markdown(markdown_path, lines)

# Write the review of a repository to a markdown file: the summary, then one section for each file that has findings or something that could not be reviewed.
# Returns None, or an Exception if it could not be written.
def write_repository_report(target_path: Path, chunked: ChunkedRepository, file_results: list[tuple[Path, ReviewResult]], markdown_path: Path) -> None | Exception:
    lines = [f"# Code review: {target_path}", "", "## Summary", "", describe_repository_review(chunked, file_results, None), ""]
    # one section for each file that has something to say
    for file_path, result in file_results:
        if result.findings or result.skipped_notes or result.failed_notes:
            # each finding's heading goes one level below the file's parts
            deeper = ReviewResult(findings=[text.replace("### ", "#### ", 1) if text.startswith("### ") else text for text in result.findings], skipped_notes=result.skipped_notes, failed_notes=result.failed_notes)
            lines.extend([f"## {file_path}", ""] + build_result_sections(deeper, "###"))
    # the files that could not be read
    if chunked.skipped_files:
        lines.extend(["## Files that could not be read", ""] + chunked.skipped_files + [""])
    lines.append(REPORT_REMINDER)
    return write_markdown(markdown_path, lines)

# Review chunked code with the model, file by file, and write the report if a markdown path was given.
# chunked is given for a repository (None for a single file). Returns None, or an Exception if the review could not be done.
def review_chunks(target_path: Path, repository_data: RepositoryData, model: LocalModel, markdown_path: Path | None, chunked: ChunkedRepository | None = None) -> None | Exception:
    # embed the chunks of all the files so that related chunks can be found (embeddings are saved, so unchanged code is not embedded again)
    embedder = Embedder(repository_data)
    embedding_result = embedder.create_chunk_vector_embeddings()
    if isinstance(embedding_result, Exception):
        return Exception(f"Unable to embed the chunks: {embedding_result}")
    # create the retriever used to find related chunks
    retriever = Retriever(repository_data.chunks, embedder.embedding_state)
    # review the files one at a time and say how it went
    file_results = CodeReviewer(model, retriever, repository_data).run(chunked is not None)
    if chunked is not None:
        # a repository: a brief summary of all the files
        print(describe_repository_review(chunked, file_results, REVIEW_SUMMARY_MAX_FILE_ROWS))
        # there is no report to write
        if markdown_path is None:
            return None
        written = write_repository_report(target_path, chunked, file_results, markdown_path)
    else:
        # a single file: the summary of its review
        result = file_results[0][1] if file_results else ReviewResult()
        print(result.describe())
        if markdown_path is None:
            return None
        written = write_review_report(target_path, result, markdown_path)
    if isinstance(written, Exception):
        return written
    print(f"Wrote the review to {markdown_path}")