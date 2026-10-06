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
from config import (
    REVIEW_N_PREDICT, REVIEW_REPEAT_PENALTY, REVIEW_TIME_BUDGET_SECONDS, REVIEW_MIN_CALL_TIMEOUT_SECONDS,
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

# Reviews code one piece at a time, most worthwhile first, until everything is done or the time guideline is used up.
# Each review gets the code itself, the suspicious lines found by a pattern search, and its most similar chunks as background, so every prompt is small enough for the small model.
class CodeReviewer:
    def __init__(self, model: LocalModel, retriever: Retriever, repository_data: RepositoryData):
        # the model that does the reviews, and the retriever that finds related chunks
        self.model = model
        self.retriever = retriever
        # which chunk each chunk is inside
        self.parents = get_parent_chunks(repository_data)
        # the reviews to do, most worthwhile first (only flagged lines are reviewed)
        plan = rank_chunks_for_review(repository_data)
        self.units = plan.units
        # start recording the result (chunks too large for one prompt that contain other chunks are reviewed through those other chunks)
        self.result = ReviewResult(unit_count=len(self.units), unflagged=plan.unflagged_count)
        for chunk in plan.not_reviewable:
            self.result.skipped_notes.append(f"- {describe_chunk_location(chunk)}")
        # how many model failures there have been in a row
        self.consecutive_failures = 0

    # Show a message without breaking the progress bar
    def say(self, message: str) -> None:
        tqdm.write(message)

    # do the reviews and return what was found
    def run(self) -> ReviewResult:
        print(f"Doing {len(self.units)} reviews of flagged lines, most worthwhile first, with a guideline of {describe_duration(REVIEW_TIME_BUDGET_SECONDS)} for the model.")
        print("Press Ctrl+C to stop early: reviews already done are saved and will not be repeated next time.")
        # the progress bar counts reviews done
        progress = tqdm(self.units, desc="Reviewing", unit="review")
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
            self.result.stop_reason = "it was stopped by the user"
        finally:
            # stop the progress bar
            progress.close()
        return self.result

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
        if self.result.model_seconds >= REVIEW_TIME_BUDGET_SECONDS:
            self.result.stop_reason = f"the {describe_duration(REVIEW_TIME_BUDGET_SECONDS)} time guideline was reached"
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
        remaining = REVIEW_TIME_BUDGET_SECONDS - self.result.model_seconds
        call_timeout = min(MODEL_TIMEOUT_SECONDS, max(REVIEW_MIN_CALL_TIMEOUT_SECONDS, remaining))
        # ask the model (its answer is forced into the shape of the grammar)
        started = time.time()
        response = self.model.query_model(prompt, REVIEW_N_PREDICT, call_timeout, grammar, REVIEW_REPEAT_PENALTY)
        # the time counts whether the call worked or not
        self.result.model_seconds += time.time() - started
        return response

    # record that the model failed to do a review, and stop if it keeps failing
    def record_failure(self, unit: ReviewUnit, response: str | Exception) -> None:
        # show why (an empty answer has no exception to show)
        reason = str(response) if isinstance(response, Exception) else "The model gave an empty answer"
        self.say(f"{reason} ({describe_chunk_location(unit.chunk)})")
        self.result.failed += 1
        self.result.failed_notes.append(f"- {unit.kind.lower()} review of {describe_chunk_location(unit.chunk)}")
        # too many failures in a row means something is wrong (for example the model can not be started)
        self.consecutive_failures += 1
        if self.consecutive_failures >= REVIEW_MAX_CONSECUTIVE_FAILURES:
            self.result.stop_reason = f"the model failed {self.consecutive_failures} times in a row"

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

# Write the review to a markdown file. Returns None, or an Exception if it could not be written.
def write_review_report(target_path: Path, result: ReviewResult, markdown_path: Path) -> None | Exception:
    # the title and the summary
    lines = [f"# Code review: {target_path}", "", result.describe(), "", "## Findings", ""]
    # the findings, or a note that there are none
    lines.extend(result.findings if result.findings else ["No findings were reported.", ""])
    # the code that could not be reviewed
    if result.skipped_notes:
        lines.extend(["## Not reviewed as a whole: too large for one prompt (functions inside them are reviewed on their own)", ""] + result.skipped_notes + [""])
    if result.failed_notes:
        lines.extend(["## Not reviewed: the model failed", ""] + result.failed_notes + [""])
    # a reminder of how far the findings can be trusted
    lines.append("_Reviewed by a small local model. Each finding is about a line that a pattern search flagged and the model judged, but whether it is truly a defect still has to be checked._")
    try:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        return Exception(f"Unable to write the review to {markdown_path}. Error: {e}")

# Review chunked code with the model and write the report if a markdown path was given. Returns None, or an Exception if the review could not be done.
def review_chunks(target_path: Path, repository_data: RepositoryData, model: LocalModel, markdown_path: Path | None) -> None | Exception:
    # embed the chunks so that related chunks can be found (embeddings are saved, so an unchanged file is not embedded again)
    embedder = Embedder(repository_data)
    embedding_result = embedder.create_chunk_vector_embeddings()
    if isinstance(embedding_result, Exception):
        return Exception(f"Unable to embed the chunks: {embedding_result}")
    # create the retriever used to find related chunks
    retriever = Retriever(repository_data.chunks, embedder.embedding_state)
    # review the chunks one at a time and say how it went
    result = CodeReviewer(model, retriever, repository_data).run()
    print(result.describe())
    # there is no report to write
    if markdown_path is None:
        return None
    # write the report
    written = write_review_report(target_path, result, markdown_path)
    if isinstance(written, Exception):
        return written
    print(f"Wrote the review to {markdown_path}")