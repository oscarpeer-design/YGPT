# This file documents code with the local model, one file at a time, and writes the documentation.
# Everything that can be found out without the model is written by code: the dependencies, the files that use a file, the functions that matter and why (see chunk_documenting_importance.py).
# The model only puts into words what a file is for, how it works, what a change to it could affect, and what each important function does.

# Import External Dependencies
# dataclass, field: used to create struct-like classes
from dataclasses import dataclass, field
# Path: library used to find file paths on computer
from pathlib import Path
# Callable: used to say that a value is a function
from typing import Callable
# re: used to split text into words
import re
# time: used to measure how long the model takes
import time
# tqdm: draws the progress bar (installed with pip, together with the other dependencies)
from tqdm import tqdm

# Import Internal Depenencies
# the facts found without the model, and the functions that put them together
from chunk_documenting_importance import (
    FileFacts, ImportantFunction, build_documentation_plan, get_most_used_files, get_repository_external_dependencies, get_entry_points
)
# the chunks of code and the kinds of symbol
from code_chunker import RepositoryData, SymbolType
# the result of chunking a whole repository
from code_parser import ChunkedRepository
# the settings: time guidelines, answer lengths, instructions and limits
from config import (
    DOCUMENT_TIME_BUDGET_SECONDS, DOCUMENT_REPOSITORY_TIME_BUDGET_SECONDS, DOCUMENT_REPOSITORY_OVERVIEW_RESERVE_SECONDS, DOCUMENT_OVERVIEW_RESERVE_SECONDS,
    DOCUMENT_OVERVIEW_MAX_CHARS, DOCUMENT_FUNCTION_MAX_CHARS, DOCUMENT_REPOSITORY_MAX_CHARS,
    DOCUMENT_OVERVIEW_INSTRUCTION, DOCUMENT_FUNCTION_INSTRUCTION, DOCUMENT_SUBJECT_PATTERN, DOCUMENT_FRAGMENT_START_PATTERN, DOCUMENT_CUT_OFF_MARK, DOCUMENT_CLAUSE_MARKS,
    DOCUMENT_FUNCTION_EXAMPLE, DOCUMENT_REPOSITORY_INSTRUCTION, DOCUMENT_OVERVIEW_SENTENCES, DOCUMENT_FUNCTION_SENTENCES, DOCUMENT_REPOSITORY_SENTENCES,
    DOCUMENT_MIN_ANSWER_CHARS, DOCUMENT_REPEAT_PENALTY, DOCUMENT_TEMPERATURE, DOCUMENT_COPY_WORD_RUN, DOCUMENT_MAX_LISTED_SYMBOLS, DOCUMENT_MAX_NAMED_FILES, DOCUMENT_MAX_LISTED_DEPENDENCIES, DOCUMENT_FUNCTIONS_PER_FILE, SECONDS_PER_MINUTE,
    DOCUMENT_MAX_DIAGRAM_EDGES, DOCUMENT_MAX_CORE_FILES, DOCUMENT_SUMMARY_MAX_FILE_ROWS, REVIEW_MAX_CONSECUTIVE_FAILURES,
    REVIEW_MIN_CALL_TIMEOUT_SECONDS, MODEL_TIMEOUT_SECONDS, REVIEW_PROGRESS_NAME_CHARS
)
# the prompts and the grammar
from context_builder import (
    build_file_overview_prompt, build_function_documentation_prompt, build_repository_overview_prompt,
    build_document_grammar, describe_duration, describe_stop_reasons, get_answer_tokens
)
# the model, and the function that tidies one line of its answer
from model import LocalModel, tidy_field

# The words used for a part that could not be written
NOT_WRITTEN_TEXT = "Not written: the time guideline was reached or the model gave no usable answer."

# DocumentationResult: how the model's part of the documentation of one file went
@dataclass
class DocumentationResult:
    planned: int = 0 # how many parts the model was to write (what the file is for, and what each important function does)
    written: int = 0 # how many the model wrote
    unusable: int = 0 # how many answers were thrown away: too short, or they copied the instructions
    failed: int = 0 # how many the model failed to write
    model_seconds: float = 0.0 # time spent waiting for the model
    stop_reason: str = "" # why the file was not finished (empty if it was)

    # add the result of another file to this one (used to total a repository)
    def add(self, other: "DocumentationResult") -> None:
        # the counts of parts are added up
        self.planned += other.planned
        self.written += other.written
        self.unusable += other.unusable
        self.failed += other.failed
        # the time is added up too
        self.model_seconds += other.model_seconds

    # how many parts have no text from the model
    def not_written(self) -> int:
        # planned parts that were not written (never below zero)
        return max(0, self.planned - self.written)

# DocumentedFile: the facts about a file and the text the model wrote for it
@dataclass
class DocumentedFile:
    facts: FileFacts # what is known about the file without the model
    overview: str = "" # what the file is for and how it works (empty if not written)
    implications: str = "" # what a change to the file could affect (written by code from the facts)
    function_texts: dict[int, str] = field(default_factory=dict) # chunk id of an important function -> what it does
    result: DocumentationResult = field(default_factory=DocumentationResult) # how the model's part went
    is_repository: bool = False # whether the file was documented as part of a repository (otherwise it is the only file that was read)

# DocumentPart: one thing the model is asked to write
@dataclass
class DocumentPart:
    name: str # what is shown in the progress bar
    build_prompt: Callable[[], str] # builds the whole prompt given to the model (when it is needed, because it can depend on parts written before)
    instruction: str # the instruction inside the prompt (used to notice an answer that copies it)
    max_chars: int # the longest the answer may be
    max_sentences: int # how many sentences of the answer are kept
    store: Callable[[str], None] # puts the finished text where it belongs

# Split a text into lower case words (letters, digits and apostrophes only)
def split_words(text: str) -> list[str]:
    # everything that is not part of a word separates words
    return re.findall(r"[a-z0-9']+", text.lower())

# Say whether an answer repeats a run of words from the instruction it was given
def copies_instruction(answer: str, instruction: str) -> bool:
    # the words of both texts
    answer_words = split_words(answer)
    instruction_words = split_words(instruction)
    # every run of this many words in a row in the instruction
    instruction_runs = set()
    for start in range(len(instruction_words) - DOCUMENT_COPY_WORD_RUN + 1):
        instruction_runs.add(tuple(instruction_words[start:start + DOCUMENT_COPY_WORD_RUN]))
    # the answer copied the instruction if any run of the same length in the answer is one of those
    for start in range(len(answer_words) - DOCUMENT_COPY_WORD_RUN + 1):
        if tuple(answer_words[start:start + DOCUMENT_COPY_WORD_RUN]) in instruction_runs:
            return True
    return False

# Remove the start of an answer that only names the function ("The function `X` builds ..." becomes "Builds ..."). The answer is left alone if nothing useful would remain.
def strip_subject(text: str) -> str:
    stripped = re.sub(DOCUMENT_SUBJECT_PATTERN, "", text, count=1, flags=re.IGNORECASE)
    # keep the original if what remains is too short to say anything, or is a fragment ("is designed to ...")
    if len(stripped) < DOCUMENT_MIN_ANSWER_CHARS or re.match(DOCUMENT_FRAGMENT_START_PATTERN, stripped, flags=re.IGNORECASE):
        return text
    # a sentence starts with a capital letter
    return stripped[0].upper() + stripped[1:]

# The first sentence of a text
def first_sentence(text: str) -> str:
    # look at every character with its position
    for index, character in enumerate(text):
        # a sentence ends at . ! or ? followed by a space or the end of the text
        if character in ".!?" and (index + 1 == len(text) or text[index + 1].isspace()):
            return text[:index + 1]
    # no end found, so the whole text is the sentence
    return text

# The first few sentences of a text (an answer that runs on is cut off after the sentences that were asked for)
def first_sentences(text: str, count: int) -> str:
    kept: list[str] = []
    rest = text.strip()
    while rest and len(kept) < count:
        sentence = first_sentence(rest)
        kept.append(sentence)
        rest = rest[len(sentence):].strip()
    return " ".join(kept)

# Finish an answer that was cut off: the unfinished end is dropped back to the last clause, so that the sentence is complete (an answer with no clause to cut back to is left as it is)
def finish_cut_off(text: str) -> str:
    if not text.endswith(DOCUMENT_CUT_OFF_MARK):
        return text
    body = text[:-len(DOCUMENT_CUT_OFF_MARK)]
    # the last place where a clause ended
    cut = -1
    for mark in DOCUMENT_CLAUSE_MARKS:
        cut = max(cut, body.rfind(mark))
    # cutting back is only worth it if enough text is left
    if cut < DOCUMENT_MIN_ANSWER_CHARS:
        return text
    return body[:cut].rstrip() + "."

# Clean up the model's answer. Returns the text, or an empty string if the answer can not be used (too short, or it copied the instruction).
def clean_answer(raw: str, instruction: str, max_sentences: int) -> str:
    # only the first line is used, with a cut-off ending removed
    text = first_sentences(finish_cut_off(tidy_field(raw.strip().split("\n")[0])), max_sentences)
    # an answer that is too short says nothing
    if len(text) < DOCUMENT_MIN_ANSWER_CHARS:
        return ""
    # an answer that repeats the instruction says nothing about the code
    if copies_instruction(text, instruction):
        return ""
    # otherwise the answer is used
    return text

# Documents the files one after another until everything is done or the time guideline is used up.
class DocumentationWriter:
    def __init__(self, model: LocalModel, repository_data: RepositoryData):
        # the model that writes the text
        self.model = model
        # what is known about every file, in a stable order
        self.plan = build_documentation_plan(repository_data)
        # the result of the file being documented (a new one is started for each file)
        self.result = DocumentationResult()
        # the time (in seconds) the file being documented may spend waiting for the model, and how to describe that limit
        self.time_budget = float(DOCUMENT_TIME_BUDGET_SECONDS)
        self.time_budget_description = f"the {describe_duration(DOCUMENT_TIME_BUDGET_SECONDS)} time guideline"
        # how many model failures there have been in a row, and why everything had to stop (empty if it did not)
        self.consecutive_failures = 0
        self.abort_reason = ""
        # whether a whole repository is being documented (set when the documenting starts)
        self.is_repository = False

    # Show a message without breaking the progress bar
    def say(self, message: str) -> None:
        # tqdm.write prints above the progress bar
        tqdm.write(message)

    # document every file, one after another, and return what was written for each. A single file gets the single-file guideline, a repository the repository guideline.
    def run(self, is_repository: bool) -> list[DocumentedFile]:
        # whether other files were read matters for what can be said about who uses a file
        self.is_repository = is_repository
        # the time the model may take in total
        total_budget = float(DOCUMENT_REPOSITORY_TIME_BUDGET_SECONDS if is_repository else DOCUMENT_TIME_BUDGET_SECONDS)
        # in a repository some time is kept back for the overview of the whole project
        file_budget = total_budget - (DOCUMENT_REPOSITORY_OVERVIEW_RESERVE_SECONDS if is_repository else 0)
        # the files that have anything to describe (a file with no functions or classes only has its dependencies, which need no model)
        files_to_document = []
        for facts in self.plan:
            if facts.has_content():
                files_to_document.append(facts)
        # the position of each of those files in that list (found by identity because the facts are not hashable)
        positions = {}
        for index, facts in enumerate(files_to_document):
            positions[id(facts)] = index
        # tell the user what is about to happen
        print(f"Documenting {len(files_to_document)} of {len(self.plan)} files one after another, with a guideline of {describe_duration(total_budget)} for the model in total.")
        print("Press Ctrl+C to stop early: the documentation still covers what is already written.")
        # what was written for each file, and the model time used so far
        documented: list[DocumentedFile] = []
        spent = 0.0
        # every file is in the documentation, in the order of the plan
        for facts in self.plan:
            # each file starts with a new result
            self.result = DocumentationResult()
            document = DocumentedFile(facts, result=self.result, is_repository=is_repository)
            # only a file with something to describe needs the model
            if facts.has_content():
                # the file gets the time that is left divided by the files still to do (time that earlier files did not use goes to the later ones)
                files_left = len(files_to_document) - positions[id(facts)]
                self.time_budget = max(0.0, file_budget - spent) / files_left
                self.time_budget_description = f"the {describe_duration(total_budget)} time guideline"
                # a repository shows the name of each file, a single file does not need to
                self.document_file(document, facts.file_path.name if is_repository else None, positions[id(facts)] + 1, len(files_to_document))
            # the time this file used is no longer available to the next ones
            spent += self.result.model_seconds
            documented.append(document)
        return documented

    # write the model's parts of one file: what it is for and how it works, what a change could affect, and what each important function does
    def document_file(self, document: DocumentedFile, file_label: str | None, position: int, file_count: int) -> None:
        facts = document.facts
        # the parts to write, in order of importance
        parts = self.plan_parts(document)
        self.result.planned = len(parts)
        # a repository tells the user which file is next
        if file_label is not None:
            print(f"File {position} of {file_count}: {facts.file_path} ({len(parts)} parts)")
        # a failure that had to stop everything leaves the rest of the files unwritten
        if self.abort_reason:
            self.result.stop_reason = self.abort_reason
            return
        # the progress bar counts the parts of this file
        progress = tqdm(total=len(parts), desc=f"Documenting {facts.file_path.name}", unit="part")
        try:
            # write the parts one after another (the functions first, and the overview of the file last because it is built from them)
            for index, part in enumerate(parts):
                is_last = index == len(parts) - 1
                # a failure that had to stop everything stops the file
                if self.abort_reason:
                    break
                # once time has run short the remaining functions are skipped, but the overview of the file is still tried
                if self.result.stop_reason and not is_last:
                    progress.update(1)
                    continue
                # show which part is being written and how long the model has taken so far
                progress.set_postfix_str(f"{part.name[:REVIEW_PROGRESS_NAME_CHARS]} | model {self.result.model_seconds:.0f}s")
                # get the text from the model. The functions leave room for the overview.
                text = self.get_text(part, 0 if is_last else DOCUMENT_OVERVIEW_RESERVE_SECONDS)
                progress.update(1)
                # a part with usable text is kept
                if text:
                    part.store(text)
        except KeyboardInterrupt:
            # Ctrl+C stops everything, and what is already written is kept
            self.say("\nDocumenting stopped early by the user.")
            self.abort_reason = "it was stopped by the user"
            self.result.stop_reason = self.abort_reason
        finally:
            # stop the progress bar
            progress.close()

    # work out the parts that the model writes for a file: what each important function does, then what the file is for (built from those)
    def plan_parts(self, document: DocumentedFile) -> list[DocumentPart]:
        facts = document.facts
        parts: list[DocumentPart] = []
        # the example is part of what an answer must not copy
        function_instruction = DOCUMENT_FUNCTION_INSTRUCTION + "\n" + DOCUMENT_FUNCTION_EXAMPLE
        for item in facts.important_functions:
            # the item is bound as a default so that each part keeps its own function
            parts.append(DocumentPart(
                item.chunk.chunk_name,
                lambda item=item: build_function_documentation_prompt(facts.file_path.name, item.chunk.signature or item.chunk.chunk_name, item.chunk.source, item.callee_signatures),
                function_instruction,
                DOCUMENT_FUNCTION_MAX_CHARS,
                DOCUMENT_FUNCTION_SENTENCES,
                lambda text, chunk_id=item.chunk.chunk_id: document.function_texts.__setitem__(chunk_id, strip_subject(text))
            ))
        # the overview is last, and is asked for with whatever was written about the functions
        parts.append(DocumentPart(
            f"what {facts.file_path.name} is for",
            lambda: self.build_overview_prompt(document),
            DOCUMENT_OVERVIEW_INSTRUCTION,
            DOCUMENT_OVERVIEW_MAX_CHARS,
            DOCUMENT_OVERVIEW_SENTENCES,
            lambda text: setattr(document, "overview", text)
        ))
        # what a change could affect is written by code from the facts, so the model is not asked
        document.implications = describe_implications(facts, self.is_repository)
        return parts

    # the prompt that asks what a file is for: what its main functions do (as far as that was written) and the names of everything else it defines
    def build_overview_prompt(self, document: DocumentedFile) -> str:
        facts = document.facts
        described = []
        for item in facts.important_functions:
            if item.chunk.chunk_id in document.function_texts:
                described.append((item.chunk.chunk_name, document.function_texts[item.chunk.chunk_id]))
        # everything the file defines, once each and in source order (the model must see all of it to describe the file as a whole)
        names: list[str] = []
        for symbol in facts.symbols:
            if symbol.chunk_name not in names:
                names.append(symbol.chunk_name)
        for name in facts.constants:
            if name not in names:
                names.append(name)
        # the classes, structs and enums of the file
        types: list[str] = []
        for symbol in facts.symbols:
            if symbol.symbol_type in (SymbolType.CLASS, SymbolType.STRUCT, SymbolType.ENUM) and symbol.chunk_name not in types:
                types.append(symbol.chunk_name)
        return build_file_overview_prompt(facts.file_path.name, facts.language, facts.line_count, described, types, names)

    # get the model's text for a part. Returns "" if there is no usable text.
    def get_text(self, part: DocumentPart, reserve: float = 0.0) -> str:
        # the prompt is built now, because it can depend on parts written before
        prompt = part.build_prompt()
        # how long the answer may be, as tokens
        n_predict = get_answer_tokens(part.max_chars)
        # the grammar that keeps the answer to one line of the right length
        grammar = build_document_grammar(part.max_chars)
        # the time guideline is checked before each model call: the remaining parts are left for another run
        if self.result.model_seconds >= self.time_budget - reserve:
            self.result.stop_reason = f"{self.time_budget_description} was reached"
            return ""
        # a call may not run far past the time guideline, but always gets a minimum amount of time
        remaining = self.time_budget - self.result.model_seconds
        call_timeout = min(MODEL_TIMEOUT_SECONDS, max(REVIEW_MIN_CALL_TIMEOUT_SECONDS, remaining))
        # ask the model, and measure how long it takes
        started = time.time()
        response = self.model.query_model(prompt, n_predict, call_timeout, grammar, DOCUMENT_REPEAT_PENALTY, DOCUMENT_TEMPERATURE)
        # the time counts whether the call worked or not
        self.result.model_seconds += time.time() - started
        # a failure (or an empty answer) is recorded
        if isinstance(response, Exception) or not response.strip():
            self.record_failure(part.name, response)
            return ""
        # the model worked, so the run of failures is over
        self.consecutive_failures = 0
        # check the answer, and count it as written or thrown away
        text = clean_answer(response, part.instruction, part.max_sentences)
        if text:
            self.result.written += 1
        else:
            self.result.unusable += 1
        return text

    # record that the model failed to write a part, and stop everything if it keeps failing
    def record_failure(self, name: str, response: str | Exception) -> None:
        # show why (an empty answer has no exception to show)
        reason = str(response) if isinstance(response, Exception) else "The model gave an empty answer"
        self.say(f"{reason} ({name})")
        self.result.failed += 1
        # too many failures in a row means something is wrong (for example the model can not be started), so no file is worth continuing
        self.consecutive_failures += 1
        if self.consecutive_failures >= REVIEW_MAX_CONSECUTIVE_FAILURES:
            self.abort_reason = f"the model failed {self.consecutive_failures} times in a row"
            self.result.stop_reason = self.abort_reason

    # write the overview of a whole project from the first sentence of what each file is for. Returns "" if there is nothing to say or no time.
    def document_repository(self, documented: list[DocumentedFile], spent: float, total_budget: float) -> tuple[str, DocumentationResult]:
        # this call has its own result
        self.result = DocumentationResult()
        # one note per file whose overview was written
        notes = []
        for document in documented:
            if document.overview:
                notes.append(f"{document.facts.file_path.name}: {first_sentence(document.overview)}")
        # with no notes, or after a failure that stopped everything, there is nothing to write
        if not notes or self.abort_reason:
            return "", self.result
        self.result.planned = 1
        # the overview may use whatever is left of the guideline, but never less than the time kept back for it
        self.time_budget = max(float(DOCUMENT_REPOSITORY_OVERVIEW_RESERVE_SECONDS), total_budget - spent)
        self.time_budget_description = f"the {describe_duration(total_budget)} time guideline"
        # the one part of the project overview
        part = DocumentPart("the whole project", lambda: build_repository_overview_prompt(notes), DOCUMENT_REPOSITORY_INSTRUCTION, DOCUMENT_REPOSITORY_MAX_CHARS, DOCUMENT_REPOSITORY_SENTENCES, lambda text: None)
        try:
            text = self.get_text(part)
        except KeyboardInterrupt:
            # Ctrl+C stops here too
            self.say("\nDocumenting stopped early by the user.")
            self.abort_reason = "it was stopped by the user"
            text = ""
        return text, self.result

# Join clauses in plain words ("a", "a and b", "a, b and c")
def join_clauses(clauses: list[str]) -> str:
    if len(clauses) <= 1:
        return "".join(clauses)
    return ", ".join(clauses[:-1]) + " and " + clauses[-1]

# Say in one sentence what a change to a file could affect (written by code from the facts, so it can not be wrong about them)
def describe_implications(facts: FileFacts, is_repository: bool) -> str:
    # the files that use it
    if facts.used_by:
        used_by_names = []
        for path in facts.used_by[:DOCUMENT_MAX_NAMED_FILES]:
            used_by_names.append(path.name)
        text = f"A change can affect the {len(facts.used_by)} file{'s' if len(facts.used_by) != 1 else ''} that use it ({', '.join(used_by_names)})"
    elif is_repository:
        text = "No other file in the repository uses it"
    else:
        # only this file was read, so nothing is known about who uses it
        text = "Which other files use it is not known because only this file was read"
    # what its code does with files, the console or the network
    if facts.effects:
        text += f", and its code {join_clauses(facts.effects)}"
    return text + "."

# List names for a sentence: the first few, then how many more there are
def list_names(names: list[str]) -> str:
    shown = ", ".join(names[:DOCUMENT_MAX_LISTED_DEPENDENCIES])
    if len(names) > DOCUMENT_MAX_LISTED_DEPENDENCIES:
        shown += f" and {len(names) - DOCUMENT_MAX_LISTED_DEPENDENCIES} more"
    return shown

# Say in one sentence what a file depends on (written by code, from the facts)
def describe_dependencies(facts: FileFacts, is_repository: bool) -> str:
    clauses: list[str] = []
    # what it needs from outside the repository (when only one file was read, "outside the repository" means not found next to the file)
    if facts.external_dependencies:
        where = "from outside the repository" if is_repository else "(not found next to the file)"
        clauses.append(f"{list_names(facts.external_dependencies)} {where}")
    # what it needs from inside the repository
    if facts.internal_dependencies:
        internal_names = []
        for path in facts.internal_dependencies:
            internal_names.append(path.name)
        clauses.append(f"{list_names(internal_names)} from the repository")
    # headers and modules next to the file that were not among the files read
    if facts.local_dependencies:
        clauses.append(f"{list_names(facts.local_dependencies)} from the same folder")
    # a file with no dependencies says so
    if not clauses:
        return "It depends on nothing else."
    return f"It depends on {join_clauses(clauses)}."

# Say what a file contains in a sentence (written by code, used when the model's text is missing)
def describe_contents(facts: FileFacts) -> str:
    classes = 0
    for symbol in facts.symbols:
        if symbol.symbol_type in (SymbolType.CLASS, SymbolType.STRUCT, SymbolType.ENUM, SymbolType.NAMESPACE):
            classes += 1
    # what the file defines, as far as there is anything
    items: list[str] = []
    if facts.function_count:
        items.append(f"{facts.function_count} function{'s' if facts.function_count != 1 else ''}")
    if classes:
        items.append(f"{classes} class{'es' if classes != 1 else ''} or other type{'s' if classes != 1 else ''}")
    if facts.constants:
        items.append(f"{len(facts.constants)} constant{'s' if len(facts.constants) != 1 else ''}")
    defined = join_clauses(items) if items else "nothing with a name"
    return f"This {facts.language} file has {facts.line_count} lines and defines {defined}."

# Put the paragraph of a file together: what it is responsible for, how it works, what it depends on, then what a change affects. The model's text is used where it was written.
def compose_file_paragraph(document: DocumentedFile) -> str:
    # what it is responsible for and how it works (code's sentence when the model's is missing), then its dependencies, then its implications
    parts = [document.overview or describe_contents(document.facts), describe_dependencies(document.facts, document.is_repository)]
    if document.implications:
        parts.append(document.implications)
    return " ".join(parts)

# Write the entry of one important function: its signature, what it does and why it matters
def build_function_entry(item: ImportantFunction, text: str) -> str:
    chunk = item.chunk
    # what the model wrote, or a note that it is missing
    what = text if text else f"_{NOT_WRITTEN_TEXT}_"
    # why it matters is written by code from the reasons found
    why = " ".join(item.reasons) if item.reasons else "It is one of the larger functions of the file."
    entry = f"- **`{chunk.chunk_name}`** (lines {chunk.start_line}-{chunk.end_line}): `{chunk.signature or chunk.chunk_name}`\n  - What it does: {what}"
    # the functions it calls are found by code, so they show whether the sentence above matches the code
    if item.callee_names:
        entry += f"\n  - Calls: {', '.join(item.callee_names)}"
    return entry + f"\n  - What this funtcion affects: {why}"

# Write the documentation of one file as markdown
def build_file_section(document: DocumentedFile, shown_name: str) -> list[str]:
    facts = document.facts
    # the heading and the paragraph
    lines = [f"## {shown_name}", "", compose_file_paragraph(document), ""]
    # say if the model's text for the file is missing
    if facts.has_content() and not document.overview:
        lines.extend([f"_Summary of the file: {NOT_WRITTEN_TEXT}_", ""])
    # the important functions, each with what it does and why it matters
    if facts.important_functions:
        lines.extend(["**Important functions**", ""])
        for item in facts.important_functions:
            lines.append(build_function_entry(item, document.function_texts.get(item.chunk.chunk_id, "")))
        lines.append("")
    # everything the file defines, by signature
    listed = []
    for symbol in facts.symbols:
        if symbol.signature:
            listed.append(symbol)
    if listed:
        lines.extend(["**Everything it defines**", ""])
        for symbol in listed[:DOCUMENT_MAX_LISTED_SYMBOLS]:
            lines.append(f"- `{symbol.signature}`")
        # say how many are left out
        if len(listed) > DOCUMENT_MAX_LISTED_SYMBOLS:
            lines.append(f"- ...and {len(listed) - DOCUMENT_MAX_LISTED_SYMBOLS} more")
        lines.append("")
    # its constants and settings, by name
    if facts.constants:
        lines.extend(["**Constants**", ""])
        for name in facts.constants[:DOCUMENT_MAX_LISTED_SYMBOLS]:
            lines.append(f"- `{name}`")
        if len(facts.constants) > DOCUMENT_MAX_LISTED_SYMBOLS:
            lines.append(f"- ...and {len(facts.constants) - DOCUMENT_MAX_LISTED_SYMBOLS} more")
        lines.append("")
    return lines

# Draw which file includes or imports which, as a Mermaid diagram (an arrow means "uses")
def build_dependency_diagram(plan: list[FileFacts]) -> list[str]:
    # one arrow for each file a file uses
    edges = []
    for facts in plan:
        for dependency in facts.internal_dependencies:
            edges.append((facts.file_path, dependency))
    # with no arrows there is nothing to draw
    if not edges:
        return []
    # every file in an arrow gets a short id, because paths do not work as ids
    diagram_paths = set()
    for start, end in edges:
        diagram_paths.add(start)
        diagram_paths.add(end)
    ids = {}
    for index, path in enumerate(sorted(diagram_paths)):
        ids[path] = f"f{index}"
    lines = ["```mermaid", "graph LR"]
    # draw the arrows, up to the limit
    for start, end in edges[:DOCUMENT_MAX_DIAGRAM_EDGES]:
        lines.append(f'    {ids[start]}["{start.name}"] --> {ids[end]}["{end.name}"]')
    lines.append("```")
    # say if arrows are left out
    if len(edges) > DOCUMENT_MAX_DIAGRAM_EDGES:
        lines.append(f"_Only the first {DOCUMENT_MAX_DIAGRAM_EDGES} of {len(edges)} arrows are drawn._")
    return lines

# Write the overview of a whole repository: the model's sentences if there are any, then what the code found out
def build_repository_paragraph(plan: list[FileFacts], model_text: str) -> str:
    # the languages and the size of the repository
    language_set = set()
    line_total = 0
    for facts in plan:
        if facts.has_content():
            language_set.add(facts.language)
        line_total += facts.line_count
    languages = sorted(language_set)
    sentences = [f"The repository has {len(plan)} code files ({', '.join(languages)}) and about {line_total} lines."] if plan else []
    # where it starts
    entry_points = get_entry_points(plan)
    if entry_points:
        entry_texts = []
        for path, name in entry_points:
            entry_texts.append(f"{name} in {path.name}")
        sentences.append(f"It starts from {', '.join(entry_texts)}.")
    # the files the others use most
    core = get_most_used_files(plan, DOCUMENT_MAX_CORE_FILES)
    if core:
        core_texts = []
        for facts in core:
            core_texts.append(f"{facts.file_path.name} (used by {len(facts.used_by)})")
        sentences.append(f"The files that the others use most are {', '.join(core_texts)}.")
    # what it needs from outside
    external = get_repository_external_dependencies(plan)
    if external:
        sentences.append(f"Outside the repository it depends on {', '.join(external)}.")
    # the model's sentences come first
    return " ".join(([model_text] if model_text else []) + sentences)

# Describe how documenting went in a few lines: the files, the model's parts, and the time
def describe_documentation(documented: list[DocumentedFile], repository_result: DocumentationResult | None, skipped_files: list[str], file_count: int, budget_seconds: float, max_rows: int | None) -> str:
    # add up the results of every file and of the project overview
    total = DocumentationResult()
    for document in documented:
        total.add(document.result)
    if repository_result is not None:
        total.add(repository_result)
    # the files that had something for the model to describe
    with_text = []
    for document in documented:
        if document.facts.has_content():
            with_text.append(document)
    lines = [
        f"Files: {file_count} found, {len(with_text)} documented, {len(documented) - len(with_text)} with nothing to describe, {len(skipped_files)} could not be read.",
        f"Parts written by the model: {total.written} of {total.planned}, {total.not_written()} not written, {total.unusable} answers thrown away, {total.failed} failed.",
        f"Time spent on the model: {total.model_seconds / SECONDS_PER_MINUTE:.1f} minutes (guideline {describe_duration(budget_seconds)})."
    ]
    # why anything stopped early (the same reason for many files is said once)
    stop_reasons = []
    for document in documented:
        if document.result.stop_reason:
            stop_reasons.append(document.result.stop_reason)
    if stop_reasons:
        lines.append("Stopped early: " + describe_stop_reasons(stop_reasons) + ".")
    # tell the user how to finish what is missing
    if total.not_written() > 0:
        lines.append("Raise the time guideline in config.py, or document fewer files at a time, to write the parts that are missing.")
    # a table of the files with model text
    if with_text:
        shown = with_text if max_rows is None else with_text[:max_rows]
        lines.extend(["", "| File | Parts written | Not written |", "| --- | --- | --- |"])
        for document in shown:
            lines.append(f"| {document.facts.file_path.name} | {document.result.written} of {document.result.planned} | {document.result.not_written()} |")
        # say how many rows are left out
        if len(shown) < len(with_text):
            lines.append(f"| ...and {len(with_text) - len(shown)} more files (all of them are in the documentation) | | |")
    return "\n".join(lines)

# Write the whole documentation as markdown lines. For a repository, file names are shown relative to the repository folder.
def build_documentation_markdown(target_path: Path, is_repository: bool, documented: list[DocumentedFile], repository_text: str, summary: str, skipped_files: list[str]) -> list[str]:
    plan = []
    for document in documented:
        plan.append(document.facts)
    lines = [f"# Documentation: {target_path}", ""]
    # a repository starts with its overview and a diagram of which file uses which
    if is_repository:
        lines.extend(["## Overview", "", build_repository_paragraph(plan, repository_text), ""])
        diagram = build_dependency_diagram(plan)
        if diagram:
            lines.extend(["## Which file uses which", ""] + diagram + [""])
        lines.extend(["## Files", ""])
    # then one section for each file
    for document in documented:
        try:
            # the name relative to the repository folder, so that two files with the same name can be told apart
            shown_name = str(document.facts.file_path.relative_to(target_path)) if is_repository else document.facts.file_path.name
        except ValueError:
            # a path outside the folder is shown whole
            shown_name = str(document.facts.file_path)
        section = build_file_section(document, shown_name)
        # a file section is a level lower inside a repository's "Files"
        for line in section:
            if is_repository and line.startswith("## "):
                lines.append("#" + line)
            else:
                lines.append(line)
    # files that could not be read
    if skipped_files:
        lines.extend(["## Files that could not be read", ""] + skipped_files + [""])
    # how it was written, and what to trust
    lines.extend(["## How this was written", "", summary, "", "_The sentences about what a file is for and what each function does were written by a small local model and have to be checked. The dependencies, the files that use a file, what a change could affect, and the reasons a function matters were found by code._"])
    return lines

# Write the documentation to a markdown file. Returns None, or an Exception if it could not be written.
def write_documentation(markdown_path: Path, lines: list[str]) -> None | Exception:
    try:
        # make the folder if it does not exist
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        # write the lines
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        return Exception(f"Unable to write the documentation to {markdown_path}. Error: {e}")

# Document chunked code with the model and write it to the markdown file if a path was given (otherwise it is shown).
# chunked is given for a repository (None for a single file). Returns None, or an Exception if the documentation could not be done.
def document_chunks(target_path: Path, repository_data: RepositoryData, model: LocalModel, markdown_path: Path | None, chunked: ChunkedRepository | None = None) -> None | Exception:
    # a repository is documented differently from a single file
    is_repository = chunked is not None
    # document every file
    writer = DocumentationWriter(model, repository_data)
    documented = writer.run(is_repository)
    total_budget = float(DOCUMENT_REPOSITORY_TIME_BUDGET_SECONDS if is_repository else DOCUMENT_TIME_BUDGET_SECONDS)
    repository_text = ""
    repository_result: DocumentationResult | None = None
    # the overview of a whole project is written last, from what the files are for
    if is_repository:
        spent = 0.0
        for document in documented:
            spent += document.result.model_seconds
        repository_text, repository_result = writer.document_repository(documented, spent, total_budget)
    # the files that could not be read, and how many files there were
    skipped_files = chunked.skipped_files if chunked is not None else []
    file_count = chunked.file_count if chunked is not None else len(documented)
    # a brief summary of how it went
    summary = describe_documentation(documented, repository_result, skipped_files, file_count, total_budget, DOCUMENT_SUMMARY_MAX_FILE_ROWS)
    print(summary)
    # the documentation file has the summary in full
    full_summary = describe_documentation(documented, repository_result, skipped_files, file_count, total_budget, None)
    lines = build_documentation_markdown(target_path, is_repository, documented, repository_text, full_summary, skipped_files)
    # with no markdown file the documentation is shown instead
    if markdown_path is None:
        print("\n" + "\n".join(lines))
        return None
    # otherwise it is written to the file
    written = write_documentation(markdown_path, lines)
    if isinstance(written, Exception):
        return written
    print(f"Wrote the documentation to {markdown_path}")
    return None