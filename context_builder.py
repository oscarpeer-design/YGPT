# This file builds context which is sent to LLM queries

# Import external dependencies
# dataclass, replace: used to make struct-like classes and to make edited copies of them
from dataclasses import dataclass, replace
# os: used to swap a finished file into place
import os
# Path: library used to find file paths on computer
from pathlib import Path

# Import internal dependencies
from code_chunker import CodeChunk, RepositoryData, RelationshipType
from code_chunk_utils import get_chunk_text, get_text_digest
from config import *

# Takes a list of retrieved chunks and builds a context string from that
def build_context(retrieved_chunks: list[CodeChunk]) -> str:
    # Check retrived_chunks has information we need
    if not retrieved_chunks:
        raise ValueError(f"Expected 'retrieved_chunks' to have enough context. Instead got {retrieved_chunks}.")
    # Initialise no context parts
    context_parts = []
    # Go through each retrieved chunk and get formatted chunk text
    for chunk in retrieved_chunks:
        context_parts.append(get_chunk_text(chunk))
    # Return this context as a single string
    return "\n\n".join(context_parts)


# Builds documentation prompt from context
def build_documentation_prompt(context: str) -> str:
    return (
        f"{DOCUMENTATION_INSTRUCTION}\n\n"
        f"CODEBASE CONTEXT:\n"
        f"{context}"
    )

# Builds review prompt from context
def build_review_prompt(context: str) -> str:
    return (
        f"{CODE_REVIEW_INSTRUCTION}\n\n"
        f"CODEBASE CONTEXT:\n"
        f"{context}"
    )


# --- Reviewing a file one chunk at a time ---
# The small model cannot read a whole file at once, so each chunk is reviewed in its own small prompt.
# Everything that decides what goes into those prompts (and which chunks are reviewed first) is in this file.

# ReviewHint: a line that looks suspicious, shown to the model so that it checks that line instead of searching on its own
@dataclass
class ReviewHint:
    line: int # real line number in the source file
    text: str # the line of code (shortened for display)
    description: str # why the line looks suspicious
    points: int # how strong the signal is (used to rank reviews and to pick the strongest hints)

# ContextLine: a line from elsewhere in a function that uses the same names as a flagged line (where a value comes from, or a check on it)
@dataclass
class ContextLine:
    line: int # real line number in the source file
    text: str # the line of code (shortened for display)

# Get the instruction for a kind of review
def get_review_instruction(kind: str) -> str:
    # performance reviews have their own instruction; everything else is a security review
    if kind == REVIEW_KIND_PERFORMANCE:
        return CODE_REVIEW_PERFORMANCE_CHUNK_INSTRUCTION
    return CODE_REVIEW_SECURITY_CHUNK_INSTRUCTION

# How many characters a review prompt may use so that the prompt AND the model's answer both fit in the context window
def get_prompt_char_budget() -> int:
    # the prompt and the answer must both fit in the context window
    tokens_available = MODEL_CTX - REVIEW_N_PREDICT
    # we also cap the prompt so that each review stays quick
    tokens_for_prompt = min(tokens_available, REVIEW_MAX_PROMPT_TOKENS)
    # turn tokens into characters (an estimate), keeping some room to spare
    return int(tokens_for_prompt * CHARS_PER_TOKEN * PROMPT_SAFETY_FACTOR)

# Windows line endings (\r\n) only waste space in a prompt
def normalise_line_endings(text: str) -> str:
    # replace each \r\n with \n
    return text.replace("\r\n", "\n")

# Show a chunk's source with its real line numbers at the left, so the model can say which line a problem is on (the code itself is unchanged)
def number_source_lines(chunk: CodeChunk) -> str:
    # split the code into lines
    lines = normalise_line_endings(chunk.source).split("\n")
    # work out how wide the largest line number is so that the numbers line up
    width = len(str(chunk.start_line + len(lines) - 1))
    # put each line's real line number in front of it
    numbered_lines = []
    for offset, line in enumerate(lines):
        numbered_lines.append(f"{chunk.start_line + offset:>{width}} | {line}")
    # join the lines back together
    return "\n".join(numbered_lines)

# Describe where a chunk is in words, e.g. "function 'parse', lines 10-42"
def describe_chunk_location(chunk: CodeChunk) -> str:
    return f"{chunk.symbol_type.value} '{chunk.chunk_name}', lines {chunk.start_line}-{chunk.end_line}"

# Show a chunk as background code
def format_related_chunk(chunk: CodeChunk, target: CodeChunk) -> str:
    # a heading saying what the code is (and which file it is in, if that is not the file under review), then the code
    where = describe_chunk_location(chunk) if chunk.file_path == target.file_path else f"{describe_chunk_location(chunk)} of {Path(chunk.file_path).name}"
    return f"[{where}]\n{normalise_line_endings(chunk.source)}"

# Build the heading that goes above the code under review
def build_review_header(target: CodeChunk, enclosing: CodeChunk | None) -> str:
    # say which file the code is in and where
    header = f"CODE UNDER REVIEW ({describe_chunk_location(target)} of {Path(target.file_path).name})"
    # say what contains the code, if something does
    if enclosing is not None:
        header += f"\nIt is defined inside {enclosing.symbol_type.value} '{enclosing.chunk_name}'."
    return header

# Show the flagged lines by number and reason. The code of each line is already shown with its number, and repeating it would only tempt the model to copy it.
def format_review_hints(hints: list[ReviewHint]) -> str:
    heading = "\nFLAGGED LINES (from a simple pattern search, they may be false alarms):\n"
    hint_lines = [f"line {hint.line}: {hint.description}" for hint in hints]
    return heading + "\n".join(hint_lines) + "\n"

# Show lines from elsewhere in the function that use the same names as a flagged line
def format_context_lines(context_lines: list[ContextLine]) -> str:
    heading = "\nOTHER LINES THAT USE THE SAME NAMES (from elsewhere in the function):\n"
    return heading + "\n".join(f"line {context.line}: {context.text}" for context in context_lines) + "\n"

# Build the prompt for reviewing ONE chunk, adding lines that use the same names and related chunks as background for as long as they fit.
# Returns (prompt, the related chunks that were included), or None if the code and the flagged lines do not fit in the prompt.
def build_chunk_review_prompt(
    target: CodeChunk,
    related_candidates: list[CodeChunk],
    enclosing: CodeChunk | None,
    kind: str = REVIEW_KIND_SECURITY,
    hints: list[ReviewHint] | None = None,
    context_lines: list[ContextLine] | None = None
) -> tuple[str, list[CodeChunk]] | None:
    # find how much room the prompt has
    char_budget = get_prompt_char_budget()
    # start with the instruction, then the code under review
    prompt = (
        f"{get_review_instruction(kind)}\n"
        f"{build_review_header(target, enclosing)}\n"
        f"{number_source_lines(target)}\n"
    )
    # if the chunk alone is too large it can not be reviewed in one prompt
    if len(prompt) > char_budget:
        return None
    # add the flagged lines: the model is asked about them, so the review can not be done without them
    if hints:
        prompt += format_review_hints(hints)
        if len(prompt) > char_budget:
            return None
    # add the lines that use the same names, if they fit (they help, but the code and the flagged lines matter more)
    if context_lines:
        context_block = format_context_lines(context_lines)
        if len(prompt) + len(context_block) <= char_budget:
            prompt += context_block
    # add related chunks (best match first) until we have enough of them or run out of room
    related_heading = "\nRELATED CODE (background only, do not review):\n"
    included: list[CodeChunk] = []
    for candidate in related_candidates:
        # stop when we have enough related chunks
        if len(included) >= REVIEW_RELATED_CHUNK_COUNT:
            break
        # work out how much room this chunk would take (the first one also needs the heading)
        block = format_related_chunk(candidate, target) + "\n\n"
        extra = len(block)
        if not included:
            extra += len(related_heading)
        # chunks are added whole or not at all: source code is never cut short
        if len(prompt) + extra > char_budget:
            continue
        # add the heading before the first related chunk
        if not included:
            prompt += related_heading
        prompt += block
        included.append(candidate)
    # remove trailing blank lines
    return prompt.rstrip() + "\n", included

# Cut the part of a chunk around some of its lines (the flagged lines) into windows. Each window holds REVIEW_FOCUS_RADIUS_LINES lines above and below a flagged line,
# and windows that touch are joined while they stay within REVIEW_FOCUS_MAX_LINES. Each window keeps its real line numbers.
# The chunk itself is not changed: each window is a copy.
def get_focus_windows(chunk: CodeChunk, focus_lines: list[int]) -> list[CodeChunk]:
    # split the code into lines, and find the real line number of the last one
    lines = normalise_line_endings(chunk.source).split("\n")
    last_line = chunk.start_line + len(lines) - 1
    # work out the first and last line of each window
    ranges: list[list[int]] = []
    for focus_line in sorted(set(focus_lines)):
        first = max(chunk.start_line, focus_line - REVIEW_FOCUS_RADIUS_LINES)
        last = min(last_line, focus_line + REVIEW_FOCUS_RADIUS_LINES)
        # join this window to the one before it if they touch and the joined window is not too long
        if ranges and first <= ranges[-1][1] + 1 and last - ranges[-1][0] + 1 <= REVIEW_FOCUS_MAX_LINES:
            ranges[-1][1] = max(ranges[-1][1], last)
        else:
            ranges.append([first, last])
    # build a copy of the chunk for each window
    windows = []
    for first, last in ranges:
        windows.append(replace(
            chunk,
            source="\n".join(lines[first - chunk.start_line:last - chunk.start_line + 1]),
            start_line=first,
            end_line=last
        ))
    return windows

# Build the grammar that forces the model's answer into the shape the review asks for: either NO_ISSUES_TEXT, or up to REVIEW_MAX_FINDINGS_PER_REVIEW findings,
# each naming one of the flagged lines with a reason and a fix of limited length. Naming only flagged lines means the model can not blame code that was not flagged.
def build_review_grammar(hints: list[ReviewHint]) -> str:
    # the line numbers the model may name
    line_choices = " | ".join(f'"{line}"' for line in sorted({hint.line for hint in hints}))
    # a finding, then up to the allowed number of further findings, nested so that each one needs the one before it
    findings = "finding" + " (finding" * (REVIEW_MAX_FINDINGS_PER_REVIEW - 1) + ")?" * (REVIEW_MAX_FINDINGS_PER_REVIEW - 1)
    # a reason or a fix is one line of limited length
    field = "[^\\n]{1," + str(REVIEW_FIELD_MAX_CHARS) + "}"
    return (
        f"root ::= none | {findings}\n"
        f'none ::= "{NO_ISSUES_TEXT}"\n'
        f'finding ::= "LINE: " line "\\nWHY: " field "\\nFIX: " field "\\n"\n'
        f"line ::= {line_choices}\n"
        f"field ::= {field}\n"
    )

# Get the ids of the chunks that have other chunks inside them
def get_container_ids(repository_data: RepositoryData) -> set[int]:
    container_ids = set()
    # a chunk is a container if it is the start of a CONTAINS relationship
    for relationship in repository_data.relationships:
        if relationship.relationship_type == RelationshipType.CONTAINS:
            container_ids.add(relationship.start_chunk_id)
    return container_ids

# Get the chunk that contains each chunk, as {chunk id: containing chunk}. (Chunk ids are not positions in the list of chunks, so chunks are looked up by id.)
def get_parent_chunks(repository_data: RepositoryData) -> dict[int, CodeChunk]:
    # look chunks up by id
    chunks_by_id = {chunk.chunk_id: chunk for chunk in repository_data.chunks}
    parents = {}
    # each CONTAINS relationship says which chunk is inside which
    for relationship in repository_data.relationships:
        if relationship.relationship_type == RelationshipType.CONTAINS and relationship.start_chunk_id in chunks_by_id:
            parents[relationship.end_chunk_id] = chunks_by_id[relationship.start_chunk_id]
    return parents

# --- Saving reviews so that an interrupted or repeated review picks up where it left off ---

# Work out where the saved review for a prompt lives. Everything that can change the model's answer is part of the name.
def get_review_cache_path(model_name: str, prompt: str, grammar: str) -> Path:
    # the folder sits next to this file
    cache_folder = Path(__file__).resolve().parent / REVIEW_CACHE_FOLDER_NAME
    # the name is a digest of the cache version, the model, its settings (including the repeat penalty), the grammar that shapes the answer and the prompt
    digest = get_text_digest([str(REVIEW_CACHE_VERSION), model_name, str(REVIEW_N_PREDICT), str(REVIEW_REPEAT_PENALTY), str(MODEL_CTX), str(TEMPERATURE), grammar, prompt])
    return cache_folder / f"{digest}.txt"

# Load a saved review. Returns None if there is no saved review, or an Exception if it can not be read.
def load_cached_review(cache_path: Path) -> str | None | Exception:
    try:
        # no saved review
        if not cache_path.is_file():
            return None
        return cache_path.read_text(encoding="utf-8")
    except Exception as e:
        return Exception(f"Unable to read the saved review {cache_path}. Error: {e}")

# Save a review. Returns None, or an Exception if it can not be saved (a failure to save must not stop a review).
def save_cached_review(cache_path: Path, response: str) -> None | Exception:
    try:
        # make the folder if it does not exist yet
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # write to a temporary file first, then swap it in, so that a half-written review is never read
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(response, encoding="utf-8")
        os.replace(temp_path, cache_path)
    except Exception as e:
        return Exception(f"Unable to save the review {cache_path}. Error: {e}")