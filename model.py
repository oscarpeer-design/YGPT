# This defines the local LLM used and its parameters

# External dependencies
# subprocess: allows running programs (like a local LLM) on a computer:
# We verify files before running to prevent dependency injection attacks
import subprocess
# Path: library used to find file paths on computer
from pathlib import Path
# dataclass: used to create struct-like classes
from dataclasses import dataclass
# os, tempfile: used to give llama-cli a grammar file and remove it afterwards
import os
import tempfile
# re: regular expressions, used to read the model's findings
import re

# Internal dependencies
from code_chunker import CodeChunk
from context_builder import ReviewHint
from config import *

# Text file from which we read the model_path
MODEL_PATH_FILE = "model_path.txt"

# Convert sizes from a number in bytes to a string in Gigabytes
def format_size_gb(size_bytes: int) -> str:
    one_gigabyte = 1024 ** 3
    size_gb = size_bytes / one_gigabyte
    return f"{size_gb:.2f} Gigabytes"

# Verify that the provided model path points to an actual local AI model
def validate_model_path(model_path: Path) -> Path | Exception:
    # Check if the model path exists
    if not model_path.exists():
        return Exception(f"Model does not exist at this location: {model_path}")
    # Check if the path points to a regular file
    if not model_path.is_file():
        return Exception(f"Model is not a regular file at this location: {model_path}")
    # Check if the file has the correct extension
    if model_path.suffix != ".gguf":
        return Exception(f"Expected a '.gguf' extension. Instead got {model_path.suffix}")
    # Check for an absolute path (required to prevent runtime issues)
    if not model_path.is_absolute():
        return Exception(f"Model path is not an absolute path: {model_path}")

    # Check that the model size is not too large (avoid system crash)
    model_size = model_path.stat().st_size
    if model_size > MAX_MODEL_SIZE:
        # Convert maximum model size to a readable string
        s_max_size = format_size_gb(MAX_MODEL_SIZE)
        # Convert actual model size to a readable string
        s_model_size = format_size_gb(model_size)
        # Warn that the model is oversized
        return Exception(
            f"""Model file exceeds the supported size limit.
            Expected {s_max_size} but got {s_model_size}
            For model at path {model_path}"""
        )

    # The model is valid
    return model_path

# extract path from text-file set up by user (model_path.txt)
def read_path_from_file() -> Path|Exception:
    # read the lines from the file
    try:
        with open(MODEL_PATH_FILE, 'r', encoding = 'utf-8') as file:
            # read the first line
            line = file.readline()
            # Check the line is not empty
            if line is None or line == "":
                return Exception(f"The first line of the config file, {MODEL_PATH_FILE}, is empty. The first line should be the absolute file path of the local LLM you wish to use.")
            # Get the first line which is a file path
            print(f"IT IS EXPECTED THAT THE FIRST LINE IN THIS FILE IS AN ABSOLUTE FILE PATH FOR A LOCAL LLM: {MODEL_PATH_FILE}\n")
            # Remove line endings
            line = line.strip()
            # Return the line as a path object
            path = Path(line)
            return path

    except Exception as e:
        return Exception(f"The attempt to read the file {MODEL_PATH_FILE} failed. Error: {e}")

# Gets the model name from path and runs model name extraction and file path validation
def get_and_validate_model_path() -> Path|None:
    # read the path from the file
    path = read_path_from_file()

    # check we found an actual path in the file (an Exception will be returned if it doesn't exist where required)
    if isinstance(path, Exception):
        # print the exception and return
        print(path)
        return

    # validate the path
    valid_path = validate_model_path(path)

    # check if the path is valid (an Exception will be returned if it is not)
    if isinstance(valid_path, Exception):
        # print the exception and return
        print(valid_path)
        return
    
    # the path is now read and validated
    return valid_path

# define class to run model
class LocalModel():
    def __init__(self, model_path:Path):
        # initialise model path
        self.model_path = model_path
        # get model name (stem of file path)
        self.model_name = model_path.stem
   
    # Extract response from llama-cli (ignore UI elements)
    def extract_model_response(self, output: str, prompt: str) -> str:
        response = output
        # Remove llama-cli's exit message (and anything after it)
        if "Exiting..." in response:
            response = response.split("Exiting...", 1)[0]
        # llama-cli shows the user's prompt after a '> ' marker. Short prompts are shown in full...
        prompt_marker = f"> {prompt}"
        # ...but long prompts are shortened and end with this marker
        truncated_marker = "... (truncated)"
        if prompt_marker in response:
            # the answer starts after the prompt
            response = response.split(prompt_marker, 1)[1]
        elif truncated_marker in response:
            # the answer starts after the shortened prompt
            response = response.split(truncated_marker, 1)[1]
        return response.strip()

    # run query through the model and return the model's response, or an Exception if the model could not be run
    # timeout (in seconds) defaults to MODEL_TIMEOUT_SECONDS, but a shorter one can be given
    # grammar (a llama.cpp grammar) forces the model's answer into a fixed shape
    # repeat_penalty (above 1.0) discourages the model from repeating itself
    def query_model(self, prompt: str, n_predict = 1024, timeout: float | None = None, grammar: str | None = None, repeat_penalty: float | None = None) -> str | Exception:
        # llama-cli reads a grammar from a file, so a grammar is written to a temporary file for the length of the call
        grammar_path = None
        try:
            grammar_arguments = []
            if grammar is not None:
                file_descriptor, grammar_path = tempfile.mkstemp(suffix=".gbnf", text=True)
                with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as grammar_file:
                    grammar_file.write(grammar)
                grammar_arguments = ["--grammar-file", grammar_path]
            # discourage repeating
            repeat_arguments = ["--repeat-penalty", str(repeat_penalty)] if repeat_penalty is not None else []
            # Call llama-cli to run local model
            result = subprocess.run(
                [
                    "llama-cli", # Llamma-cli: the program that will run the local model
                    "-m", str(self.model_path), # The path to the local LLM
                     "-p", prompt, # The prompt that will be run
                    "--temp", str(TEMPERATURE), # The temperature
                    "--ctx-size", str(MODEL_CTX), # The context size available
                    "--threads", str(CPU_THREADS_USED), # The number of threads used to write the answer
                    "--threads-batch", str(CPU_THREADS_USED), # The number of threads used to read the prompt (the slow part of a long prompt)
                    "--no-warmup", # Skip llama-cli's practice run when it starts (it starts again for every review)
                    "--no-display-prompt", # Don't include display prompt (less noise)
                    "--n-predict", str(n_predict), # The number of tokens consumed in the prediction
                    "--single-turn",
                    "--no-show-timings",
                    "--simple-io",
                    # By default llama-cli turns backslash sequences in the prompt into control characters
                    # (for example the \r in C:\Users\...\repos, or "\n".join(...) in source code), so we turn that off
                    "--no-escape",
                ] + grammar_arguments + repeat_arguments,
                text=True,
                # llama-cli writes UTF-8, so we decode it as UTF-8 rather than the Windows default
                encoding="utf-8",
                errors="replace",
                check=True,
                capture_output=True,
                timeout=MODEL_TIMEOUT_SECONDS if timeout is None else timeout
            )
            # Return the response without llama-cli's user interface text
            return self.extract_model_response(result.stdout, prompt)

        except Exception as e:
            # Give error message information to the caller
            return Exception(f"Tried to run the local model at path {self.model_path}. Got an error: {e}")

        finally:
            # remove the temporary grammar file
            if grammar_path is not None:
                try:
                    os.remove(grammar_path)
                except OSError:
                    pass

    # run query through the model (taking in a prompt and printing the output text)
    def run_query_through_model(self, prompt: str, n_predict = 1024) -> bool:
        # run the query
        response = self.query_model(prompt, n_predict)
        # Give error message information to user
        if isinstance(response, Exception):
            print(response)
            # We failed to run a query
            return False
        # Print result of query
        print(response)
        # We ran a query successfully
        return True

# ReviewFinding: one problem the model reported about a flagged line
@dataclass
class ReviewFinding:
    line: int # real line number of the flagged line
    code: str # the flagged line of code (taken from the source, not from the model)
    problem: str # what the model says is wrong
    fix: str # the model's suggested fix

# Find where complete sentences end in text. A full stop inside code (`j.m_data`) or between words and code does not end a sentence:
# the mark must be followed by a space or the end of the text, and must not be inside backticks.
def find_sentence_ends(text: str) -> list[int]:
    ends: list[int] = []
    inside_code = False
    for index, character in enumerate(text):
        # text between backticks is code
        if character == "`":
            inside_code = not inside_code
        # a sentence ends at . ! or ? that is outside code and followed by a space or the end of the text
        elif character in ".!?" and not inside_code and (index + 1 == len(text) or text[index + 1].isspace()):
            ends.append(index)
    return ends

# Clean up a reason or a fix written by the model: text that was cut off loses its unfinished end, and a sentence the model repeated is kept once
def tidy_field(text: str) -> str:
    text = text.strip()
    # long text that does not end like a sentence, or that ends inside a backtick, was cut off by the length limit (short text is a label such as "unsafe casts" and is left alone)
    was_cut_off = len(text) >= REVIEW_CUT_OFF_MIN_CHARS and (text[-1] not in ".!?" or text.count("`") % 2 == 1)
    if was_cut_off:
        ends = find_sentence_ends(text)
        if ends:
            # keep only the complete sentences
            text = text[:ends[-1] + 1]
        else:
            # no complete sentence: keep the words that came before the cut, without an open backtick or a dangling mark
            if text.count("`") % 2 == 1:
                text = text[:text.rfind("`")]
            text = text.rsplit(" ", 1)[0].rstrip(" ,;:(-") + "..."
    # keep each sentence once, in order
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if sentence not in kept:
            kept.append(sentence)
    return " ".join(kept)

# Normalise a reason so that it can be compared with a hint's description: lower case, no punctuation or extra spaces
def normalise_reason(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())

# Check whether a finding is worth showing: it must not argue against itself or repeat the instruction.
# A reason that matches the flag's description is kept on purpose, because a fix that fits the flagged line is still useful.
def is_informative(why: str, fix: str, description: str) -> bool:
    combined = f"{why} {fix}".lower()
    # a finding that says there is no problem (or repeats the instruction) is not a finding
    return not any(phrase in combined for phrase in REVIEW_DISCARDED_PHRASES)


# Read the model's answer. A finding is kept only if it names a flagged line and says something (see is_informative).
# The code shown for a finding is taken from the source, so the model can not misquote it.
# Returns (the findings that were kept, how many were thrown away).
def parse_review_findings(response: str, chunk: CodeChunk, hints: list[ReviewHint]) -> tuple[list[ReviewFinding], int]:
    # the lines of the chunk, so that the code of a finding can be shown, and why each flagged line was flagged
    chunk_lines = chunk.source.replace("\r\n", "\n").split("\n")
    descriptions = {hint.line: hint.description for hint in hints}
    findings: list[ReviewFinding] = []
    thrown_away = 0
    seen_lines: set[int] = set()
    # a finding is three lines: LINE, WHY, FIX
    pattern = re.compile(r"LINE:[ \t]*(\d+)[ \t]*\n[ \t]*WHY:[ \t]*(.+)\n[ \t]*FIX:[ \t]*(.+)")
    for match in pattern.finditer(response):
        line = int(match.group(1))
        # the model sometimes repeats itself, so each line is reported once
        if line in seen_lines:
            continue
        seen_lines.add(line)
        why = tidy_field(match.group(2))
        fix = tidy_field(match.group(3))
        # the model named a line that was not flagged, or the finding says nothing
        if line not in descriptions or not is_informative(why, fix, descriptions[line]):
            thrown_away += 1
            continue
        code = " ".join(chunk_lines[line - chunk.start_line].split())
        findings.append(ReviewFinding(line, code, why, fix))
    return findings, thrown_away