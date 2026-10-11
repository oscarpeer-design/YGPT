# This file decides what is worth saying when code is documented: which files depend on which, and which functions matter most.
# It does NOT parse anything. Everything that needs the syntax tree (includes, imports, the names a function calls, signatures) is read by code_chunker.py,
# and this file only counts, looks up and ranks what the chunker found. It knows nothing about the model: it gathers the facts that the model is given and
# the reasons why a function matters, so that the model only has to put them into words (see doc_writer.py).

# Import external dependencies
# dataclass: used to create struct-like classes
from dataclasses import dataclass
# Path: library used to find file paths on computer
from pathlib import Path
# re: regular expressions, used to find effects such as reading files in the text of code
import re

# Import internal dependencies
# the chunk structures, and the names of the two languages
from code_chunker import CodeChunk, RepositoryData, SymbolType, LANGUAGE_CPP, LANGUAGE_PYTHON
# strip_comments: removes comments from code so that commented-out code is not counted
from chunk_risk import strip_comments
# get_container_ids: finds the chunks that contain other chunks (such as a class that contains methods)
from context_builder import get_container_ids
# the settings that are shared with the rest of the program
from config import DOCUMENT_FUNCTIONS_PER_FILE, DOCUMENT_MAX_CALLEE_SIGNATURES, DOCUMENT_MAX_NAMED_CALLERS

# Define how functions are ranked. The score only decides which functions are documented first; it is not shown.
# A function name shared by more than this many functions says nothing about which one is called, so such names are ignored
IMPORTANCE_MAX_SAME_NAME = 3
# A name shorter than this is too common to tell which function is called
IMPORTANCE_MIN_NAME_CHARS = 4
# Points for each function that calls the function, and the most points that callers can give
IMPORTANCE_POINTS_PER_CALLER = 2
IMPORTANCE_MAX_CALLER_POINTS = 10
# Points for a function that other files can use
IMPORTANCE_POINTS_PUBLIC = 2
# Points for the function a program starts from
IMPORTANCE_POINTS_ENTRY_POINT = 6
# One point for every this many lines of a function, up to the most points that size can give
IMPORTANCE_LINES_PER_POINT = 15
IMPORTANCE_MAX_SIZE_POINTS = 2
# A function needs at least this many points to be documented
IMPORTANCE_MIN_SCORE = 1
# The names of the functions a program starts from
IMPORTANCE_ENTRY_POINT_NAMES = ("main", "wmain", "WinMain", "wWinMain")
# File endings of C++ header files (a function declared there is meant for other files)
IMPORTANCE_HEADER_ENDINGS = (".h", ".hpp")

# Define the effects that are reported for a file ("it reads or writes files"). Each is (what it does, regular expression): the effect is reported if the expression matches the code (comments removed).
# This is a simple pattern search like the risk signals: it says what the code touches, not whether that is a problem.
CPP_EFFECT_SIGNALS = [
    ("reads or writes files", r"\b(fopen|ifstream|ofstream|fstream|fread|fwrite|std::filesystem)\b"),
    ("prints to the console", r"\b(std::cout|std::cerr|printf|puts|fprintf)\b"),
    ("runs other programs", r"\b(system|popen|execv\w*|CreateProcess\w*)\s*\("),
    ("uses the network", r"\b(socket|connect|bind|listen|recv|send|curl_\w+)\s*\("),
    ("manages memory by hand", r"\b(malloc|calloc|realloc|free)\s*\(|\bnew\s+[\w:<>]+\s*[\[(]|\bdelete\b"),
    ("uses threads or locks", r"\b(std::thread|std::mutex|std::lock_guard|std::async|pthread_\w+)\b"),
    ("throws exceptions", r"\bthrow\b"),
]
PYTHON_EFFECT_SIGNALS = [
    ("reads or writes files", r"\b(open|read_text|write_text|shutil|makedirs|os\.remove)\s*\("),
    ("prints to the console", r"\bprint\s*\("),
    ("runs other programs", r"\b(subprocess|os\.system|Popen)\b"),
    ("uses the network", r"\b(requests|urllib|socket|http\.client)\b"),
    ("uses threads or processes", r"\b(threading|multiprocessing|asyncio)\b"),
    ("raises exceptions", r"\braise\b"),
    ("changes global state", r"\bglobal\s+\w+"),
]

# ImportantFunction: a function that is worth documenting, and why
@dataclass
class ImportantFunction:
    chunk: CodeChunk # the function
    score: int # how important it is (only used to put the functions in order)
    callers: list[CodeChunk] # the functions that call it
    reasons: list[str] # why it matters, in plain words, written by this file from facts
    callee_signatures: list[str] # the signatures of the functions of the repository that it calls
    callee_names: list[str] # the names of those functions, in the order they are called

# FileFacts: everything that is known about one file before the model is asked anything
@dataclass
class FileFacts:
    file_path: Path # the file
    language: str # the programming language
    line_count: int # the last line that a chunk of the file ends on
    symbols: list[CodeChunk] # every chunk of the file (classes, functions and so on) in source order
    external_dependencies: list[str] # what it includes or imports that is not in the repository
    local_dependencies: list[str] # headers (quoted includes) and Python modules that sit next to the file but are not among the files being documented (for example when only one file is documented)
    internal_dependencies: list[Path] # the files of the repository that it includes or imports
    used_by: list[Path] # the files of the repository that include or import it
    effects: list[str] # what its code does that is worth knowing ("reads or writes files")
    important_functions: list[ImportantFunction] # the functions to document, most important first
    function_count: int # how many functions and methods the file has
    constants: list[str] # the names of its top-level constants and settings, in source order

    # whether there is anything in the file to describe
    def has_content(self) -> bool:
        return bool(self.symbols or self.constants)

# Get the last part of a function's name (Class::method gives method)
def get_short_name(chunk: CodeChunk) -> str:
    # split on the scope marker and keep what comes last
    return chunk.chunk_name.split("::")[-1]

# Find the functions that can be told apart by name: name -> the functions with that name. Names that are too short or shared by too many functions are left out.
def index_functions_by_name(repository_data: RepositoryData) -> dict[str, list[CodeChunk]]:
    # collect every function and method under its short name
    by_name: dict[str, list[CodeChunk]] = {}
    for chunk in repository_data.chunks:
        # only functions and methods can be called
        if chunk.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
            by_name.setdefault(get_short_name(chunk), []).append(chunk)
    # keep the names that say which function is meant
    return {name: chunks for name, chunks in by_name.items() if len(name) >= IMPORTANCE_MIN_NAME_CHARS and len(chunks) <= IMPORTANCE_MAX_SAME_NAME}

# Find the callers of every function: chunk id -> the functions that call it (each once, and never itself)
def find_callers(repository_data: RepositoryData, by_name: dict[str, list[CodeChunk]]) -> dict[int, list[CodeChunk]]:
    callers: dict[int, list[CodeChunk]] = {}
    for caller in repository_data.chunks:
        # only functions and methods make calls
        if caller.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
            continue
        # the names that the chunker found in the calls of this function
        for name in caller.called_names:
            for callee in by_name.get(name, []):
                # a function can only call a function written in the same language, and a call to itself does not count
                if callee.chunk_id == caller.chunk_id or callee.language != caller.language:
                    continue
                # remember this caller for the callee, once
                known = callers.setdefault(callee.chunk_id, [])
                if all(existing.chunk_id != caller.chunk_id for existing in known):
                    known.append(caller)
    return callers

# Say whether a Python module (a, or a.b.c) exists as a file or package in a folder
def is_local_python_module(folder: Path, target: str) -> bool:
    relative = Path(*target.split("."))
    return (folder / relative).with_suffix(".py").is_file() or (folder / relative / "__init__.py").is_file()

# Find which file each include or import refers to. Returns (internal: file -> files of the repository it uses, external: file -> what it uses outside the repository, local: file -> headers next to it that are not among the files being documented).
def resolve_includes(repository_data: RepositoryData, file_paths: list[Path]) -> tuple[dict[Path, list[Path]], dict[Path, list[str]], dict[Path, list[str]]]:
    # every file starts with nothing found
    internal: dict[Path, list[Path]] = {path: [] for path in file_paths}
    external: dict[Path, list[str]] = {path: [] for path in file_paths}
    local: dict[Path, list[str]] = {path: [] for path in file_paths}
    # look files up by their name (util.h) and by their name without the ending (util)
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in file_paths:
        by_name.setdefault(path.name, []).append(path)
        by_stem.setdefault(path.stem, []).append(path)
    for include in repository_data.includes:
        # the files of the repository that this include or import could mean
        matches: list[Path] = []
        if include.file_path.suffix == ".py":
            # import a.b.c refers to c.py, and the folders before it (a and b) must match the folders of the file
            parts = include.target.split(".")
            wanted_folders = parts[:-1]
            for path in by_stem.get(parts[-1], []):
                # the last folders of the file's path must be the wanted ones (nothing is wanted for a plain import)
                if not wanted_folders or path.parent.parts[len(path.parent.parts) - len(wanted_folders):] == tuple(wanted_folders):
                    matches.append(path)
        else:
            # #include "dir/header.h" refers to header.h, and the folders before it must match
            target_parts = Path(include.target).parts
            for path in by_name.get(target_parts[-1], []):
                # the end of the file's path must be the path that was written
                if path.parts[len(path.parts) - len(target_parts):] == target_parts:
                    matches.append(path)
        # a file never depends on itself
        matches = [path for path in matches if path != include.file_path]
        if matches:
            # the include is a file of the repository
            for path in matches:
                if path not in internal[include.file_path]:
                    internal[include.file_path].append(path)
        elif include.file_path.suffix == ".py" and is_local_python_module(include.file_path.parent, include.target):
            # a module that sits next to the file is the user's own code, not an outside library
            name = include.target.split(".")[0]
            if name not in local[include.file_path]:
                local[include.file_path].append(name)
        elif not include.is_system and include.file_path.suffix != ".py" and (include.file_path.parent / include.target).is_file():
            # a quoted include of a header that exists next to the file is the user's own code, not an outside library
            if include.target not in local[include.file_path]:
                local[include.file_path].append(include.target)
        else:
            # the include is outside the repository, named by its first part (os.path is os)
            name = include.target if include.file_path.suffix != ".py" else include.target.split(".")[0]
            if name not in external[include.file_path]:
                external[include.file_path].append(name)
    return internal, external, local

# Find the effects in the code of a file (each once, in the order of the signals)
def find_effects(chunks: list[CodeChunk], language: str, container_ids: set[int]) -> list[str]:
    # the signals of the language of the file (a language without signals has no effects to find)
    signals = CPP_EFFECT_SIGNALS if language == LANGUAGE_CPP else PYTHON_EFFECT_SIGNALS if language == LANGUAGE_PYTHON else []
    # all the code of the file without comments; a chunk that contains other chunks would repeat the code inside them, so it is left out
    code = "\n".join(strip_comments(chunk.source, language) for chunk in chunks if chunk.chunk_id not in container_ids)
    # an effect is reported if its pattern is found anywhere in that code
    return [description for description, pattern in signals if re.search(pattern, code)]

# Decide whether other files can use a function. Returns True, False, or None if it can not be told (a C++ method may be private, which needs the access labels).
def is_public_function(chunk: CodeChunk) -> bool | None:
    # by convention a Python name that starts with an underscore is private
    if chunk.language == LANGUAGE_PYTHON:
        return not get_short_name(chunk).startswith("_")
    # a C++ function that is static is only for its own file; any other function can be used by other files
    if chunk.symbol_type == SymbolType.FUNCTION:
        return re.search(r"\bstatic\b", chunk.signature) is None
    # a C++ method may be private, and that can not be told here
    return None

# Describe in plain words where a function is called from, naming a few callers
def describe_callers(function: CodeChunk, callers: list[CodeChunk]) -> str:
    # the callers that are in other files
    other_files = {caller.file_path for caller in callers if caller.file_path != function.file_path}
    # the first few callers, each with its file
    named = ", ".join(f"{caller.chunk_name} in {Path(caller.file_path).name}" for caller in callers[:DOCUMENT_MAX_NAMED_CALLERS])
    # say how many more there are if not all are named
    more = f" and {len(callers) - DOCUMENT_MAX_NAMED_CALLERS} more" if len(callers) > DOCUMENT_MAX_NAMED_CALLERS else ""
    # say how many of them are in other files, if any are
    where = f", {len(other_files)} of them in other files" if other_files else ""
    return f"It is called by {len(callers)} function{'s' if len(callers) != 1 else ''} ({named}{more}){where}."

# Say what kind of special member a C++ function is: "constructor", "destructor", or "" for any other function.
# The chunk's name is only the short name (CSR::CSR is named CSR), so the class is read from the signature.
def get_special_member(chunk: CodeChunk) -> str:
    # only C++ has them
    if chunk.language != LANGUAGE_CPP:
        return ""
    # the first qualified name before the parameters (Class::Name( or Class::~Name()
    match = re.search(r"(\w+)::(~?)(\w+)\s*\(", chunk.signature)
    if match is None:
        return ""
    # a name that starts with ~ is a destructor, and a function named like its class is a constructor
    if match.group(2):
        return "destructor"
    if match.group(1) == match.group(3):
        return "constructor"
    return ""

# Describe in plain words why other files can use a function
def describe_public(chunk: CodeChunk) -> str:
    # code that creates or destroys an object calls these, and that can not be seen from the names of called functions
    special = get_special_member(chunk)
    if special == "constructor":
        return "It is a constructor, so it runs whenever an object of the class is created."
    if special == "destructor":
        return "It is a destructor, so it runs whenever an object of the class is destroyed."
    # a Python function is public by convention
    if chunk.language == LANGUAGE_PYTHON:
        return "No caller was found in the code that was read. It is public: its name does not start with an underscore."
    # a function in a header file is meant to be used by the files that include it
    if Path(chunk.file_path).suffix in IMPORTANCE_HEADER_ENDINGS:
        return "No caller was found in the code that was read. It is defined in a header file, so any file that includes it can use it."
    # any other C++ function that is not static can be used by other files
    return "No caller was found in the code that was read. It is not static, so other files can use it."

# Score a function and say why it matters. Returns (score, reasons).
def score_function(chunk: CodeChunk, callers: list[CodeChunk]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    # the function a program starts from
    if get_short_name(chunk) in IMPORTANCE_ENTRY_POINT_NAMES:
        score += IMPORTANCE_POINTS_ENTRY_POINT
        reasons.append("It is the entry point of the program.")
    # functions that call it
    if callers:
        score += min(IMPORTANCE_MAX_CALLER_POINTS, IMPORTANCE_POINTS_PER_CALLER * len(callers))
        reasons.append(describe_callers(chunk, callers))
    # other files can use it (this is only said when nothing more telling is known, because it is true of most functions)
    if is_public_function(chunk):
        score += IMPORTANCE_POINTS_PUBLIC
        if not reasons:
            reasons.append(describe_public(chunk))
    # a long function does more
    lines = chunk.end_line - chunk.start_line + 1
    score += min(IMPORTANCE_MAX_SIZE_POINTS, lines // IMPORTANCE_LINES_PER_POINT)
    return score, reasons

# Find the functions of the repository that a function calls, in the order it calls them (each once)
def find_callees(chunk: CodeChunk, by_name: dict[str, list[CodeChunk]]) -> list[CodeChunk]:
    callees: list[CodeChunk] = []
    for name in chunk.called_names:
        for callee in by_name.get(name, []):
            # the callee must be another function in the same language that has a signature, and each one is kept once
            if callee.chunk_id != chunk.chunk_id and callee.language == chunk.language and callee.signature and callee not in callees:
                callees.append(callee)
    # only the first few are kept
    return callees[:DOCUMENT_MAX_CALLEE_SIGNATURES]

# Choose the functions to document in one file: the highest scoring ones, most important first
def choose_important_functions(functions: list[CodeChunk], callers: dict[int, list[CodeChunk]], by_name: dict[str, list[CodeChunk]]) -> list[ImportantFunction]:
    scored: list[ImportantFunction] = []
    for function in functions:
        # score the function from the functions that call it
        function_callers = callers.get(function.chunk_id, [])
        score, reasons = score_function(function, function_callers)
        # a function with too few points is not worth documenting
        if score >= IMPORTANCE_MIN_SCORE:
            callees = find_callees(function, by_name)
            scored.append(ImportantFunction(function, score, function_callers, reasons, [callee.signature for callee in callees], list(dict.fromkeys(get_short_name(callee) for callee in callees))))
    # highest score first; for equal scores, source order so that the result is stable
    scored.sort(key=lambda item: (-item.score, item.chunk.start_line))
    # only the best few are kept
    return scored[:DOCUMENT_FUNCTIONS_PER_FILE]

# Gather the facts about every file of the repository, in a stable order (sorted by path)
def build_documentation_plan(repository_data: RepositoryData) -> list[FileFacts]:
    # every file that has a chunk, an include or a constant
    file_paths = sorted({chunk.file_path for chunk in repository_data.chunks} | {include.file_path for include in repository_data.includes} | {constant.file_path for constant in repository_data.constants})
    # facts that need the whole repository: containers, functions by name, callers, and where includes point
    container_ids = get_container_ids(repository_data)
    by_name = index_functions_by_name(repository_data)
    callers = find_callers(repository_data, by_name)
    internal, external, local = resolve_includes(repository_data, file_paths)
    # which files use each file: the reverse of what each file uses
    used_by: dict[Path, list[Path]] = {path: [] for path in file_paths}
    for path, dependencies in internal.items():
        for dependency in dependencies:
            used_by[dependency].append(path)
    # the chunks of each file
    chunks_by_file: dict[Path, list[CodeChunk]] = {path: [] for path in file_paths}
    for chunk in repository_data.chunks:
        chunks_by_file[chunk.file_path].append(chunk)
    constants_by_file: dict[Path, list] = {path: [] for path in file_paths}
    for constant in repository_data.constants:
        constants_by_file[constant.file_path].append(constant)
    plan: list[FileFacts] = []
    for path in file_paths:
        # the chunks of the file in source order
        chunks = sorted(chunks_by_file[path], key=lambda chunk: (chunk.start_line, chunk.chunk_id))
        functions = [chunk for chunk in chunks if chunk.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)]
        # a file has one language: its chunks say which (a file without chunks has no effects to find, so its ending is enough)
        language = chunks[0].language if chunks else (LANGUAGE_PYTHON if path.suffix == ".py" else LANGUAGE_CPP)
        plan.append(FileFacts(
            file_path=path,
            language=language,
            line_count=repository_data.line_counts.get(path, max((chunk.end_line for chunk in chunks), default=0)),
            symbols=chunks,
            external_dependencies=sorted(external[path]),
            local_dependencies=sorted(local[path]),
            internal_dependencies=sorted(internal[path]),
            used_by=sorted(used_by[path]),
            effects=find_effects(chunks, language, container_ids),
            important_functions=choose_important_functions(functions, callers, by_name),
            function_count=len(functions),
            constants=[constant.name for constant in sorted(constants_by_file[path], key=lambda constant: constant.line)]
        ))
    return plan

# Find the files that the most other files use (the core of the repository), most used first
def get_most_used_files(plan: list[FileFacts], count: int) -> list[FileFacts]:
    # only files that something uses
    used = [facts for facts in plan if facts.used_by]
    # most used first; for equal numbers, by path so that the result is stable
    used.sort(key=lambda facts: (-len(facts.used_by), str(facts.file_path)))
    return used[:count]

# Find everything that the repository includes or imports from outside itself (each once, sorted)
def get_repository_external_dependencies(plan: list[FileFacts]) -> list[str]:
    return sorted({dependency for facts in plan for dependency in facts.external_dependencies})

# Find the places the program starts from: (file, function) for every entry point
def get_entry_points(plan: list[FileFacts]) -> list[tuple[Path, str]]:
    entry_points: list[tuple[Path, str]] = []
    for facts in plan:
        for symbol in facts.symbols:
            # a function with one of the entry point names
            if symbol.symbol_type == SymbolType.FUNCTION and get_short_name(symbol) in IMPORTANCE_ENTRY_POINT_NAMES:
                entry_points.append((facts.file_path, symbol.chunk_name))
    return entry_points