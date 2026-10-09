# This file decides which lines of code the model is asked about, and which reviews are done first.
# It is NOT a vulnerability scanner and it knows nothing about the model: it flags the lines that look risky or slow, and puts the reviews in order,
# so that when time is limited the most worthwhile reviews are done first. The model (given the code around each line) does the judging.

# Import external dependencies
# dataclass: used to create struct-like classes
from dataclasses import dataclass
# Path: library used to find file paths on computer
from pathlib import Path
# re: regular expressions, used to find risk signals in code
import re

# Import internal dependencies
from code_chunker import CodeChunk, RepositoryData, SymbolType
from context_builder import ReviewHint, ContextLine, build_chunk_review_prompt, get_focus_windows, get_container_ids
from config import (
    REVIEW_KIND_SECURITY, REVIEW_KIND_PERFORMANCE, REVIEW_HINTS_PER_SIGNAL, REVIEW_MAX_HINTS, REVIEW_HINT_MAX_CHARS,
    REVIEW_WHOLE_CHUNK_CHARS, REVIEW_MIN_HINT_POINTS, REVIEW_MAX_CONTEXT_LINES, REVIEW_IGNORED_NAMES
)

# Define how lines are flagged as risky. This is NOT a vulnerability scanner: it only decides which lines the model is asked about and the ORDER of the reviews.
# Each signal is (points, description, regular expression): a line is flagged if the expression matches it (comments removed), and the points rank the review
CPP_RISK_SIGNALS = [
    (3, "raw memory/C-string calls", r"\b(memcpy|memmove|memset|strcpy|strncpy|strcat|strncat|sprintf|vsprintf|gets|alloca|malloc|calloc|realloc|free)\s*\("),
    (3, "runs files/processes", r"\b(system|popen|execv\w*|execl\w*|fopen|CreateProcess\w*)\s*\("),
    (2, "parses numbers from text", r"\b(strtod|strtof|strtold|strtoll|strtoull|strtol|strtoul|atoi|atol|atof|stoi|stol|stoul|stoll|stoull|stod|stof)\s*\("),
    (2, "unsafe casts", r"\b(reinterpret_cast|const_cast)\b"),
    (2, "allocation sized by data", r"\.(resize|reserve)\s*\(|\bnew\s+[\w:<>]+\s*\["),
    (1, "narrowing casts", r"static_cast\s*<\s*(unsigned\s+)?(char|short|int|std::u?int(8|16|32)_t)\s*>"),
    (1, "assert is the only check", r"\bassert\s*\("),
    (1, "indexes with a variable", r"\w\s*\[\s*[A-Za-z_][^\]\n]*\]"),
    (1, "union / launder", r"\b(union|std::launder)\b"),
]
PYTHON_RISK_SIGNALS = [
    (4, "runs dynamic code", r"\b(eval|exec|compile|__import__)\s*\("),
    (4, "runs shell commands", r"\bos\.(system|popen)\s*\(|shell\s*=\s*True"),
    (4, "unsafe deserialisation", r"\b(pickle|cPickle|marshal|shelve|dill)\.loads?\s*\(|\byaml\.load\s*\("),
    (3, "extracts archives", r"\.(extractall|extract)\s*\("),
    (3, "disables TLS checks", r"verify\s*=\s*False"),
    (3, "builds SQL from strings", r"\.execute\s*\(\s*(f[\"']|[\"'][^\"']*[\"']\s*(%|\+|\.format))"),
    (3, "insecure temp file", r"\bmktemp\s*\("),
    (2, "runs processes", r"\bsubprocess\.\w+\s*\("),
    (1, "opens files", r"\bopen\s*\("),
    (1, "network access", r"\b(socket|requests|urllib|http\.client)\b"),
    (1, "weak hash/random", r"\b(md5|sha1)\s*\(|\brandom\.(random|randint|choice)\s*\("),
    (1, "assert is the only check", r"(?m)^\s*assert\b"),
    (1, "swallows exceptions", r"(?m)except\s*(Exception)?\s*:\s*(pass)?\s*$"),
]
# Points added when a function calls itself (hostile input can make it recurse without limit)
RISK_POINTS_RECURSION = 2
# Points added when a function's name suggests it handles data from outside the program
RISK_POINTS_OUTSIDE_DATA_NAME = 2
# Name words that suggest a function handles outside data (a word in a name matches if it starts with one of these)
RISK_OUTSIDE_DATA_NAME_STEMS = ("parse", "read", "load", "decode", "deserial", "from", "scan", "recv", "receive", "handle", "request", "input", "token", "extract", "unpack")
# Keywords that make code harder to reason about (counted to measure how many branches a chunk has)
RISK_BRANCH_WORDS = ("if", "for", "while", "switch", "case", "catch", "except", "elif")
# Number of branches from which a chunk gets 1 extra point, and the number from which it gets 2 extra points
RISK_SOME_BRANCHES = 8
RISK_MANY_BRANCHES = 20

# Define how chunks are checked for performance problems. These signals do not decide risk: they find lines that LOOK slow, with their real line numbers,
# so that the model is asked about specific lines instead of being asked to find problems on its own.
# Each signal is (points, description, regular expression, where, unless): the expression is matched against one line of code (comments removed) and
# 'where' says which lines count: "loop" (only lines inside a loop), "signature" (only the lines up to the start of the body) or "anywhere".
# If 'unless' is given, the signal is ignored in a chunk where that expression matches somewhere (for example a push_back is fine if the chunk also calls reserve).
CPP_PERFORMANCE_SIGNALS = [
    (3, "allocates memory on every loop iteration", r"\bnew\s+\w|\bstd::make_(unique|shared)\s*<", "loop", None),
    (3, "builds a regex or string stream inside a loop", r"\bstd::(regex|stringstream|ostringstream|istringstream)\b", "loop", None),
    (3, "searches linearly inside a loop (may be O(n^2))", r"\bstd::(find|find_if|count|count_if|search)\s*\(|\.(insert|erase)\s*\([^;]*\.begin\s*\(\)", "loop", None),
    (2, "string length recomputed inside a loop", r"\bstrlen\s*\(", "loop", None),
    (2, "grows a container in a loop without reserving space", r"\.(push_back|emplace_back)\s*\(", "loop", r"\.reserve\s*\("),
    (2, "builds a new container or string on every loop iteration", r"^\s*(const\s+)?std::(string|vector<[^;]*>|map<[^;]*>|unordered_map<[^;]*>|set<[^;]*>|unordered_set<[^;]*>|deque<[^;]*>|list<[^;]*>)\s+\w+\s*(=|\(|\{|;)", "loop", None),
    (1, "flushes output on every loop iteration", r"\bstd::endl\b", "loop", None),
    (2, "passes a large container by value (a copy is made)", r"\bstd::(string|vector|map|unordered_map|set|unordered_set|deque|list)\b(<[^()]*>)?\s+\w+\s*[,)]", "signature", None),
]
PYTHON_PERFORMANCE_SIGNALS = [
    (3, "runs a query or request inside a loop", r"\.(execute|executemany|fetchone|fetchall)\s*\(|\brequests\.\w+\s*\(|\burlopen\s*\(", "loop", None),
    (3, "opens or reads a file inside a loop", r"\bopen\s*\(|\.read_text\s*\(|\.read_bytes\s*\(", "loop", None),
    (3, "compiles a regex inside a loop", r"\bre\.compile\s*\(", "loop", None),
    (2, "builds a string by repeated concatenation", r"\b\w+\s*\+=\s*(f?[\"']|str\s*\()|\b(\w+)\s*=\s*\2\s*\+\s*(f?[\"']|str\s*\()", "loop", None),
    (3, "scans or sorts a whole collection inside a loop (may be O(n^2))", r"\.(index|count|remove)\s*\(|\.insert\s*\(\s*0\s*,|\b(sorted|sum|max|min)\s*\(", "loop", None),
    (2, "copies data inside a loop", r"\bcopy\.deepcopy\s*\(|\blist\s*\(|\bdict\s*\(|\.copy\s*\(", "loop", None),
    (2, "parses or formats JSON inside a loop", r"\bjson\.(loads|dumps)\s*\(", "loop", None),
    (1, "indexes a sequence instead of iterating it", r"\bfor\s+\w+\s+in\s+range\s*\(\s*len\s*\(", "anywhere", None),
]
# Points added when a loop sits inside another loop (the work grows with the product of both sizes), and the extra point when it sits three or more deep
RISK_POINTS_NESTED_LOOP = 2
RISK_POINTS_DEEP_LOOP = 1
# Points added when a function is called from inside a loop somewhere else (its cost is multiplied by the number of iterations)
RISK_POINTS_HOT_CALLEE = 2
# A function name is only trusted to find its callers if at most this many chunks share it and it has at least this many characters (so that 'get' or 'at' do not link everything)
RISK_HOT_CALLEE_MAX_SAME_NAME = 3
RISK_HOT_CALLEE_MIN_NAME_CHARS = 4

# ReviewUnit: one review of one piece of code (one prompt)
@dataclass
class ReviewUnit:
    chunk_index: int # position, in the repository's list of chunks, of the chunk this code comes from (used to find related chunks)
    chunk: CodeChunk # the code to review: a whole chunk, or a window of a longer chunk around its flagged lines
    score: int # how worthwhile the review looks (higher is reviewed sooner)
    reasons: list[str] # the reasons for the score
    kind: str # which review this is: REVIEW_KIND_SECURITY or REVIEW_KIND_PERFORMANCE
    hints: list[ReviewHint] # the flagged lines the model is asked about
    notes: list[str] # extra facts about where the code is used, shown in the report (not to the model, which tends to copy them)
    context_lines: list[ContextLine] # other lines of the function that use the same names as the flagged lines

# ReviewPlan: everything that is going to be reviewed, and what is not
@dataclass
class ReviewPlan:
    units: list[ReviewUnit] # the reviews to do, most worthwhile first
    unflagged_count: int # how many chunks had no flagged lines, so there was nothing to ask the model
    unflagged_by_file: dict[Path, int] # the same count for each file
    not_reviewable: list[CodeChunk] # chunks too large to review as a whole (the chunks inside them are reviewed on their own)

# Pick the chunks to review: every function/method, plus any other chunk that has no chunks inside it.
# (A class or namespace that contains functions is not reviewed as a whole: its functions are reviewed on their own.)
def get_reviewable_chunk_indices(repository_data: RepositoryData) -> list[int]:
    container_ids = get_container_ids(repository_data)
    reviewable = []
    for index, chunk in enumerate(repository_data.chunks):
        is_function = chunk.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)
        if is_function or chunk.chunk_id not in container_ids:
            reviewable.append(index)
    return reviewable

# Remove comments so that a word in a comment does not count as a signal (approximate, which is good enough for flagging lines)
def strip_comments(source: str, language: str) -> str:
    # Python comments start with # and run to the end of the line
    if language == "Python":
        return re.sub(r"(?m)#.*$", "", source)
    # C++ block comments (keeping their line endings, so that line numbers stay right), then C++ line comments
    without_blocks = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_blocks)

# Cut a line of code down for display as a hint
def shorten_for_hint(text: str) -> str:
    # remove the indentation and cut very long lines
    text = " ".join(text.split())
    if len(text) > REVIEW_HINT_MAX_CHARS:
        text = text[:REVIEW_HINT_MAX_CHARS] + "..."
    return text

# Find every line where a security signal matches. Only signals worth REVIEW_MIN_HINT_POINTS or more flag a line (weaker ones flag too many lines to be useful).
def find_security_hints(chunk: CodeChunk) -> list[ReviewHint]:
    # look at the code without its comments (line endings are kept, so line numbers stay right)
    code = strip_comments(chunk.source, chunk.language)
    lines = chunk.source.replace("\r\n", "\n").split("\n")
    signals = PYTHON_RISK_SIGNALS if chunk.language == "Python" else CPP_RISK_SIGNALS
    hints: list[ReviewHint] = []
    for points, description, pattern in signals:
        # weak signals do not flag lines
        if points < REVIEW_MIN_HINT_POINTS:
            continue
        # the line numbers (counted from the top of the chunk) where this signal matches, without repeats
        line_offsets = []
        for match in re.finditer(pattern, code):
            offset = code.count("\n", 0, match.start())
            if offset not in line_offsets:
                line_offsets.append(offset)
        for offset in line_offsets:
            hints.append(ReviewHint(chunk.start_line + offset, shorten_for_hint(lines[offset]), description, points))
    return hints

# Work out, for every line of a chunk, how many loops it is inside, and which lines start a loop.
# Returns (loop depth of each line, whether each line starts a loop). A loop's own line is counted at the depth OUTSIDE the loop.
# This is approximate (it counts braces in C++ and indentation in Python) but good enough to tell which lines repeat.
def get_loop_depths(lines: list[str], language: str) -> tuple[list[int], list[bool]]:
    depths: list[int] = []
    starts_loop: list[bool] = []
    # Python: a loop body is every following line that is indented deeper than the loop line
    if language == "Python":
        loop_indents: list[int] = []
        for line in lines:
            stripped = line.strip()
            # blank lines stay at the depth of the line before them
            if not stripped:
                depths.append(len(loop_indents))
                starts_loop.append(False)
                continue
            indent = len(line) - len(line.lstrip())
            # leave every loop that this line is no longer inside
            while loop_indents and indent <= loop_indents[-1]:
                loop_indents.pop()
            depths.append(len(loop_indents))
            is_loop = re.match(r"(async\s+)?(for|while)\b.*:\s*$", stripped) is not None
            starts_loop.append(is_loop)
            if is_loop:
                loop_indents.append(indent)
        return depths, starts_loop
    # C++: a loop body is the braces that follow a for/while/do header
    brace_depth = 0
    loop_body_depths: list[int] = []
    # we have seen a loop header and are waiting for its opening brace
    waiting_for_body = False
    paren_depth = 0
    for line in lines:
        # string and character literals are replaced so that braces inside them are not counted
        code = re.sub(r"\"(\\.|[^\"\\])*\"|'(\\.|[^'\\])*'", '""', line)
        depths.append(len(loop_body_depths))
        # a '} while (x);' that ends a do-loop is not a new loop
        is_loop = (re.search(r"\b(for|while)\s*\(", code) is not None or re.match(r"\s*do\b", code) is not None) and re.match(r"\s*\}\s*while\b", code) is None
        starts_loop.append(is_loop)
        if is_loop:
            waiting_for_body = True
            paren_depth = 0
        # walk the characters to follow parentheses and braces
        for character in code:
            if character == "(":
                paren_depth += 1
            elif character == ")":
                paren_depth -= 1
            elif character == "{":
                brace_depth += 1
                # the first opening brace after the header (outside its parentheses) starts the loop body
                if waiting_for_body and paren_depth <= 0:
                    loop_body_depths.append(brace_depth)
                    waiting_for_body = False
            elif character == "}":
                # the closing brace that matches the loop body ends the loop
                if loop_body_depths and loop_body_depths[-1] == brace_depth:
                    loop_body_depths.pop()
                brace_depth -= 1
            elif character == ";" and waiting_for_body and paren_depth <= 0:
                # a loop with no braces: we do not follow it
                waiting_for_body = False
    return depths, starts_loop

# Find every line that looks slow. Only signals worth REVIEW_MIN_HINT_POINTS or more flag a line.
def find_performance_hints(chunk: CodeChunk) -> list[ReviewHint]:
    # look at the code without its comments (line endings are kept, so line numbers stay right)
    code_lines = strip_comments(chunk.source, chunk.language).split("\n")
    source_lines = chunk.source.replace("\r\n", "\n").split("\n")
    depths, starts_loop = get_loop_depths(code_lines, chunk.language)
    signals = PYTHON_PERFORMANCE_SIGNALS if chunk.language == "Python" else CPP_PERFORMANCE_SIGNALS
    # the lines up to the start of the body (the function's signature)
    body_start = len(code_lines)
    for index, line in enumerate(code_lines):
        # in Python the signature ends at the first line ending in ':'; in C++ it ends at the first line with an opening brace
        ends_signature = line.rstrip().endswith(":") if chunk.language == "Python" else "{" in line
        if ends_signature:
            body_start = index
            break
    # the whole chunk's code, used by signals that are ignored when something else is present
    whole_code = "\n".join(code_lines)
    hints: list[ReviewHint] = []
    for points, description, pattern, where, unless in signals:
        # weak signals do not flag lines, and a signal is ignored in a chunk that does something that makes it fine
        if points < REVIEW_MIN_HINT_POINTS or (unless is not None and re.search(unless, whole_code)):
            continue
        for index, line in enumerate(code_lines):
            # only lines in the right place count
            if where == "loop" and depths[index] == 0:
                continue
            if where == "signature" and index > body_start:
                continue
            if re.search(pattern, line):
                hints.append(ReviewHint(chunk.start_line + index, shorten_for_hint(source_lines[index]), description, points))
    # a loop inside a loop: the work grows with the product of the two sizes
    for index in range(len(code_lines)):
        if starts_loop[index] and depths[index] >= 1:
            points = RISK_POINTS_NESTED_LOOP + (RISK_POINTS_DEEP_LOOP if depths[index] >= 2 else 0)
            description = "loop nested inside another loop (the work grows with both sizes)"
            hints.append(ReviewHint(chunk.start_line + index, shorten_for_hint(source_lines[index]), description, points))
    return hints

# Find the functions that are called from inside a loop in another function.
# Returns {chunk id of the called function: a sentence saying where it is called in a loop}.
# Names are matched without looking at types, so only names that few chunks share are trusted.
def find_hot_callees(repository_data: RepositoryData) -> dict[int, str]:
    container_ids = get_container_ids(repository_data)
    functions = [chunk for chunk in repository_data.chunks if chunk.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)]
    # which functions have each name
    by_name: dict[str, list[CodeChunk]] = {}
    for function in functions:
        short_name = function.chunk_name.split("::")[-1]
        if len(short_name) >= RISK_HOT_CALLEE_MIN_NAME_CHARS:
            by_name.setdefault(short_name, []).append(function)
    hot: dict[int, str] = {}
    for caller in functions:
        # a function that contains other chunks would repeat the loops of the chunks inside it
        if caller.chunk_id in container_ids:
            continue
        code_lines = strip_comments(caller.source, caller.language).split("\n")
        depths, starts_loop = get_loop_depths(code_lines, caller.language)
        for index, line in enumerate(code_lines):
            # only lines inside a loop count
            if depths[index] == 0:
                continue
            for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", line):
                candidates = by_name.get(name, [])
                # a name shared by many chunks tells us nothing about which one is called
                if not candidates or len(candidates) > RISK_HOT_CALLEE_MAX_SAME_NAME:
                    continue
                for callee in candidates:
                    if callee.chunk_id != caller.chunk_id and callee.chunk_id not in hot:
                        # say which file the caller is in if it is not the callee's file
                        caller_file = "" if caller.file_path == callee.file_path else f" in {Path(caller.file_path).name}"
                        hot[callee.chunk_id] = f"It is called inside a loop at line {caller.start_line + index} of {caller.symbol_type.value} '{caller.chunk_name}'{caller_file}."
    return hot

# Score what makes a whole chunk matter more, whatever its lines are: it calls itself, it handles outside data, or it is complicated.
# Returns (points, reasons).
def score_chunk_context(chunk: CodeChunk) -> tuple[int, list[str]]:
    # look at the code without its comments
    code = strip_comments(chunk.source, chunk.language)
    points = 0
    reasons: list[str] = []
    # a function that calls itself can recurse without limit on hostile input (its name appears in a call besides its definition)
    short_name = chunk.chunk_name.split("::")[-1]
    if short_name and len(re.findall(r"\b" + re.escape(short_name) + r"\s*\(", code)) >= 2:
        points += RISK_POINTS_RECURSION
        reasons.append("may recurse")
    # a function whose name suggests it handles outside data matters more (words are matched by their start so that 'thread' does not match 'read')
    name_words = [word for word in re.split(r"[^a-z0-9]+", chunk.chunk_name.lower()) if word]
    if any(word.startswith(RISK_OUTSIDE_DATA_NAME_STEMS) for word in name_words):
        points += RISK_POINTS_OUTSIDE_DATA_NAME
        reasons.append("handles outside data")
    # complicated code hides more mistakes
    branch_count = len(re.findall(r"\b(" + "|".join(RISK_BRANCH_WORDS) + r")\b", code))
    if branch_count >= RISK_MANY_BRANCHES:
        points += 2
        reasons.append("many branches")
    elif branch_count >= RISK_SOME_BRANCHES:
        points += 1
        reasons.append("several branches")
    return points, reasons

# Score a list of hints: each kind of problem counts once, however many lines show it. Returns (points, reasons).
def score_hints(hints: list[ReviewHint]) -> tuple[int, list[str]]:
    # the points of each kind of problem
    points_by_description: dict[str, int] = {}
    for hint in hints:
        points_by_description[hint.description] = max(points_by_description.get(hint.description, 0), hint.points)
    return sum(points_by_description.values()), list(points_by_description.keys())

# Keep the hints for one prompt: a few lines for each kind of problem, the strongest first, and no more than the prompt should carry (shown in line order)
def limit_hints(hints: list[ReviewHint]) -> list[ReviewHint]:
    kept: list[ReviewHint] = []
    count_by_description: dict[str, int] = {}
    # the strongest signals first, then source order
    for hint in sorted(hints, key=lambda hint: (-hint.points, hint.line)):
        if count_by_description.get(hint.description, 0) >= REVIEW_HINTS_PER_SIGNAL:
            continue
        count_by_description[hint.description] = count_by_description.get(hint.description, 0) + 1
        kept.append(hint)
    # no more than the prompt should carry, shown in line order
    kept = kept[:REVIEW_MAX_HINTS]
    kept.sort(key=lambda hint: hint.line)
    return kept

# Find lines elsewhere in a function that explain the flagged lines: the lines that give a name its value, the checks on it, and the function's own parameters.
# The window only shows the code around a flagged line, but the size used here may have been set (or checked) far above it.
# Returns the closest such lines, in line order.
def find_context_lines(chunk: CodeChunk, window: CodeChunk, hints: list[ReviewHint]) -> list[ContextLine]:
    # the names used on the flagged lines
    window_lines = window.source.replace("\r\n", "\n").split("\n")
    names: set[str] = set()
    for hint in hints:
        flagged_code = strip_comments(window_lines[hint.line - window.start_line], window.language)
        names.update(re.findall(r"[A-Za-z_]\w*", flagged_code))
    names -= set(REVIEW_IGNORED_NAMES)
    # no names worth following
    if not names:
        return []
    code_lines = strip_comments(chunk.source, chunk.language).split("\n")
    source_lines = chunk.source.replace("\r\n", "\n").split("\n")
    # the function's own parameters are on the lines before its body starts
    body_start = len(code_lines)
    for index, line in enumerate(code_lines):
        if line.rstrip().endswith(":") if chunk.language == "Python" else "{" in line:
            body_start = index
            break
    candidates: list[tuple[int, ContextLine]] = []
    for index, code in enumerate(code_lines):
        line_number = chunk.start_line + index
        # lines the model already sees are not repeated
        if window.start_line <= line_number <= window.end_line:
            continue
        # the line must use one of the names
        mentioned = names & set(re.findall(r"[A-Za-z_]\w*", code))
        if not mentioned:
            continue
        # it must give a name a value, be a check, or be part of the function's signature
        gives_value = any(re.search(r"\b" + re.escape(name) + r"\b\s*(\[[^\]]*\])?\s*([-+*/|&^]|<<|>>)?=(?!=)", code) for name in mentioned)
        is_check = re.search(r"\b(if|while|assert|JSON_ASSERT|raise|throw)\b", code) is not None
        if gives_value or is_check or index <= body_start:
            distance = min(abs(line_number - window.start_line), abs(line_number - window.end_line))
            candidates.append((distance, ContextLine(line_number, shorten_for_hint(source_lines[index]))))
    # keep the closest lines, shown in line order
    candidates.sort(key=lambda candidate: candidate[0])
    closest = [context for distance, context in candidates[:REVIEW_MAX_CONTEXT_LINES]]
    return sorted(closest, key=lambda context: context.line)

# Get the hints whose lines are inside a window of code
def get_hints_in_window(hints: list[ReviewHint], window: CodeChunk) -> list[ReviewHint]:
    return [hint for hint in hints if window.start_line <= hint.line <= window.end_line]

# Check whether a review's code and flagged lines fit in a prompt
def unit_fits_in_prompt(unit: ReviewUnit) -> bool:
    return build_chunk_review_prompt(unit.chunk, [], None, unit.kind, unit.hints, unit.context_lines) is not None

# Make the reviews for one window of code, using the lines flagged anywhere in its chunk.
# A window gets a security review if it holds flagged security lines, and a performance review if it holds flagged slow lines.
def make_units_for_window(index: int, chunk: CodeChunk, window: CodeChunk, security_hints: list[ReviewHint], performance_hints: list[ReviewHint],
                          context_points: int, context_reasons: list[str], hot_note: str | None) -> list[ReviewUnit]:
    units: list[ReviewUnit] = []
    # only the flagged lines inside this window count
    window_security = limit_hints(get_hints_in_window(security_hints, window))
    window_performance = limit_hints(get_hints_in_window(performance_hints, window))
    # the security review: the flagged lines' points, plus what makes the whole chunk matter
    if window_security:
        points, reasons = score_hints(window_security)
        context_lines = find_context_lines(chunk, window, window_security)
        units.append(ReviewUnit(index, window, points + context_points, reasons + context_reasons, REVIEW_KIND_SECURITY, window_security, [], context_lines))
    # the performance review: the flagged lines' points, plus a bonus if the function is called inside a loop elsewhere
    if window_performance:
        points, reasons = score_hints(window_performance)
        notes = []
        if hot_note is not None:
            points += RISK_POINTS_HOT_CALLEE
            reasons.append("called inside a loop elsewhere")
            notes.append(hot_note)
        context_lines = find_context_lines(chunk, window, window_performance)
        units.append(ReviewUnit(index, window, points, reasons, REVIEW_KIND_PERFORMANCE, window_performance, notes, context_lines))
    # a review whose code and flagged lines do not fit in a prompt can not be done
    return [unit for unit in units if unit_fits_in_prompt(unit)]

# Put every review in the order it should be done: the most worthwhile first.
# Only lines flagged by a pattern search are reviewed: the model is asked about those lines, with the code around them, because a small model can not find problems on its own.
# A chunk that is short is reviewed whole; a longer one is reviewed in windows around its flagged lines. A chunk that contains other chunks and is not short is left to the chunks inside it.
def rank_chunks_for_review(repository_data: RepositoryData) -> ReviewPlan:
    container_ids = get_container_ids(repository_data)
    # which functions are called from inside loops elsewhere
    hot_callees = find_hot_callees(repository_data)
    plan = ReviewPlan([], 0, {}, [])
    for index in get_reviewable_chunk_indices(repository_data):
        chunk = repository_data.chunks[index]
        # the lines flagged anywhere in the chunk (loops are followed across the whole chunk, so this is done before cutting it into windows)
        security_hints = find_security_hints(chunk)
        performance_hints = find_performance_hints(chunk)
        # nothing flagged: nothing to ask the model
        if not security_hints and not performance_hints:
            plan.unflagged_count += 1
            plan.unflagged_by_file[chunk.file_path] = plan.unflagged_by_file.get(chunk.file_path, 0) + 1
            continue
        # a chunk that contains other chunks and is not short is left to the chunks inside it
        is_short = len(chunk.source) <= REVIEW_WHOLE_CHUNK_CHARS
        if chunk.chunk_id in container_ids and not is_short:
            plan.not_reviewable.append(chunk)
            continue
        # a short chunk is reviewed whole; a longer one in windows around its flagged lines
        windows = [chunk] if is_short else get_focus_windows(chunk, [hint.line for hint in security_hints + performance_hints])
        context_points, context_reasons = score_chunk_context(chunk)
        hot_note = hot_callees.get(chunk.chunk_id)
        for window in windows:
            window_units = make_units_for_window(index, chunk, window, security_hints, performance_hints, context_points, context_reasons, hot_note)
            # a window with flagged lines but no review that fits in a prompt can not be reviewed
            if not window_units:
                plan.not_reviewable.append(window)
            plan.units.extend(window_units)
    # highest score first; for equal scores, source order and review kind so the result is stable
    plan.units.sort(key=lambda unit: (-unit.score, unit.chunk_index, unit.chunk.start_line, unit.kind))
    return plan