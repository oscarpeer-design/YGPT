# This file is responsible for defining CONFIGURATION variables used to run the model

# os: get system-specific hyperparameters
import os

# Max allowable size of the model: this can be altered based on user needs. For now we want something that can run in under a gigabyte
MAX_MODEL_SIZE = 1073741824 # 1024 * 1024 * 1024, or 1 Gigabyte

# Define model hyperparameters
# Temperature controls the randomness of token selection during response generation
TEMPERATURE = 0.1
# Stream causes model to show outputs to the user
STREAM = True
# Context window sets the number of tokens available to the model
MODEL_CTX = 4096
# Longest time (in seconds) one call to the model may take before it is stopped
MODEL_TIMEOUT_SECONDS = 600
# Get the number of CPU cores to run the model on so it can do concurrent processing for faster results (use minimum 1 core)
# (Physical cores are faster than logical cores for the model: two logical cores share one physical core's maths units, so extra threads only add waiting. psutil can count physical cores; without it we use the logical count.)
try:
    import psutil
    CPU_THREADS_USED = psutil.cpu_count(logical=False) or os.cpu_count() or 1
except ImportError:
    CPU_THREADS_USED = os.cpu_count() or 1

# Define retrieval parameters
RETRIEVAL_RESULT_COUNT = 5 # How many chunks to retrieve (can change based on query)
RETRIEVAL_SIMILARITY_THRESHOLD = 0.50 # How similar chunks should be to query
RETRIEVAL_RELATED_CHUNK_COUNT = 2 # For each semantically retrieved chunk, how many structurally related chunks should we consider adding (connect retrieved chunks to relationships)
RETRIEVAL_MAX_CONTEXT_CHUNKS = 10 # How many chunks to retrieve

# Define the settings for CHAT
# How similar (cosine similarity, 0 to 1) a chunk must be to a question to be used. This is a first guess: the similarity of each source is printed under the answer so a better value can be chosen.
CHAT_SIMILARITY_THRESHOLD = 0.30
# The most tokens the model may use to answer a question. This is only a safety net against a model that never stops: an answer normally ends by itself well before it. The prompt keeps its own room (CHAT_MAX_PROMPT_TOKENS), so a larger value here does not shrink the code the model is shown
CHAT_MAX_ANSWER_TOKENS = 1024
# The most tokens a chat prompt may use. The code of the best matching chunks fills what the question leaves
CHAT_MAX_PROMPT_TOKENS = 1536
# The longest question (in characters) that is sent to the model
CHAT_MAX_QUESTION_CHARS = 500
# The room for code in a chat prompt is shared between the chunks found, so that one large chunk can not crowd out the rest. This is the most chunks it is shared between (one chunk alone may use all of it)
CHAT_CHUNKS_PER_PROMPT = 3
# The model is discouraged from repeating itself (above 1.0), and picks its most likely wording (temperature 0)
CHAT_REPEAT_PENALTY = 1.1
CHAT_TEMPERATURE = 0.0
# What the model is told when code was found for the question, and when it was not
CHAT_INSTRUCTION = "Answer the question using only the code below. If the code does not contain the answer, say so. Name the function you used."
CHAT_NO_CODE_INSTRUCTION = "Answer the question. No code was found for it, so say if you are not sure."

# Define parameters for the embedding model and the saved (cached) embeddings
# Name of the sentence-transformers model that turns code chunks into vectors (also part of the cache key, so changing model never reuses old vectors)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Name of the folder (created next to the code) where computed embeddings are saved so that unchanged files are not embedded again
EMBEDDING_CACHE_FOLDER_NAME = ".embedding_cache"
# A cache file holds two things: the vectors and the chunks they were made from (in the same order). These are the file ending and the names of the two things.
EMBEDDING_CACHE_SUFFIX = ".npz"
EMBEDDING_CACHE_VECTORS_KEY = "embeddings"
EMBEDDING_CACHE_CHUNKS_KEY = "chunks"
# A cache file is written under this mark in its name first, then swapped into place, so a half-written file is never read
EMBEDDING_CACHE_TEMP_MARK = ".tmp"

# Define parameters for the iterative, one-chunk-at-a-time code review (needed because the model is small and its context window is limited)
# How strongly the model is discouraged from repeating words it has already written (1.0 means not at all). A small model otherwise tends to repeat a sentence until it runs out of tokens
REVIEW_REPEAT_PENALTY = 1.15
# Maximum number of tokens the model may write for ONE review. The model writes about 10 to 15 tokens a second on a CPU, so every token costs time.
# A finding is about 80 tokens, and the answer when there is no problem is about 6
REVIEW_N_PREDICT = 160
# Most findings the model may write for ONE review. A limit also stops the model repeating itself until it runs out of tokens, which is slow
REVIEW_MAX_FINDINGS_PER_REVIEW = 2
# Largest prompt (in tokens) used for ONE chunk's review. Smaller prompts are reviewed faster. It is also limited by the context window
REVIEW_MAX_PROMPT_TOKENS = 1024
# Rough number of characters per token for source code. Deliberately pessimistic so that prompts do not overflow the context window
CHARS_PER_TOKEN = 3
# Only fill this fraction of the estimated prompt space, since the characters-per-token figure is only an estimate
PROMPT_SAFETY_FACTOR = 0.9
# How many related chunks (found by similarity) may be added as background to the review of one chunk
REVIEW_RELATED_CHUNK_COUNT = 3
# How many similar chunks are considered before choosing the ones that fit in the prompt
REVIEW_RELATED_CANDIDATES = 8
# How many times more neighbours than candidates are searched for, because some neighbours are skipped for overlapping the chunk under review
REVIEW_RELATED_SEARCH_MULTIPLIER = 4
# How similar (cosine similarity, 0 to 1) a chunk must be to be used as background for another chunk
REVIEW_RELATED_SIMILARITY_THRESHOLD = 0.40
# Guideline for the total time (in seconds) a review spends waiting for the model. It is checked before each model call, so the last call may finish after it.
# A review that reaches this guideline leaves the remaining reviews undone, so raise it (or review fewer files) to reach them
REVIEW_TIME_BUDGET_SECONDS = 600
# The same guideline for reviewing a whole repository. Files are reviewed one after another: each file gets the time that is left divided by the number of files still to do, so time a file does not use goes to the files after it
REVIEW_REPOSITORY_TIME_BUDGET_SECONDS = 1200
# How many files the summary of a repository review lists (the report lists all of them)
REVIEW_SUMMARY_MAX_FILE_ROWS = 15

# Define parameters for documenting code (the DOCUMENT command)
# How many functions of each file are documented one by one (the most important ones; the rest are only listed by signature)
DOCUMENT_FUNCTIONS_PER_FILE = 6
# How many signatures of the functions that an important function calls are shown to the model as background
DOCUMENT_MAX_CALLEE_SIGNATURES = 4
# How many callers are named when saying where an important function is called from
DOCUMENT_MAX_NAMED_CALLERS = 3
# Guideline for the total time (in seconds) documenting spends waiting for the model: one file, or a whole repository (files are done one after another, each with its share of the time left)
DOCUMENT_TIME_BUDGET_SECONDS = 300
DOCUMENT_REPOSITORY_TIME_BUDGET_SECONDS = 600
# Time (in seconds) kept back from the files for the overview of a whole repository, which is written last
DOCUMENT_REPOSITORY_OVERVIEW_RESERVE_SECONDS = 30
# Time (in seconds) each file keeps back for its own overview, which is written after its functions. The functions of a file stop when its share of the time has only this much left (time the overview does not use goes to the files after it)
DOCUMENT_OVERVIEW_RESERVE_SECONDS = 30
# The most characters the model may write for each part of the documentation (a part that reaches this length is cut back to its last whole sentence)
DOCUMENT_OVERVIEW_MAX_CHARS = 450
DOCUMENT_FUNCTION_MAX_CHARS = 300
DOCUMENT_REPOSITORY_MAX_CHARS = 450
# The shortest answer that is used (anything shorter says nothing)
DOCUMENT_MIN_ANSWER_CHARS = 20
# How the model is asked: it repeats itself less with a penalty, and the prompt is capped so that each call stays quick
DOCUMENT_REPEAT_PENALTY = 1.1
# Room (in tokens) kept beyond the longest answer, for the end-of-answer marker and a miscounted token or two
DOCUMENT_ANSWER_SLACK_TOKENS = 10
# The characters a list line adds to a prompt besides its text ("- " and the line end)
DOCUMENT_LIST_LINE_CHARS = 3
# How many file names are listed when saying which files use a file
DOCUMENT_MAX_NAMED_FILES = 6
# How many names are listed in a sentence about what a file depends on, before it says how many more there are
DOCUMENT_MAX_LISTED_DEPENDENCIES = 8
# Maximum characters of a signature that is kept (a long declaration is cut here)
DOCUMENT_MAX_SIGNATURE_CHARS = 240
# Durations: a time is described in minutes from this many seconds, and this many seconds make a minute
DURATION_MINUTES_FROM_SECONDS = 120
SECONDS_PER_MINUTE = 60
# Documentation uses no randomness: the model always picks its most likely wording, which invents less than a sampled one
DOCUMENT_TEMPERATURE = 0.0
DOCUMENT_MAX_PROMPT_TOKENS = 1024
# How many signatures are listed for a file, and how many include arrows are drawn in the diagram of a repository
DOCUMENT_MAX_LISTED_SYMBOLS = 30
DOCUMENT_MAX_DIAGRAM_EDGES = 40
# How many files are named when saying which files matter most in a repository
DOCUMENT_MAX_CORE_FILES = 4
# How many files are listed in the summary of documenting a repository
DOCUMENT_SUMMARY_MAX_FILE_ROWS = 15

# The instructions for the parts of the documentation. They are kept short and plain because the small model copies wording it can quote.
DOCUMENT_OVERVIEW_INSTRUCTION = """
Write exactly two sentences about this file as a whole, not about one function.
The first says what the file is responsible for. If the file defines a type, it says what the type is for.
The second says how it works.
Use only the function descriptions and names below.
"""
DOCUMENT_FUNCTION_INSTRUCTION = """
Say what this function does in one sentence of about 25 words.
Describe only what the code does, not why it is useful.
Use only the code below.
"""
# One short example of a good answer for each kind of part. It is about something else on purpose, because the model copies the style of an example
DOCUMENT_FUNCTION_EXAMPLE = "Adds up every number in the list and returns the total."
DOCUMENT_REPOSITORY_INSTRUCTION = """
Write two sentences saying what this project does, using the notes on its files below.
"""
# Many answers about a function start by naming it ("The function `X` ..."), which wastes room. This pattern (case does not matter) matches that start so that it can be removed
DOCUMENT_SUBJECT_PATTERN = r"^(?:the\s+)?(?:function|method|constructor|destructor)\s+`[^`]*`(?:\s+(?:function|method))?\s*"
# An answer that starts with one of these words after the function's name was removed is a fragment ("is designed to ..."), so the name is kept
DOCUMENT_FRAGMENT_START_PATTERN = r"^(?:is|are|was|were|has|have|can|will|does|do)\b"
# The mark that cleaning a model's answer puts at a cut-off end, and the marks where an unfinished end can be cut back to
DOCUMENT_CUT_OFF_MARK = "..."
DOCUMENT_CLAUSE_MARKS = ",;:"
# How many sentences of an answer are kept (the model sometimes adds more, and the rest is cut off)
DOCUMENT_OVERVIEW_SENTENCES = 2
DOCUMENT_FUNCTION_SENTENCES = 2
DOCUMENT_REPOSITORY_SENTENCES = 2
# The model tends to copy wording from its instructions. An answer that repeats this many words in a row from its own instruction is thrown away
DOCUMENT_COPY_WORD_RUN = 5
# The shortest time (in seconds) a single model call is given, even when the time guideline is nearly used up
REVIEW_MIN_CALL_TIMEOUT_SECONDS = 30
# Stop the review after this many model failures in a row (for example if the model cannot be started)
REVIEW_MAX_CONSECUTIVE_FAILURES = 3
# How many characters of a function's name are shown in the progress bar
REVIEW_PROGRESS_NAME_CHARS = 30
# A chunk of code is reviewed whole if it is no longer than this many characters. A longer chunk is reviewed in windows around its flagged lines
REVIEW_WHOLE_CHUNK_CHARS = 1500
# How many lines of code the model sees above and below a flagged line
REVIEW_FOCUS_RADIUS_LINES = 12
# Neighbouring windows are joined into one while the joined window is no longer than this many lines
REVIEW_FOCUS_MAX_LINES = 60
# How many other lines of a function (outside the window the model sees) are shown to the model because they use the same names as a flagged line
REVIEW_MAX_CONTEXT_LINES = 6
# Words that are not worth following to other lines because they are not the program's own names: keywords, types and very common library names
REVIEW_IGNORED_NAMES = (
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "return", "goto", "try", "catch", "throw", "const", "static", "auto", "void",
    "int", "char", "bool", "long", "short", "unsigned", "signed", "float", "double", "size_t", "true", "false", "nullptr", "new", "delete", "sizeof", "this",
    "struct", "class", "enum", "typename", "template", "namespace", "using", "std", "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast",
    "def", "elif", "in", "not", "and", "or", "is", "None", "True", "False", "self", "lambda", "pass", "raise", "with", "as", "from", "import", "len", "range", "print", "str"
)
# If the reason or fix of a finding contains one of these phrases, the finding argues against itself or only repeats the instruction, so it is thrown away
REVIEW_DISCARDED_PHRASES = (
    "no issues found", "false alarm", "false positive", "not a performance", "not a security", "not an issue", "not a problem", "is not a",
    "does not contain any", "does not perform any", "does not use any", "does not grow", "does not already protect"
)
# A line is only flagged for review by a signal worth at least this many points (weak signals flag too many lines to be useful)
REVIEW_MIN_HINT_POINTS = 2
# The most characters the model may write in the reason or the fix of a finding (the answer is forced to respect this, see build_review_grammar)
REVIEW_FIELD_MAX_CHARS = 140
# Text of at least this many characters that does not end with . ! or ? is treated as cut off by the length limit
REVIEW_CUT_OFF_MIN_CHARS = 100
# The two kinds of review. Each piece of code is reviewed once for each kind that is worth doing (see chunk_risk.py)
REVIEW_KIND_SECURITY = "SECURITY"
REVIEW_KIND_PERFORMANCE = "PERFORMANCE"
# How many flagged lines (found by a simple pattern search) one signal may point the model at, and how many in total in one prompt
REVIEW_HINTS_PER_SIGNAL = 2
REVIEW_MAX_HINTS = 8
# Flagged lines are shortened to this many characters when shown to the model as a hint (the code itself is never shortened)
REVIEW_HINT_MAX_CHARS = 120

# What the model is told to write when it finds nothing wrong (the review loop also looks for this text)
NO_ISSUES_TEXT = "No issues found."

# Used in both review instructions: the exact shape of an answer. The answer is forced into this shape (see build_review_grammar in context_builder.py)
REVIEW_OUTPUT_FORMAT = f"""Answer exactly like this for each real problem (at most {REVIEW_MAX_FINDINGS_PER_REVIEW}):
LINE: <flagged line number>
WHY: <one short sentence>
FIX: <one short sentence>
If there is no real problem, answer exactly: {NO_ISSUES_TEXT}
"""

# CODE_REVIEW_SECURITY: Used to check the flagged lines of ONE piece of code for security defects. Written for a small model: short, strict, and explicit about what not to do.
# (Every character is read by the model on every review, so it is kept short.)
CODE_REVIEW_SECURITY_CHUNK_INSTRUCTION = f"""
Check the FLAGGED LINES of this code for SECURITY defects: memory safety, injection, unsafe handling of outside data, unbounded recursion or allocation, unsafe use of files or processes.
Most flagged lines are false alarms: report one only if the code around it does not already protect against the problem. RELATED CODE is background only.
{REVIEW_OUTPUT_FORMAT}"""

# CODE_REVIEW_PERFORMANCE: Used to check the flagged lines of ONE piece of code for performance defects
CODE_REVIEW_PERFORMANCE_CHUNK_INSTRUCTION = f"""
Check the FLAGGED LINES of this code for PERFORMANCE problems: needless copying, repeated work inside loops, wasteful allocation, accidental O(n^2) or worse, repeated file access or queries.
Most flagged lines are false alarms: report one only if it is really slow. The fix must be low-risk and must not change what the code does. RELATED CODE is background only.
{REVIEW_OUTPUT_FORMAT}"""
# The characters that may be put around a path typed in a command (for example a path with spaces in it)
PATH_QUOTE_CHARACTERS = "\"'"