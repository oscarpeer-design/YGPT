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

# Define different model instructions for different tasks
# DOCUMENTATION: Used to generate markdowns describing previously unseen repositories
DOCUMENTATION_INSTRUCTION = """
Examine the entire codebase and document how it works. 
Describe external dependencies and the roles of key modules. 
Describe how various components interact together to provide required functionality. 
Be concise.
"""

# CODE_REVIEW: Used to analyse pull requests to identify security and performance defects
CODE_REVIEW_INSTRUCTION = """
Conduct an advesarial review of the codebase, examining security and performance defects.
For performance review, suggest improvements that are low-risk and do not modify existing functionality.
For security review, list the file and line of code where a CVE occurs; describe how it can cause an exploit and suggest the simplest possible fix.
"""

# Define parameters for the embedding model and the saved (cached) embeddings
# Name of the sentence-transformers model that turns code chunks into vectors (also part of the cache key, so changing model never reuses old vectors)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Name of the folder (created next to the code) where computed embeddings are saved so that unchanged files are not embedded again
EMBEDDING_CACHE_FOLDER_NAME = ".embedding_cache"

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
# Reviews saved from an earlier run cost no time, so running the same review again carries on where it stopped
REVIEW_TIME_BUDGET_SECONDS = 300
# The shortest time (in seconds) a single model call is given, even when the time guideline is nearly used up
REVIEW_MIN_CALL_TIMEOUT_SECONDS = 30
# Stop the review after this many model failures in a row (for example if the model cannot be started)
REVIEW_MAX_CONSECUTIVE_FAILURES = 3
# How many characters of a function's name are shown in the progress bar
REVIEW_PROGRESS_NAME_CHARS = 30
# Name of the folder (created next to the code) where the model's review of each chunk is saved
REVIEW_CACHE_FOLDER_NAME = ".review_cache"
# Change this number to make every saved review stale at once (for example after changing how the model is asked)
REVIEW_CACHE_VERSION = 1
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
REVIEW_CUT_OFF_MIN_CHARS = 60
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