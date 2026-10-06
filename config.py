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
# Get the number of CPU cores to run the model on so it can do concurrent processing for faster results (use minimum 1 core)
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