# This file builds context which is sent to LLM queries

# Import internal dependencies
from code_chunker import CodeChunk
from code_chunk_utils import get_chunk_text
from config import DOCUMENTATION_INSTRUCTION, CODE_REVIEW_INSTRUCTION

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

