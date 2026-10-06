# This file contains utility functionality used in creating and building chunks
# This is used during both embedding generation and context building

# Internal dependencies
from code_chunker import CodeChunk

# Create chunk text using necessary information
def get_chunk_text(chunk: CodeChunk) -> str:
    # Create a string of code chunk information
    return (
        f"File: {chunk.file_path}\n"
        f"Language: {chunk.language}\n"
        f"Type: {chunk.symbol_type.value}\n"
        f"Name: {chunk.chunk_name}\n\n"
        f"{chunk.source}"
    )