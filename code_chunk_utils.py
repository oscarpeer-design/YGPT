# This file contains utility functionality used in creating and building chunks
# This is used during both embedding generation and context building

# External dependencies
# hashlib: used to turn text into a short fixed-length key
import hashlib

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

# Turn pieces of text into one short key that is safe to use as a file name (used to name saved embeddings and saved reviews)
def get_text_digest(parts: list[str]) -> str:
    # join the pieces with a null character so that where one piece ends and the next begins is part of the key
    joined = "\0".join(parts)
    # hash the text: the same text always gives the same key and any change gives a different one
    digest = hashlib.sha256(joined.encode("utf-8"))
    # return the key as text
    return digest.hexdigest()