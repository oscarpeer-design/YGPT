# This file is responsible for embedding code chunks in a vector database and retrieving embeddings

# Import External Dependencies
# Set embedding model -> don't load immediately but when needed only
EMBEDDING_MODEL = None
# faiss: vector database used for retrieval and storage of chunks
import faiss
# numpy: dependency used for embedding state
import numpy as np
# dataclass: used to create struct-like classes
from dataclasses import dataclass
# json: used to save the chunks as text inside the cache file
import json
# os: used to swap a finished file into place
import os
# Path: library used to find file paths on computer
from pathlib import Path

# Import Internal Depenencies
from code_chunker import RepositoryData, CodeChunk, SymbolType
from config import RETRIEVAL_RESULT_COUNT, RETRIEVAL_SIMILARITY_THRESHOLD, EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_FOLDER_NAME, REVIEW_RELATED_SEARCH_MULTIPLIER, EMBEDDING_CACHE_SUFFIX, EMBEDDING_CACHE_VECTORS_KEY, EMBEDDING_CACHE_CHUNKS_KEY, EMBEDDING_CACHE_TEMP_MARK
from code_chunk_utils import get_chunk_text, get_text_digest

# EmbeddingState: state of shared vector embeddings used in retrieval and embedding creation
@dataclass
class EmbeddingState:
    embeddings: np.ndarray
    vector_index: faiss.Index

# Get embedding model that is initialised
def get_embedding_model():
    """
    Get the shared SentenceTransformer embedding model.

    The embedding model is loaded lazily rather than at module import time.
    This prevents YitzGPT from paying the model startup cost when semantic
    embedding or retrieval is not required.

    Returns:
        SentenceTransformer: The shared embedding model instance.
    """
    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        # import sentencetransformrers: vector embedding models for text
        from sentence_transformers import SentenceTransformer
        EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return EMBEDDING_MODEL

# Work out where the saved embeddings for a list of chunk texts live.
# The name is a digest of the model name and the exact text of every chunk, so if the file or the model changes the saved embeddings are not reused.
def get_embedding_cache_path(chunk_texts: list[str]) -> Path:
    # the folder sits next to this file
    cache_folder = Path(__file__).resolve().parent / EMBEDDING_CACHE_FOLDER_NAME
    # the name is a digest of the model name followed by every chunk text, in order
    digest = get_text_digest([EMBEDDING_MODEL_NAME] + chunk_texts)
    return cache_folder / f"{digest}{EMBEDDING_CACHE_SUFFIX}"

# Turn a chunk into a plain record that can be saved as text. The record keeps what CHAT needs to show the chunk again (its code, where it is and what it is called).
def chunk_to_record(chunk: CodeChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_name": chunk.chunk_name,
        "source": chunk.source,
        "file_path": str(chunk.file_path),
        "language": chunk.language,
        "symbol_type": chunk.symbol_type.value,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "signature": chunk.signature
    }

# Turn a saved record back into a chunk. Returns an Exception if the record is damaged.
def record_to_chunk(record: dict) -> CodeChunk | Exception:
    try:
        return CodeChunk(
            chunk_id=record["chunk_id"],
            chunk_name=record["chunk_name"],
            source=record["source"],
            file_path=Path(record["file_path"]),
            language=record["language"],
            symbol_type=SymbolType(record["symbol_type"]),
            start_line=record["start_line"],
            end_line=record["end_line"],
            signature=record["signature"]
        )
    except Exception as e:
        return Exception(f"A saved chunk is damaged. Error: {e}")

# Load saved embeddings. Returns None if there are none, or an Exception if they can not be read.
def load_cached_embeddings(cache_path: Path, expected_count: int) -> np.ndarray | None | Exception:
    try:
        # no saved embeddings
        if not cache_path.is_file():
            return None
        # the cache file holds the vectors and the chunks they were made from, and only the vectors are needed here
        with np.load(cache_path) as saved_file:
            embeddings = saved_file[EMBEDDING_CACHE_VECTORS_KEY]
        # check that the saved embeddings match the chunks we have
        if embeddings.ndim != 2 or embeddings.shape[0] != expected_count:
            return Exception(f"The saved embeddings {cache_path} do not match the chunks, so they will not be used.")
        # FAISS needs contiguous float32 vectors
        return np.ascontiguousarray(embeddings, dtype=np.float32)
    except Exception as e:
        return Exception(f"Unable to read the saved embeddings {cache_path}. Error: {e}")

# Save embeddings together with the chunks they were made from (row i of the embeddings is chunk i). Returns None, or an Exception if they can not be saved (a failure to save must not stop a review).
def save_cached_embeddings(cache_path: Path, embeddings: np.ndarray, chunks: list[CodeChunk]) -> None | Exception:
    try:
        # make the folder if it does not exist yet
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # turn every chunk into a record, in order, and write them all as one piece of text
        records = []
        for chunk in chunks:
            records.append(chunk_to_record(chunk))
        chunks_text = json.dumps(records)
        # write to a temporary file first, then swap it in, so that a half-written file is never read
        temp_path = cache_path.with_name(cache_path.stem + EMBEDDING_CACHE_TEMP_MARK + EMBEDDING_CACHE_SUFFIX)
        np.savez_compressed(temp_path, **{EMBEDDING_CACHE_VECTORS_KEY: embeddings, EMBEDDING_CACHE_CHUNKS_KEY: np.array(chunks_text)})
        os.replace(temp_path, cache_path)
    except Exception as e:
        return Exception(f"Unable to save the embeddings {cache_path}. Error: {e}")

# Take in repository data and embed the chunks in a vector database using faiss
class Embedder:
    def __init__(self, repository_data: RepositoryData):
        # set up all chunks
        self.chunks = repository_data.chunks
        # set up embedding state (none for now)
        self.embedding_state : EmbeddingState|None = None

    # create vector embeddings based on the available chunks
    def create_chunk_vector_embeddings(self) -> None|Exception:
        # Convert all code chunks into vector embeddings for semantic retrieval.
        # The embedding index position corresponds to the chunk's position in self.chunks.
        # first check if we have empty list (fail early here)
        if not self.chunks:
            return ValueError("No chunks available to embed")

        # Create text representations of the chunks.
        chunk_texts = []
        for chunk in self.chunks:
            chunk_text = get_chunk_text(chunk)
            chunk_texts.append(chunk_text)
        # Reuse the embeddings from an earlier run if every chunk text is identical
        cache_path = get_embedding_cache_path(chunk_texts)
        embeddings = load_cached_embeddings(cache_path, len(chunk_texts))
        # a problem with the saved embeddings is only a warning: we embed again
        if isinstance(embeddings, Exception):
            print(f"Warning: {embeddings}")
            embeddings = None
        if embeddings is None:
            # Embed using the shared embedding model (with a progress bar, as this can take a while on a CPU).
            print(f"Embedding {len(chunk_texts)} chunks...")
            embeddings = get_embedding_model().encode(
                chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            # FAISS needs contiguous float32 vectors
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            # save the embeddings for next time (a problem saving is only a warning)
            saved = save_cached_embeddings(cache_path, embeddings, self.chunks)
            if isinstance(saved, Exception):
                print(f"Warning: {saved}")
        else:
            print(f"Loaded {len(chunk_texts)} chunk embeddings from the saved copy.")
        # Create a FAISS index using inner product similarity.
        vector_index = faiss.IndexFlatIP(embeddings.shape[1])
        # Add all chunk embeddings to the index.
        vector_index.add(embeddings)
        # Store the embedding state.
        self.embedding_state = EmbeddingState(
            embeddings=embeddings,
            vector_index=vector_index
        )

# Retrieve the code chunks most semantically relevant to a query
class Retriever:
    def __init__(self, all_chunks: list[CodeChunk], embedding_state: EmbeddingState):
        # all the chunks taken from source
        self.all_chunks = all_chunks
        # all the chunks retrieved by a query
        self.retrieved_chunks: list[CodeChunk] = []
        # how similar each retrieved chunk was to the query, in the same order as the retrieved chunks
        self.retrieved_scores: list[float] = []
        # State of embeddings
        self.embedding_state = embedding_state

    # clear retrieved chunks list inbetween queries
    def clear_retrieval(self) -> None:
        self.retrieved_chunks = []
        self.retrieved_scores = []

    # Retrieve chunks based on whatever is most semantically relevant to the passed query. Returns None, or an Exception if the query could not be searched.
    def retrieve(self, query: str, result_count: int = RETRIEVAL_RESULT_COUNT, similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD) -> None | Exception:
        # clear retrieved chunks
        self.clear_retrieval()

        # Convert the query into an embedding. Note that models/faiss can fail
        try:
            query_embedding = get_embedding_model().encode([query], normalize_embeddings=True)
        except Exception as e:
            return Exception(f"Failed to encode query: {str(e)}")

        # get distances and indices. Note that retrieval can be corrupted
        try:
            distances, indices = self.embedding_state.vector_index.search(query_embedding, result_count)
        except Exception as e:
            return Exception(f"FAISS search failed: {str(e)}")
        # Go through the closest chunk embeddings.
        for distance, index in zip(distances[0], indices[0]):
            # If index = -1, there is no embedding
            if index == -1:
                continue
            # Convert float to int for list indexing
            index = int(index)
            # make distance a float (type specificity) and check for similarity issues
            if float(distance) < similarity_threshold:
                continue

            # Try to retrieve specific chunk (handle out of bounds access)
            try:
                self.retrieved_chunks.append(self.all_chunks[index])
                self.retrieved_scores.append(float(distance))
            except IndexError:
                print(f"Warning: Index {index} out of bounds for chunks list (size: {len(self.all_chunks)})")
            

    # Find chunks similar to a chunk that has already been embedded (its stored embedding is reused, so the embedding model is not needed).
    # Chunks that share lines with the chunk (itself, anything inside it, anything containing it) are skipped because they would only repeat its code.
    # Returns up to candidate_count chunks, most similar first.
    def retrieve_related_to_chunk(self, chunk_index: int, candidate_count: int, similarity_threshold: float) -> list[CodeChunk]:
        # the chunk we want neighbours of
        target = self.all_chunks[chunk_index]
        # look at extra neighbours because some will be skipped for overlapping the chunk
        search_count = min(self.embedding_state.vector_index.ntotal, candidate_count * REVIEW_RELATED_SEARCH_MULTIPLIER + 1)
        # the chunk's stored embedding (kept as a 2D array of one row, which is what FAISS expects)
        stored_embedding = self.embedding_state.embeddings[chunk_index:chunk_index + 1]
        # search the FAISS index for the closest embeddings. Note that retrieval can be corrupted
        try:
            distances, indices = self.embedding_state.vector_index.search(stored_embedding, search_count)
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {str(e)}")
        related: list[CodeChunk] = []
        for distance, index in zip(distances[0], indices[0]):
            # If index = -1, there is no embedding
            if index == -1:
                continue
            # results run from most to least similar, so once one is below the threshold we are done
            if float(distance) < similarity_threshold:
                break
            # Convert to int for list indexing and skip the chunk itself
            index = int(index)
            if index == chunk_index or index >= len(self.all_chunks):
                continue
            candidate = self.all_chunks[index]
            # skip chunks that share lines with the chunk (chunks of other files can not overlap it)
            if candidate.file_path == target.file_path and candidate.start_line <= target.end_line and candidate.end_line >= target.start_line:
                continue
            related.append(candidate)
            # stop when we have enough
            if len(related) >= candidate_count:
                break
        return related

# Load the newest saved embeddings together with the chunks they were made from, and build a retriever from them (used by CHAT, which is not given any paths).
# Returns an Exception if there is nothing saved, or what is saved can not be used.
def load_latest_index() -> Retriever | Exception:
    try:
        # the folder where embeddings are saved
        cache_folder = Path(__file__).resolve().parent / EMBEDDING_CACHE_FOLDER_NAME
        if not cache_folder.is_dir():
            return Exception("No saved embeddings were found.")
        # find the newest finished cache file (a file still being written has the temporary mark in its name)
        newest_path = None
        newest_time = 0.0
        for path in cache_folder.glob("*" + EMBEDDING_CACHE_SUFFIX):
            if path.stem.endswith(EMBEDDING_CACHE_TEMP_MARK):
                continue
            modified_time = path.stat().st_mtime
            if newest_path is None or modified_time > newest_time:
                newest_path = path
                newest_time = modified_time
        if newest_path is None:
            return Exception("No saved embeddings were found.")
        # read the vectors and the chunks from the file
        with np.load(newest_path) as saved_file:
            embeddings = saved_file[EMBEDDING_CACHE_VECTORS_KEY]
            chunks_text = saved_file[EMBEDDING_CACHE_CHUNKS_KEY].item()
        # turn every record back into a chunk, in order
        chunks = []
        for record in json.loads(chunks_text):
            chunk = record_to_chunk(record)
            if isinstance(chunk, Exception):
                return chunk
            chunks.append(chunk)
        # row i of the vectors must be chunk i, so there must be one of each
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            return Exception(f"The saved embeddings {newest_path} do not match their chunks, so they will not be used.")
        # build the search index from the vectors (FAISS needs contiguous float32 vectors)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        vector_index = faiss.IndexFlatIP(embeddings.shape[1])
        vector_index.add(embeddings)
        return Retriever(chunks, EmbeddingState(embeddings=embeddings, vector_index=vector_index))
    except Exception as e:
        return Exception(f"Unable to read the saved embeddings. Error: {e}")