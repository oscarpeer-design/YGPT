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

# Import Internal Depenencies
from code_chunker import RepositoryData, CodeChunk
from config import RETRIEVAL_RESULT_COUNT, RETRIEVAL_SIMILARITY_THRESHOLD
from code_chunk_utils import get_chunk_text

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
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    return EMBEDDING_MODEL

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
        # Embed using the shared embedding model.
        embeddings = get_embedding_model().encode(
            chunk_texts,
            normalize_embeddings=True
        )
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
        # State of embeddings
        self.embedding_state = embedding_state

    # clear retrieved chunks list inbetween queries
    def clear_retrieval(self) -> None:
        self.retrieved_chunks = []

    # Retrieve chunks based on whatever is most semantically relevant to the passed query
    def retrieve(self, query: str, result_count: int = RETRIEVAL_RESULT_COUNT, similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD) -> None:
        # clear retrieved chunks
        self.clear_retrieval()

        # Convert the query into an embedding. Note that models/faiss can fail
        try:
            query_embedding = get_embedding_model().encode([query], normalize_embeddings=True)
        except Exception as e:
            raise RuntimeError(f"Failed to encode query: {str(e)}")

        # get distances and indices. Note that retrieval can be corrupted
        try:
            distances, indices = self.embedding_state.vector_index.search(query_embedding, result_count)
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {str(e)}")
        # Search the FAISS index for the closest chunk embeddings.
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
            except IndexError:
                print(f"Warning: Index {index} out of bounds for chunks list (size: {len(self.all_chunks)})")
            