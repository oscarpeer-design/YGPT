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
# tqdm: shows a progress bar while chunks are embedded
from tqdm import tqdm

# Import Internal Depenencies
from code_chunker import RepositoryData, CodeChunk, SymbolType
from config import RETRIEVAL_RESULT_COUNT, RETRIEVAL_SIMILARITY_THRESHOLD, EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_FOLDER_NAME, REVIEW_RELATED_SEARCH_MULTIPLIER, EMBEDDING_CACHE_SUFFIX, EMBEDDING_CACHE_VECTORS_KEY, EMBEDDING_CACHE_CHUNKS_KEY, EMBEDDING_CACHE_TEMP_MARK, EMBEDDING_MODEL_FILE_NAME, EMBEDDING_TOKENIZER_FILE_NAME, EMBEDDING_MAX_TOKENS, CPU_THREADS_USED
from code_chunk_utils import get_chunk_text, get_text_digest

# EmbeddingState: state of shared vector embeddings used in retrieval and embedding creation
@dataclass
class EmbeddingState:
    embeddings: np.ndarray
    vector_index: faiss.Index

# OnnxEmbedder: turns text into vectors with the MiniLM model. It uses onnxruntime (runs the model file) and tokenizers (cuts text into pieces), so neither PyTorch nor sentence-transformers is needed.
#
# What happens to one text, for example "int x = 5;":
#   1. The tokenizer cuts it into pieces (a short word is one piece) and gives each piece a number. It also adds a start mark and an end mark that the model expects.
#      A text longer than EMBEDDING_MAX_TOKENS pieces is cut off.
#   2. The model reads those numbers and gives back one vector (a list of 384 numbers) for every piece.
#   3. The vectors of the pieces are averaged into one vector for the whole text.
#   4. That vector is scaled to length 1. FAISS compares vectors with an inner product, and for vectors of length 1 the inner product is the cosine similarity.
# The vector says what the text is about, so two chunks about the same thing have vectors that point the same way.
class OnnxEmbedder:
    def __init__(self, session, tokenizer):
        # the onnxruntime session: the model file, loaded and ready to run
        self.session = session
        # the tokenizer: cuts text into pieces and turns each piece into the number the model knows it by
        self.tokenizer = tokenizer

    # Turn texts into vectors of length 1 (one row per text, in the order given). Returns an Exception if the model could not run.
    # show_progress shows a progress bar (only worth it for many texts, such as all the chunks of a repository).
    def embed_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray | Exception:
        # nothing to embed, so there is nothing to return
        if not texts:
            return ValueError("No text to embed")
        # the vector of each text, in the order the texts were given
        vectors = []
        # the progress bar counts finished texts (disable=True hides it when show_progress is False)
        progress = tqdm(total=len(texts), desc="Embedding", unit="chunk", disable=not show_progress)
        # the progress bar is closed afterwards even if something fails half way
        try:
            # embed the texts one after another
            for text in texts:
                # embed_text returns an Exception when it fails, which is passed on to our caller
                vector = self.embed_text(text)
                if isinstance(vector, Exception):
                    return vector
                vectors.append(vector)
                # one more text is done
                progress.update(1)
        finally:
            # stop and remove the progress bar
            progress.close()
        # the vectors become one grid with a row for each text (FAISS needs 32-bit decimals)
        return np.array(vectors, dtype=np.float32)

    # Embed one text. Returns one vector of length 1, or an Exception if the model could not run.
    def embed_text(self, text: str) -> np.ndarray | Exception:
        # anything that goes wrong in the tokenizer or the model is returned as an Exception (see the end of this method)
        try:
            # Cut the text into pieces and number them. This adds the start and end marks the model expects,
            # and cuts the text off after EMBEDDING_MAX_TOKENS pieces (set when the tokenizer was loaded).
            encoding = self.tokenizer.encode(text)
            # The model reads a grid of numbers with one row per text. We give it one text, so the grid has one row.
            input_ids = np.array([encoding.ids], dtype=np.int64)
            # The model file asks for three inputs, each under a fixed name:
            #   input_ids      = the number of each piece
            #   attention_mask = 1 for a real piece. Texts are not mixed in a batch here, so nothing is padding and every piece is real.
            #   token_type_ids = which sentence each piece belongs to. The model was trained on pairs of sentences; we give it one text, so these are all 0.
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": np.ones_like(input_ids),
                "token_type_ids": np.zeros_like(input_ids)
            }
            # Run the model. None means "give me all outputs", and [0] takes the first one, which holds one vector for every piece (shape: 1 text x pieces x 384).
            piece_vectors = self.session.run(None, model_inputs)[0]
            # The vector of the text is the average of the vectors of its pieces. [0] picks our one text, and averaging over the pieces leaves 384 numbers.
            average = piece_vectors[0].mean(axis=0)
            # The length of that vector (the tiny minimum only stops a division by zero)
            length = max(float(np.linalg.norm(average)), 1e-12)
            # Dividing by the length scales the vector to length 1, so that the inner product of two vectors is their cosine similarity.
            return average / length
        # the tokenizer or the model failed: say so instead of stopping the program
        except Exception as e:
            return Exception(f"Failed to embed text: {e}")

# Get the embedding model, loading it the first time it is needed (not at import, so a run that never embeds does not pay for it).
# Returns an Exception if the two model files are missing or can not be loaded.
def get_embedding_model() -> OnnxEmbedder | Exception:
    # EMBEDDING_MODEL (at the top of this file) holds the loaded model, so every caller shares one copy
    global EMBEDDING_MODEL
    # only load it if it is not loaded yet
    if EMBEDDING_MODEL is None:
        # the two model files sit in the same folder as this file
        model_folder = Path(__file__).resolve().parent
        # the model itself
        model_path = model_folder / EMBEDDING_MODEL_FILE_NAME
        # the tokenizer: cuts text into pieces and numbers them the way the model was trained
        tokenizer_path = model_folder / EMBEDDING_TOKENIZER_FILE_NAME
        # both files must be there, and a clear message says which one is missing
        for required_path in (model_path, tokenizer_path):
            if not required_path.is_file():
                return FileNotFoundError(f"The embedding model file {required_path} was not found. Both {EMBEDDING_MODEL_FILE_NAME} and {EMBEDDING_TOKENIZER_FILE_NAME} (all-MiniLM-L6-v2 from Hugging Face) must be in {model_folder}.")
        # loading can fail (for example a damaged file), and that is returned as an Exception
        try:
            # onnxruntime runs the model file, tokenizers cuts text into pieces (imported here so they are only loaded when embedding is needed)
            import onnxruntime
            from tokenizers import Tokenizer
            # load the tokenizer from its file
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            # a text longer than the most pieces the model reads is cut off after that many pieces
            tokenizer.enable_truncation(max_length=EMBEDDING_MAX_TOKENS)
            # settings for running the model
            options = onnxruntime.SessionOptions()
            # how many CPU threads the model may use: the same number as the language model
            options.intra_op_num_threads = CPU_THREADS_USED
            # load the model file, to run on the CPU
            session = onnxruntime.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
            # keep the finished embedder for the next caller
            EMBEDDING_MODEL = OnnxEmbedder(session, tokenizer)
        except Exception as e:
            return Exception(f"Failed to load the embedding model {model_path}: {e}")
    # the loaded embedder (new or from an earlier call)
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
            embedder = get_embedding_model()
            if isinstance(embedder, Exception):
                return embedder
            embeddings = embedder.embed_texts(chunk_texts, True)
            if isinstance(embeddings, Exception):
                return embeddings
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

        # Convert the query into an embedding. Note that the model can fail
        embedder = get_embedding_model()
        if isinstance(embedder, Exception):
            return Exception(f"Failed to encode query: {embedder}")
        query_embedding = embedder.embed_texts([query])
        if isinstance(query_embedding, Exception):
            return Exception(f"Failed to encode query: {query_embedding}")

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