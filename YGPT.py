# This is the MAIN file for the YitzGPT Program.
# This program is a RAG project around a small local LLM, which performs code review, documentation generation and code generation capabilities.
# The purpose of this program is to automate essential engineering tasks and enhance CI/CD.

from pathlib import Path
import re
import shlex
from enum import Enum

# Imported internal dependencies
from code_parser import get_language, parse_source_file, chunk_repository, ChunkedRepository
from model import*
from config import *
from code_chunker import CodeChunker, RepositoryData
from embedder_and_retriever import Embedder, Retriever
from context_builder import *
from code_reviewer import review_chunks

# Set the model now (will be loaded at a later stage)
LOCAL_MODEL = None
# Toggle whether model is loaded/valid
VALID_MODEL = False

# Enum for query types
class QueryType(Enum):
    REVIEW = "REVIEW"
    DOCUMENT = "DOCUMENT"
    CHAT = "CHAT"

# Load the model when needed
def load_local_model() -> bool:
    global LOCAL_MODEL
    # Check for uninitialised model
    if isinstance(LOCAL_MODEL, LocalModel) is False:
        # get model path
        model_path = get_and_validate_model_path()
        # if model_path is still none, this is an error (we return false)
        if model_path is None:
            return False
        # Initialise model class
        LOCAL_MODEL = LocalModel(model_path)
        print("Intialised model successfully.")
    return True

# Validate the local model by running a query
def validate_local_model() -> bool:
    # First check the model is a LocalModel
    if isinstance(LOCAL_MODEL, LocalModel) is False:
        print("Local Model uninitialised")
        return False
    # Run simple query to warm the model and check it is valid
    simple_query = "Hello"
    query_ran = LOCAL_MODEL.run_query_through_model(simple_query, 10)
    # Return whether it ran
    return query_ran

# Load and validate the local model
def load_and_validate_model() -> None:
    global VALID_MODEL
    if VALID_MODEL is False:
        # Load and validate the model if we haven't already
        model_loaded:bool = load_local_model()
        # Validate the model
        validated_model:bool = validate_local_model()
        # Check both are valid before toggling
        if model_loaded is True and validated_model is True:
            # determine the model is validated
            VALID_MODEL = True

# CHAT command -> run chat with full system context access
def run_chat(query: str):
    # check validated model
    global VALID_MODEL
    global LOCAL_MODEL
    if VALID_MODEL is True and LOCAL_MODEL is not None:
        # run chat with model
        print(f"\nRunning CHAT with model {LOCAL_MODEL.model_name}")
        LOCAL_MODEL.run_query_through_model(query)

# Parses paths given in DOCUMENT/REVIEW commands
def parse_paths(command_arguments: str) -> tuple[Path, Path | None] | Exception:
    """
    Precondition:
        command_arguments contains everything after the DOCUMENT or REVIEW
        command.

    Postcondition:
        Returns the target path and an optional markdown path.

    Purpose:
        Extract and validate the paths supplied to a DOCUMENT or REVIEW command.
    """
    # Break down the file path using shlex, a Python library that allows parsing of commands with spaces in folder/file names
    try:
        arguments = shlex.split(command_arguments, posix=False)
    except ValueError as e:
        return Exception(f"Unable to parse command arguments. Error: {e}")
    # At least one path must be provided
    if len(arguments) == 0:
        return Exception("A target file or repository path must be provided.")
    # No more than one optional markdown path can be provided
    if len(arguments) > 2:
        return Exception("Only a target path and an optional markdown path can be provided.")
    # get target path
    target_path = Path(arguments[0])
    # check it is absolute (no absolute causes errors later on)
    if not target_path.is_absolute():
        return Exception(f"Target path must be absolute: {target_path}")
    # get path of markdown file in which review/documentation will be written
    markdown_path = None
    # get markdown path
    if len(arguments) == 2:
        markdown_path = Path(arguments[1])
        # check markdown path is absolute (no absolute causes errors later on)
        if not markdown_path.is_absolute():
            return Exception(f"Markdown path must be absolute: {markdown_path}")
    # return target/markdown paths
    return target_path, markdown_path

# reviews chunked code with the model, once there is something to review (shared by the file and repository reviews)
def review_chunked_code(target_path: Path, repository_data: RepositoryData, markdown_path: Path | None, chunked: ChunkedRepository | None = None) -> None:
    # nothing to embed (e.g. an empty file or one with no named functions/classes)
    if not repository_data.chunks:
        print(f"No code chunks were found in {target_path}, so there is nothing to review.")
        return
    # we need the model before doing any expensive work
    if LOCAL_MODEL is None:
        print("Unable to run code review because of Error: local model uninitialised.")
        return
    # review the chunks (and write the markdown file if a path was given)
    review_error = review_chunks(target_path, repository_data, LOCAL_MODEL, markdown_path, chunked)
    # Output any exceptions
    if isinstance(review_error, Exception):
        print(f"Unable to run code review because of {review_error}")

# performs code review on a source file
def review_source_file(target_path:Path, markdown_path: Path | None = None) -> None:
    # initialise repository data
    repository_data = RepositoryData([], [])
    # parse file and get bytes
    parsing_results = parse_source_file(target_path)
    # check for errors -> return None if they are encountered
    if isinstance(parsing_results, Exception):
        print(parsing_results)
        return
    # get bytes and parse tree
    parse_tree, source_bytes = parsing_results
    # get language used
    language = get_language(target_path)
    print(f"Attempting to chunk and parse {target_path} using {language}.")
    # create code chunker
    chunker = CodeChunker(
        file_path=target_path,
        source=source_bytes,
        language_used=language,
        repository_data=repository_data
    )
    # chunk repository
    chunker.chunk(parse_tree.root_node)
    # get data from repository
    repository_data = chunker.repository_data
    print(f"Successfully chunked {len(repository_data.chunks)} chunks within {target_path}.")
    # review the chunks
    review_chunked_code(target_path, repository_data, markdown_path)

# performs code review on a whole repoitory, one file after another
def review_repository(target_path:Path, markdown_path: Path | None = None) -> None:
    print(f"Attempting to chunk and parse the code files in {target_path}.")
    # chunk every code file in the repository (a file that can not be read is noted and skipped)
    chunked = chunk_repository(target_path)
    # check for errors -> return None if they are encountered
    if isinstance(chunked, Exception):
        print(chunked)
        return
    print(f"Successfully chunked {len(chunked.repository_data.chunks)} chunks within {chunked.file_count - len(chunked.skipped_files)} of {chunked.file_count} code files.")
    # say which files could not be read
    for skipped_file in chunked.skipped_files:
        print(f"Skipped {skipped_file[2:]}")
    # review the chunks, file by file
    review_chunked_code(target_path, chunked.repository_data, markdown_path, chunked)

# runs a code review
def run_review(input_str: str) -> None:
    # Get the rest of the input after the first command
    remaining_input = input_str.partition(' ')[2]
    # Parse the target and markdown paths
    paths_result = parse_paths(remaining_input)
    # if paths_result is exception we print it and discontinue
    if isinstance(paths_result, Exception):
        print(paths_result)
        return
    # get target and markdown paths
    target_path, markdown_path = paths_result
    # if markdown_path has nonstandard file ending we discontinue
    if markdown_path is not None:
        markdown_suffix = markdown_path.suffix
        if markdown_suffix not in [".md", ".txt"]:
            print(f"Nonstandard markdown file ending. Expected '.txt' or '.md'. Instead got {markdown_suffix}")
            return
    # Determine whether the target is a file or repository
    if target_path.is_file():
        # conduct code review on single file
        review_source_file(target_path, markdown_path)

    elif target_path.is_dir():
        # conduct code review on a whole repository
        review_repository(target_path, markdown_path)
    else:
        print(f"Target path does not exist: {target_path}")

# creates software documentation
def create_documentation(input_str: str) -> None:
    # Get the rest of the input after the first command
    remaining_input = input_str.partition(' ')[2]
    # Parse the target and markdown paths
    document_path = parse_paths(remaining_input)
    # if document_path is exception we print it and discontinue
    if isinstance(document_path, Exception):
        print(document_path)
        return
    target_path, markdown_path = document_path
    # if markdown_path has nonstandard file ending we discontinue
    if markdown_path is not None:
        markdown_suffix = markdown_path.suffix
        if markdown_suffix != ".txt" or markdown_suffix != ".md":
            print(f"Nonstandard markdown file ending. Expected '.txt' or '.md'. Instead got {markdown_suffix}")
            return

# Checks query type
def get_query_type(input_str: str) -> QueryType:
    # Get first word (command)
    first_word = input_str.partition(' ')[0]
    if first_word is None:
        first_word = input_str
    # Convert this to uppercase
    first_word = first_word.upper()
    # Get the rest of the input
    #remaining_input = input_str.partition(' ')[2]
    # Check if we need documentation
    if first_word == "DOCUMENT":
        return QueryType.DOCUMENT
    # check if we need review
    elif first_word == "REVIEW":
        return QueryType.REVIEW
    # we are chatting directly
    return QueryType.CHAT

# Help function for the user
def output_user_help():
    # Output user information regarding how to use the RAG system
    print(f"""
    - YitzGPT is a CI/CD tool and RAG system around a small local AI model. 
    - It requires a local AI model ('.gguf' file) to be installed on your computer. The model must be less than {format_size_gb(MAX_MODEL_SIZE)}. We recommend a code-facing model like qwen2.5-coder.
    - Within this codebase there should be a 'model_path.txt' file which contains the absolute path of the local LLM you wish to use on the first line.
    - All documentation generated is saved in this folder, for the model's future reference. 

    - This system supports DOCUMENTATION, CODE REVIEW, and CHAT functionalities

    The options to use the system are as follows:
    * DOCUMENT <absolute file path> <absolute path for the markown>: writes a markdown file documenting a single code file. Note that any file paths provided must be absolute.
    * DOCUMENT <absolute repository path> <absolute path for the markown>: writes a markdown file documenting a local repository on your PC. Note that any folder paths provided must be absolute.
    
    * REVIEW <absolute file path> [absolute path for markdown]: does performance and security analysis on a single code file. If a additional folder path is provided it will write the review in a markdown file. This opion requires absolute paths.
    * REVIEW <absolute repository path> [absolute path for markdown]: does performance and security analysis on a local repository. If a additional folder path is provided it will write the review in a markdown file. This opion requires absolute paths. 

    * CHAT: allows chat functionality for the local model, with full access to generated system documentation. 
    
    * HELP: displays this help message.

    * EXIT: Exits the program

    """)

# CLI Runner
def run_cli() -> None:
    # load and validate model
    load_and_validate_model()
    # repeatedly get input from user
    input_str:str = ""
    while (input_str:= input("Enter Query Type: ")) != "EXIT":
        # Check for HELP command -> output user help
        if input_str == "HELP":
            output_user_help()
        # Run multi-part commands like CHAT/DOCUMENT/REVIEW
        else:
            # Check query type
            query_type = get_query_type(input_str)
            # if query_type is chat, run chat
            if query_type == QueryType.CHAT:
                run_chat(input_str)
            # if query_type is review, run review
            elif query_type == QueryType.REVIEW:
                run_review(input_str)
            # document source code
            else:
                create_documentation(input_str)

# Main entry point
if __name__ == "__main__":
    # Print user help
    output_user_help()
    # Run CLI and commands
    run_cli()
