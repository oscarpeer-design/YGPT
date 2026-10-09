# This file reads code repositories and is responsible for parsing

# Import external dependencies 
# Path: library used to find file paths on computer
from pathlib import Path
# tree_sitter: library used to parse languages
# Note that currently we only support codebases in Python and C++
from tree_sitter import Language, Parser, Tree, Node
import tree_sitter_python
import tree_sitter_cpp
from collections import deque
# dataclass annotation: used for data-class structs
from dataclasses import dataclass

from code_chunker import LANGUAGE_CPP, LANGUAGE_PYTHON, CodeChunker, RepositoryData

# set up valid file endings
VALID_FILE_ENDINGS = ['.cpp', '.hpp','.h', '.py']
CPP_FILE_ENDINGS = ['.cpp', '.h', '.hpp']
# set up languages and parsers
PYTHON_LANGUAGE = Language(tree_sitter_python.language())
PYTHON_PARSER = Parser(PYTHON_LANGUAGE)

CPP_LANGUAGE = Language(tree_sitter_cpp.language())
CPP_PARSER = Parser(CPP_LANGUAGE)

# Parse code from a source file written in python, returning a syntax Tree
def parse_python(source_code: bytes) -> Tree:
    # Parse source code bytes ('utf-8' flag is for bytes) and get syntax tree
    tree = PYTHON_PARSER.parse(source_code)
    # return syntax tree
    return tree

# Parse code from a source code file written in C++, returning a syntax Tree
def parse_cpp(source_code: bytes) -> Tree:
    # Parse source code bytes ('utf-8' flag is for bytes) and get syntax tree
    tree = CPP_PARSER.parse(source_code)
    # return syntax tree
    return tree

# route parsing to its constituent functions (return parse Tree). Note that at this point the file must have a valid ending which we recognise.
def parse_source_code(source_code: bytes, FILE_ENDING: str) -> Tree|Exception:
    # parse C++ file
    if FILE_ENDING in CPP_FILE_ENDINGS:
        return parse_cpp(source_code)
    # parse Python file
    elif FILE_ENDING == '.py':
        return parse_python(source_code)

    # unusual file suffix: note that we shouldn't get here because of earlier file suffix validation
    return Exception(f"Unable to parse the source code because an unusual file ending was encountered. Got {FILE_ENDING}. Anticipated these file endings: {VALID_FILE_ENDINGS}.")

# format errors in syntax tree into a string
def format_errors(errors) -> str:
     return "\n".join(
        f"Line {error['line']}, {error['type']}"
        for error in errors
     )

# find errors within tree with hard cap
def find_errors(node: Node, errors: list[dict]) -> None:
    """
    Precondition:
        node is a valid Tree-sitter node.
        errors is a mutable list used to store discovered errors.

    Postcondition:
        Up to 10 Tree-sitter ERROR nodes are appended to errors.

    Purpose:
        Find syntax errors by checking the current node and its immediate
        named children without traversing the entire syntax tree.
    """
    max_error_number = 10
    max_child_number = 10

    if node.type == "ERROR":
        errors.append({
            "line": node.start_point.row + 1,
            "type": node.type
        })

    if len(errors) >= max_error_number:
        return

    child_limit = min(node.named_child_count, max_child_number)

    for index in range(child_limit):
        child = node.named_child(index)

        if child.type == "ERROR":
            errors.append({
                "line": child.start_point.row + 1,
                "type": child.type
            })

            if len(errors) >= max_error_number:
                return

# check syntax tree for errors
def get_errors(syntax_tree: Tree) -> Exception|None:
    # check if errors exist
    if syntax_tree.root_node.has_error is False:
        return

    # set up list of errors to catch
    errors = []
    # find errors and append them to the list
    find_errors(syntax_tree.root_node, errors)

    # format these errors into a string that is readable
    error_messages = format_errors(errors)
    # create an exception and return it
    return Exception(f"""During parsing the following errors were detected:
    {error_messages}                
    """)

# read the source code from the file, given the file path (note that the file must be valid) at this point
def read_source_file(file_path: Path) -> bytes|Exception:
    try:
        # open and read file contents (as utf-8 string). Also note we want to read bytes so we read as 'rb'.
        with open(file_path, 'rb') as file:
            # read all lines as a string
            return file.read()

    except Exception as e:
        return Exception(f"The attempt to read the file {file_path} failed. Error: {e}")

# check code file suffix given file path and return the suffix (return Exception if not available)
def check_code_file_suffix(code_file_path: Path) -> str|Exception:
    # Check exception is valid
    file_suffix = code_file_path.suffix
    if file_suffix not in VALID_FILE_ENDINGS:
        return Exception(f"Got a {file_suffix} file ending. Anticipated these file endings: {VALID_FILE_ENDINGS}.")
    # The suffix is supported
    return file_suffix

# parse source file and return syntax tree and bytes (or an Exception if found). Output Exception if there are errors. in the caller function.
def parse_source_file(source_file_path:Path) -> tuple[Tree, bytes]|Exception:
    # First check file suffix is valid
    FILE_ENDING = check_code_file_suffix(source_file_path)
    # Handle invalid suffix
    if isinstance(FILE_ENDING, Exception):
        return FILE_ENDING

    # Then read source from file
    source_bytes = read_source_file(source_file_path)
    # Handle exception when reading from file
    if isinstance(source_bytes, Exception):
        return source_bytes

    # Next parse the source code
    parse_tree = parse_source_code(source_bytes, FILE_ENDING)
    # Handle exception (this shouldn't happen)
    if isinstance(parse_tree, Exception):
        return parse_tree

    # Then check parse tree for errors
    #errors = get_errors(parse_tree)
    # Handle source code errors
    #if isinstance(errors, Exception):
        #return errors

    # Finally, the source tree is valid and usable, so we return it along with the source bytes
    return parse_tree, source_bytes

# get code files in repository
def get_code_files(repository_path: Path) -> list[Path]:
    """
    Precondition:
        repository_path exists and points to a directory.

    Postcondition:
        Returns a list containing only files whose extensions are present
        in VALID_FILE_ENDINGS, in a stable order (sorted by path).

    Purpose:
        Find all supported source-code files within a repository.
    """

    code_files = set()
    # get list of code file paths that only have valid endings (global children of the respository path) by recursively scanning through this repository
    for file_ending in VALID_FILE_ENDINGS:
        # only files count (a folder could be named like a code file)
        code_files.update(path for path in repository_path.rglob(f"*{file_ending}") if path.is_file())

    # a stable order means that the same repository is always reviewed in the same order
    return sorted(code_files)

# The result of chunking a repository: the chunks of every code file that could be read, and the files that could not
@dataclass
class ChunkedRepository:
    repository_data: RepositoryData # the chunks and relationships of all the files that could be chunked
    file_count: int # how many code files were found
    skipped_files: list[str] # one note for each file that could not be read or chunked, with the reason

# chunk one code file into the shared repository data. Returns None, or an Exception if the file could not be read or chunked.
def chunk_code_file(file_path: Path, repository_data: RepositoryData) -> None|Exception:
    # parse the file and get its bytes
    parsing_results = parse_source_file(file_path)
    if isinstance(parsing_results, Exception):
        return parsing_results
    parse_tree, source_bytes = parsing_results
    # remember how much data there was, so that a file that fails half way leaves nothing behind
    chunk_count = len(repository_data.chunks)
    relationship_count = len(repository_data.relationships)
    try:
        # the chunker adds the chunks of the file to the shared repository data
        chunker = CodeChunker(
            file_path=file_path,
            source=source_bytes,
            language_used=get_language(file_path),
            repository_data=repository_data
        )
        chunker.chunk(parse_tree.root_node)
    except Exception as e:
        del repository_data.chunks[chunk_count:]
        del repository_data.relationships[relationship_count:]
        return Exception(f"Unable to chunk the file {file_path}. Error: {e}")

# chunk every code file in a repository into one shared RepositoryData, one file at a time (a parse tree is dropped as soon as its file is chunked).
# A file that can not be read or chunked is noted and the other files carry on. Returns an Exception only if there are no code files at all.
def chunk_repository(repository_path: Path) -> ChunkedRepository|Exception:
    # get relevant code files
    code_files = get_code_files(repository_path)
    if not code_files:
        return Exception(f"No code files were found in {repository_path}. Anticipated these file endings: {VALID_FILE_ENDINGS}.")
    repository_data = RepositoryData([], [])
    skipped_files: list[str] = []
    for file_path in code_files:
        chunked = chunk_code_file(file_path, repository_data)
        # note the file and carry on with the others
        if isinstance(chunked, Exception):
            skipped_files.append(f"- {chunked}")
    return ChunkedRepository(repository_data, len(code_files), skipped_files)

# get the language of a source file
def get_language(source_file: Path) -> str:
    suffix = source_file.suffix
    if suffix == '.py':
        return LANGUAGE_PYTHON
    if suffix in CPP_FILE_ENDINGS:
        return LANGUAGE_CPP