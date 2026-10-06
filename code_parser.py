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

from code_chunker import LANGUAGE_CPP, LANGUAGE_PYTHON

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
        in VALID_FILE_ENDINGS.

    Purpose:
        Find all supported source-code files within a repository.
    """

    code_files = []
    # get list of code file paths that only have valid endings (global children of the respository path) by recursively scanning through this repository
    for file_ending in VALID_FILE_ENDINGS:
        code_files.extend(repository_path.rglob(f"*{file_ending}"))

    return code_files

# parse a repository to find all relevant code files
def parse_repository(repository_path: Path) -> dict[Path, Tree]|Exception:
    # get relevant code files
    code_files = get_code_files(repository_path)
    # initialise dictionary of parse trees
    parsed_files = {}
    # read source code for each file
    for file_path in code_files:
        #source = file_path.read_text(encoding="utf-8")
        # get parse tree of source
        parse_tree, source_bytes = parse_source_file(file_path)
        # check we have no error encountered
        if isinstance(parse_tree, Exception):
            return parse_tree
        # add parsed output (each path signifies a unique code file)
        parsed_files[file_path] = parse_tree
    # return parsed output
    return parsed_files

# get the language of a source file
def get_language(source_file: Path) -> str:
    suffix = source_file.suffix
    if suffix == '.py':
        return LANGUAGE_PYTHON
    if suffix in CPP_FILE_ENDINGS:
        return LANGUAGE_CPP