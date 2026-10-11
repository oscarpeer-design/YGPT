# This file is responsible for building chunks from a ParseTree of source code

# Import External Dependencies
# enum: ENUM library
from enum import Enum
# dataclass annotation: used for data-class structs
from dataclasses import dataclass, field
# tree-sitter: used for nodes
from tree_sitter import Language, Node, Tree
# Path: library used to find file paths on computer
from pathlib import Path
# deque: used for walking parse tree
from collections import deque
# bisect_right: used to turn byte offsets into line numbers
# re: regular expressions, used to remove comments from signatures
import re
from bisect import bisect_right
# Import Internal Dependencies
# DOCUMENT_MAX_SIGNATURE_CHARS: the longest signature kept for a chunk (a longer one is cut)
from config import DOCUMENT_MAX_SIGNATURE_CHARS

# set enums for code module types
class SymbolType(Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    INCLUDE = "include"
    STRUCT = "struct"
    ENUM = "enum"
    NAMESPACE = "namespace"


# set enums for relationship types
class RelationshipType(Enum):
    CONTAINS = "contains"
    CALLS = "calls"
    INCLUDES = "includes"
    REFERENCES = "references"
    INHERITS = "inherits"

# set language for C++
LANGUAGE_CPP = "C++"
# set independent C++ specifiers from a parse tree (these are independent nodes that may form relationships with other nodes)
INDEPENDENT_CPP_SPECIFIERS = {"class_specifier": SymbolType.CLASS, 
                              "struct_specifier": SymbolType.STRUCT, 
                              "enum_specifier": SymbolType.ENUM, 
                              "namespace_definition":SymbolType.NAMESPACE, 
                              "function_definition":SymbolType.FUNCTION}
# set C++ specifiers from a parse tree (these are nodes which are declarations so may be worth looking into)
CPP_DECLARATION_CONTAINERS = {"class_specifier", "struct_specifier", "namespace_definition", "enum_specifier"}
# set up identifier nodes where we can identify names of chunks
CPP_IDENTIFIER_NODES = {
    "identifier",
    "type_identifier",
    "field_identifier"
}

# set the node types that include a header, and that call a function, in C++
CPP_INCLUDE_NODES = {"preproc_include"}
CPP_CALL_NODES = {"call_expression"}
# set the node types that define a constant at the top level of a file (a #define in C++, an assignment statement in Python)
CPP_CONSTANT_NODES = {"preproc_def", "preproc_function_def"}

# set language for Python
LANGUAGE_PYTHON = "Python"
# set independent Python specifiers from a parse tree (these are independent nodes that may form relationships with other nodes)
INDEPENDENT_PYTHON_SPECIFIERS = {
    "class_definition": SymbolType.CLASS,
    "function_definition": SymbolType.FUNCTION,
    #"import_statement": SymbolType.INCLUDE,
    #"import_from_statement": SymbolType.INCLUDE
}
# set Python specifiers from a parse tree (these are nodes which are declarations so may be worth looking into)
PYTHON_DECLARATION_CONTAINERS = {"class_definition", "function_definition"}
# set up identifier nodes where we can identify names of chunks
PYTHON_IDENTIFIER_NODES = {"identifier"}
# set the node types that import a module, and that call a function, in Python
PYTHON_INCLUDE_NODES = {"import_statement", "import_from_statement"}
PYTHON_CALL_NODES = {"call"}
PYTHON_CONSTANT_NODES = {"expression_statement"}
# The CodeChunk structure is the smallest unit of analysis within a code base
@dataclass
class CodeChunk:
    chunk_id: int # unique chunk identifier
    chunk_name: str # name of chunk
    source: str # name of source file containing chunk
    file_path: Path # path of source file containing chunk
    language: str # name of programming language
    symbol_type: SymbolType # type of chunk
    start_line: int # line in code where chunk starts
    end_line: int # line of code where chunk ends
    called_names: list[str] = field(default_factory=list) # names of the functions this chunk calls (read from the call nodes of the parse tree, each name once)
    signature: str = "" # the declaration of the chunk without its body, on one line (for example the name, parameters and return type of a function)

# The Relationship structure defines how chunks interact
@dataclass
class Relationship:
    start_chunk_id: int # unique identifier for start chunk
    end_chunk_id: int # unique identifier for end chunk
    relationship_type: RelationshipType # type of relationship

# Distinguish between code chunks and raw discovered nodes that are interpreted from the source tree
@dataclass
class DiscoveredNode:
    node: Node # node within parse Tree
    symbol_type: SymbolType # type of chunk
    parent_chunk_id: int | None # id of parent node (may not have a parent)

# A FileInclude is one #include (C++) or import (Python) line found in a code file
@dataclass
class FileInclude:
    file_path: Path # path of the file that includes or imports
    target: str # what is included (a header such as 'vector' or 'my.h') or imported (a module such as 'os.path')
    is_system: bool # True for a C++ include in angle brackets <like this> (such an include is never part of the repository)
    line: int # line of the include or import

# A FileConstant is one top-level constant or setting of a file: a Python assignment such as NAME = 3, or a C++ #define
@dataclass
class FileConstant:
    file_path: Path # path of the file that defines it
    name: str # its name
    line: int # line it is defined on

# Hold repository data in a shareable struct
@dataclass
class RepositoryData:
    chunks: list[CodeChunk]
    relationships: list[Relationship]
    includes: list[FileInclude] = field(default_factory=list)
    constants: list[FileConstant] = field(default_factory=list) # the top-level constants and settings of every file
    line_counts: dict[Path, int] = field(default_factory=dict) # how many lines each file has

# Chunker responsible for taking a code file in a language and its source and creating a structure of important chunks
class CodeChunker:
    def __init__(self, file_path: Path, source: bytes, language_used: str, repository_data:RepositoryData):
        # intialise file path and source
        self.file_path = file_path 
        self.source = source
        # initialise chunk ids: they continue after the ids already in the shared repository data, so chunks from different files never share an id
        self.next_chunk_id = max((chunk.chunk_id for chunk in repository_data.chunks), default=-1) + 1
        # create lists of chunks and relationships between those chunks -> stored as RepositoryData object
        self.repository_data = repository_data
        # set language used
        self.language_used = language_used
        # set containers, identifier nodes and specifers from language provided
        self.independent_specifiers = {}
        self.declaration_containers = {}
        self.identifier_nodes = {}
        self.include_nodes = {}
        self.call_nodes = {}
        self.constant_nodes = {}
        self.set_language_features()
        # the chunks of this file by id (calls found in a chunk are added to it)
        self.chunks_by_id: dict[int, CodeChunk] = {}
        # track number of visited nodes
        self.nodes_visited = 0
        # get the byte offset at which each line of the source starts (used to work out line numbers)
        # note that we avoid node.start_point and node.end_point for this: reading them crashes Python (segfault) in tree-sitter 0.26.x
        self.line_starts = [0] + [index + 1 for index, byte in enumerate(source) if byte == 10]

    # set language features
    def set_language_features(self):
        # C++ used
        if self.language_used == LANGUAGE_CPP:
            self.independent_specifiers = INDEPENDENT_CPP_SPECIFIERS
            self.declaration_containers = CPP_DECLARATION_CONTAINERS
            self.identifier_nodes = CPP_IDENTIFIER_NODES
            self.include_nodes = CPP_INCLUDE_NODES
            self.call_nodes = CPP_CALL_NODES
            self.constant_nodes = CPP_CONSTANT_NODES
        # Python used 
        elif self.language_used == LANGUAGE_PYTHON:
            self.independent_specifiers = INDEPENDENT_PYTHON_SPECIFIERS
            self.declaration_containers = PYTHON_DECLARATION_CONTAINERS
            self.identifier_nodes = PYTHON_IDENTIFIER_NODES
            self.include_nodes = PYTHON_INCLUDE_NODES
            self.call_nodes = PYTHON_CALL_NODES
            self.constant_nodes = PYTHON_CONSTANT_NODES
        else:
            # The language used is not handled yet
            raise Exception(f"Unable to create code chunks for programming language {self.language_used} yet.")

    # chunk a parse Tree, starting from the root node
    def chunk(self, root_node: Node) -> None:
        self.walk_node(root_node, None)
        # remember how many lines the file has (a final line end does not start another line)
        self.repository_data.line_counts[self.file_path] = len(self.line_starts) - (1 if self.source.endswith(b"\n") else 0)

    # walk the parse Tree, identifying important features and adding those features as code chunks
    def walk_node(self, root_node: Node,parent_chunk_id: int|None) -> None:
        # we walk the parse tree depth-first using a cursor: this avoids recursion and avoids building a list of children
        # for every node (which was slow on large files), and every node is visited exactly once
        cursor = root_node.walk()
        # parent_stack holds, for each level of the tree we are inside, the id of the chunk that contains that level
        parent_stack: list[int | None] = [parent_chunk_id]
        # a root node with no children has nothing to chunk
        if not cursor.goto_first_child():
            return
        # visit every node beneath the root
        while True:
            # get the node under the cursor
            node = cursor.node
            # count the visited node
            self.nodes_visited += 1
            # get the id of the chunk that contains this node (None at the top level)
            current_parent_id = parent_stack[-1]
            # nodes beneath this one belong to the same chunk, unless this node becomes a chunk itself
            child_parent_id = current_parent_id
            # only named nodes can become chunks
            if node.is_named:
                # an include or import line belongs to the file, and a call belongs to the chunk it is inside
                if node.type in self.include_nodes:
                    self.record_includes(node)
                elif node.type in self.call_nodes:
                    self.record_call(node, current_parent_id)
                # a constant or setting at the top level of the file belongs to the file
                if len(parent_stack) == 1 and node.type in self.constant_nodes:
                    self.record_constants(node)
                # check if we have discovered a node that should become a code chunk
                discovered = self.discover_node(node, current_parent_id)
                if discovered is not None:
                    # create new chunk
                    chunk = self.create_chunk(discovered)
                    # a nameless node is not a chunk, but we still walk beneath it so that nothing inside it is lost
                    if chunk is not None:
                        # append the chunk
                        self.repository_data.chunks.append(chunk)
                        self.chunks_by_id[chunk.chunk_id] = chunk
                        # check if we discovered a parent of the current chunk
                        if current_parent_id is not None:
                            # add the new relationship
                            self.repository_data.relationships.append(
                                Relationship(
                                    start_chunk_id=current_parent_id,
                                    end_chunk_id=chunk.chunk_id,
                                    relationship_type=RelationshipType.CONTAINS
                                )
                            )
                        # declarations discovered beneath this node belong to the new chunk
                        child_parent_id = chunk.chunk_id
            # walk into the children of the node: wrapper nodes (declaration lists, blocks, template declarations)
            # must be walked through to reach the declarations inside them
            if cursor.goto_first_child():
                # remember which chunk contains the children
                parent_stack.append(child_parent_id)
                # visit the first child
                continue
            # the node has no children: move to the next sibling, climbing back up the tree when there are no more siblings
            while not cursor.goto_next_sibling():
                # climb to the parent: if there is none then the whole tree has been visited
                if not cursor.goto_parent():
                    return
                # leave this level of the tree
                parent_stack.pop()
                # once we are back at the root there is nothing left to visit
                if not parent_stack:
                    return

    # classify node and add its discovered variant
    def discover_node(self, node: Node, parent_chunk_id: int | None) -> DiscoveredNode | None:
        # get node type
        symbol_type = self.independent_specifiers.get(node.type)
        # for a function with a parent id, it is a method: otherwise it is a single function
        if node.type == "function_definition":
            if parent_chunk_id is not None:
                symbol_type = SymbolType.METHOD
            else:
                symbol_type = SymbolType.FUNCTION
        # if it is not a recognised type exit early
        if symbol_type is None:
            return None
        # return discovered node
        return DiscoveredNode(
            node=node,
            symbol_type=symbol_type,
            parent_chunk_id=parent_chunk_id
        )

    # create a new chunk
    def create_chunk(self,discovered: DiscoveredNode) -> CodeChunk|None:
        # assign new node
        node = discovered.node
        # grab chunk ID and increment chunk ID
        chunk_id = self.next_chunk_id
        self.next_chunk_id += 1
        # get source code from chunk
        source = self.source[node.start_byte:node.end_byte].decode("utf-8")
        # extract name of chunk
        chunk_name = self.extract_name(node)
        if chunk_name is None:
            return None
        # return discovered chunk (line numbers come from byte offsets, see the note in __init__)
        return CodeChunk(
            chunk_id=chunk_id,
            chunk_name=chunk_name,
            source=source,
            file_path=self.file_path,
            language=self.language_used,
            symbol_type=discovered.symbol_type,
            start_line=bisect_right(self.line_starts, node.start_byte),
            end_line=bisect_right(self.line_starts, node.end_byte),
            signature=self.extract_signature(node)
        )

    # get the declaration of a chunk without its body, on one line
    def extract_signature(self, node: Node) -> str:
        # the declaration ends where the body starts (a node without a body is all declaration)
        body = node.child_by_field_name("body")
        end_byte = body.start_byte if body is not None else node.end_byte
        text = self.source[node.start_byte:end_byte].decode("utf-8", errors="replace")
        # comments between the declaration and the body are not part of it
        text = re.sub(r"#[^\n]*" if self.language_used == LANGUAGE_PYTHON else r"//[^\n]*", "", text)
        # one line with single spaces, and not too long
        text = " ".join(text.split())
        return text[:DOCUMENT_MAX_SIGNATURE_CHARS]

    # record the constants that a top-level node defines as FileConstants
    def record_constants(self, node: Node) -> None:
        line = bisect_right(self.line_starts, node.start_byte)
        names: list[Node] = []
        if self.language_used == LANGUAGE_CPP:
            # #define NAME value
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                names.append(name_node)
        else:
            # NAME = value (only a plain name on the left counts, not NAME[0] = value or a, b = 1, 2)
            for child in node.children:
                if child.type == "assignment":
                    left = child.child_by_field_name("left")
                    if left is not None and left.type == "identifier":
                        names.append(left)
        for name_node in names:
            self.repository_data.constants.append(FileConstant(self.file_path, self.source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace"), line))

    # record the include or imports of an include node as FileIncludes
    def record_includes(self, node: Node) -> None:
        line = bisect_right(self.line_starts, node.start_byte)
        targets: list[tuple[str, bool]] = []
        if self.language_used == LANGUAGE_CPP:
            # #include <header> or #include "header"
            path = node.child_by_field_name("path")
            if path is not None:
                targets.append((path.text.decode("utf-8", errors="replace").strip('<>"'), path.type == "system_lib_string"))
        elif node.type == "import_statement":
            # import a, b.c as d
            for child in node.children_by_field_name("name"):
                # an aliased import holds the module in its own name field
                module = child.child_by_field_name("name") if child.type == "aliased_import" else child
                if module is not None:
                    targets.append((module.text.decode("utf-8", errors="replace"), False))
        else:
            # from module import names: the module is what is imported, unless it is only dots (from . import names), when the names are
            module = node.child_by_field_name("module_name")
            module_text = module.text.decode("utf-8", errors="replace") if module is not None else ""
            if module_text.strip("."):
                targets.append((module_text, False))
            else:
                for child in node.children_by_field_name("name"):
                    imported = child.child_by_field_name("name") if child.type == "aliased_import" else child
                    if imported is not None:
                        targets.append((imported.text.decode("utf-8", errors="replace"), False))
        for target, is_system in targets:
            self.repository_data.includes.append(FileInclude(self.file_path, target, is_system, line))

    # record the name of the function a call node calls in the chunk that contains the call
    def record_call(self, node: Node, chunk_id: int | None) -> None:
        # a call outside every chunk (for example at the top of a file) belongs to no chunk
        chunk = self.chunks_by_id.get(chunk_id) if chunk_id is not None else None
        if chunk is None:
            return
        name = self.extract_called_name(node.child_by_field_name("function"))
        if name and name not in chunk.called_names:
            chunk.called_names.append(name)

    # get the name that a call calls: the last part of a qualified or member name (ns::f, object.f, object->f)
    def extract_called_name(self, function_node: Node | None) -> str | None:
        current = function_node
        while current is not None:
            if current.type in self.identifier_nodes:
                return current.text.decode("utf-8", errors="replace")
            # the part after the last :: , . or ->
            if current.type in ("qualified_identifier", "template_function"):
                current = current.child_by_field_name("name")
            elif current.type == "field_expression":
                current = current.child_by_field_name("field")
            elif current.type == "attribute":
                current = current.child_by_field_name("attribute")
            else:
                # a call through something else (a lambda, a function pointer expression) has no name
                return None
        return None

    # get the name of the relevant chunk
    def extract_name(self, node: Node) -> str | None:
        """
        Precondition:
            node is a structural node selected for chunking.

        Postcondition:
            Returns the name of the node if one can be identified.
            Returns None if the node has no identifiable name.

        Purpose:
            Extract only the name of a structural code node.
        """
        # get name
        name = node.child_by_field_name("name")
        if name is not None:
            return name.text.decode("utf-8")
        # get declarator
        declarator = node.child_by_field_name("declarator")
        # check if we found a declarator
        if declarator is not None:
            return self.extract_identifier(declarator)

    # find the identifier within a declarator
    def extract_identifier(self, node: Node) -> str | None:
        """
        Precondition:
            node is a C++ declarator.

        Postcondition:
            Returns the identifier associated with the declarator,
            or None if no identifier can be found.

        Purpose:
         Follow the declarator structure to obtain the identifier.
        """

        current = node

        while current is not None:
            if current.type in self.identifier_nodes:
                return current.text.decode("utf-8")

            if current.type == "qualified_identifier":
                name = current.child_by_field_name("name")

                if name is not None:
                    return name.text.decode("utf-8")

                return None

            current = current.child_by_field_name("declarator")

        return None