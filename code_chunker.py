# This file is responsible for building chunks from a ParseTree of source code

# Import External Dependencies
# enum: ENUM library
from enum import Enum
# dataclass annotation: used for data-class structs
from dataclasses import dataclass
# tree-sitter: used for nodes
from tree_sitter import Language, Node, Tree
# Path: library used to find file paths on computer
from pathlib import Path
# deque: used for walking parse tree
from collections import deque

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

# Hold repository data in a shareable struct
@dataclass
class RepositoryData:
    chunks: list[CodeChunk]
    relationships: list[Relationship]

# Chunker responsible for taking a code file in a language and its source and creating a structure of important chunks
class CodeChunker:
    def __init__(self, file_path: Path, source: bytes, language_used: str, repository_data:RepositoryData):
        # intialise file path and source
        self.file_path = file_path 
        self.source = source
        # initialise chunk ids (we will increment from zero)
        self.next_chunk_id = 0
        # create lists of chunks and relationships between those chunks -> stored as RepositoryData object
        self.repository_data = repository_data
        # set language used
        self.language_used = language_used
        # set containers, identifier nodes and specifers from language provided
        self.independent_specifiers = {}
        self.declaration_containers = {}
        self.identifier_nodes = {}
        self.set_language_features()
        # track number of visited nodes
        self.nodes_visited = 0

    # set language features
    def set_language_features(self):
        # C++ used
        if self.language_used == LANGUAGE_CPP:
            self.independent_specifiers = INDEPENDENT_CPP_SPECIFIERS
            self.declaration_containers = CPP_DECLARATION_CONTAINERS
            self.identifier_nodes = CPP_IDENTIFIER_NODES
        # Python used 
        elif self.language_used == LANGUAGE_PYTHON:
            self.independent_specifiers = INDEPENDENT_PYTHON_SPECIFIERS
            self.declaration_containers = PYTHON_DECLARATION_CONTAINERS
            self.identifier_nodes = PYTHON_IDENTIFIER_NODES
        else:
            # The language used is not handled yet
            raise Exception(f"Unable to create code chunks for programming language {self.language_used} yet.")

    # chunk a parse Tree, starting from the root node
    def chunk(self, root_node: Node) -> None:
        self.walk_node(root_node, None)

    # walk the parse Tree, identifying important features and adding those features as code chunks
    def walk_node(self, root_node: Node,parent_chunk_id: int|None) -> None:
        # we walk the parse tree using BFS -> avoid issues with recursion
        # and only focus on the nodes we need, making control flow predictable
        # set up queue of nodes to visit and initialise it
        nodes_to_visit: deque[tuple[Node, int | None]] = deque(
            [(root_node, parent_chunk_id)]
        )
        # conduct breadth-first-search on parse Tree
        while nodes_to_visit:
            # get node and the id of its parent
            node, current_parent_id = nodes_to_visit.popleft()
            # check how many nodes visted
            self.nodes_visited += 1
            print(f"visited nodes: {self.nodes_visited}")
            # classify the direct children of the current node
            for child in node.named_children:
                # set current parent ID
                child_parent_id = current_parent_id
                # check if we have discovered a node that should become a code chunk
                discovered = self.discover_node(child, current_parent_id)
                if discovered is not None:
                    # create new chunk and append it
                    chunk = self.create_chunk(discovered)
                    # continue on empty chunk
                    if chunk is None:
                        continue
                    self.repository_data.chunks.append(chunk)
                    # check if we discovered a parent of the current chunk
                    if discovered.parent_chunk_id is not None:
                        # add the new relationship
                        self.repository_data.relationships.append(
                            Relationship(
                                start_chunk_id=discovered.parent_chunk_id,
                                end_chunk_id=chunk.chunk_id,
                                relationship_type=RelationshipType.CONTAINS
                            )
                        )
                    # reset child parent ID so that any declarations discovered beneath this node belong to the new chunk
                    child_parent_id = chunk.chunk_id
                # check the node type and add declaration-containing nodes to the queue
                if child.type in self.declaration_containers:
                    nodes_to_visit.append((child, child_parent_id))

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
        # return discovered chunk
        return CodeChunk(
            chunk_id=chunk_id,
            chunk_name=chunk_name,
            source=source,
            file_path=self.file_path,
            language=self.language_used,
            symbol_type=discovered.symbol_type,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1
        )

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