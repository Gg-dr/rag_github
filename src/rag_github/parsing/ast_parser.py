from __future__ import annotations

import logging
from dataclasses import dataclass, field

import tree_sitter
from tree_sitter_language_pack import get_parser

logger = logging.getLogger(__name__)


@dataclass
class ASTSymbol:
    node_type: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    parent_name: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    children: list[ASTSymbol] = field(default_factory=list)


LANGUAGE_NODE_TYPES = {
    "python": ["function_definition", "class_definition", "decorated_definition"],
    "javascript": ["function_declaration", "class_declaration", "method_definition"],
    "typescript": ["function_declaration", "class_declaration", "method_definition"],
    "java": [
        "class_declaration",
        "method_declaration",
        "constructor_declaration",
        "interface_declaration",
    ],
    "c": ["function_definition", "class_specifier", "struct_specifier"],
    "cpp": ["function_definition", "class_specifier", "struct_specifier"],
    "go": ["function_declaration", "method_declaration", "type_declaration"],
    "rust": ["function_item", "impl_item", "struct_item", "enum_item", "trait_item"],
    "ruby": ["method", "class", "module"],
    "php": ["function_definition", "class_declaration", "method_declaration"],
}


class ASTParser:
    def __init__(self):
        self._parsers = {}

    def _get_parser(self, language: str) -> tree_sitter.Parser | None:
        if language not in self._parsers:
            try:
                self._parsers[language] = get_parser(language)
            except Exception as e:
                logger.warning(f"Could not load grammar for {language}: {e}")
                self._parsers[language] = None
        return self._parsers[language]

    def parse_file(self, source_code: str, language: str) -> tree_sitter.Tree | None:
        parser = self._get_parser(language)
        if parser is None:
            return None
        return parser.parse(source_code.encode("utf-8"))

    def extract_symbols(
        self, tree: tree_sitter.Tree, source_code: str, language: str
    ) -> list[ASTSymbol]:
        if tree is None:
            return []

        source_bytes = source_code.encode("utf-8")

        def walk(node, parent_name=None):
            extracted = []
            node_type = node.type

            is_target = False
            lang_types = LANGUAGE_NODE_TYPES.get(language, [])
            if node_type in lang_types:
                is_target = True

            if is_target:
                name_node = self._find_name_node(node)
                name = name_node.text.decode("utf-8") if name_node else "unknown"

                qual_name = f"{parent_name}.{name}" if parent_name else name

                symbol = ASTSymbol(
                    node_type=node_type,
                    name=name,
                    qualified_name=qual_name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    parent_name=parent_name,
                )

                symbol.docstring = self._extract_docstring(node, language, source_bytes)

                for child in node.children:
                    if child != name_node:
                        child_symbols = walk(child, qual_name)
                        symbol.children.extend(child_symbols)

                extracted.append(symbol)
            else:
                for child in node.children:
                    extracted.extend(walk(child, parent_name))

            return extracted

        try:
            symbols = walk(tree.root_node)
            # Decorators are wrappers around a declaration; keep just the declaration node.
            if language == "python":

                def unwrap(items):
                    result = []
                    for item in items:
                        if item.node_type == "decorated_definition" and item.children:
                            result.extend(item.children)
                        else:
                            result.append(item)
                    return result

                symbols = unwrap(symbols)
            return symbols
        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
            return []

    def _find_name_node(self, node: tree_sitter.Node) -> tree_sitter.Node | None:
        for child in node.children:
            if child.type in ["identifier", "name"]:
                return child
            if child.type == "type_identifier":
                return child
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ["function_definition", "class_definition"]:
                    return self._find_name_node(child)
        return None

    @staticmethod
    def chunk_type(symbol: ASTSymbol, language: str, parent: str | None = None) -> str:
        node_type = symbol.node_type
        if language == "python":
            if node_type == "class_definition":
                return "class"
            if node_type == "function_definition":
                return "method" if parent else "function"
        if node_type in {
            "class_declaration",
            "class_specifier",
            "struct_specifier",
            "interface_declaration",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
            "type_declaration",
        }:
            return "class"
        if "method" in node_type or (
            parent and ("function" in node_type or "function_item" == node_type)
        ):
            return "method"
        if "function" in node_type:
            return "method" if parent else "function"
        return "module"

    def extract_imports(self, tree: tree_sitter.Tree, source_code: str, language: str) -> list[str]:
        node_types = {
            "python": {"import_statement", "import_from_statement"},
            "javascript": {"import_statement"},
            "typescript": {"import_statement"},
            "tsx": {"import_statement"},
            "java": {"import_declaration"},
            "go": {"import_declaration"},
            "rust": {"use_declaration"},
            "cpp": {"preproc_include"},
            "c": {"preproc_include"},
        }.get(language, set())
        source = source_code.encode("utf-8")
        found: list[str] = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in node_types:
                found.append(source[node.start_byte : node.end_byte].decode("utf-8", "replace"))
            else:
                stack.extend(reversed(node.children))
        return found

    def extract_calls(self, symbol: ASTSymbol, source_code: str, language: str) -> list[str]:
        """Return syntactic call names inside a symbol (best-effort)."""
        tree = self.parse_file(source_code, language)
        if tree is None:
            return []
        call_types = {"call", "call_expression", "method_invocation"}
        source = source_code.encode("utf-8")
        calls: set[str] = set()
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.start_byte < symbol.start_byte or node.end_byte > symbol.end_byte:
                continue
            if node.type in call_types and node.children:
                calls.add(
                    source[node.children[0].start_byte : node.children[0].end_byte].decode(
                        "utf-8", "replace"
                    )
                )
            stack.extend(node.children)
        return sorted(calls)

    def extract_bases(self, symbol: ASTSymbol, source_code: str, language: str) -> list[str]:
        tree = self.parse_file(source_code, language)
        if tree is None or symbol.node_type not in {
            "class_definition",
            "class_declaration",
            "class_specifier",
        }:
            return []
        source = source_code.encode("utf-8")
        node = tree.root_node
        stack = [node]
        target = None
        while stack:
            current = stack.pop()
            if current.start_byte == symbol.start_byte and current.end_byte == symbol.end_byte:
                target = current
                break
            stack.extend(current.children)
        if target is None:
            return []
        bases: list[str] = []
        for child in target.children:
            if child.type in {
                "argument_list",
                "superclasses",
                "base_class_clause",
                "class_heritage",
            }:
                text = source[child.start_byte : child.end_byte].decode("utf-8", "replace")
                bases.extend(text.strip("(): ").split(","))
        return [base.strip() for base in bases if base.strip()]

    def _extract_docstring(
        self, node: tree_sitter.Node, language: str, source_bytes: bytes
    ) -> str | None:
        if language == "python":
            if node.type in ["function_definition", "class_definition"]:
                block = node.child_by_field_name("body")
                if block and block.children:
                    first_stmt = block.children[0]
                    if first_stmt.type == "expression_statement":
                        if first_stmt.children and first_stmt.children[0].type == "string":
                            return first_stmt.children[0].text.decode("utf-8")
        elif language in ["javascript", "typescript", "java", "cpp", "c"]:
            prev = node.prev_sibling
            if prev and prev.type == "comment":
                return prev.text.decode("utf-8")
        return None
