from __future__ import annotations

import logging

from rag_github.parsing.ast_parser import ASTParser
from rag_github.parsing.models import ChunkMetadata, CodeChunk

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Split source at symbol boundaries and retain uncovered module code."""

    def __init__(self, max_chunk_tokens: int = 6000):
        self.max_chunk_tokens = max_chunk_tokens
        self.parser = ASTParser()

    @staticmethod
    def _complexity(line_count: int) -> str:
        return "low" if line_count < 50 else "medium" if line_count <= 150 else "high"

    def chunk_file(
        self, source_code: str, language: str, repo_name: str, file_path: str
    ) -> list[CodeChunk]:
        tree = self.parser.parse_file(source_code, language)
        if tree is None:
            lines = source_code.splitlines()
            if not source_code.strip():
                return []
            metadata = ChunkMetadata(
                repo_name=repo_name,
                file_path=file_path,
                language=language,
                chunk_type="module",
                start_line=1,
                end_line=max(1, len(lines)),
            )
            return [CodeChunk(content=source_code, metadata=metadata)]
        symbols = self.parser.extract_symbols(tree, source_code, language)
        lines = source_code.splitlines()
        imports = tuple(self.parser.extract_imports(tree, source_code, language))
        chunks: list[CodeChunk] = []
        covered: list[tuple[int, int]] = []

        def add(
            text: str,
            start: int,
            end: int,
            *,
            kind: str,
            name: str | None = None,
            parent: str | None = None,
            docstring: str | None = None,
            calls: tuple[str, ...] = (),
            bases: tuple[str, ...] = (),
        ) -> None:
            if not text.strip():
                return
            metadata = ChunkMetadata(
                repo_name=repo_name,
                file_path=file_path,
                language=language,
                chunk_type=kind,
                symbol_name=name,
                start_line=start,
                end_line=end,
                parent_symbol=parent,
                imports=imports,
                docstring=docstring,
                complexity_hint=self._complexity(end - start + 1),
                calls=calls,
                bases=bases,
            )
            chunks.append(CodeChunk(content=text, metadata=metadata))

        def visit(symbols_to_visit, parent: str | None = None) -> None:
            for symbol in symbols_to_visit:
                name = symbol.name if symbol.name != "unknown" else symbol.qualified_name
                kind = self.parser.chunk_type(symbol, language, parent)
                # Container symbols are represented by their own chunk if small; for large
                # containers retain child symbols and let module chunks cover the rest.
                body = source_code.encode("utf-8")[symbol.start_byte : symbol.end_byte].decode(
                    "utf-8"
                )
                if len(body) // 4 <= self.max_chunk_tokens:
                    add(
                        body,
                        symbol.start_line,
                        symbol.end_line,
                        kind=kind,
                        name=name,
                        parent=parent,
                        docstring=symbol.docstring,
                        calls=tuple(self.parser.extract_calls(symbol, source_code, language)),
                        bases=tuple(self.parser.extract_bases(symbol, source_code, language)),
                    )
                    covered.append((symbol.start_line, symbol.end_line))
                elif symbol.children:
                    visit(symbol.children, name)
                else:
                    # Never split a symbol; unusually large symbols remain whole.
                    add(
                        body,
                        symbol.start_line,
                        symbol.end_line,
                        kind=kind,
                        name=name,
                        parent=parent,
                        docstring=symbol.docstring,
                    )
                    covered.append((symbol.start_line, symbol.end_line))

        if symbols:
            visit(symbols)
        if not symbols:
            add(source_code, 1, max(1, len(lines)), kind="module")

        # Keep file-level code not represented by any symbol, including imports and constants.
        covered.sort()
        cursor = 1
        module_parts: list[tuple[int, int]] = []
        for start, end in covered:
            if start > cursor:
                module_parts.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= len(lines):
            module_parts.append((cursor, len(lines)))
        for start, end in module_parts:
            add("\n".join(lines[start - 1 : end]), start, end, kind="module")
        # Keep all source content either in a semantic symbol chunk or a module chunk.
        return chunks
