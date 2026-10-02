from __future__ import annotations

import logging
from typing import Any

from rag_github.graph.neo4j_client import Neo4jClient
from rag_github.parsing.models import CodeChunk

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Write a portable Neo4j graph using parameterized, APOC-free Cypher."""

    def __init__(self, neo4j_client: Neo4jClient):
        self.client = neo4j_client
        self.client.run_write_query(
            "CREATE INDEX file_repo_path IF NOT EXISTS FOR (n:File) ON (n.repo_name, n.file_path)"
        )
        self.client.run_write_query(
            "CREATE INDEX symbol_repo_qname IF NOT EXISTS FOR (n:Symbol) "
            "ON (n.repo_name, n.qualified_name)"
        )
        self.client.run_write_query(
            "CREATE INDEX chunk_repo_file_symbol IF NOT EXISTS FOR (n:CodeChunk) "
            "ON (n.repo_name, n.file_path, n.symbol_name)"
        )

    def build_graph(self, chunks: list[CodeChunk], repo_name: str) -> dict[str, int]:
        files: dict[str, dict[str, Any]] = {}
        symbols: dict[str, dict[str, Any]] = {}
        contains: set[tuple[str, str]] = set()
        imports: set[tuple[str, str]] = set()
        calls: set[tuple[str, str]] = set()
        inherits: set[tuple[str, str]] = set()

        for chunk in chunks:
            meta = chunk.metadata
            path = meta.file_path
            files[path] = {
                "repo": repo_name,
                "path": path,
                "name": path.rsplit("/", 1)[-1],
                "language": meta.language,
            }
            if meta.chunk_type not in {"class", "function", "method"} or not meta.symbol_name:
                continue
            parent = f"{meta.parent_symbol}." if meta.parent_symbol else ""
            qname = f"{path}::{parent}{meta.symbol_name}"
            symbols[qname] = {
                "repo": repo_name,
                "qname": qname,
                "name": meta.symbol_name,
                "kind": "Function" if meta.chunk_type in {"function", "method"} else "Class",
                "path": path,
                "start": meta.start_line,
                "end": meta.end_line,
            }
            symbols[qname]["content"] = chunk.content
            contains.add((f"FILE::{path}", qname))
            if meta.parent_symbol:
                parent_qname = f"{path}::{meta.parent_symbol}"
                contains.add((parent_qname, qname))
            for statement in meta.imports:
                imports.add((path, statement))
        symbol_by_name: dict[str, list[str]] = {}
        for qname, row in symbols.items():
            symbol_by_name.setdefault(row["name"], []).append(qname)
        # Resolve syntactic call/base names against symbols in this repository only.
        # Ambiguous short names are deliberately not connected.
        for chunk in chunks:
            meta = chunk.metadata
            if not meta.symbol_name:
                continue
            parent = f"{meta.parent_symbol}." if meta.parent_symbol else ""
            source = f"{meta.file_path}::{parent}{meta.symbol_name}"
            for name in meta.calls:
                targets = symbol_by_name.get(name.split(".")[-1], [])
                if len(targets) == 1:
                    calls.add((source, targets[0]))
            for name in meta.bases:
                targets = symbol_by_name.get(name.split(".")[-1], [])
                if len(targets) == 1:
                    inherits.add((source, targets[0]))

        file_rows = list(files.values())
        symbol_rows = list(symbols.values())
        import_rows = [
            {"repo": repo_name, "qname": f"IMPORT::{statement}", "name": statement, "path": path}
            for path, statement in imports
        ]
        self._write_batches(
            "UNWIND $rows AS row MERGE (f:File {repo_name: row.repo, file_path: row.path}) "
            "SET f.name = row.name, f.language = row.language",
            file_rows,
        )
        self._write_batches(
            "UNWIND $rows AS row MERGE (s:Symbol {repo_name: row.repo, qualified_name: row.qname}) "
            "SET s.name = row.name, s.kind = row.kind, s.file_path = row.path, "
            "s.start_line = row.start, s.end_line = row.end, s:CodeSymbol",
            symbol_rows,
        )
        self._write_batches(
            "UNWIND $rows AS row MATCH (s:Symbol {repo_name: row.repo, qualified_name: row.qname}) "
            "SET s:Class",
            [row for row in symbol_rows if row["kind"] == "Class"],
        )
        self._write_batches(
            "UNWIND $rows AS row MATCH (s:Symbol {repo_name: row.repo, qualified_name: row.qname}) "
            "SET s:Function",
            [row for row in symbol_rows if row["kind"] == "Function"],
        )
        self._write_batches(
            "UNWIND $rows AS row MERGE (i:Import {repo_name: row.repo, qualified_name: row.qname}) "
            "SET i.name = row.name, i.file_path = row.path",
            import_rows,
        )
        self._write_batches(
            "UNWIND $rows AS row MATCH (s:Symbol {repo_name: row.repo, qualified_name: row.qname}) "
            "MERGE (c:CodeChunk {repo_name: row.repo, file_path: row.path, symbol_name: row.name}) "
            "SET c.content = row.content, c.start_line = row.start, c.end_line = row.end "
            "MERGE (s)-[:DEFINED_IN]->(c)",
            [
                {
                    "repo": row["repo"],
                    "qname": row["qname"],
                    "path": row["path"],
                    "name": row["name"],
                    "content": row["content"],
                    "start": row["start"],
                    "end": row["end"],
                }
                for row in symbol_rows
            ],
        )
        self._write_batches(
            "UNWIND $rows AS row MATCH (f:File {repo_name: row.repo, file_path: row.path}) "
            "MATCH (i:Import {repo_name: row.repo, qualified_name: row.qname}) "
            "MERGE (f)-[:IMPORTS]->(i)",
            [
                {"repo": repo_name, "path": path, "qname": f"IMPORT::{statement}"}
                for path, statement in imports
            ],
        )

        rel_rows: list[dict[str, str]] = []
        for source, target in contains:
            if not source.startswith("FILE::"):
                rel_rows.append(
                    {"repo": repo_name, "source": source, "target": target, "kind": "CONTAINS"}
                )
        for source, target in calls:
            rel_rows.append(
                {"repo": repo_name, "source": source, "target": target, "kind": "CALLS"}
            )
        for source, target in inherits:
            rel_rows.append(
                {"repo": repo_name, "source": source, "target": target, "kind": "INHERITS"}
            )
        # File containment uses a file-specific match; symbol relationships share a compact query.
        symbol_relations = rel_rows
        file_relations = [
            {"repo": repo_name, "path": source.removeprefix("FILE::"), "target": target}
            for source, target in contains
            if source.startswith("FILE::")
        ]
        self._write_batches(
            "UNWIND $rows AS row MATCH (f:File {repo_name: row.repo, file_path: row.path}) "
            "MATCH (s:Symbol {repo_name: row.repo, qualified_name: row.target}) "
            "MERGE (f)-[:CONTAINS]->(s)",
            file_relations,
        )
        self._write_batches(
            "UNWIND $rows AS row "
            "MATCH (a:Symbol {repo_name: row.repo, qualified_name: row.source}) "
            "MATCH (b:Symbol {repo_name: row.repo, qualified_name: row.target}) "
            "MERGE (a)-[:CONTAINS]->(b)",
            [row for row in symbol_relations if row["kind"] == "CONTAINS"],
        )
        self._write_batches(
            "UNWIND $rows AS row "
            "MATCH (a:Symbol {repo_name: row.repo, qualified_name: row.source}) "
            "MATCH (b:Symbol {repo_name: row.repo, qualified_name: row.target}) "
            "MERGE (a)-[:CALLS]->(b)",
            [row for row in symbol_relations if row["kind"] == "CALLS"],
        )
        self._write_batches(
            "UNWIND $rows AS row "
            "MATCH (a:Symbol {repo_name: row.repo, qualified_name: row.source}) "
            "MATCH (b:Symbol {repo_name: row.repo, qualified_name: row.target}) "
            "MERGE (a)-[:INHERITS]->(b)",
            [row for row in symbol_relations if row["kind"] == "INHERITS"],
        )
        return {
            "files": len(file_rows),
            "symbols": len(symbol_rows),
            "imports": len(import_rows),
            "relationships": len(file_relations) + len(symbol_relations) + len(imports),
        }

    def _write_batches(self, cypher: str, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.client.batch_write(cypher, rows)
