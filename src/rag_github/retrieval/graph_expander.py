from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GraphExpander:
    """Expand retrieved symbols with connected code locations stored in Neo4j."""

    def __init__(self, neo4j_client: Any, max_depth: int = 2):
        self.client = neo4j_client
        self.max_depth = max(1, min(max_depth, 4))

    def expand_context(self, documents: list[Document], repo_name: str,
                       max_expansions: int = 5) -> list[Document]:
        if not documents or max_expansions <= 0:
            return documents
        names = list(dict.fromkeys(doc.metadata.get("symbol_name") for doc in documents
                                   if doc.metadata.get("symbol_name")))
        if not names:
            return documents
        query = f"""
        UNWIND $names AS name
        MATCH (s:Symbol {{repo_name: $repo, name: name}})-[*1..{self.max_depth}]-(related:Symbol)
        WITH DISTINCT related LIMIT $limit
        MATCH (related)-[:DEFINED_IN]->(c:CodeChunk {{repo_name: $repo}})
        RETURN c.content AS content, c.file_path AS file_path,
               c.start_line AS start_line, c.end_line AS end_line,
               related.name AS symbol_name
        """
        rows = self.client.run_query(query, {"names": names, "repo": repo_name,
                                             "limit": max_expansions})
        expanded = [Document(page_content=row["content"], metadata={
            "repo_name": repo_name, "file_path": row["file_path"],
            "start_line": row["start_line"], "end_line": row["end_line"],
            "symbol_name": row["symbol_name"], "source": "graph",
        }) for row in rows if row.get("content")]
        return documents + expanded
