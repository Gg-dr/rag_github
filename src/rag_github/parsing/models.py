"""Data models for code chunks and their metadata.

These models are the core data structures shared across the entire pipeline:
ingestion → parsing → embedding → storage → retrieval → generation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkMetadata:
    """Structured metadata attached to every code/text chunk.

    This metadata enables precise file:line citations in generated answers
    and powers filtered search in Qdrant.
    """

    repo_name: str
    """Repository identifier, e.g. 'fastapi/fastapi'."""

    file_path: str
    """Relative path within the repo, e.g. 'src/auth/service.py'."""

    language: str
    """Programming language or file type, e.g. 'python', 'javascript', 'markdown'."""

    chunk_type: str
    """AST node type: 'module', 'class', 'function', 'method', 'block', 'text'."""

    symbol_name: str | None = None
    """Fully qualified symbol name, e.g. 'AuthService.validate_token'."""

    start_line: int = 1
    """1-indexed start line in the source file."""

    end_line: int = 1
    """1-indexed end line in the source file (inclusive)."""

    parent_symbol: str | None = None
    """Enclosing symbol name (e.g. class name for a method)."""

    imports: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    notebook_cell: int | None = None
    notebook_cell_type: str | None = None
    """Import statements found in or relevant to this chunk."""

    docstring: str | None = None
    """Extracted docstring or leading comment block."""

    complexity_hint: str = "low"
    """Heuristic complexity: 'low', 'medium', 'high'."""

    def to_dict(self) -> dict:
        """Convert to a flat dict suitable for Qdrant payload."""
        return {
            "repo_name": self.repo_name,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "symbol_name": self.symbol_name or "",
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_symbol": self.parent_symbol or "",
            "imports": list(self.imports),
            "calls": list(self.calls),
            "bases": list(self.bases),
            "notebook_cell": self.notebook_cell,
            "notebook_cell_type": self.notebook_cell_type,
            "docstring": self.docstring or "",
            "complexity_hint": self.complexity_hint,
        }


@dataclass(frozen=True)
class CodeChunk:
    """A single chunk of source code or text with its metadata.

    The chunk ID is a deterministic hash of (repo, file, start_line, end_line)
    so re-indexing produces stable IDs.
    """

    content: str
    """Raw source code or text content of the chunk."""

    metadata: ChunkMetadata
    """Structured metadata for this chunk."""

    id: str = field(default="", init=False)
    """Deterministic chunk ID (SHA-256 of repo+file+lines)."""

    def __post_init__(self) -> None:
        # Compute deterministic ID from provenance
        key = (
            f"{self.metadata.repo_name}:"
            f"{self.metadata.file_path}:"
            f"{self.metadata.start_line}:"
            f"{self.metadata.end_line}"
            f":{self.metadata.symbol_name or self.metadata.chunk_type}"
            f":{self.metadata.notebook_cell or ''}"
        )
        chunk_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        # Bypass frozen dataclass restriction for init
        object.__setattr__(self, "id", chunk_id)

    @property
    def citation(self) -> str:
        """Format as a file:line citation string."""
        path = self.metadata.file_path
        if self.metadata.notebook_cell is not None:
            path += f"#cell-{self.metadata.notebook_cell}"
        return f"{path}:{self.metadata.start_line}-{self.metadata.end_line}"

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (chars / 4)."""
        return len(self.content) // 4


@dataclass
class FileInfo:
    """Metadata about a single file discovered during repo scanning."""

    file_path: str
    """Path relative to the repo root (posix-style)."""

    repo_name: str
    """Repository identifier, e.g. 'owner/repo'."""

    language: str | None
    """Detected language name (tree-sitter grammar), or None."""

    chunk_strategy: str
    """'ast' for AST-aware chunking, 'text' for line-based fallback."""

    size_bytes: int
    """File size in bytes."""


@dataclass
class GraphNode:
    """A node in the code knowledge graph."""

    node_type: str
    """'File', 'Module', 'Class', 'Function', 'Import'."""

    name: str
    """Symbol name."""

    qualified_name: str
    """Fully qualified name including parent symbols."""

    file_path: str
    """Relative file path in the repo."""

    repo_name: str
    """Repository identifier."""

    start_line: int = 0
    end_line: int = 0
    properties: dict = field(default_factory=dict)
    """Additional properties (language, is_method, etc.)."""


@dataclass
class GraphEdge:
    """An edge (relationship) in the code knowledge graph."""

    edge_type: str
    """'CONTAINS', 'CALLS', 'IMPORTS', 'INHERITS', 'REFERENCES'."""

    source_name: str
    """Qualified name of the source node."""

    target_name: str
    """Qualified name of the target node."""

    properties: dict = field(default_factory=dict)
    """Additional edge properties."""
