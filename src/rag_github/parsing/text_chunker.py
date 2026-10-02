from __future__ import annotations

from rag_github.parsing.models import ChunkMetadata, CodeChunk


class TextChunker:
    """Line-based chunking for documents and source without a usable AST grammar."""

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 10):
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    def chunk_file(
        self, content: str, repo_name: str, file_path: str, language: str | None = None
    ) -> list[CodeChunk]:
        lines = content.splitlines()
        if not lines:
            return []
        chunks: list[CodeChunk] = []
        step = self.chunk_size - self.chunk_overlap
        for offset in range(0, len(lines), step):
            section = lines[offset:offset + self.chunk_size]
            metadata = ChunkMetadata(
                repo_name=repo_name, file_path=file_path, language=language or "text",
                chunk_type="text", start_line=offset + 1,
                end_line=offset + len(section),
            )
            chunks.append(CodeChunk(content="\n".join(section), metadata=metadata))
        return chunks
