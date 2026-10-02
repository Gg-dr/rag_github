from __future__ import annotations

import json

from rag_github.parsing.models import ChunkMetadata, CodeChunk


class NotebookChunker:
    """Index notebook cell sources while ignoring saved execution outputs."""

    def __init__(self, max_chars: int = 6000):
        self.max_chars = max_chars

    def chunk_file(self, content: str, repo_name: str, file_path: str) -> list[CodeChunk]:
        notebook = json.loads(content)
        chunks = []
        for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            if not source.strip():
                continue
            cell_type = cell.get("cell_type", "code")
            start = 0
            while start < len(source):
                end = min(start + self.max_chars, len(source))
                if end < len(source):
                    newline = source.rfind("\n", start, end)
                    if newline > start:
                        end = newline + 1
                text = source[start:end]
                line_start = source.count("\n", 0, start) + 1
                line_end = line_start + text.count("\n")
                metadata = ChunkMetadata(
                    repo_name=repo_name,
                    file_path=file_path,
                    language="notebook",
                    chunk_type="notebook_cell",
                    symbol_name=f"cell {cell_number}",
                    start_line=line_start,
                    end_line=max(line_start, line_end),
                    notebook_cell=cell_number,
                    notebook_cell_type=cell_type,
                )
                chunks.append(CodeChunk(content=text, metadata=metadata))
                start = end
        return chunks
