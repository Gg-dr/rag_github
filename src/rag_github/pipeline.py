from __future__ import annotations

import logging
from typing import Any

from rag_github.config.settings import get_settings

logger = logging.getLogger(__name__)


def ingest_repository(repo_url: str) -> dict[str, Any]:
    from rag_github.embeddings.encoder import CodeEmbedder
    from rag_github.embeddings.sparse_encoder import SparseEncoder
    from rag_github.graph.graph_builder import GraphBuilder
    from rag_github.graph.neo4j_client import Neo4jClient
    from rag_github.ingestion.file_scanner import scan_repository
    from rag_github.ingestion.repo_cloner import clone_repo
    from rag_github.parsing.chunker import SemanticChunker
    from rag_github.parsing.notebook_chunker import NotebookChunker
    from rag_github.parsing.text_chunker import TextChunker
    from rag_github.vectorstore.qdrant_store import QdrantStore

    repo_path = clone_repo(repo_url)
    repo_name = "/".join(repo_path.parts[-2:])
    files = scan_repository(repo_path, repo_name)
    ast_chunker, text_chunker, notebook_chunker = (
        SemanticChunker(),
        TextChunker(),
        NotebookChunker(),
    )
    chunks = []
    for info in files:
        try:
            source = (repo_path / info.file_path).read_text(encoding="utf-8")
            if info.file_path.lower().endswith(".ipynb"):
                file_chunks = notebook_chunker.chunk_file(source, repo_name, info.file_path)
            elif info.chunk_strategy == "ast" and info.language:
                file_chunks = ast_chunker.chunk_file(
                    source, info.language, repo_name, info.file_path
                )
                # Grammar/parser gaps should not silently drop a supported file.
                if not file_chunks:
                    file_chunks = text_chunker.chunk_file(
                        source, repo_name, info.file_path, info.language
                    )
            else:
                file_chunks = text_chunker.chunk_file(
                    source, repo_name, info.file_path, info.language
                )
            chunks.extend(file_chunks)
        except (OSError, UnicodeError) as error:
            logger.warning("Skipping %s: %s", info.file_path, error)

    settings = get_settings()
    qdrant = QdrantStore(settings)
    qdrant.ensure_collection()
    try:
        dense = CodeEmbedder().encode([chunk.content for chunk in chunks])
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Could not load the local embedding model. Check that the ML dependencies are "
            "installed in the active environment. The Jina v2 model requires "
            "transformers>=4.41,<4.49; run `python -m pip install -e .` to apply the "
            "project's compatible version constraints, then retry."
        ) from error
    sparse = SparseEncoder().encode([chunk.content for chunk in chunks])
    # Only replace existing indexed data after all local encoders succeeded.
    qdrant.delete_repo(repo_name)
    qdrant.upsert_chunks(chunks, dense, sparse)
    with Neo4jClient(settings) as neo4j:
        neo4j.clear_repo(repo_name)
        graph = GraphBuilder(neo4j).build_graph(chunks, repo_name)
    return {
        "repo_name": repo_name,
        "repo_path": str(repo_path),
        "files": len(files),
        "chunks": len(chunks),
        "graph": graph,
    }


def query_repository(question: str, repo_name: str, top_k: int = 5) -> dict[str, Any]:
    from rag_github.embeddings.encoder import CodeEmbedder
    from rag_github.embeddings.sparse_encoder import SparseEncoder
    from rag_github.generation.chain import RAGChain
    from rag_github.graph.neo4j_client import Neo4jClient
    from rag_github.retrieval.graph_expander import GraphExpander
    from rag_github.retrieval.hybrid_retriever import HybridRetriever
    from rag_github.vectorstore.qdrant_store import QdrantStore

    settings = get_settings()
    with Neo4jClient(settings) as neo4j:
        retriever = HybridRetriever(QdrantStore(settings), CodeEmbedder(), SparseEncoder())
        expander = GraphExpander(neo4j, settings.graph_expansion_depth)
        return RAGChain(
            retriever,
            expander,
            min_relevance_score=settings.min_relevance_score,
            context_budget_tokens=settings.context_budget_tokens,
        ).invoke(question, repo_name, top_k)
