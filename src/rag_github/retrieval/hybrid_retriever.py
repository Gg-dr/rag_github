from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from rag_github.embeddings.encoder import CodeEmbedder
from rag_github.embeddings.sparse_encoder import SparseEncoder
from rag_github.retrieval.reranker import CodeReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense and sparse Qdrant retrieval, rank fusion, then optional local reranking."""

    def __init__(
        self,
        qdrant_store: Any,
        dense_encoder: CodeEmbedder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        reranker: CodeReranker | None = None,
    ):
        self.qdrant_store = qdrant_store
        self.dense_encoder = dense_encoder or CodeEmbedder()
        self.sparse_encoder = sparse_encoder or SparseEncoder()
        self.reranker = reranker

    def retrieve(
        self, query: str, repo_name: str | None = None, top_k: int = 10, fetch_k: int = 60
    ) -> list[Document]:
        candidates = self.qdrant_store.hybrid_search(
            self.dense_encoder.encode_query(query),
            self.sparse_encoder.encode_query(query),
            top_k=fetch_k if self.reranker else top_k,
            repo_name=repo_name,
            candidate_limit=fetch_k,
        )
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        return [
            Document(
                page_content=item.get("page_content", item.get("content", "")),
                metadata={
                    **item.get("metadata", {}),
                    "id": item.get("id"),
                    "dense_score": item.get("dense_score"),
                    "rrf_score": item.get("rrf_score"),
                    "rerank_score": item.get("rerank_score"),
                },
            )
            for item in candidates
        ]
