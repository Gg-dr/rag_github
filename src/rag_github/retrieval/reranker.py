"""
Module for reranking search results using CrossEncoder models.
"""
from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class CodeReranker:
    """Reranks retrieved code chunks using a CrossEncoder model."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize the CodeReranker.
        
        Args:
            model_name: The HuggingFace model name for the CrossEncoder.
        """
        self.model_name = model_name
        logger.info(f"Initializing CodeReranker with model: {model_name}")
        try:
            # Load the heavy ML stack only when reranking is explicitly enabled.
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model {model_name}: {e}")
            raise

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
        """
        Rerank a list of chunks based on a query.
        
        Args:
            query: The search query.
            chunks: A list of dicts, where each dict has a 'page_content' or 'text' key.
            top_k: Number of top chunks to return.
            
        Returns:
            The reranked list of chunks (up to top_k).
        """
        if not chunks:
            return []
            
        # Extract text content from chunks for pairing with the query
        pairs = []
        for chunk in chunks:
            text = chunk.get('page_content', chunk.get('text', ''))
            pairs.append([query, text])
            
        # Predict scores
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Error during reranking prediction: {e}")
            return chunks[:top_k]
            
        # Combine chunks with scores and sort
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            new_chunk = chunk.copy()
            new_chunk['rerank_score'] = float(scores[i])
            scored_chunks.append(new_chunk)
            
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return scored_chunks[:top_k]
