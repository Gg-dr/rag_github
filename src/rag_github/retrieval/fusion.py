"""
Module for Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(results_lists: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion on multiple lists of search results.
    
    Args:
        results_lists: List of lists of ranked results. Each result must be a dict
                       with an 'id' key to identify unique documents.
        k: Smoothing parameter for RRF formula (default 60).
        
    Returns:
        A list of results sorted by their RRF score.
    """
    rrf_scores: dict[str, float] = {}
    docs_by_id: dict[str, dict[str, Any]] = {}
    
    for results in results_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = str(doc.get('id', ''))
            if not doc_id:
                continue
                
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                docs_by_id[doc_id] = doc
                
            rrf_scores[doc_id] += 1.0 / (k + rank)
            
    # Sort documents by RRF score descending
    sorted_docs = []
    for doc_id, score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True):
        doc = docs_by_id[doc_id].copy()
        doc['rrf_score'] = score
        sorted_docs.append(doc)
        
    return sorted_docs
