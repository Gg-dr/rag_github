from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

class SparseEncoder:
    """Sparse BM25 encoder using fastembed."""
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding
            logger.info(f"Loading sparse embedding model: {self.model_name}")
            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model
    
    def encode(self, texts: List[str]) -> List[SparseVector]:
        if not texts:
            return []
        logger.debug(f"Encoding {len(texts)} texts sparsely.")
        embeddings = self.model.embed(texts)
        
        results = []
        for emb in embeddings:
            results.append(SparseVector(indices=list(emb.indices), values=list(emb.values)))
        return results
    
    def encode_query(self, query: str) -> SparseVector:
        return self.encode([query])[0]
