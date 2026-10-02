from __future__ import annotations

import logging

import numpy as np

from rag_github.config.settings import get_settings

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """Dense embeddings using sentence-transformers with Jina code model."""

    def __init__(self, model_name: str | None = None, max_seq_length: int | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.max_seq_length = max_seq_length or settings.max_seq_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading dense embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self._model.max_seq_length = self.max_seq_length
            actual_dim = self._model.get_sentence_embedding_dimension()
            configured_dim = get_settings().embedding_dim
            if actual_dim != configured_dim:
                raise ValueError(
                    f"Embedding model outputs {actual_dim} dimensions, but "
                    f"EMBEDDING_DIM is set to {configured_dim}. Update .env to match "
                    "and recreate the Qdrant collection if it already exists."
                )
        return self._model

    def encode(
        self, texts: list[str], batch_size: int = 4, show_progress: bool = True
    ) -> np.ndarray:
        if not texts:
            return np.array([])
        logger.debug(f"Encoding {len(texts)} texts densely.")
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]
