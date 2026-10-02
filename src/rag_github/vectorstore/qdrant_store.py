from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_github.config.settings import Settings, get_settings
from rag_github.parsing.models import CodeChunk

logger = logging.getLogger(__name__)


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert chunk id (string/hash) to UUID format for Qdrant point IDs."""
    try:
        return str(uuid.UUID(chunk_id))
    except ValueError:
        h = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class QdrantStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        self.collection_name = self.settings.qdrant_collection

    def health_check(self) -> bool:
        self.client.get_collections()
        return True

    def ensure_collection(self) -> None:
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.settings.embedding_dim, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={"sparse": models.SparseVectorParams()},
                )

                # Payload indices
                for field in ["repo_name", "language", "chunk_type", "file_path"]:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
            else:
                info = self.client.get_collection(self.collection_name)
                configured_size = self.settings.embedding_dim
                dense_config = info.config.params.vectors
                if isinstance(dense_config, dict) and "dense" in dense_config:
                    actual_size = dense_config["dense"].size
                    if actual_size != configured_size:
                        raise ValueError(
                            f"Qdrant collection vector size is {actual_size}, but settings require "
                            f"{configured_size}; migrate or recreate the collection."
                        )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}")
            raise

    def upsert_chunks(
        self, chunks: list[CodeChunk], dense_vectors: np.ndarray, sparse_vectors: list
    ) -> None:
        if not chunks:
            return

        points = []
        for idx, chunk in enumerate(chunks):
            point_id = _chunk_id_to_uuid(chunk.id)
            payload = chunk.metadata.to_dict()
            payload["content"] = chunk.content

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vectors[idx].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vectors[idx].indices, values=sparse_vectors[idx].values
                        ),
                    },
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.debug(f"Upserted {len(batch)} points to {self.collection_name}")

    def hybrid_search(
        self,
        dense_vector: np.ndarray,
        sparse_vector: Any,
        top_k: int = 20,
        repo_name: str | None = None,
    ) -> list[dict]:
        query_filter = self._repo_filter(repo_name)
        prefetch = [
            models.Prefetch(
                query=dense_vector.tolist(),
                using="dense",
                limit=top_k * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices, values=sparse_vector.values
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [self._point_result(point) for point in results.points]

    def search_dense(
        self, query_vector: np.ndarray, limit: int = 20, filter_repo: str | None = None
    ) -> list[dict]:
        return self._search_vector(query_vector.tolist(), "dense", limit, filter_repo)

    def search_sparse(
        self, query_vector: Any, limit: int = 20, filter_repo: str | None = None
    ) -> list[dict]:
        vector = models.SparseVector(indices=query_vector.indices, values=query_vector.values)
        return self._search_vector(vector, "sparse", limit, filter_repo)

    def _search_vector(
        self, vector: Any, using: str, limit: int, repo_name: str | None
    ) -> list[dict]:
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=using,
            query_filter=self._repo_filter(repo_name),
            limit=limit,
            with_payload=True,
        ).points
        return [self._point_result(point) for point in points]

    @staticmethod
    def _repo_filter(repo_name: str | None) -> models.Filter | None:
        if not repo_name:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="repo_name", match=models.MatchValue(value=repo_name))]
        )

    @staticmethod
    def _point_result(point: Any) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            "id": str(point.id),
            "score": point.score,
            "page_content": payload.get("content", ""),
            "metadata": {key: value for key, value in payload.items() if key != "content"},
        }

    def search_by_metadata(self, filters: dict[str, Any], limit: int = 10) -> list[dict]:
        must_conditions = []
        for k, v in filters.items():
            must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

        query_filter = models.Filter(must=must_conditions)

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )[0]  # scroll returns (results, next_page_offset)

        return [
            {
                "id": res.id,
                "content": res.payload.get("content", ""),
                "metadata": {k: v for k, v in res.payload.items() if k != "content"},
            }
            for res in results
        ]

    def delete_repo(self, repo_name: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_name", match=models.MatchValue(value=repo_name)
                        )
                    ]
                )
            ),
        )

    def delete_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def collection_info(self) -> dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.model_dump()
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
