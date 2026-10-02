"""Application settings loaded from environment variables via pydantic-settings.

Usage:
    from rag_github.config.settings import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration, loaded from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Qdrant Cloud ──────────────────────────────────────────────────────
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant Cloud API key")
    qdrant_collection: str = Field(default="code_chunks", description="Collection name")

    # ── Neo4j AuraDB ──────────────────────────────────────────────────────
    neo4j_uri: str = Field(..., description="Neo4j AuraDB bolt+s URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")

    # ── Embeddings (local) ────────────────────────────────────────────────
    embedding_model: str = Field(
        default="jinaai/jina-embeddings-v2-base-code",
        description="HuggingFace model ID for code embeddings",
    )
    embedding_dim: int = Field(default=768, description="Embedding vector dimension")
    max_seq_length: int = Field(default=2048, description="Max token sequence length")

    # ── Reranker (local) ──────────────────────────────────────────────────
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace model ID for cross-encoder reranker",
    )

    # ── LLM (Groq free tier) ─────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")
    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        description="LLM model name on Groq",
    )

    # ── LangSmith Observability ───────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=True, description="Enable LangSmith tracing")
    langchain_api_key: str | None = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="rag-github", description="LangSmith project name")

    # ── Retrieval Tuning ──────────────────────────────────────────────────
    top_k_vector: int = Field(default=20, description="Top-K for initial vector search")
    hnsw_ef: int = Field(
        default=64,
        description="Qdrant HNSW search breadth; higher values favor recall over speed",
    )
    top_k_rerank: int = Field(default=10, description="Top-K after reranking")
    top_k_final: int = Field(default=5, description="Top-K results returned to user")
    min_relevance_score: float = Field(
        default=0.35,
        description="Minimum Qdrant cosine similarity required to answer a repository query",
    )
    context_budget_tokens: int = Field(
        default=3500,
        description="Approximate token budget for retrieved context sent to the LLM",
    )
    graph_expansion_depth: int = Field(
        default=2, description="Max hops for graph context expansion"
    )

    # ── Repo Limits ───────────────────────────────────────────────────────
    max_file_size_mb: float = Field(default=1.0, description="Max single file size in MB")
    max_repo_size_mb: float = Field(default=500.0, description="Max total repo size in MB")
    clone_dir: str = Field(default="/tmp/rag_repos", description="Directory for cloned repos")

    @field_validator("clone_dir")
    @classmethod
    def ensure_clone_dir_exists(cls, v: str) -> str:
        """Create the clone directory if it doesn't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def max_repo_size_bytes(self) -> int:
        return int(self.max_repo_size_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
