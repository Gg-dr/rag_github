from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    repo_path: str | None = Field(None, description="GitHub repository URL to ingest")
    repo_name: str | None = Field(None, description="Repository URL or owner/name identifier")


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query string")
    repo_name: str | None = Field(
        None, description="Optional repository name to restrict the search space"
    )
    top_k: int = Field(10, description="Number of results to retrieve")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated response from the LLM")
    is_relevant: bool = Field(
        True, description="Whether repository search found enough relevant context"
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="List of source chunks used for the answer"
    )
    graph_context: dict[str, Any] = Field(
        default_factory=dict, description="Metadata retrieved from the graph traversal"
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the API")
    db_connectivity: bool = Field(..., description="True if databases are reachable")
