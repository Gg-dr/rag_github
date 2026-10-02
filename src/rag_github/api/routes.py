from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException

from rag_github.graph.neo4j_client import Neo4jClient
from rag_github.pipeline import ingest_repository, query_repository
from rag_github.vectorstore.qdrant_store import QdrantStore

from .schemas import HealthResponse, IngestRequest, QueryRequest, QueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)
ingest_jobs: dict[str, dict] = {}


def _ingest(job_id: str, repo_url: str):
    try:
        result = ingest_repository(repo_url)
        ingest_jobs[job_id] = {"status": "completed", **result}
    except Exception as error:
        ingest_jobs[job_id] = {"status": "failed", "error": str(error)}
        logger.exception("Repository ingestion failed for %s", repo_url)


@router.post("/ingest", response_model=dict)
async def ingest_repo(request: IngestRequest, background_tasks: BackgroundTasks):
    repo_url = request.repo_path or request.repo_name
    if not repo_url:
        raise HTTPException(status_code=422, detail="Provide repo_path as a GitHub URL")
    if "://" not in repo_url and "/" in repo_url:
        repo_url = "https://github.com/" + repo_url.strip("/")
    job_id = uuid4().hex
    ingest_jobs[job_id] = {"status": "running", "repo": repo_url}
    background_tasks.add_task(_ingest, job_id, repo_url)
    return {"job_id": job_id, "status": "running", "repo": repo_url}


@router.get("/ingest/{job_id}", response_model=dict)
async def ingest_status(job_id: str):
    job = ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job


@router.get("/repositories", response_model=dict)
async def list_repositories():
    try:
        return {"repositories": QdrantStore().list_repositories()}
    except Exception as error:
        logger.exception("Could not list indexed repositories")
        raise HTTPException(
            status_code=503, detail="Could not load indexed repositories"
        ) from error


@router.post("/query", response_model=QueryResponse)
async def query_repo(request: QueryRequest):
    try:
        if not request.repo_name:
            raise HTTPException(
                status_code=422, detail="repo_name is required for repository queries"
            )
        result = query_repository(request.query, request.repo_name or "", request.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            is_relevant=result["is_relevant"],
            graph_context={},
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/health", response_model=HealthResponse)
async def health_check():
    connectivity = {}
    try:
        connectivity["qdrant"] = QdrantStore().health_check()
    except Exception:
        connectivity["qdrant"] = False
    try:
        with Neo4jClient() as neo4j:
            connectivity["neo4j"] = neo4j.verify_connectivity()
    except Exception:
        connectivity["neo4j"] = False
    healthy = all(connectivity.values())
    return HealthResponse(status="healthy" if healthy else "unhealthy", db_connectivity=healthy)
