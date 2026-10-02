from __future__ import annotations

import logging
import sys

import typer

from rag_github.graph.neo4j_client import Neo4jClient
from rag_github.pipeline import ingest_repository, query_repository
from rag_github.vectorstore.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO)
app = typer.Typer(help="AI developer assistant for GitHub repositories.")


@app.command()
def health():
    """Check Qdrant and Neo4j connectivity."""
    for name, check in (
        ("Qdrant", lambda: QdrantStore().health_check()),
        ("Neo4j", lambda: _neo4j_health()),
    ):
        try:
            typer.echo(f"✅ {name} connected" if check() else f"❌ {name} unavailable")
        except Exception as error:
            typer.echo(f"❌ {name} error: {error}")


@app.command()
def doctor():
    """Show which Python environment runs the CLI and test core ML imports."""
    typer.echo(f"Python: {sys.version.split()[0]}")
    try:
        import scipy
        import sentence_transformers

        typer.echo(f"SciPy: {scipy.__version__} ({scipy.__file__})")
        typer.echo(f"Sentence Transformers: {sentence_transformers.__version__}")
    except Exception as error:
        typer.echo(f"ML import failed: {type(error).__name__}: {error}")
        raise typer.Exit(code=1) from error


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", help="Address to bind the API server"),
    port: int = typer.Option(8000, help="Port for the API server"),
    reload: bool = typer.Option(False, help="Restart the server when source files change"),
):
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("rag_github.main:app", host=host, port=port, reload=reload)


def _neo4j_health() -> bool:
    with Neo4jClient() as client:
        return client.verify_connectivity()


@app.command()
def ingest(repo_url: str):
    """Clone, index, and graph a GitHub repository."""
    result = ingest_repository(repo_url)
    typer.echo(
        f"Ingested {result['repo_name']}: {result['files']} files, "
        f"{result['chunks']} chunks; graph={result['graph']}"
    )


@app.command()
def query(q: str, repo: str = typer.Option(..., "--repo", "-r", help="Repository identifier")):
    """Ask a grounded question about an indexed repository."""
    result = query_repository(q, repo)
    typer.echo(result["answer"])
    for source in result["sources"]:
        if source.get("file_path"):
            typer.echo(
                f"  {source['file_path']}:{source.get('start_line')}-{source.get('end_line')}"
            )


if __name__ == "__main__":
    app()
