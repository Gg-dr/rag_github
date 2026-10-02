# RAG GitHub

An AI developer assistant that indexes GitHub repositories and answers questions about their code with source citations.

## One-time setup

Use Python 3.12 for the local embedding stack. In Terminal:

```bash
conda create -n rag-github python=3.12 -y  # only the first time
conda activate rag-github
cd /path/to/Rag_github
python -m pip install -e .
```

Set Qdrant, Neo4j, and Groq credentials in `.env` before ingesting or querying. The project pins Transformers below 4.49 for compatibility with the Jina v2 embedding model.

## Everyday use

From the project folder, use `./run`. It uses the project `.venv` if present or active Conda Python, checks the supported Python version, and bypasses stale global CLI commands.

```bash
./run doctor
./run health
./run api --reload
./run ingest https://github.com/owner/repo
./run query "How does this work?" --repo owner/repo
```

The API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The browser UI is at [http://127.0.0.1:8000](http://127.0.0.1:8000). Paste a GitHub URL, wait for indexing to finish, then ask questions on the same page.

Ingestion stores chunks and metadata in Qdrant and creates code nodes and relationships in Neo4j. If a Tree-sitter grammar cannot load, the file is retained as a module chunk so its content remains searchable.
