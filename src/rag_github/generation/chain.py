from __future__ import annotations

from rag_github.generation.groq_client import GroqClient


class RAGChain:
    """Small explicit RAG flow: retrieve, expand, format citations, answer."""

    def __init__(self, retriever, graph_expander, llm=None):
        self.retriever = retriever
        self.graph_expander = graph_expander
        self.llm = llm or GroqClient().model

    def invoke(self, question: str, repo_name: str, top_k: int = 5) -> dict:
        docs = self.retriever.retrieve(question, repo_name=repo_name, top_k=top_k)
        docs = self.graph_expander.expand_context(docs, repo_name=repo_name)

        def citation_path(metadata):
            path = metadata.get("file_path", "unknown")
            cell = metadata.get("notebook_cell")
            return f"{path}#cell-{cell}" if cell is not None else path

        context = "\n\n".join(
            f"[{citation_path(d.metadata)}:{d.metadata.get('start_line', '?')}-"
            f"{d.metadata.get('end_line', '?')}]\n{d.page_content}"
            for d in docs
        )
        messages = [
            (
                "system",
                "Answer using only the supplied repository context. Cite important claims "
                "with [path:line-line]. Say when the context is insufficient.",
            ),
            ("human", f"Question: {question}\n\nRepository context:\n{context}"),
        ]
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            "answer": answer,
            "sources": [
                {
                    "file_path": d.metadata.get("file_path"),
                    "start_line": d.metadata.get("start_line"),
                    "end_line": d.metadata.get("end_line"),
                    "symbol_name": d.metadata.get("symbol_name"),
                    "notebook_cell": d.metadata.get("notebook_cell"),
                }
                for d in docs
            ],
        }
