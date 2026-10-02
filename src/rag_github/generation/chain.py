from __future__ import annotations

from langchain_core.documents import Document

from rag_github.generation.groq_client import GroqClient


class RAGChain:
    """Small explicit RAG flow: retrieve, expand, format citations, answer."""

    def __init__(
        self,
        retriever,
        graph_expander,
        llm=None,
        min_relevance_score: float = 0.35,
        context_budget_tokens: int = 3500,
    ):
        self.retriever = retriever
        self.graph_expander = graph_expander
        self.llm = llm or GroqClient().model
        self.min_relevance_score = min_relevance_score
        # Code tends to use more tokens per character than prose; keep this conservative.
        self.context_budget_chars = max(3000, context_budget_tokens * 3)

    def _pack_context(self, docs: list[Document]) -> list[Document]:
        """Keep ranked, unique evidence within a conservative character budget."""
        packed = []
        seen = set()
        remaining = self.context_budget_chars
        for doc in docs:
            metadata = doc.metadata
            key = (
                metadata.get("file_path"),
                metadata.get("start_line"),
                metadata.get("end_line"),
                metadata.get("symbol_name"),
            )
            if key in seen:
                continue
            seen.add(key)
            content_budget = remaining - 120  # reserve room for citation and separators
            if content_budget <= 0:
                break
            content = doc.page_content.strip()
            if not content:
                continue
            if len(content) > content_budget:
                excerpt = content[:content_budget]
                line_end = excerpt.rfind("\n")
                if line_end > 0:
                    excerpt = excerpt[:line_end]
                content = excerpt.rstrip()
                if not content:
                    break
                metadata = {
                    **metadata,
                    "end_line": metadata.get("start_line", 1) + content.count("\n"),
                    "is_excerpt": True,
                }
            packed.append(Document(page_content=content, metadata=metadata))
            remaining -= len(content) + 120
        return packed

    def invoke(self, question: str, repo_name: str, top_k: int = 5) -> dict:
        docs = self.retriever.retrieve(question, repo_name=repo_name, top_k=top_k)
        best_score = max((d.metadata.get("dense_score") or 0 for d in docs), default=0)
        if best_score < self.min_relevance_score:
            return {
                "answer": "I couldn’t find sufficiently relevant information in this repository "
                "to answer that. I can help with questions about its code and documentation.",
                "sources": [],
                "is_relevant": False,
            }
        docs = self.graph_expander.expand_context(docs, repo_name=repo_name)
        docs = self._pack_context(docs)

        def citation_path(metadata):
            path = metadata.get("file_path", "unknown")
            cell = metadata.get("notebook_cell")
            return f"{path}#cell-{cell}" if cell is not None else path

        context = "\n\n".join(
            f"[{citation_path(d.metadata)}:{d.metadata.get('start_line', '?')}-"
            f"{d.metadata.get('end_line', '?')}]"
            f"{' [excerpt]' if d.metadata.get('is_excerpt') else ''}\n{d.page_content}"
            for d in docs
        )
        messages = [
            (
                "system",
                "Answer using only the supplied repository context. Cite important claims "
                "with [path:line-line]. Some passages may be excerpts. Say when the context "
                "is insufficient; do not infer missing implementation details.",
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
            "is_relevant": True,
        }
