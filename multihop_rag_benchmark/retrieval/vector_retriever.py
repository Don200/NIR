"""Vector RAG retriever."""

from typing import Optional
from .base import BaseRetriever, RetrievalResult
from ..indexing.vector_index import VectorIndex


class VectorRetriever(BaseRetriever):
    """
    Vector RAG retriever.

    As described in the paper:
    - Top-K semantic search over chunk embeddings
    - Returns concatenated chunk texts as context
    """

    def __init__(
        self,
        index: VectorIndex,
        top_k: int = 10,
        separator: str = "\n\n---\n\n",
    ):
        self.index = index
        self.top_k = top_k
        self.separator = separator

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant chunks using vector similarity."""
        results = self.index.search(query, top_k=self.top_k)

        if not results:
            return RetrievalResult(
                context="",
                metadata={"num_results": 0},
                num_chunks=0,
                retrieval_method=self.name,
            )

        # Concatenate chunk contents
        texts = [r.chunk.content for r in results]
        context = self.separator.join(texts)

        return RetrievalResult(
            context=context,
            metadata={
                "num_results": len(results),
                "scores": [r.score for r in results],
                "chunk_ids": [r.chunk.chunk_id for r in results],
            },
            num_chunks=len(results),
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return "vector_rag"
