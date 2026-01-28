"""Knowledge Graph RAG retriever."""

from typing import Optional
from .base import BaseRetriever, RetrievalResult
from ..indexing.kg_index import KnowledgeGraphIndex


class KGRetriever(BaseRetriever):
    """
    KG-based GraphRAG retriever.

    As described in the paper (LlamaIndex approach):
    - Extract entities from query
    - Match with KG entities
    - Traverse graph to get relevant triplets
    - Optionally include original source texts
    """

    def __init__(
        self,
        index: KnowledgeGraphIndex,
        max_hops: int = 2,
        include_text: bool = True,
        max_triplets: int = 50,
        separator: str = "\n\n---\n\n",
    ):
        self.index = index
        self.max_hops = max_hops
        self.include_text = include_text
        self.max_triplets = max_triplets
        self.separator = separator

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant information from knowledge graph."""
        result = self.index.search(
            query,
            max_hops=self.max_hops,
            include_text=self.include_text,
            max_triplets=self.max_triplets,
        )

        if not result.triplets:
            return RetrievalResult(
                context="",
                metadata={
                    "num_triplets": 0,
                    "matched_entities": [],
                },
                num_chunks=0,
                retrieval_method=self.name,
            )

        # Build context
        context_parts = []

        # Add triplets as structured knowledge
        triplets_text = self.index.triplets_to_text(result.triplets)
        context_parts.append(f"Knowledge Graph Facts:\n{triplets_text}")

        # Add source texts if available
        if self.include_text and result.source_texts:
            context_parts.append("Source Documents:")
            context_parts.extend(result.source_texts)

        context = self.separator.join(context_parts)

        return RetrievalResult(
            context=context,
            metadata={
                "num_triplets": len(result.triplets),
                "matched_entities": result.matched_entities,
                "num_source_texts": len(result.source_texts),
            },
            num_chunks=len(result.source_texts),
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return "kg_rag"
