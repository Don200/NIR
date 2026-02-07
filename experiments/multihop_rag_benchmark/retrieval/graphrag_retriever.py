"""GraphRAG v2 retrievers — local (entity-driven) and global (map-reduce) search."""

from .base import BaseRetriever, RetrievalResult
from ..indexing.kg_index import GraphRAGIndex


class GraphRAGLocalRetriever(BaseRetriever):
    """
    Local search: query -> entity embeddings -> entity context + community summaries.

    Produces enriched context with:
    1. Entity descriptions and their relationships (fine-grained)
    2. Community summaries (synthesized view)
    """

    def __init__(self, index: GraphRAGIndex, top_k: int = 10):
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:
        result = self.index.local_search(query, top_k=self.top_k)

        # Assemble context: source text first (most precise), then entities, then summaries
        context_parts = []

        if result.source_chunks:
            context_parts.append("=== Source Text ===")
            context_parts.append("\n\n---\n\n".join(result.source_chunks))

        if result.entity_context:
            context_parts.append("=== Relevant Entities ===")
            context_parts.append("\n\n".join(result.entity_context))

        if result.community_summaries:
            context_parts.append("=== Community Summaries ===")
            context_parts.append("\n\n---\n\n".join(result.community_summaries))

        context = "\n\n".join(context_parts)

        return RetrievalResult(
            context=context,
            metadata={
                "matched_entities": result.matched_entities,
                "matched_community_ids": result.matched_community_ids,
                "num_entity_contexts": len(result.entity_context),
                "num_summaries": len(result.community_summaries),
                "num_source_chunks": len(result.source_chunks),
                **result.metadata,
            },
            num_chunks=len(result.source_chunks) + len(result.entity_context) + len(result.community_summaries),
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return "graphrag_local"


class GraphRAGGlobalRetriever(BaseRetriever):
    """
    Global search with map-reduce (based on Microsoft GraphRAG).

    1. Pre-filters communities by embedding similarity.
    2. Map phase: LLM extracts relevant points from each community.
    3. Returns top points sorted by relevance score.
    """

    def __init__(
        self,
        index: GraphRAGIndex,
        top_communities: int = 20,
        min_score: int = 20,
    ):
        self.index = index
        self.top_communities = top_communities
        self.min_score = min_score

    def retrieve(self, query: str) -> RetrievalResult:
        result = self.index.global_search(
            query,
            top_communities=self.top_communities,
            min_score=self.min_score,
        )

        # Context is the map-reduce extracted points (already sorted by relevance)
        context = "\n\n---\n\n".join(
            result.community_summaries
        ) if result.community_summaries else ""

        return RetrievalResult(
            context=context,
            metadata={
                "num_points": len(result.global_search_points),
                "num_communities": len(result.matched_community_ids),
                **result.metadata,
            },
            num_chunks=len(result.community_summaries),
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return "graphrag_global"
