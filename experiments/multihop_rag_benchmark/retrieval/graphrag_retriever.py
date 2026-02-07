"""GraphRAG v2 retrievers — local (entity-driven) and global (map-reduce) search."""

from .base import BaseRetriever, RetrievalResult
from ..indexing.kg_index import GraphRAGIndex


class GraphRAGLocalRetriever(BaseRetriever):
    """Local search: query -> entity embeddings -> community summaries."""

    def __init__(self, index: GraphRAGIndex, top_k: int = 10):
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:
        result = self.index.local_search(query, top_k=self.top_k)

        context = "\n\n---\n\n".join(result.community_summaries) if result.community_summaries else ""

        return RetrievalResult(
            context=context,
            metadata={
                "matched_entities": result.matched_entities,
                "matched_community_ids": result.matched_community_ids,
                "num_summaries": len(result.community_summaries),
                **result.metadata,
            },
            num_chunks=len(result.community_summaries),
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return "graphrag_local"


class GraphRAGGlobalRetriever(BaseRetriever):
    """Global search: all community summaries as context."""

    def __init__(self, index: GraphRAGIndex):
        self.index = index

    def retrieve(self, query: str) -> RetrievalResult:
        result = self.index.global_search()

        context = "\n\n---\n\n".join(result.community_summaries) if result.community_summaries else ""

        return RetrievalResult(
            context=context,
            metadata={
                "num_summaries": len(result.community_summaries),
                "num_communities": len(result.matched_community_ids),
                **result.metadata,
            },
            num_chunks=len(result.community_summaries),
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return "graphrag_global"
