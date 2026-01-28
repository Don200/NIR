"""Community-based GraphRAG retriever."""

from typing import Optional, Literal
from .base import BaseRetriever, RetrievalResult
from ..indexing.community_index import CommunityGraphIndex


class CommunityRetriever(BaseRetriever):
    """
    Community-based GraphRAG retriever.

    As described in the paper (Microsoft GraphRAG approach):
    - Local search: entities + relations + low-level community summaries
    - Global search: high-level community summaries
    """

    def __init__(
        self,
        index: CommunityGraphIndex,
        method: Literal["local", "global"] = "local",
        community_level: Optional[int] = None,
    ):
        self.index = index
        self.method = method
        self.community_level = community_level

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve using community-based search."""
        result = self.index.search(
            query,
            method=self.method,
            community_level=self.community_level,
        )

        return RetrievalResult(
            context=result.response,
            metadata={
                "search_type": result.search_type,
                "context_data": result.context_data,
            },
            num_chunks=1,  # GraphRAG returns synthesized response
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return f"community_rag_{self.method}"


class CommunityLocalRetriever(CommunityRetriever):
    """Local search retriever."""

    def __init__(self, index: CommunityGraphIndex, community_level: int = 2):
        super().__init__(index, method="local", community_level=community_level)


class CommunityGlobalRetriever(CommunityRetriever):
    """Global search retriever."""

    def __init__(self, index: CommunityGraphIndex, community_level: int = 1):
        super().__init__(index, method="global", community_level=community_level)
