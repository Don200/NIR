"""Hybrid retriever combining Vector and Graph search."""

import logging
from typing import Optional

from langfuse.decorators import observe

from ..core.config import Config
from ..core.models import SearchResult, QueryResult, RetrievalMethod
from ..core.llm import LLMClient
from ..core.prompts import format_context_block
from ..indexing.vector_index import VectorIndex
from ..indexing.graph_index import GraphIndex

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector and graph retrieval with LLM generation."""

    def __init__(
        self,
        config: Config,
        vector_index: Optional[VectorIndex] = None,
        graph_index: Optional[GraphIndex] = None,
    ):
        self.config = config
        self.vector_index = vector_index or VectorIndex(config)
        self.graph_index = graph_index or GraphIndex(config)
        self.llm = LLMClient(config.llm)

    def load_indexes(self) -> dict[str, bool]:
        """Load all indexes from disk."""
        results = {
            "vector": self.vector_index.is_indexed(),
            "graph": self.graph_index.load() if self.graph_index.is_indexed() else False,
        }
        return results

    @observe(name="hybrid_retrieve")
    def retrieve(
        self,
        query: str,
        method: RetrievalMethod = RetrievalMethod.HYBRID,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """Retrieve relevant context using specified method."""
        results = []

        if method in (RetrievalMethod.VECTOR, RetrievalMethod.HYBRID):
            vector_results = self.vector_index.search(query, top_k)
            results.extend(vector_results)
            logger.debug(f"Vector retrieval: {len(vector_results)} results")

        if method in (RetrievalMethod.GRAPH, RetrievalMethod.HYBRID):
            graph_results = self.graph_index.search(query, top_k)
            results.extend(graph_results)
            logger.debug(f"Graph retrieval: {len(graph_results)} results")

        # Deduplicate by content similarity
        if method == RetrievalMethod.HYBRID:
            results = self._deduplicate_results(results)

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _deduplicate_results(
        self,
        results: list[SearchResult],
        similarity_threshold: float = 0.9,
    ) -> list[SearchResult]:
        """Remove near-duplicate results."""
        if not results:
            return results

        unique = []
        seen_content = set()

        for r in results:
            # Simple dedup by normalized content prefix
            content_key = r.content[:200].lower().strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique.append(r)

        return unique

    @observe(name="rag_query")
    def query(
        self,
        query: str,
        method: RetrievalMethod = RetrievalMethod.HYBRID,
        top_k: Optional[int] = None,
        max_context_length: Optional[int] = None,
    ) -> QueryResult:
        """Full RAG pipeline: retrieve + generate."""
        # Use config default if not specified
        max_ctx = max_context_length or self.config.generation.max_context_length

        # Retrieve
        results = self.retrieve(query, method, top_k)

        if not results:
            return QueryResult(
                answer="В предоставленных источниках нет информации по данному вопросу.",
                sources=[],
                query=query,
                method=method,
            )

        # Build context with source metadata
        context_parts = []
        current_length = 0
        used_results = []

        for r in results:
            # Format context block with metadata
            formatted = format_context_block(r.content, r.metadata)
            if current_length + len(formatted) > max_ctx:
                break
            context_parts.append(formatted)
            current_length += len(formatted)
            used_results.append(r)

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        answer = self.llm.generate_with_context(query, context)

        return QueryResult(
            answer=answer,
            sources=used_results,
            query=query,
            method=method,
        )

    def get_status(self) -> dict:
        """Get retriever status info."""
        return {
            "vector_indexed": self.vector_index.is_indexed(),
            "vector_count": self.vector_index.collection.count() if self.vector_index.is_indexed() else 0,
            "graph_indexed": self.graph_index.is_indexed(),
        }
