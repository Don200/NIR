"""Hybrid retrieval strategies combining RAG and GraphRAG."""

from typing import Optional, List
from .base import BaseRetriever, RetrievalResult
from ..generation.llm_client import LLMClient


class SelectionRetriever(BaseRetriever):
    """
    Selection strategy from the paper.

    Classifies query as fact-based or reasoning-based:
    - Fact-based queries -> Vector RAG
    - Reasoning-based queries -> GraphRAG

    This is more resource-efficient than integration.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        graph_retriever: BaseRetriever,
        llm_client: LLMClient,
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.llm_client = llm_client

    def _classify_query(self, query: str) -> str:
        """Classify query as fact-based or reasoning-based."""
        prompt = f"""Classify the following question as either "fact-based" or "reasoning-based".

Fact-based questions:
- Ask for specific facts, names, dates, numbers
- Can be answered with a single piece of information
- Examples: "Who is the CEO of X?", "When was Y founded?"

Reasoning-based questions:
- Require connecting multiple pieces of information
- Involve comparison, inference, or temporal reasoning
- Examples: "How did X affect Y?", "Compare A and B", "What happened before/after Z?"

Question: {query}

Classification (respond with only "fact-based" or "reasoning-based"):"""

        response = self.llm_client.generate(prompt, temperature=0.0)
        response = response.lower().strip()

        if "reasoning" in response:
            return "reasoning-based"
        return "fact-based"

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve using selected strategy based on query classification."""
        classification = self._classify_query(query)

        if classification == "fact-based":
            result = self.vector_retriever.retrieve(query)
            result.metadata["query_classification"] = "fact-based"
            result.metadata["selected_retriever"] = self.vector_retriever.name
        else:
            result = self.graph_retriever.retrieve(query)
            result.metadata["query_classification"] = "reasoning-based"
            result.metadata["selected_retriever"] = self.graph_retriever.name

        result.retrieval_method = self.name
        return result

    @property
    def name(self) -> str:
        return "hybrid_selection"


class IntegrationRetriever(BaseRetriever):
    """
    Integration strategy from the paper.

    Runs both Vector RAG and GraphRAG in parallel,
    then concatenates their results.

    More resource-intensive but achieves best performance.
    Paper reports +6.4% improvement on MultiHop-RAG.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        graph_retriever: BaseRetriever,
        separator: str = "\n\n=== Additional Context ===\n\n",
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.separator = separator

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve from both sources and combine."""
        vector_result = self.vector_retriever.retrieve(query)
        graph_result = self.graph_retriever.retrieve(query)

        # Combine contexts
        contexts = []
        if vector_result.context:
            contexts.append(f"[Vector Search Results]\n{vector_result.context}")
        if graph_result.context:
            contexts.append(f"[Graph Search Results]\n{graph_result.context}")

        combined_context = self.separator.join(contexts)

        return RetrievalResult(
            context=combined_context,
            metadata={
                "vector_metadata": vector_result.metadata,
                "graph_metadata": graph_result.metadata,
                "vector_chunks": vector_result.num_chunks,
                "graph_chunks": graph_result.num_chunks,
            },
            num_chunks=vector_result.num_chunks + graph_result.num_chunks,
            retrieval_method=self.name,
        )

    @property
    def name(self) -> str:
        return "hybrid_integration"


class HybridRetriever(BaseRetriever):
    """
    Generic hybrid retriever that can use either selection or integration.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        graph_retriever: BaseRetriever,
        llm_client: Optional[LLMClient] = None,
        strategy: str = "integration",  # "selection" or "integration"
    ):
        self.strategy = strategy

        if strategy == "selection":
            if llm_client is None:
                raise ValueError("LLM client required for selection strategy")
            self._retriever = SelectionRetriever(
                vector_retriever, graph_retriever, llm_client
            )
        elif strategy == "integration":
            self._retriever = IntegrationRetriever(
                vector_retriever, graph_retriever
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve using configured strategy."""
        return self._retriever.retrieve(query)

    @property
    def name(self) -> str:
        return f"hybrid_{self.strategy}"
