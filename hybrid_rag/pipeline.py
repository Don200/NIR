"""High-level RAG pipeline for programmatic use (notebooks, tests, experiments)."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .core.config import Config, load_config
from .core.models import RetrievalMethod, SearchResult, QueryResult
from .retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Extended query result with timing and retrieval details."""
    query: str
    method: RetrievalMethod
    answer: str
    sources: list[SearchResult]
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    num_sources: int
    context_length: int


class RAGPipeline:
    """Facade for the hybrid RAG system.

    Provides a simple interface for:
    - Loading config and indexes
    - Running queries with timing
    - Batch evaluation
    - Comparing retrieval methods

    Usage::

        from hybrid_rag.pipeline import RAGPipeline

        pipeline = RAGPipeline.from_config("hybrid_rag/config.yaml")
        result = pipeline.query("Что такое латинский квадрат?")
        print(result.answer)
    """

    def __init__(self, config: Config):
        self.config = config
        self.retriever = HybridRetriever(config)
        self._loaded = False

    @classmethod
    def from_config(cls, config_path: str | Path) -> "RAGPipeline":
        """Create pipeline from a YAML config file."""
        config = load_config(Path(config_path))
        pipeline = cls(config)
        pipeline.load()
        return pipeline

    @classmethod
    def from_defaults(cls, index_dir: str | Path = "./indexes") -> "RAGPipeline":
        """Create pipeline with default config, pointing to existing indexes."""
        config = Config()
        config.index_dir = Path(index_dir)
        pipeline = cls(config)
        pipeline.load()
        return pipeline

    def load(self) -> dict[str, bool]:
        """Load indexes from disk."""
        status = self.retriever.load_indexes()
        self._loaded = True
        logger.info(f"Indexes loaded: {status}")
        return status

    def query(
        self,
        question: str,
        method: str | RetrievalMethod = RetrievalMethod.HYBRID,
        top_k: Optional[int] = None,
        max_context_length: Optional[int] = None,
    ) -> RetrievalResult:
        """Run a single query with timing information.

        Args:
            question: The question to ask.
            method: Retrieval method - "vector", "graph", "hybrid" or RetrievalMethod enum.
            top_k: Number of results to retrieve.
            max_context_length: Max context chars to pass to LLM.

        Returns:
            RetrievalResult with answer, sources, and timing.
        """
        if isinstance(method, str):
            method = RetrievalMethod(method)

        t0 = time.perf_counter()

        # Retrieval phase
        sources = self.retriever.retrieve(question, method, top_k)
        t_retrieval = time.perf_counter()

        # Generation phase
        result = self.retriever.query(question, method, top_k, max_context_length)
        t_generation = time.perf_counter()

        context_length = sum(len(s.content) for s in result.sources)

        return RetrievalResult(
            query=question,
            method=method,
            answer=result.answer,
            sources=result.sources,
            retrieval_time_s=round(t_retrieval - t0, 3),
            generation_time_s=round(t_generation - t_retrieval, 3),
            total_time_s=round(t_generation - t0, 3),
            num_sources=len(result.sources),
            context_length=context_length,
        )

    def retrieve_only(
        self,
        question: str,
        method: str | RetrievalMethod = RetrievalMethod.HYBRID,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """Run retrieval without generation (for retrieval-only evaluation)."""
        if isinstance(method, str):
            method = RetrievalMethod(method)
        return self.retriever.retrieve(question, method, top_k)

    def batch_query(
        self,
        questions: list[str],
        method: str | RetrievalMethod = RetrievalMethod.HYBRID,
        top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> list[RetrievalResult]:
        """Run multiple queries sequentially with progress output."""
        results = []
        for i, q in enumerate(questions):
            if verbose:
                print(f"[{i+1}/{len(questions)}] {q[:80]}...")
            result = self.query(q, method, top_k)
            results.append(result)
        return results

    def compare_methods(
        self,
        question: str,
        methods: list[str] | None = None,
        top_k: Optional[int] = None,
    ) -> dict[str, RetrievalResult]:
        """Run the same question with different methods for comparison."""
        if methods is None:
            methods = ["vector", "graph", "hybrid"]
        return {m: self.query(question, m, top_k) for m in methods}

    def get_status(self) -> dict:
        """Get index status."""
        return self.retriever.get_status()
