"""Base retriever interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    context: str  # Retrieved context to pass to LLM
    metadata: Dict[str, Any] = field(default_factory=dict)
    num_chunks: int = 0
    retrieval_method: str = ""


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant context for a query.

        Args:
            query: The search query

        Returns:
            RetrievalResult with context and metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the retriever for logging."""
        pass
