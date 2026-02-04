from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RetrievalMethod(Enum):
    """Retrieval method type."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class Document:
    """Source document."""
    id: str
    content: str
    title: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Document chunk for indexing."""
    id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Triplet:
    """Knowledge graph triplet."""
    subject: str
    relation: str
    object: str
    source_chunk_id: Optional[str] = None


@dataclass
class SearchResult:
    """Single search result."""
    content: str
    score: float
    source: RetrievalMethod
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    """Complete query result with answer and sources."""
    answer: str
    sources: list[SearchResult]
    query: str
    method: RetrievalMethod
