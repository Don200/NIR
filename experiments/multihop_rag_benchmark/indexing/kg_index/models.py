"""Data models for GraphRAG v2 indexing."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Entity:
    """A knowledge graph entity with description."""
    name: str
    entity_type: str
    description: str
    source_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A knowledge graph relationship with description."""
    source: str
    target: str
    relation: str
    description: str
    source_chunk_id: str = ""


@dataclass
class Community:
    """A community of related entities detected by Leiden algorithm."""
    community_id: int
    entity_names: List[str] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    summary: str = ""


@dataclass
class GraphRAGSearchResult:
    """Result from a GraphRAG search operation."""
    community_summaries: List[str] = field(default_factory=list)
    matched_entities: List[str] = field(default_factory=list)
    matched_community_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
