claud"""NetworkX-based graph store for GraphRAG entities and relationships."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from .models import Entity, Relationship

logger = logging.getLogger(__name__)


class GraphStore:
    """Manages a NetworkX graph of entities and relationships with descriptions."""

    def __init__(self):
        self._graph = nx.Graph()
        self._entities: Dict[str, Entity] = {}

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for consistent matching."""
        return name.strip().lower()

    def add_entity(self, entity: Entity) -> None:
        """Add or merge an entity into the graph."""
        key = self._normalize_name(entity.name)

        if key in self._entities:
            existing = self._entities[key]
            old_desc_len = len(existing.description)
            if entity.description and entity.description not in existing.description:
                existing.description = f"{existing.description} {entity.description}".strip()
            new_chunks = [c for c in entity.source_chunk_ids if c not in existing.source_chunk_ids]
            existing.source_chunk_ids.extend(new_chunks)
            logger.debug(
                f"[graph] MERGE entity \"{entity.name}\" (key={key}) | "
                f"desc: {old_desc_len}->{len(existing.description)} chars | "
                f"+{len(new_chunks)} chunks (total={len(existing.source_chunk_ids)})"
            )
        else:
            self._entities[key] = Entity(
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_chunk_ids=list(entity.source_chunk_ids),
            )
            logger.debug(
                f"[graph] ADD entity \"{entity.name}\" (key={key}) | "
                f"type={entity.entity_type} | "
                f"desc_len={len(entity.description)} | "
                f"chunks={entity.source_chunk_ids}"
            )

        # Update graph node
        e = self._entities[key]
        self._graph.add_node(
            key,
            name=e.name,
            entity_type=e.entity_type,
            description=e.description,
        )

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship as an edge in the graph."""
        src_key = self._normalize_name(relationship.source)
        tgt_key = self._normalize_name(relationship.target)

        # Ensure both nodes exist
        if src_key not in self._graph:
            self._graph.add_node(src_key, name=relationship.source, entity_type="Unknown", description="")
            logger.debug(f"[graph] ADD implicit node \"{relationship.source}\" (from relationship)")
        if tgt_key not in self._graph:
            self._graph.add_node(tgt_key, name=relationship.target, entity_type="Unknown", description="")
            logger.debug(f"[graph] ADD implicit node \"{relationship.target}\" (from relationship)")

        # Add or merge edge
        if self._graph.has_edge(src_key, tgt_key):
            existing = self._graph[src_key][tgt_key]
            relations = existing.get("relations", [])
            relations.append({
                "relation": relationship.relation,
                "description": relationship.description,
                "source_chunk_id": relationship.source_chunk_id,
            })
            existing["relations"] = relations
            logger.debug(
                f"[graph] MERGE edge \"{relationship.source}\" -[{relationship.relation}]-> "
                f"\"{relationship.target}\" | total_relations={len(relations)}"
            )
        else:
            self._graph.add_edge(
                src_key, tgt_key,
                relations=[{
                    "relation": relationship.relation,
                    "description": relationship.description,
                    "source_chunk_id": relationship.source_chunk_id,
                }],
            )
            logger.debug(
                f"[graph] ADD edge \"{relationship.source}\" -[{relationship.relation}]-> "
                f"\"{relationship.target}\""
            )

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        key = self._normalize_name(name)
        return self._entities.get(key)

    def get_neighbors(self, entity_name: str) -> List[str]:
        """Get neighbor entity keys."""
        key = self._normalize_name(entity_name)
        if key in self._graph:
            return list(self._graph.neighbors(key))
        return []

    def get_edge_data(self, source: str, target: str) -> Optional[dict]:
        """Get edge data between two entities."""
        src_key = self._normalize_name(source)
        tgt_key = self._normalize_name(target)
        if self._graph.has_edge(src_key, tgt_key):
            return self._graph[src_key][tgt_key]
        return None

    def to_networkx(self) -> nx.Graph:
        """Return a copy of the underlying NetworkX graph for algorithms."""
        return self._graph.copy()

    @property
    def entities(self) -> Dict[str, Entity]:
        """All entities keyed by normalized name."""
        return self._entities

    @property
    def stats(self) -> dict:
        """Graph statistics."""
        return {
            "num_entities": len(self._entities),
            "num_nodes": self._graph.number_of_nodes(),
            "num_edges": self._graph.number_of_edges(),
        }

    def save(self, path: Path) -> None:
        """Save graph to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entities": {
                key: {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "description": e.description,
                    "source_chunk_ids": e.source_chunk_ids,
                }
                for key, e in self._entities.items()
            },
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relations": d.get("relations", []),
                }
                for u, v, d in self._graph.edges(data=True)
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[graph] Saved to {path}: {self.stats}")

    def load(self, path: Path) -> None:
        """Load graph from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._graph = nx.Graph()
        self._entities = {}

        for key, e_data in data.get("entities", {}).items():
            entity = Entity(
                name=e_data["name"],
                entity_type=e_data["entity_type"],
                description=e_data["description"],
                source_chunk_ids=e_data.get("source_chunk_ids", []),
            )
            self._entities[key] = entity
            self._graph.add_node(
                key,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
            )

        for edge in data.get("edges", []):
            self._graph.add_edge(
                edge["source"],
                edge["target"],
                relations=edge.get("relations", []),
            )

        logger.info(f"[graph] Loaded from {path}: {self.stats}")

    def export_graphml(
        self,
        path: Path,
        community_mapping: dict = None,
    ) -> None:
        """Export graph to GraphML format (for Gephi, Cytoscape, yEd, etc.).

        GraphML supports only scalar node/edge attributes, so we flatten:
        - Node attrs: name, entity_type, description, source_chunks (comma-joined)
        - Edge attrs: one edge per relation (multi-edges become separate edges)
        - Optionally: community_id as node attribute

        Args:
            path: Output .graphml file path.
            community_mapping: Optional dict {entity_key: [community_ids]}
                to add community_id as node attribute.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build a fresh MultiDiGraph — one edge per relation
        export_graph = nx.MultiDiGraph()

        # Add nodes with scalar attributes
        for key, entity in self._entities.items():
            attrs = {
                "label": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "source_chunks": ", ".join(entity.source_chunk_ids),
            }
            if community_mapping and key in community_mapping:
                cids = community_mapping[key]
                attrs["community_id"] = cids[0] if len(cids) == 1 else min(cids)
                attrs["community_ids"] = ", ".join(str(c) for c in cids)
            export_graph.add_node(key, **attrs)

        # Add edges — one per relation (flatten the relations list)
        for u, v, data in self._graph.edges(data=True):
            for rel in data.get("relations", []):
                export_graph.add_edge(
                    u, v,
                    label=rel.get("relation", "RELATED_TO"),
                    description=rel.get("description", ""),
                    source_chunk_id=rel.get("source_chunk_id", ""),
                )

        nx.write_graphml(export_graph, str(path))
        logger.info(
            f"[graph] Exported GraphML to {path}: "
            f"{export_graph.number_of_nodes()} nodes, "
            f"{export_graph.number_of_edges()} edges"
        )
