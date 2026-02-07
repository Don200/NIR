"""Community detection using hierarchical Leiden algorithm."""

import logging
from typing import Dict, List, Tuple

from .models import Community, Relationship
from .graph_store import GraphStore

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detect communities in the graph using hierarchical Leiden algorithm."""

    def __init__(self, max_cluster_size: int = 10):
        self.max_cluster_size = max_cluster_size

    def detect(
        self, graph_store: GraphStore
    ) -> Tuple[Dict[int, Community], Dict[str, List[int]]]:
        """
        Run hierarchical Leiden community detection.

        Returns:
            Tuple of:
            - communities: Dict mapping community_id to Community
            - entity_to_community_ids: Dict mapping entity key to list of community IDs
        """
        nx_graph = graph_store.to_networkx()

        if nx_graph.number_of_nodes() == 0:
            logger.warning("[communities] Empty graph, no communities to detect")
            return {}, {}

        logger.debug(
            f"[communities] Running Leiden on graph: "
            f"{nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges | "
            f"max_cluster_size={self.max_cluster_size}"
        )

        # Run hierarchical Leiden
        try:
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            raise ImportError(
                "graspologic is required for community detection. "
                "Install it with: pip install graspologic"
            )

        community_mapping = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )

        # Build community structures
        community_nodes: Dict[int, List[str]] = {}
        entity_to_community_ids: Dict[str, List[int]] = {}

        for item in community_mapping:
            node = item.node
            cluster_id = item.cluster

            if cluster_id not in community_nodes:
                community_nodes[cluster_id] = []
            community_nodes[cluster_id].append(node)

            if node not in entity_to_community_ids:
                entity_to_community_ids[node] = []
            if cluster_id not in entity_to_community_ids[node]:
                entity_to_community_ids[node].append(cluster_id)

        logger.debug(
            f"[communities] Leiden result: {len(community_nodes)} clusters | "
            f"cluster_sizes={[len(v) for v in community_nodes.values()]}"
        )

        # Build Community objects with intra-community relationships
        communities: Dict[int, Community] = {}
        for cid, nodes in community_nodes.items():
            node_set = set(nodes)
            intra_relationships = []

            for u in nodes:
                for v in graph_store.get_neighbors(u):
                    if v in node_set and u < v:
                        edge_data = graph_store.get_edge_data(u, v)
                        if edge_data:
                            for rel in edge_data.get("relations", []):
                                src_entity = graph_store.get_entity(u)
                                tgt_entity = graph_store.get_entity(v)
                                src_name = src_entity.name if src_entity else u
                                tgt_name = tgt_entity.name if tgt_entity else v
                                intra_relationships.append(Relationship(
                                    source=src_name,
                                    target=tgt_name,
                                    relation=rel.get("relation", "RELATED_TO"),
                                    description=rel.get("description", ""),
                                    source_chunk_id=rel.get("source_chunk_id", ""),
                                ))

            entity_names = []
            for n in nodes:
                entity = graph_store.get_entity(n)
                entity_names.append(entity.name if entity else n)

            communities[cid] = Community(
                community_id=cid,
                entity_names=entity_names,
                relationships=intra_relationships,
            )

            logger.debug(
                f"[communities] Community {cid}: "
                f"{len(entity_names)} entities={entity_names} | "
                f"{len(intra_relationships)} intra-edges"
            )

        logger.info(
            f"[communities] Detected {len(communities)} communities from "
            f"{nx_graph.number_of_nodes()} nodes"
        )
        return communities, entity_to_community_ids
