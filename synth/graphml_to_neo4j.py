from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
from neo4j import GraphDatabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load GraphML KG into Neo4j.")
    parser.add_argument("--graphml", type=Path, required=True, help="Path to GraphML file.")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="Neo4j bolt URI.")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", type=str, required=True, help="Neo4j password.")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for node/edge upserts.")
    return parser.parse_args()


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _prepare_nodes(g: nx.Graph) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for node_id, data in g.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "page_content": data.get("page_content", ""),
                "metadata": data.get("metadata", "{}"),
                "claims": data.get("claims", "[]"),
            }
        )
    return nodes


def _prepare_edges(g: nx.Graph) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for src, dst, data in g.edges(data=True):
        edges.append(
            {
                "source": src,
                "target": dst,
                "weight": float(data.get("weight", 0.0)),
                "relation": data.get("relation", "embedding_similarity"),
            }
        )
    return edges


def _upsert_nodes(tx, batch: List[Dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $batch AS n
        MERGE (c:Chunk {id: n.id})
        SET c.page_content = n.page_content,
            c.metadata = n.metadata,
            c.claims = n.claims
        """,
        batch=batch,
    )


def _upsert_edges(tx, batch: List[Dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $batch AS r
        MATCH (s:Chunk {id: r.source})
        MATCH (t:Chunk {id: r.target})
        MERGE (s)-[e:SIMILAR {relation: r.relation}]->(t)
        SET e.weight = r.weight
        """,
        batch=batch,
    )


def load_to_neo4j(graphml_path: Path, uri: str, user: str, password: str, batch_size: int) -> None:
    g = nx.read_graphml(graphml_path)
    nodes = _prepare_nodes(g)
    edges = _prepare_edges(g)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")

        for batch in _chunks(nodes, batch_size):
            session.execute_write(_upsert_nodes, batch)

        for batch in _chunks(edges, batch_size):
            session.execute_write(_upsert_edges, batch)

    driver.close()
    print(f"Imported {len(nodes)} nodes and {len(edges)} edges into Neo4j at {uri}")


def main() -> None:
    args = parse_args()
    load_to_neo4j(
        graphml_path=args.graphml,
        uri=args.uri,
        user=args.user,
        password=args.password,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
