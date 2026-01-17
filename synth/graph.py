from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from synth.api_clients import EmbedderClient, OpenAILLMClient, ensure_embeddings_endpoint
from synth.chunking import ChunkRecord


@dataclass
class GraphNode:
    node_id: str
    page_content: str
    metadata: dict
    claims: Optional[List[str]] = None


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    relation: str


@dataclass
class KnowledgeGraph:
    graph: nx.Graph
    nodes: List[GraphNode]
    edges: List[GraphEdge]


CLAIMS_PROMPT = """
Выдели 2-4 кратких пункта, передающих основные факты или смысл фрагмента.
Формат: один пункт на строку, без нумерации.
Фрагмент:
{context}
"""


def _node_id(chunk: ChunkRecord, index: int) -> str:
    doc_id = chunk.metadata.get("doc_id", "unknown")
    return f"{doc_id}::chunk_{index}"


def _build_embeddings(chunks: List[ChunkRecord], *, model: str, api_base: Optional[str], api_key: str) -> np.ndarray:
    embeddings_endpoint = ensure_embeddings_endpoint(api_base)
    embedder = EmbedderClient(
        embeddings_endpoint=embeddings_endpoint,
        model=model,
        timeout=90,
        api_key=api_key,
    )
    vectors = embedder.get_embeddings([chunk.page_content for chunk in chunks])
    return np.array(vectors, dtype=float)


def _tfidf_embeddings(chunks: List[ChunkRecord]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for TF-IDF fallback.") from exc

    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform([chunk.page_content for chunk in chunks])
    return matrix.toarray().astype(float)


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    normalized = vectors / norm
    return normalized @ normalized.T


def _extract_claims(chunks: List[ChunkRecord], *, model: str, api_base: Optional[str], api_key: str) -> List[List[str]]:
    if not api_base:
        raise RuntimeError("claims api_base must be set for enrichment_mode=embeddings_plus_llm.")
    llm = OpenAILLMClient(
        llm_url=api_base,
        api_key=api_key,
        path2llm=model,
    )
    claims_list: List[List[str]] = []
    for chunk in chunks:
        response = llm.ask(
            messages=[{"role": "user", "content": CLAIMS_PROMPT.format(context=chunk.page_content)}],
            temperature=0.1,
            top_p=0.95,
            timeout=90,
        )
        claims = [line.strip() for line in response.splitlines() if line.strip()]
        claims_list.append(claims)
    return claims_list


def build_kg(
    chunks: List[ChunkRecord],
    *,
    top_k: int,
    min_similarity: float,
    use_embeddings: bool,
    use_tfidf_fallback: bool,
    embeddings_config: dict,
    enrichment_mode: str,
    claims_config: dict,
) -> KnowledgeGraph:
    nodes: List[GraphNode] = []
    g = nx.Graph()
    for idx, chunk in enumerate(chunks):
        node = GraphNode(
            node_id=_node_id(chunk, idx),
            page_content=chunk.page_content,
            metadata=chunk.metadata,
        )
        nodes.append(node)
        g.add_node(
            node.node_id,
            page_content=node.page_content,
            metadata=json.dumps(node.metadata, ensure_ascii=False),
            claims=json.dumps(node.claims or [], ensure_ascii=False),
        )

    vectors: Optional[np.ndarray] = None
    if use_embeddings:
        api_key = embeddings_config["api_key"]
        vectors = _build_embeddings(
            chunks,
            model=embeddings_config["model"],
            api_base=embeddings_config.get("api_base"),
            api_key=api_key,
        )
    elif use_tfidf_fallback:
        vectors = _tfidf_embeddings(chunks)

    if vectors is not None:
        similarity = _cosine_similarity_matrix(vectors)
        for i in range(similarity.shape[0]):
            row = similarity[i]
            indices = np.argsort(row)[::-1]
            added = 0
            for j in indices:
                if i == j:
                    continue
                if row[j] < min_similarity:
                    break
                g.add_edge(
                    nodes[i].node_id,
                    nodes[j].node_id,
                    weight=float(row[j]),
                    relation="embedding_similarity",
                )
                added += 1
                if added >= top_k:
                    break

    if enrichment_mode == "embeddings_plus_llm":
        api_key = claims_config["api_key"]
        claims = _extract_claims(
            chunks,
            model=claims_config["model"],
            api_base=claims_config.get("api_base"),
            api_key=api_key,
        )
        for node, claim_list in zip(nodes, claims):
            node.claims = claim_list
            g.nodes[node.node_id]["claims"] = json.dumps(claim_list, ensure_ascii=False)

    edges: List[GraphEdge] = []
    for src, dst, data in g.edges(data=True):
        edges.append(
            GraphEdge(
                source=src,
                target=dst,
                weight=float(data.get("weight", 0.0)),
                relation=data.get("relation", "embedding_similarity"),
            )
        )

    return KnowledgeGraph(graph=g, nodes=nodes, edges=edges)


def graph_stats(graph: KnowledgeGraph) -> Dict[str, float]:
    g = graph.graph
    degree = dict(g.degree())
    avg_degree = sum(degree.values()) / max(len(degree), 1)
    components = nx.number_connected_components(g) if len(g) > 0 else 0
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "avg_degree": avg_degree,
        "components": components,
    }


def save_kg(path: Path, graph: KnowledgeGraph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph.graph, path)


def load_kg(path: Path) -> KnowledgeGraph:
    g = nx.read_graphml(path)
    nodes: List[GraphNode] = []
    for node_id, data in g.nodes(data=True):
        metadata = json.loads(data.get("metadata", "{}"))
        claims = json.loads(data.get("claims", "[]"))
        nodes.append(
            GraphNode(
                node_id=node_id,
                page_content=data.get("page_content", ""),
                metadata=metadata,
                claims=claims,
            )
        )
    edges: List[GraphEdge] = []
    for src, dst, data in g.edges(data=True):
        edges.append(
            GraphEdge(
                source=src,
                target=dst,
                weight=float(data.get("weight", 0.0)),
                relation=data.get("relation", "embedding_similarity"),
            )
        )
    return KnowledgeGraph(graph=g, nodes=nodes, edges=edges)


def build_adjacency(graph: KnowledgeGraph) -> Dict[str, List[str]]:
    g = graph.graph
    adjacency: Dict[str, List[str]] = {node: [] for node in g.nodes()}
    for src, dst in g.edges():
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    return adjacency


def path_exists(adjacency: Dict[str, List[str]], path: List[str]) -> bool:
    for idx in range(len(path) - 1):
        if path[idx + 1] not in adjacency.get(path[idx], []):
            return False
    return True


def find_path(adjacency: Dict[str, List[str]], length: int) -> Optional[List[str]]:
    if length < 1:
        return None

    nodes = list(adjacency.keys())
    if not nodes:
        return None

    for start in nodes:
        stack: List[Tuple[str, List[str]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == length + 1:
                return path
            for neighbor in adjacency.get(node, []):
                if neighbor in path:
                    continue
                stack.append((neighbor, path + [neighbor]))
    return None
