"""Knowledge Graph Index using LlamaIndex for KG-based GraphRAG."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import json
import hashlib

from ..data.preprocessing import Chunk, Document
from ..generation.llm_client import LLMClient, EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """A knowledge graph triplet."""
    subject: str
    relation: str
    object: str
    source_chunk_id: str


@dataclass
class KGSearchResult:
    """Result from KG search."""
    triplets: List[Triplet]
    source_texts: List[str]  # Original chunk texts if include_text=True
    matched_entities: List[str]


class KnowledgeGraphIndex:
    """
    Knowledge Graph Index for KG-based GraphRAG.

    Implements the LlamaIndex-style approach from the paper:
    - Extract triplets (subject, relation, object) from chunks using LLM
    - Build graph from triplets
    - Retrieve by matching query entities and traversing graph
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: Optional[EmbeddingClient] = None,
        max_triplets_per_chunk: int = 10,
    ):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Please install networkx: pip install networkx")

        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.max_triplets_per_chunk = max_triplets_per_chunk

        self.graph = nx.DiGraph()
        self.triplets: List[Triplet] = []
        self.chunk_map: Dict[str, Chunk] = {}  # chunk_id -> Chunk
        self.entity_to_chunks: Dict[str, List[str]] = {}  # entity -> chunk_ids
        self._is_built = False

    def _extract_triplets_prompt(self, text: str) -> str:
        """Generate prompt for triplet extraction."""
        return f"""Extract knowledge graph triplets from the following text.
Each triplet should be in the format: (subject, relation, object)

Rules:
- Extract up to {self.max_triplets_per_chunk} most important triplets
- Subject and object should be named entities or important concepts
- Relations should be verbs or short phrases describing the relationship
- Output one triplet per line in format: subject | relation | object

Text:
{text}

Triplets:"""

    def _parse_triplets(self, response: str, chunk_id: str) -> List[Triplet]:
        """Parse LLM response into triplets."""
        triplets = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                triplet = Triplet(
                    subject=parts[0],
                    relation=parts[1],
                    object=parts[2],
                    source_chunk_id=chunk_id,
                )
                triplets.append(triplet)

        return triplets[:self.max_triplets_per_chunk]

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from a query using LLM."""
        prompt = f"""Extract the main entities (people, organizations, places, concepts) from this question.
Output one entity per line, nothing else.

Question: {query}

Entities:"""

        response = self.llm_client.generate(prompt, temperature=0.0)
        entities = [e.strip() for e in response.strip().split('\n') if e.strip()]
        return entities

    def build_from_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> None:
        """
        Build knowledge graph from chunks.

        Args:
            chunks: List of document chunks
            show_progress: Whether to show progress
        """
        import networkx as nx

        logger.info(f"Extracting triplets from {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Processing chunk {i + 1}/{len(chunks)}")

            # Store chunk
            self.chunk_map[chunk.chunk_id] = chunk

            # Extract triplets using LLM
            prompt = self._extract_triplets_prompt(chunk.content)
            response = self.llm_client.generate(prompt, temperature=0.0)
            chunk_triplets = self._parse_triplets(response, chunk.chunk_id)

            # Add to graph
            for triplet in chunk_triplets:
                self.triplets.append(triplet)

                # Add nodes
                subj_lower = triplet.subject.lower()
                obj_lower = triplet.object.lower()

                if not self.graph.has_node(subj_lower):
                    self.graph.add_node(subj_lower, label=triplet.subject)
                if not self.graph.has_node(obj_lower):
                    self.graph.add_node(obj_lower, label=triplet.object)

                # Add edge
                self.graph.add_edge(
                    subj_lower,
                    obj_lower,
                    relation=triplet.relation,
                    chunk_id=chunk.chunk_id,
                )

                # Update entity -> chunks mapping
                for entity in [subj_lower, obj_lower]:
                    if entity not in self.entity_to_chunks:
                        self.entity_to_chunks[entity] = []
                    if chunk.chunk_id not in self.entity_to_chunks[entity]:
                        self.entity_to_chunks[entity].append(chunk.chunk_id)

        self._is_built = True
        logger.info(f"Built KG with {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges, {len(self.triplets)} triplets")

    def _find_matching_entities(self, query_entities: List[str]) -> List[str]:
        """Find graph entities that match query entities."""
        matched = []
        graph_entities = set(self.graph.nodes())

        for qe in query_entities:
            qe_lower = qe.lower()

            # Exact match
            if qe_lower in graph_entities:
                matched.append(qe_lower)
                continue

            # Partial match
            for ge in graph_entities:
                if qe_lower in ge or ge in qe_lower:
                    matched.append(ge)
                    break

        return list(set(matched))

    def _get_neighborhood_triplets(
        self,
        entities: List[str],
        max_hops: int = 2,
    ) -> List[Triplet]:
        """Get triplets from neighborhood of entities."""
        import networkx as nx

        visited_edges = set()
        result_triplets = []

        # BFS to find neighbors
        frontier = set(entities)
        for hop in range(max_hops):
            next_frontier = set()
            for entity in frontier:
                if entity not in self.graph:
                    continue

                # Outgoing edges
                for _, target, data in self.graph.out_edges(entity, data=True):
                    edge_key = (entity, target, data.get('relation', ''))
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        triplet = Triplet(
                            subject=self.graph.nodes[entity].get('label', entity),
                            relation=data.get('relation', ''),
                            object=self.graph.nodes[target].get('label', target),
                            source_chunk_id=data.get('chunk_id', ''),
                        )
                        result_triplets.append(triplet)
                        next_frontier.add(target)

                # Incoming edges
                for source, _, data in self.graph.in_edges(entity, data=True):
                    edge_key = (source, entity, data.get('relation', ''))
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        triplet = Triplet(
                            subject=self.graph.nodes[source].get('label', source),
                            relation=data.get('relation', ''),
                            object=self.graph.nodes[entity].get('label', entity),
                            source_chunk_id=data.get('chunk_id', ''),
                        )
                        result_triplets.append(triplet)
                        next_frontier.add(source)

            frontier = next_frontier - set(entities)

        return result_triplets

    def search(
        self,
        query: str,
        max_hops: int = 2,
        include_text: bool = True,
        max_triplets: int = 50,
    ) -> KGSearchResult:
        """
        Search knowledge graph for relevant information.

        Args:
            query: Search query
            max_hops: Maximum hops from matched entities
            include_text: Whether to include original chunk texts
            max_triplets: Maximum triplets to return

        Returns:
            KGSearchResult with triplets and optionally source texts
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_from_chunks first.")

        # Extract entities from query
        query_entities = self._extract_query_entities(query)
        logger.debug(f"Extracted query entities: {query_entities}")

        # Find matching entities in graph
        matched_entities = self._find_matching_entities(query_entities)
        logger.debug(f"Matched graph entities: {matched_entities}")

        if not matched_entities:
            return KGSearchResult(
                triplets=[],
                source_texts=[],
                matched_entities=[],
            )

        # Get neighborhood triplets
        triplets = self._get_neighborhood_triplets(matched_entities, max_hops)
        triplets = triplets[:max_triplets]

        # Get source texts if requested
        source_texts = []
        if include_text:
            chunk_ids = set(t.source_chunk_id for t in triplets if t.source_chunk_id)
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_map:
                    source_texts.append(self.chunk_map[chunk_id].content)

        return KGSearchResult(
            triplets=triplets,
            source_texts=source_texts,
            matched_entities=matched_entities,
        )

    def triplets_to_text(self, triplets: List[Triplet]) -> str:
        """Convert triplets to readable text format."""
        lines = []
        for t in triplets:
            lines.append(f"- {t.subject} {t.relation} {t.object}")
        return "\n".join(lines)

    @property
    def stats(self) -> Dict[str, int]:
        """Get index statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "triplets": len(self.triplets),
            "chunks": len(self.chunk_map),
        }

    def _compute_fingerprint(self, chunks: List[Chunk]) -> str:
        """Compute fingerprint of chunks for cache validation."""
        content = f"{len(chunks)}:" + "".join(c.chunk_id for c in chunks[:100])
        return hashlib.md5(content.encode()).hexdigest()

    def save(self, cache_dir: Path) -> None:
        """Save KG index to disk for caching."""
        import networkx as nx

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save graph
        nx.write_graphml(self.graph, cache_dir / "graph.graphml")

        # Save triplets
        triplets_data = [
            {
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "source_chunk_id": t.source_chunk_id,
            }
            for t in self.triplets
        ]
        with open(cache_dir / "triplets.json", "w", encoding="utf-8") as f:
            json.dump(triplets_data, f, ensure_ascii=False)

        # Save entity_to_chunks mapping
        with open(cache_dir / "entity_to_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.entity_to_chunks, f, ensure_ascii=False)

        # Save chunks (just ids and content for reconstruction)
        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "start_idx": c.start_idx,
                "end_idx": c.end_idx,
                "metadata": c.metadata,
            }
            for c in self.chunk_map.values()
        ]
        with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)

        logger.info(f"KG index saved to {cache_dir}")

    def load(self, cache_dir: Path) -> bool:
        """Load KG index from cache. Returns True if successful."""
        import networkx as nx

        cache_dir = Path(cache_dir)
        required_files = ["graph.graphml", "triplets.json", "entity_to_chunks.json", "chunks.json"]

        if not all((cache_dir / f).exists() for f in required_files):
            return False

        try:
            # Load graph
            self.graph = nx.read_graphml(cache_dir / "graph.graphml")

            # Load triplets
            with open(cache_dir / "triplets.json", "r", encoding="utf-8") as f:
                triplets_data = json.load(f)
            self.triplets = [
                Triplet(
                    subject=t["subject"],
                    relation=t["relation"],
                    object=t["object"],
                    source_chunk_id=t["source_chunk_id"],
                )
                for t in triplets_data
            ]

            # Load entity_to_chunks
            with open(cache_dir / "entity_to_chunks.json", "r", encoding="utf-8") as f:
                self.entity_to_chunks = json.load(f)

            # Load chunks
            with open(cache_dir / "chunks.json", "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunk_map = {
                c["chunk_id"]: Chunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    content=c["content"],
                    start_idx=c["start_idx"],
                    end_idx=c["end_idx"],
                    metadata=c["metadata"],
                )
                for c in chunks_data
            }

            self._is_built = True
            logger.info(f"KG index loaded from cache: {self.stats}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load KG cache: {e}")
            return False

    def build_from_chunks_cached(
        self,
        chunks: List[Chunk],
        cache_dir: Path,
        force_rebuild: bool = False,
        show_progress: bool = True,
    ) -> None:
        """Build KG from chunks with caching support."""
        cache_dir = Path(cache_dir)
        fingerprint_file = cache_dir / "fingerprint.txt"

        current_fingerprint = self._compute_fingerprint(chunks)

        # Check cache
        if not force_rebuild and fingerprint_file.exists():
            stored_fingerprint = fingerprint_file.read_text().strip()
            if stored_fingerprint == current_fingerprint:
                if self.load(cache_dir):
                    logger.info("Using cached KG index")
                    return

        # Build from scratch
        logger.info("Building KG index (not cached or cache invalid)...")
        self.build_from_chunks(chunks, show_progress=show_progress)

        # Save to cache
        self.save(cache_dir)
        fingerprint_file.write_text(current_fingerprint)
