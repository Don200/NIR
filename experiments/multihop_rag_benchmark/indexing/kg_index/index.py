"""GraphRAG v2 index — main orchestrator for build and search."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ...data.preprocessing import Chunk
from ...generation.llm_client import LLMClient, EmbeddingClient
from .models import Entity, Community, GraphRAGSearchResult
from .extractor import GraphRAGExtractor
from .graph_store import GraphStore
from .community_detector import CommunityDetector
from .community_summarizer import CommunitySummarizer

logger = logging.getLogger(__name__)


class GraphRAGIndex:
    """
    GraphRAG v2 index: entity extraction, community detection, and search.

    Build pipeline:
        1. Extract entities + relationships (LLM)
        2. Populate graph (NetworkX)
        3. Detect communities (Leiden)
        4. Generate community summaries (LLM)
        5. Build entity description embeddings (for local search)

    Search:
        - local_search: query → entity embeddings → community summaries
        - global_search: return all community summaries
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        max_paths_per_chunk: int = 10,
        max_cluster_size: int = 10,
        num_workers: int = 4,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client

        self.extractor = GraphRAGExtractor(llm_client, max_paths_per_chunk, num_workers)
        self.graph_store = GraphStore()
        self.community_detector = CommunityDetector(max_cluster_size)
        self.community_summarizer = CommunitySummarizer(llm_client, num_workers)

        self.communities: Dict[int, Community] = {}
        self.entity_to_community_ids: Dict[str, List[int]] = {}

        # Entity embeddings for local search
        self._entity_keys: List[str] = []
        self._entity_embeddings: Optional[np.ndarray] = None

    def build(self, chunks: List[Chunk], show_progress: bool = True) -> None:
        """Build the full GraphRAG index from chunks."""
        # 1. Extract entities and relationships
        logger.info("Step 1/5: Extracting entities and relationships...")
        entities, relationships = self.extractor.extract_batch(chunks, show_progress)

        # 2. Populate graph
        logger.info("Step 2/5: Building graph...")
        for entity in entities:
            self.graph_store.add_entity(entity)
        for rel in relationships:
            self.graph_store.add_relationship(rel)
        logger.info(f"Graph stats: {self.graph_store.stats}")

        # 3. Detect communities
        logger.info("Step 3/5: Detecting communities...")
        self.communities, self.entity_to_community_ids = (
            self.community_detector.detect(self.graph_store)
        )

        # 4. Generate community summaries
        logger.info("Step 4/5: Generating community summaries...")
        self.communities = self.community_summarizer.summarize(self.communities)

        # 5. Build entity embeddings
        logger.info("Step 5/5: Building entity embeddings...")
        self._build_entity_embeddings()

        logger.info(
            f"GraphRAG index built: {self.graph_store.stats}, "
            f"{len(self.communities)} communities"
        )

    def build_cached(
        self,
        chunks: List[Chunk],
        cache_dir: Path,
        force_rebuild: bool = False,
        show_progress: bool = True,
    ) -> None:
        """Build with caching support."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_path = cache_dir / "fingerprint.txt"

        # Compute fingerprint
        fingerprint = self._compute_fingerprint(chunks)

        if not force_rebuild and self._check_cache(cache_dir, fingerprint):
            logger.info("Loading GraphRAG index from cache...")
            if self.load(cache_dir):
                return
            logger.info("Cache load failed, rebuilding...")

        # Build from scratch
        self.build(chunks, show_progress=show_progress)

        # Save cache
        self.save(cache_dir)
        with open(fingerprint_path, "w") as f:
            f.write(fingerprint)

    def local_search(self, query: str, top_k: int = 10) -> GraphRAGSearchResult:
        """
        Local search: embed query → find similar entities → return their community summaries.
        """
        if self._entity_embeddings is None or len(self._entity_keys) == 0:
            return GraphRAGSearchResult()

        # Embed query
        query_embedding = np.array(self.embedding_client.embed_single(query))

        # Cosine similarity
        norms = np.linalg.norm(self._entity_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = self._entity_embeddings / norms

        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        similarities = normalized @ query_embedding
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Map to community IDs
        matched_entities = []
        matched_community_ids = set()
        for idx in top_indices:
            if idx < len(self._entity_keys):
                entity_key = self._entity_keys[idx]
                entity = self.graph_store.get_entity(entity_key)
                if entity:
                    matched_entities.append(entity.name)
                cids = self.entity_to_community_ids.get(entity_key, [])
                matched_community_ids.update(cids)

        # Collect community summaries
        summaries = []
        for cid in sorted(matched_community_ids):
            community = self.communities.get(cid)
            if community and community.summary:
                summaries.append(community.summary)

        return GraphRAGSearchResult(
            community_summaries=summaries,
            matched_entities=matched_entities,
            matched_community_ids=sorted(matched_community_ids),
            metadata={"top_k": top_k, "search_type": "local"},
        )

    def global_search(self) -> GraphRAGSearchResult:
        """Global search: return all community summaries."""
        summaries = []
        community_ids = []
        for cid in sorted(self.communities.keys()):
            community = self.communities[cid]
            if community.summary:
                summaries.append(community.summary)
                community_ids.append(cid)

        return GraphRAGSearchResult(
            community_summaries=summaries,
            matched_entities=[],
            matched_community_ids=community_ids,
            metadata={"search_type": "global"},
        )

    def save(self, path: Path) -> None:
        """Save the full index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save graph
        self.graph_store.save(path / "graph.json")

        # Save communities
        communities_data = {}
        for cid, community in self.communities.items():
            communities_data[str(cid)] = {
                "community_id": community.community_id,
                "entity_names": community.entity_names,
                "relationships": [
                    {
                        "source": r.source,
                        "target": r.target,
                        "relation": r.relation,
                        "description": r.description,
                        "source_chunk_id": r.source_chunk_id,
                    }
                    for r in community.relationships
                ],
                "summary": community.summary,
            }
        with open(path / "communities.json", "w", encoding="utf-8") as f:
            json.dump(communities_data, f, ensure_ascii=False, indent=2)

        # Save entity-to-community mapping
        with open(path / "entity_communities.json", "w", encoding="utf-8") as f:
            json.dump(self.entity_to_community_ids, f, ensure_ascii=False, indent=2)

        # Save entity embeddings
        if self._entity_embeddings is not None:
            np.save(str(path / "entity_embeddings.npy"), self._entity_embeddings)
            with open(path / "entity_keys.json", "w", encoding="utf-8") as f:
                json.dump(self._entity_keys, f)

        logger.info(f"GraphRAG index saved to {path}")

    def load(self, path: Path) -> bool:
        """Load the full index from disk."""
        path = Path(path)
        required_files = ["graph.json", "communities.json", "entity_communities.json"]
        if not all((path / f).exists() for f in required_files):
            return False

        try:
            # Load graph
            self.graph_store = GraphStore()
            self.graph_store.load(path / "graph.json")

            # Load communities
            with open(path / "communities.json", "r", encoding="utf-8") as f:
                communities_data = json.load(f)

            from .models import Relationship
            self.communities = {}
            for cid_str, c_data in communities_data.items():
                cid = int(cid_str)
                rels = [
                    Relationship(
                        source=r["source"],
                        target=r["target"],
                        relation=r["relation"],
                        description=r.get("description", ""),
                        source_chunk_id=r.get("source_chunk_id", ""),
                    )
                    for r in c_data.get("relationships", [])
                ]
                self.communities[cid] = Community(
                    community_id=cid,
                    entity_names=c_data.get("entity_names", []),
                    relationships=rels,
                    summary=c_data.get("summary", ""),
                )

            # Load entity-to-community mapping
            with open(path / "entity_communities.json", "r", encoding="utf-8") as f:
                self.entity_to_community_ids = json.load(f)

            # Load entity embeddings
            embeddings_path = path / "entity_embeddings.npy"
            keys_path = path / "entity_keys.json"
            if embeddings_path.exists() and keys_path.exists():
                self._entity_embeddings = np.load(str(embeddings_path))
                with open(keys_path, "r", encoding="utf-8") as f:
                    self._entity_keys = json.load(f)

            logger.info(f"GraphRAG index loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load GraphRAG index: {e}")
            return False

    @property
    def stats(self) -> dict:
        """Index statistics."""
        return {
            **self.graph_store.stats,
            "num_communities": len(self.communities),
            "num_entity_embeddings": len(self._entity_keys),
            "communities_with_summaries": sum(
                1 for c in self.communities.values() if c.summary
            ),
        }

    def _build_entity_embeddings(self) -> None:
        """Build embeddings for all entity descriptions."""
        entities = self.graph_store.entities
        if not entities:
            return

        self._entity_keys = []
        descriptions = []
        for key, entity in entities.items():
            text = entity.description if entity.description else entity.name
            self._entity_keys.append(key)
            descriptions.append(text)

        logger.info(f"Embedding {len(descriptions)} entity descriptions...")
        embeddings = self.embedding_client.embed(descriptions)
        self._entity_embeddings = np.array(embeddings, dtype=np.float32)

    def _compute_fingerprint(self, chunks: List[Chunk]) -> str:
        """Compute a fingerprint for cache invalidation."""
        content = f"{len(chunks)}:"
        for chunk in chunks[:100]:
            content += chunk.chunk_id + "|"
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_dir: Path, fingerprint: str) -> bool:
        """Check if cache is valid."""
        fingerprint_path = cache_dir / "fingerprint.txt"
        if not fingerprint_path.exists():
            return False
        stored = fingerprint_path.read_text().strip()
        return stored == fingerprint
