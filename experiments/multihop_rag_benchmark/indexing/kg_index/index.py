"""GraphRAG v2 index — main orchestrator for build and search."""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

        # Community summary embeddings for global search pre-filtering
        self._community_keys: List[int] = []
        self._community_embeddings: Optional[np.ndarray] = None

        # Chunk ID -> content mapping for source text retrieval
        self._chunks_map: Dict[str, str] = {}

    def build(self, chunks: List[Chunk], show_progress: bool = True) -> None:
        """Build the full GraphRAG index from chunks."""
        logger.info(
            f"[build] Starting GraphRAG build: {len(chunks)} chunks | "
            f"max_paths_per_chunk={self.extractor.max_paths_per_chunk} | "
            f"num_workers={self.extractor.num_workers} | "
            f"max_cluster_size={self.community_detector.max_cluster_size}"
        )

        # Store chunk contents for source text retrieval in local search
        self._chunks_map = {c.chunk_id: c.content for c in chunks}

        # 1. Extract entities and relationships
        logger.info("[build] Step 1/6: Extracting entities and relationships...")
        entities, relationships = self.extractor.extract_batch(chunks, show_progress)
        logger.info(
            f"[build] Step 1/6 done: {len(entities)} raw entities, "
            f"{len(relationships)} raw relationships"
        )

        # 2. Populate graph
        logger.info("[build] Step 2/6: Building graph...")
        for entity in entities:
            self.graph_store.add_entity(entity)
        for rel in relationships:
            self.graph_store.add_relationship(rel)
        logger.info(f"[build] Step 2/6 done: {self.graph_store.stats}")

        # 3. Detect communities
        logger.info("[build] Step 3/6: Detecting communities...")
        self.communities, self.entity_to_community_ids = (
            self.community_detector.detect(self.graph_store)
        )
        logger.info(
            f"[build] Step 3/6 done: {len(self.communities)} communities, "
            f"{len(self.entity_to_community_ids)} entities mapped"
        )

        # 4. Generate community summaries
        logger.info("[build] Step 4/6: Generating community summaries...")
        self.communities = self.community_summarizer.summarize(self.communities)
        summaries_count = sum(1 for c in self.communities.values() if c.summary)
        logger.info(
            f"[build] Step 4/6 done: {summaries_count}/{len(self.communities)} "
            f"communities have summaries"
        )

        # 5. Build entity embeddings
        logger.info("[build] Step 5/6: Building entity embeddings...")
        self._build_entity_embeddings()

        # 6. Build community summary embeddings (for global search pre-filtering)
        logger.info("[build] Step 6/6: Building community summary embeddings...")
        self._build_community_embeddings()

        logger.info(f"[build] GraphRAG index complete: {self.stats}")

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
        Local search: embed query → find similar entities → return entity context + community summaries.

        Returns enriched context: entity descriptions, relationships, AND community summaries.
        This matches MS GraphRAG local search which returns entities + relations + community reports.
        """
        logger.debug(f"[local_search] query=\"{query}\" | top_k={top_k}")

        if self._entity_embeddings is None or len(self._entity_keys) == 0:
            logger.debug("[local_search] No entity embeddings available, returning empty")
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

        # Map to entities and communities
        matched_entities = []
        matched_entity_keys = []
        matched_community_ids = set()
        for idx in top_indices:
            if idx < len(self._entity_keys):
                entity_key = self._entity_keys[idx]
                score = float(similarities[idx])
                entity = self.graph_store.get_entity(entity_key)
                if entity:
                    matched_entities.append(entity.name)
                    matched_entity_keys.append(entity_key)
                    logger.debug(
                        f"[local_search] Match: \"{entity.name}\" "
                        f"(score={score:.4f}, type={entity.entity_type})"
                    )
                cids = self.entity_to_community_ids.get(entity_key, [])
                matched_community_ids.update(cids)

        # Build enriched entity context (descriptions + relationships)
        entity_context = self._build_entity_context(matched_entity_keys)

        # Collect source text chunks for matched entities
        source_chunks = []
        seen_chunk_ids = set()
        if self._chunks_map:
            for entity_key in matched_entity_keys:
                entity = self.graph_store.get_entity(entity_key)
                if entity:
                    for chunk_id in entity.source_chunk_ids:
                        if chunk_id not in seen_chunk_ids and chunk_id in self._chunks_map:
                            seen_chunk_ids.add(chunk_id)
                            source_chunks.append(self._chunks_map[chunk_id])

        # Collect community summaries
        summaries = []
        for cid in sorted(matched_community_ids):
            community = self.communities.get(cid)
            if community and community.summary:
                summaries.append(community.summary)

        logger.debug(
            f"[local_search] Result: "
            f"{len(matched_entities)} entities -> "
            f"{len(entity_context)} entity contexts -> "
            f"{len(matched_community_ids)} communities -> "
            f"{len(summaries)} summaries | "
            f"{len(source_chunks)} source chunks | "
            f"total_summary_chars={sum(len(s) for s in summaries)}"
        )

        return GraphRAGSearchResult(
            community_summaries=summaries,
            matched_entities=matched_entities,
            matched_community_ids=sorted(matched_community_ids),
            entity_context=entity_context,
            source_chunks=source_chunks,
            metadata={"top_k": top_k, "search_type": "local"},
        )

    def global_search(
        self,
        query: str,
        top_communities: int = 20,
        min_score: int = 20,
    ) -> GraphRAGSearchResult:
        """
        Global search with map-reduce (based on Microsoft GraphRAG).

        1. Pre-filter communities by embedding similarity to the query.
        2. Map phase: for each community, LLM extracts relevant points with scores.
        3. Filter by min_score, sort by relevance.
        4. Return top points as context.
        """
        logger.debug(
            f"[global_search] query=\"{query}\" | "
            f"top_communities={top_communities} | min_score={min_score}"
        )

        if not self.communities:
            return GraphRAGSearchResult()

        # Pre-filter communities by embedding similarity
        if self._community_embeddings is not None and len(self._community_keys) > 0:
            query_embedding = np.array(self.embedding_client.embed_single(query))

            norms = np.linalg.norm(self._community_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = self._community_embeddings / norms

            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

            similarities = normalized @ query_embedding
            n = min(top_communities, len(self._community_keys))
            top_indices = np.argsort(similarities)[-n:][::-1]

            filtered_cids = [self._community_keys[i] for i in top_indices]
            logger.debug(
                f"[global_search] Pre-filtered to {len(filtered_cids)} communities "
                f"by embedding similarity"
            )
        else:
            # Fallback: use all communities
            filtered_cids = sorted(self.communities.keys())
            logger.debug(
                f"[global_search] No community embeddings, using all "
                f"{len(filtered_cids)} communities"
            )

        # Map phase: extract relevant points from each community
        all_points = self._map_communities_parallel(query, filtered_cids)

        # Filter by score and sort
        relevant_points = [p for p in all_points if p.get("score", 0) >= min_score]
        relevant_points.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.debug(
            f"[global_search] Map result: {len(all_points)} total points, "
            f"{len(relevant_points)} above threshold ({min_score})"
        )

        # Build context from top points
        context_texts = [
            p["description"] for p in relevant_points if p.get("description")
        ]
        community_ids = sorted(set(
            p.get("community_id", -1) for p in relevant_points
            if p.get("community_id", -1) >= 0
        ))

        return GraphRAGSearchResult(
            community_summaries=context_texts,
            matched_community_ids=community_ids,
            global_search_points=relevant_points,
            metadata={
                "search_type": "global_map_reduce",
                "total_communities_checked": len(filtered_cids),
                "total_points_extracted": len(all_points),
                "points_above_threshold": len(relevant_points),
            },
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

        # Save community summary embeddings
        if self._community_embeddings is not None:
            np.save(str(path / "community_embeddings.npy"), self._community_embeddings)
            with open(path / "community_keys.json", "w", encoding="utf-8") as f:
                json.dump(self._community_keys, f)

        # Save chunk content map for source text retrieval
        if self._chunks_map:
            with open(path / "chunks_map.json", "w", encoding="utf-8") as f:
                json.dump(self._chunks_map, f, ensure_ascii=False)

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

            # Load community summary embeddings
            comm_emb_path = path / "community_embeddings.npy"
            comm_keys_path = path / "community_keys.json"
            if comm_emb_path.exists() and comm_keys_path.exists():
                self._community_embeddings = np.load(str(comm_emb_path))
                with open(comm_keys_path, "r", encoding="utf-8") as f:
                    self._community_keys = json.load(f)

            # Load chunk content map (graceful fallback for old caches)
            chunks_map_path = path / "chunks_map.json"
            if chunks_map_path.exists():
                with open(chunks_map_path, "r", encoding="utf-8") as f:
                    self._chunks_map = json.load(f)
            else:
                self._chunks_map = {}

            logger.info(f"GraphRAG index loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load GraphRAG index: {e}")
            return False

    def export_graphml(self, path: Path) -> None:
        """Export the knowledge graph to GraphML format.

        Includes community_id as node attribute if communities have been built.
        """
        self.graph_store.export_graphml(
            path,
            community_mapping=self.entity_to_community_ids or None,
        )

    @property
    def stats(self) -> dict:
        """Index statistics."""
        return {
            **self.graph_store.stats,
            "num_communities": len(self.communities),
            "num_entity_embeddings": len(self._entity_keys),
            "num_community_embeddings": len(self._community_keys),
            "communities_with_summaries": sum(
                1 for c in self.communities.values() if c.summary
            ),
        }

    def _build_entity_context(self, entity_keys: List[str]) -> List[str]:
        """Build rich entity context with descriptions and relationships.

        For each entity, produces a block like:
            Entity: Tim Cook (Person)
            Description: CEO of Apple Inc., appointed in 2011...
            Relationships:
              - Tim Cook -> Apple Inc. [CEO_OF]: Leads Apple as CEO
              - Tim Cook -> iPhone 15 [ANNOUNCED]: Presented at 2023 keynote
        """
        contexts = []
        for entity_key in entity_keys:
            entity = self.graph_store.get_entity(entity_key)
            if not entity:
                continue

            lines = [f"Entity: {entity.name} ({entity.entity_type})"]
            if entity.description:
                lines.append(f"Description: {entity.description}")

            # Gather relationships
            neighbors = self.graph_store.get_neighbors(entity_key)
            rel_lines = []
            for neighbor_key in neighbors:
                edge_data = self.graph_store.get_edge_data(entity_key, neighbor_key)
                if not edge_data:
                    continue
                for rel in edge_data.get("relations", []):
                    neighbor_entity = self.graph_store.get_entity(neighbor_key)
                    neighbor_name = neighbor_entity.name if neighbor_entity else neighbor_key
                    rel_label = rel.get("relation", "RELATED_TO")
                    desc = rel.get("description", "")
                    rel_lines.append(
                        f"  - {entity.name} -> {neighbor_name} [{rel_label}]: {desc}"
                    )

            if rel_lines:
                lines.append("Relationships:")
                lines.extend(rel_lines)

            contexts.append("\n".join(lines))

        return contexts

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

    def _build_community_embeddings(self) -> None:
        """Build embeddings for community summaries (used for global search pre-filtering)."""
        summaries = []
        keys = []
        for cid in sorted(self.communities.keys()):
            community = self.communities[cid]
            if community.summary:
                keys.append(cid)
                summaries.append(community.summary)

        if not summaries:
            self._community_keys = []
            self._community_embeddings = None
            return

        logger.info(f"Embedding {len(summaries)} community summaries...")
        embeddings = self.embedding_client.embed(summaries)
        self._community_keys = keys
        self._community_embeddings = np.array(embeddings, dtype=np.float32)

    def _map_community_points(
        self, query: str, community_id: int, summary: str
    ) -> List[dict]:
        """Map phase: extract relevant points with scores from a single community."""
        from .prompts import GLOBAL_SEARCH_MAP_SYSTEM_PROMPT, GLOBAL_SEARCH_MAP_PROMPT

        prompt = GLOBAL_SEARCH_MAP_PROMPT.format(
            context_data=summary,
            query=query,
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=GLOBAL_SEARCH_MAP_SYSTEM_PROMPT,
                max_tokens=512,
            )

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                points = data.get("points", [])
                valid_points = []
                for p in points:
                    if isinstance(p, dict) and "description" in p:
                        p["community_id"] = community_id
                        p["score"] = int(p.get("score", 0))
                        valid_points.append(p)
                logger.debug(
                    f"[global_search] Map community {community_id}: "
                    f"{len(valid_points)} points extracted"
                )
                return valid_points
            else:
                logger.debug(
                    f"[global_search] Map community {community_id}: "
                    f"no JSON found in response"
                )
                return []
        except Exception as e:
            logger.warning(
                f"[global_search] Map failed for community {community_id}: {e}"
            )
            return []

    def _map_communities_parallel(
        self, query: str, community_ids: List[int]
    ) -> List[dict]:
        """Run map phase in parallel across communities."""
        work_items = []
        for cid in community_ids:
            community = self.communities.get(cid)
            if community and community.summary:
                work_items.append((cid, community.summary))

        if not work_items:
            return []

        all_points = []
        logger.debug(
            f"[global_search] Map phase: {len(work_items)} communities, "
            f"{self.extractor.num_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=self.extractor.num_workers) as executor:
            futures = {}
            for cid, summary in work_items:
                future = executor.submit(
                    self._map_community_points, query, cid, summary
                )
                futures[future] = cid

            for future in as_completed(futures):
                cid = futures[future]
                try:
                    points = future.result()
                    all_points.extend(points)
                except Exception as e:
                    logger.warning(
                        f"[global_search] Map failed for community {cid}: {e}"
                    )

        return all_points

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
