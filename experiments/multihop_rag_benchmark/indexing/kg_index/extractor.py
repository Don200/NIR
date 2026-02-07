"""GraphRAG entity/relationship extractor — based on LlamaIndex GraphRAG v2 cookbook.

Key differences from the previous version:
1. Prompt: step-by-step with detailed 3-entity example (from notebook)
2. Parse: regex r"{.*}" (re.DOTALL) to find JSON, then json.loads (from notebook)
3. Parallel: ThreadPoolExecutor(num_workers) instead of sequential loop
   (notebook uses asyncio.run_jobs with semaphore — equivalent for sync client)
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from ...data.preprocessing import Chunk
from ...generation.llm_client import LLMClient
from .models import Entity, Relationship
from .prompts import KG_TRIPLET_EXTRACT_TMPL

logger = logging.getLogger(__name__)


class GraphRAGExtractor:
    """Extract entities and relationships with descriptions from text chunks.

    Mirrors the notebook's GraphRAGExtractor but uses ThreadPoolExecutor
    instead of asyncio (our LLMClient is synchronous).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ):
        self.llm_client = llm_client
        self.max_paths_per_chunk = max_paths_per_chunk
        self.num_workers = num_workers

    def extract(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single chunk."""
        prompt = KG_TRIPLET_EXTRACT_TMPL.format(
            text=chunk.content,
            max_knowledge_triplets=self.max_paths_per_chunk,
        )

        text_preview = chunk.content[:200].replace('\n', ' ')
        logger.debug(
            f"[extract] chunk={chunk.chunk_id} | "
            f"text_preview=\"{text_preview}...\" | "
            f"max_triplets={self.max_paths_per_chunk}"
        )

        try:
            response = self.llm_client.generate(prompt, max_tokens=2048)
            logger.debug(
                f"[extract] chunk={chunk.chunk_id} | "
                f"llm_response_len={len(response)} | "
                f"response_preview=\"{response[:300].replace(chr(10), ' ')}...\""
            )
            entities, relationships = self._parse_fn(response, chunk.chunk_id)
            logger.debug(
                f"[extract] chunk={chunk.chunk_id} | "
                f"parsed: {len(entities)} entities, {len(relationships)} relationships"
            )
            if entities:
                names = [e.name for e in entities]
                logger.debug(
                    f"[extract] chunk={chunk.chunk_id} | "
                    f"entities: {names}"
                )
            if relationships:
                rels = [f"{r.source} -[{r.relation}]-> {r.target}" for r in relationships]
                logger.debug(
                    f"[extract] chunk={chunk.chunk_id} | "
                    f"relationships: {rels}"
                )
            return entities, relationships
        except Exception as e:
            logger.warning(f"[extract] FAILED chunk={chunk.chunk_id}: {e}")
            return [], []

    def extract_batch(
        self, chunks: List[Chunk], show_progress: bool = True
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract from multiple chunks in parallel (ThreadPoolExecutor)."""
        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []

        logger.info(
            f"[extract_batch] Starting extraction: "
            f"{len(chunks)} chunks, {self.num_workers} workers, "
            f"max_paths_per_chunk={self.max_paths_per_chunk}"
        )

        if self.num_workers <= 1:
            return self._extract_sequential(chunks, show_progress)

        # Parallel extraction
        failed_count = 0
        empty_count = 0
        futures = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for chunk in chunks:
                future = executor.submit(self.extract, chunk)
                futures[future] = chunk.chunk_id

            try:
                from tqdm import tqdm
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Extracting entities/relations ({self.num_workers} workers)",
                )
            except ImportError:
                iterator = as_completed(futures)

            for future in iterator:
                chunk_id = futures[future]
                try:
                    entities, relationships = future.result()
                    if not entities and not relationships:
                        empty_count += 1
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"[extract_batch] Worker failed for chunk {chunk_id}: {e}")

        logger.info(
            f"[extract_batch] Done: "
            f"{len(all_entities)} entities, {len(all_relationships)} relationships "
            f"from {len(chunks)} chunks | "
            f"empty={empty_count}, failed={failed_count}"
        )
        return all_entities, all_relationships

    def _extract_sequential(
        self, chunks: List[Chunk], show_progress: bool = True
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Fallback sequential extraction."""
        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        empty_count = 0

        iterator = chunks
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(chunks, desc="Extracting entities/relations")
            except ImportError:
                pass

        for chunk in iterator:
            entities, relationships = self.extract(chunk)
            if not entities and not relationships:
                empty_count += 1
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        logger.info(
            f"[extract_batch] Done (sequential): "
            f"{len(all_entities)} entities, {len(all_relationships)} relationships "
            f"from {len(chunks)} chunks | empty={empty_count}"
        )
        return all_entities, all_relationships

    def _parse_fn(
        self, response_str: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response — notebook's parse_fn approach.

        Uses regex r"{.*}" with re.DOTALL to find the JSON object,
        then json.loads to parse it.
        """
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        # Notebook approach: find first JSON object with regex
        match = re.search(r"\{.*\}", response_str, re.DOTALL)
        if not match:
            logger.warning(
                f"[parse] No JSON found for chunk {chunk_id} | "
                f"raw_response=\"{response_str[:500].replace(chr(10), ' ')}\""
            )
            return entities, relationships

        json_str = match.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"[parse] JSON decode error for chunk {chunk_id}: {e} | "
                f"json_str=\"{json_str[:500].replace(chr(10), ' ')}\""
            )
            return entities, relationships

        # Parse entities — notebook field names
        raw_entities = data.get("entities", [])
        for e in raw_entities:
            if not isinstance(e, dict):
                logger.debug(f"[parse] chunk={chunk_id} | skipping non-dict entity: {e}")
                continue
            name = e.get("entity_name", "").strip()
            if not name:
                logger.debug(f"[parse] chunk={chunk_id} | skipping entity with empty name: {e}")
                continue
            entities.append(Entity(
                name=name,
                entity_type=e.get("entity_type", "Unknown"),
                description=e.get("entity_description", ""),
                source_chunk_ids=[chunk_id],
            ))

        # Parse relationships — notebook field names
        raw_rels = data.get("relationships", [])
        for r in raw_rels:
            if not isinstance(r, dict):
                logger.debug(f"[parse] chunk={chunk_id} | skipping non-dict relationship: {r}")
                continue
            source = r.get("source_entity", "").strip()
            target = r.get("target_entity", "").strip()
            if not source or not target:
                logger.debug(
                    f"[parse] chunk={chunk_id} | skipping relationship with empty "
                    f"source/target: {r}"
                )
                continue
            relationships.append(Relationship(
                source=source,
                target=target,
                relation=r.get("relation", "RELATED_TO"),
                description=r.get("relationship_description", ""),
                source_chunk_id=chunk_id,
            ))

        logger.debug(
            f"[parse] chunk={chunk_id} | "
            f"raw_entities={len(raw_entities)}, parsed={len(entities)} | "
            f"raw_rels={len(raw_rels)}, parsed={len(relationships)}"
        )

        return entities, relationships
