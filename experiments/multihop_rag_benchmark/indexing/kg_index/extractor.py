"""GraphRAG entity and relationship extractor using LLM."""

import json
import logging
import re
from typing import List, Tuple

from ...data.preprocessing import Chunk
from ...generation.llm_client import LLMClient
from .models import Entity, Relationship
from .prompts import KG_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class GraphRAGExtractor:
    """Extract entities and relationships with descriptions from text chunks using an LLM."""

    def __init__(self, llm_client: LLMClient, max_paths_per_chunk: int = 10):
        self.llm_client = llm_client
        self.max_paths_per_chunk = max_paths_per_chunk

    def extract(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single chunk."""
        prompt = KG_EXTRACTION_PROMPT.format(text=chunk.content)

        try:
            response = self.llm_client.generate(prompt, max_tokens=1024)
            entities, relationships = self._parse_llm_response(response, chunk.chunk_id)
            # Limit to max_paths_per_chunk
            entities = entities[:self.max_paths_per_chunk]
            relationships = relationships[:self.max_paths_per_chunk]
            return entities, relationships
        except Exception as e:
            logger.warning(f"Extraction failed for chunk {chunk.chunk_id}: {e}")
            return [], []

    def extract_batch(
        self, chunks: List[Chunk], show_progress: bool = True
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from multiple chunks."""
        all_entities = []
        all_relationships = []

        iterator = chunks
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(chunks, desc="Extracting entities/relations")
            except ImportError:
                pass

        for chunk in iterator:
            entities, relationships = self.extract(chunk)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        logger.info(
            f"Extracted {len(all_entities)} entities and "
            f"{len(all_relationships)} relationships from {len(chunks)} chunks"
        )
        return all_entities, all_relationships

    def _parse_llm_response(
        self, response: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM JSON response into Entity and Relationship objects."""
        # Try to extract JSON from response (handle markdown code blocks)
        json_str = response.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)```', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()

        # Try to find JSON object in response
        brace_start = json_str.find('{')
        brace_end = json_str.rfind('}')
        if brace_start != -1 and brace_end != -1:
            json_str = json_str[brace_start:brace_end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for chunk {chunk_id}")
            return [], []

        entities = []
        for e in data.get("entities", []):
            if not isinstance(e, dict):
                continue
            name = e.get("name", "").strip()
            if not name:
                continue
            entities.append(Entity(
                name=name,
                entity_type=e.get("type", "Unknown"),
                description=e.get("description", ""),
                source_chunk_ids=[chunk_id],
            ))

        relationships = []
        for r in data.get("relationships", []):
            if not isinstance(r, dict):
                continue
            source = r.get("source", "").strip()
            target = r.get("target", "").strip()
            if not source or not target:
                continue
            relationships.append(Relationship(
                source=source,
                target=target,
                relation=r.get("relation", "RELATED_TO"),
                description=r.get("description", ""),
                source_chunk_id=chunk_id,
            ))

        return entities, relationships
