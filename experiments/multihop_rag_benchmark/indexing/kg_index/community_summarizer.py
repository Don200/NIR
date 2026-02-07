"""LLM-based community summarization for GraphRAG."""

import logging
from typing import Dict

from ...generation.llm_client import LLMClient
from .models import Community
from .prompts import COMMUNITY_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    """Generate LLM summaries for detected communities."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def summarize(self, communities: Dict[int, Community]) -> Dict[int, Community]:
        """Generate summaries for all communities."""
        try:
            from tqdm import tqdm
            iterator = tqdm(communities.items(), desc="Summarizing communities")
        except ImportError:
            iterator = communities.items()

        for cid, community in iterator:
            if not community.relationships and not community.entity_names:
                community.summary = ""
                continue

            edges_text = self._format_community_edges(community)
            if not edges_text.strip():
                # Fall back to entity list if no relationships
                community.summary = (
                    f"This community contains the following entities: "
                    f"{', '.join(community.entity_names)}."
                )
                continue

            community.summary = self._generate_summary(edges_text)

        logger.info(f"Generated summaries for {len(communities)} communities")
        return communities

    def _format_community_edges(self, community: Community) -> str:
        """Format community relationships as text lines."""
        lines = []
        for rel in community.relationships:
            line = f"{rel.source} -> {rel.target} [{rel.relation}]: {rel.description}"
            lines.append(line)

        # Also include entity descriptions if available
        entity_lines = []
        for name in community.entity_names:
            entity_lines.append(f"Entity: {name}")

        return "\n".join(lines) if lines else "\n".join(entity_lines)

    def _generate_summary(self, edges_text: str) -> str:
        """Generate a community summary using the LLM."""
        prompt = COMMUNITY_SUMMARY_PROMPT.format(edges_text=edges_text)
        try:
            return self.llm_client.generate(prompt, max_tokens=512)
        except Exception as e:
            logger.warning(f"Failed to generate community summary: {e}")
            return ""
