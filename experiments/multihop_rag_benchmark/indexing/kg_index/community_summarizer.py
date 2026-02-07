"""LLM-based community summarization — based on LlamaIndex GraphRAG v2 cookbook.

The notebook formats edges as "entity1->entity2->relation->description"
and uses a system prompt asking the LLM to synthesize relationships.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from ...generation.llm_client import LLMClient
from .models import Community
from .prompts import COMMUNITY_SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    """Generate LLM summaries for detected communities."""

    def __init__(self, llm_client: LLMClient, num_workers: int = 4):
        self.llm_client = llm_client
        self.num_workers = num_workers

    def summarize(self, communities: Dict[int, Community]) -> Dict[int, Community]:
        """Generate summaries for all communities."""
        # Prepare work items: (community_id, edges_text)
        work_items = []
        for cid, community in communities.items():
            if not community.relationships and not community.entity_names:
                community.summary = ""
                continue
            edges_text = self._format_community_edges(community)
            if not edges_text.strip():
                community.summary = (
                    f"This community contains the following entities: "
                    f"{', '.join(community.entity_names)}."
                )
                continue
            work_items.append((cid, edges_text))

        if not work_items:
            return communities

        # Generate summaries (parallel if num_workers > 1)
        if self.num_workers > 1 and len(work_items) > 1:
            self._summarize_parallel(communities, work_items)
        else:
            self._summarize_sequential(communities, work_items)

        logger.info(f"Generated summaries for {len(communities)} communities")
        return communities

    def _summarize_parallel(
        self,
        communities: Dict[int, Community],
        work_items: list,
    ) -> None:
        """Generate summaries in parallel."""
        futures = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for cid, edges_text in work_items:
                future = executor.submit(self._generate_summary, edges_text)
                futures[future] = cid

            try:
                from tqdm import tqdm
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Summarizing communities ({self.num_workers} workers)",
                )
            except ImportError:
                iterator = as_completed(futures)

            for future in iterator:
                cid = futures[future]
                try:
                    communities[cid].summary = future.result()
                except Exception as e:
                    logger.warning(f"Failed to summarize community {cid}: {e}")
                    communities[cid].summary = ""

    def _summarize_sequential(
        self,
        communities: Dict[int, Community],
        work_items: list,
    ) -> None:
        """Generate summaries sequentially."""
        try:
            from tqdm import tqdm
            iterator = tqdm(work_items, desc="Summarizing communities")
        except ImportError:
            iterator = work_items

        for cid, edges_text in iterator:
            communities[cid].summary = self._generate_summary(edges_text)

    def _format_community_edges(self, community: Community) -> str:
        """Format edges as 'entity1->entity2->relation->description' (notebook style)."""
        lines = []
        for rel in community.relationships:
            line = f"{rel.source}->{rel.target}->{rel.relation}->{rel.description}"
            lines.append(line)
        return "\n".join(lines)

    def _generate_summary(self, edges_text: str) -> str:
        """Generate a community summary using the LLM (notebook approach: system + user)."""
        try:
            return self.llm_client.generate(
                prompt=edges_text,
                system_prompt=COMMUNITY_SUMMARY_SYSTEM_PROMPT,
                max_tokens=512,
            )
        except Exception as e:
            logger.warning(f"Failed to generate community summary: {e}")
            return ""
