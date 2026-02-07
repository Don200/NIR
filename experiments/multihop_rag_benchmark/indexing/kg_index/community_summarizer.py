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
        work_items = []
        skipped_empty = 0
        skipped_no_edges = 0

        for cid, community in communities.items():
            if not community.relationships and not community.entity_names:
                community.summary = ""
                skipped_empty += 1
                continue
            edges_text = self._format_community_edges(community)
            if not edges_text.strip():
                community.summary = (
                    f"This community contains the following entities: "
                    f"{', '.join(community.entity_names)}."
                )
                skipped_no_edges += 1
                logger.debug(
                    f"[summarize] Community {cid}: no edges, "
                    f"using entity list fallback | entities={community.entity_names}"
                )
                continue
            work_items.append((cid, edges_text))

        logger.info(
            f"[summarize] {len(work_items)} communities to summarize | "
            f"skipped_empty={skipped_empty}, skipped_no_edges={skipped_no_edges}"
        )

        if not work_items:
            return communities

        if self.num_workers > 1 and len(work_items) > 1:
            self._summarize_parallel(communities, work_items)
        else:
            self._summarize_sequential(communities, work_items)

        logger.info(f"[summarize] Generated summaries for {len(communities)} communities")
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
                future = executor.submit(self._generate_summary, cid, edges_text)
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
                    logger.warning(f"[summarize] FAILED community {cid}: {e}")
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
            communities[cid].summary = self._generate_summary(cid, edges_text)

    def _format_community_edges(self, community: Community) -> str:
        """Format edges as 'entity1->entity2->relation->description' (notebook style)."""
        lines = []
        for rel in community.relationships:
            line = f"{rel.source}->{rel.target}->{rel.relation}->{rel.description}"
            lines.append(line)
        return "\n".join(lines)

    def _generate_summary(self, cid: int, edges_text: str) -> str:
        """Generate a community summary using the LLM (notebook approach: system + user)."""
        logger.debug(
            f"[summarize] Community {cid} | "
            f"edges_text_len={len(edges_text)} | "
            f"edges_preview=\"{edges_text[:300].replace(chr(10), ' | ')}...\""
        )
        try:
            summary = self.llm_client.generate(
                prompt=edges_text,
                system_prompt=COMMUNITY_SUMMARY_SYSTEM_PROMPT,
                max_tokens=512,
            )
            logger.debug(
                f"[summarize] Community {cid} | "
                f"summary_len={len(summary)} | "
                f"summary_preview=\"{summary[:200].replace(chr(10), ' ')}...\""
            )
            return summary
        except Exception as e:
            logger.warning(f"[summarize] FAILED community {cid}: {e}")
            return ""
