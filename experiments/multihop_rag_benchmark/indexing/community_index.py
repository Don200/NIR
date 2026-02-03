"""Community-based GraphRAG using Microsoft GraphRAG library."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import json

logger = logging.getLogger(__name__)


@dataclass
class CommunitySearchResult:
    """Result from community-based search."""
    response: str
    context_data: Dict[str, Any]
    search_type: str  # "local" or "global"


class CommunityGraphIndex:
    """
    Community-based GraphRAG using Microsoft's graphrag library.

    Implements the approach from the paper:
    - Build KG with entity/relation extraction
    - Detect hierarchical communities using Leiden algorithm
    - Generate summaries for each community
    - Local search: entities + relations + low-level summaries
    - Global search: high-level community summaries
    """

    def __init__(
        self,
        root_dir: Path,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        embedding_api_key: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.llm_model = llm_model
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_api_base = llm_api_base

        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_api_base = embedding_api_base

        self._is_indexed = False
        self._check_graphrag_available()

    def _check_graphrag_available(self):
        """Check if graphrag is installed."""
        try:
            import graphrag
            self._graphrag_available = True
        except ImportError:
            logger.warning(
                "graphrag not installed. Install with: pip install graphrag"
            )
            self._graphrag_available = False

    def _create_settings_yaml(self) -> None:
        """Create settings.yaml for graphrag."""
        settings = {
            "llm": {
                "api_key": "${GRAPHRAG_API_KEY}",
                "type": "openai_chat",
                "model": self.llm_model,
                "model_supports_json": True,
            },
            "embeddings": {
                "llm": {
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "type": "openai_embedding",
                    "model": self.embedding_model,
                }
            },
            "input": {
                "type": "file",
                "file_type": "text",
                "base_dir": "input",
                "file_pattern": ".*\\.txt$",
            },
            "cache": {
                "type": "file",
                "base_dir": "cache",
            },
            "storage": {
                "type": "file",
                "base_dir": "output",
            },
            "reporting": {
                "type": "file",
                "base_dir": "logs",
            },
            "entity_extraction": {
                "prompt": None,  # Use default
                "max_gleanings": 1,
            },
            "community_reports": {
                "prompt": None,  # Use default
                "max_length": 2000,
            },
            "cluster_graph": {
                "max_cluster_size": 10,
            },
        }

        # Add api_base if specified
        if self.llm_api_base:
            settings["llm"]["api_base"] = self.llm_api_base
        if self.embedding_api_base:
            settings["embeddings"]["llm"]["api_base"] = self.embedding_api_base

        import yaml
        settings_path = self.root_dir / "settings.yaml"
        with open(settings_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

        # Create .env file
        env_path = self.root_dir / ".env"
        with open(env_path, "w") as f:
            f.write(f"GRAPHRAG_API_KEY={self.llm_api_key}\n")

    def _prepare_input_documents(self, documents: List[Any]) -> None:
        """Prepare input documents for graphrag indexing."""
        input_dir = self.root_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Write each document as a separate text file
        for i, doc in enumerate(documents):
            content = doc.content if hasattr(doc, 'content') else str(doc)
            title = doc.title if hasattr(doc, 'title') else f"doc_{i}"

            # Clean filename
            safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
            filename = f"{i:04d}_{safe_title[:50]}.txt"

            filepath = input_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        logger.info(f"Prepared {len(documents)} documents in {input_dir}")

    def index_documents(self, documents: List[Any]) -> None:
        """
        Index documents using graphrag.

        This runs the full graphrag indexing pipeline:
        1. Entity extraction
        2. Relationship extraction
        3. Community detection
        4. Community summarization

        Args:
            documents: List of documents with .content attribute
        """
        if not self._graphrag_available:
            raise RuntimeError("graphrag not installed")

        logger.info("Setting up graphrag project...")
        self._create_settings_yaml()
        self._prepare_input_documents(documents)

        logger.info("Running graphrag indexing (this may take a while)...")

        # Run indexing via CLI (most reliable method)
        import subprocess
        result = subprocess.run(
            ["python", "-m", "graphrag.index", "--root", str(self.root_dir)],
            capture_output=True,
            text=True,
            env={**os.environ, "GRAPHRAG_API_KEY": self.llm_api_key},
        )

        if result.returncode != 0:
            logger.error(f"Indexing failed: {result.stderr}")
            raise RuntimeError(f"GraphRAG indexing failed: {result.stderr}")

        logger.info("Indexing completed successfully")
        self._is_indexed = True

    def local_search(
        self,
        query: str,
        community_level: int = 2,
    ) -> CommunitySearchResult:
        """
        Perform local search.

        Local search uses entities, relationships, and low-level community summaries.
        Best for specific, detail-oriented queries.

        Args:
            query: Search query
            community_level: Which community level to search

        Returns:
            CommunitySearchResult
        """
        if not self._graphrag_available:
            raise RuntimeError("graphrag not installed")

        import subprocess
        result = subprocess.run(
            [
                "python", "-m", "graphrag.query",
                "--root", str(self.root_dir),
                "--method", "local",
                "--community-level", str(community_level),
                query,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GRAPHRAG_API_KEY": self.llm_api_key},
        )

        if result.returncode != 0:
            logger.error(f"Local search failed: {result.stderr}")
            return CommunitySearchResult(
                response=f"Search failed: {result.stderr}",
                context_data={},
                search_type="local",
            )

        return CommunitySearchResult(
            response=result.stdout.strip(),
            context_data={"community_level": community_level},
            search_type="local",
        )

    def global_search(
        self,
        query: str,
        community_level: int = 1,
    ) -> CommunitySearchResult:
        """
        Perform global search.

        Global search uses high-level community summaries.
        Best for broad, thematic queries.

        Args:
            query: Search query
            community_level: Which community level to search (higher = more abstract)

        Returns:
            CommunitySearchResult
        """
        if not self._graphrag_available:
            raise RuntimeError("graphrag not installed")

        import subprocess
        result = subprocess.run(
            [
                "python", "-m", "graphrag.query",
                "--root", str(self.root_dir),
                "--method", "global",
                "--community-level", str(community_level),
                query,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GRAPHRAG_API_KEY": self.llm_api_key},
        )

        if result.returncode != 0:
            logger.error(f"Global search failed: {result.stderr}")
            return CommunitySearchResult(
                response=f"Search failed: {result.stderr}",
                context_data={},
                search_type="global",
            )

        return CommunitySearchResult(
            response=result.stdout.strip(),
            context_data={"community_level": community_level},
            search_type="global",
        )

    def search(
        self,
        query: str,
        method: str = "local",
        community_level: Optional[int] = None,
    ) -> CommunitySearchResult:
        """
        Unified search interface.

        Args:
            query: Search query
            method: "local" or "global"
            community_level: Community level (default: 2 for local, 1 for global)

        Returns:
            CommunitySearchResult
        """
        if method == "local":
            level = community_level if community_level is not None else 2
            return self.local_search(query, level)
        elif method == "global":
            level = community_level if community_level is not None else 1
            return self.global_search(query, level)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'local' or 'global'")

    @property
    def is_indexed(self) -> bool:
        """Check if index is built."""
        return self._is_indexed or (self.root_dir / "output").exists()
