"""LlamaIndex PropertyGraphIndex for knowledge graph."""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import (
    Document as LlamaDocument,
    PropertyGraphIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

# Try to import OpenAILikeEmbedding, fallback to OpenAIEmbedding
try:
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    HAS_OPENAI_LIKE_EMBEDDING = True
except ImportError:
    HAS_OPENAI_LIKE_EMBEDDING = False

from ..core.config import Config
from ..core.models import Document, SearchResult, RetrievalMethod

logger = logging.getLogger(__name__)


class GraphIndex:
    """LlamaIndex PropertyGraphIndex for triplet-based retrieval."""

    def __init__(
        self,
        config: Config,
        persist_dir: Optional[Path] = None,
    ):
        self.config = config
        self.persist_dir = persist_dir or config.index_dir / "graph"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[PropertyGraphIndex] = None
        self._setup_llama_settings()

    def _setup_llama_settings(self) -> None:
        """Configure LlamaIndex global settings."""
        # Use OpenAILike for OpenAI-compatible APIs (OpenRouter, etc.)
        Settings.llm = OpenAILike(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            api_base=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
            is_chat_model=True,
        )

        # Use OpenAILikeEmbedding if available, otherwise try OpenAIEmbedding with model_name
        if HAS_OPENAI_LIKE_EMBEDDING:
            Settings.embed_model = OpenAILikeEmbedding(
                model_name=self.config.embedding.model,
                api_key=self.config.embedding.api_key,
                api_base=self.config.embedding.base_url,
            )
        else:
            # Fallback: use OpenAIEmbedding with model_name to bypass validation
            Settings.embed_model = OpenAIEmbedding(
                model_name=self.config.embedding.model,
                api_key=self.config.embedding.api_key,
                api_base=self.config.embedding.base_url,
            )

    @property
    def index(self) -> Optional[PropertyGraphIndex]:
        """Get loaded index."""
        return self._index

    def index_documents(self, documents: list[Document]) -> int:
        """Build graph index from documents."""
        # Convert to LlamaIndex documents
        llama_docs = [
            LlamaDocument(
                text=doc.content,
                doc_id=doc.id,
                metadata={
                    "title": doc.title or "",
                    **doc.metadata,
                },
            )
            for doc in documents
        ]

        logger.info(f"Building graph index from {len(llama_docs)} documents")

        # Create extractor for triplets
        kg_extractor = SimpleLLMPathExtractor(
            llm=Settings.llm,
            max_paths_per_chunk=self.config.graph.max_triplets_per_chunk,
            num_workers=4,
        )

        # Build index
        self._index = PropertyGraphIndex.from_documents(
            llama_docs,
            kg_extractors=[kg_extractor],
            embed_kg_nodes=self.config.graph.include_embeddings,
            show_progress=True,
        )

        # Persist to disk
        self._index.storage_context.persist(persist_dir=str(self.persist_dir))
        logger.info(f"Graph index saved to {self.persist_dir}")

        # Return approximate count
        try:
            graph_store = self._index.property_graph_store
            return len(graph_store.get_triplets())
        except Exception:
            return len(llama_docs)

    def load(self) -> bool:
        """Load index from disk."""
        index_file = self.persist_dir / "index_store.json"
        if not index_file.exists():
            logger.warning(f"No index found at {self.persist_dir}")
            return False

        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.persist_dir)
            )
            self._index = load_index_from_storage(storage_context)
            logger.info("Graph index loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load graph index: {e}")
            return False

    def search(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        """Search graph for relevant triplets and nodes."""
        if self._index is None:
            logger.warning("Graph index not loaded")
            return []

        k = top_k or self.config.graph.similarity_top_k

        # Use retriever
        retriever = self._index.as_retriever(
            include_text=True,
            similarity_top_k=k,
        )

        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append(
                SearchResult(
                    content=node.get_content(),
                    score=node.score or 0.0,
                    source=RetrievalMethod.GRAPH,
                    metadata=node.metadata,
                )
            )

        return results

    def is_indexed(self) -> bool:
        """Check if index exists and is loaded."""
        if self._index is not None:
            return True
        # Check if persisted
        return (self.persist_dir / "index_store.json").exists()

    def get_triplets_sample(self, limit: int = 10) -> list[tuple[str, str, str]]:
        """Get sample of triplets from graph (for debugging)."""
        if self._index is None:
            return []

        try:
            graph_store = self._index.property_graph_store
            triplets = graph_store.get_triplets()
            return [
                (t[0].id, t[1].label, t[2].id)
                for t in triplets[:limit]
            ]
        except Exception as e:
            logger.error(f"Error getting triplets: {e}")
            return []

    def get_graph_data(self, limit: int = 100) -> dict:
        """Get graph data for visualization (nodes + edges)."""
        if self._index is None:
            return {"nodes": [], "edges": []}

        try:
            graph_store = self._index.property_graph_store
            triplets = graph_store.get_triplets()[:limit]

            # Collect unique nodes
            nodes_dict = {}
            edges = []

            for triplet in triplets:
                subject = triplet[0]  # EntityNode
                relation = triplet[1]  # Relation
                obj = triplet[2]  # EntityNode

                # Add nodes
                if subject.id not in nodes_dict:
                    nodes_dict[subject.id] = {
                        "id": subject.id,
                        "label": subject.label or subject.id,
                        "properties": getattr(subject, "properties", {}),
                    }

                if obj.id not in nodes_dict:
                    nodes_dict[obj.id] = {
                        "id": obj.id,
                        "label": obj.label or obj.id,
                        "properties": getattr(obj, "properties", {}),
                    }

                # Add edge
                edges.append({
                    "source": subject.id,
                    "target": obj.id,
                    "label": relation.label,
                    "properties": getattr(relation, "properties", {}),
                })

            return {
                "nodes": list(nodes_dict.values()),
                "edges": edges,
            }
        except Exception as e:
            logger.error(f"Error getting graph data: {e}")
            return {"nodes": [], "edges": []}
