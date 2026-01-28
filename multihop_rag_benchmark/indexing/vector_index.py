"""FAISS-based vector index for Vector RAG."""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from ..data.preprocessing import Chunk
from ..generation.llm_client import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk: Chunk
    score: float
    rank: int


class VectorIndex:
    """
    FAISS-based in-memory vector index.

    Implements the Vector RAG approach from the paper:
    - Chunks indexed by embeddings
    - Top-K semantic search
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        dimension: int = 1536,  # ada-002 dimension
    ):
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-cpu")

        self.embedding_client = embedding_client
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        self.chunks: List[Chunk] = []
        self._is_built = False

    def add_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> None:
        """Add chunks to the index."""
        if not chunks:
            return

        logger.info(f"Embedding {len(chunks)} chunks...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_client.embed(texts)

        # Normalize for cosine similarity
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)

        self.index.add(embeddings_np)
        self.chunks.extend(chunks)
        self._is_built = True

        logger.info(f"Index now contains {len(self.chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for most similar chunks.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult ordered by relevance
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call add_chunks first.")

        # Embed query
        query_embedding = self.embedding_client.embed_single(query)
        query_np = np.array([query_embedding], dtype=np.float32)

        import faiss
        faiss.normalize_L2(query_np)

        # Search
        scores, indices = self.index.search(query_np, min(top_k, len(self.chunks)))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank,
            ))

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[SearchResult]]:
        """Search for multiple queries at once."""
        if not self._is_built:
            raise RuntimeError("Index not built. Call add_chunks first.")

        # Embed all queries
        query_embeddings = self.embedding_client.embed(queries)
        queries_np = np.array(query_embeddings, dtype=np.float32)

        import faiss
        faiss.normalize_L2(queries_np)

        # Batch search
        scores, indices = self.index.search(queries_np, min(top_k, len(self.chunks)))

        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for rank, (score, idx) in enumerate(zip(q_scores, q_indices)):
                if idx < 0:
                    continue
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank,
                ))
            all_results.append(results)

        return all_results

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunks)

    def save(self, path: str) -> None:
        """Save index to disk."""
        import faiss
        import pickle

        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str) -> None:
        """Load index from disk."""
        import faiss
        import pickle

        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self._is_built = True
