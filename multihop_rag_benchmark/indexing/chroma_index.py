"""ChromaDB-based persistent vector index."""

from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import hashlib
import json

from ..data.preprocessing import Chunk
from ..generation.llm_client import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk: Chunk
    score: float
    rank: int


class ChromaVectorIndex:
    """
    ChromaDB-based persistent vector index.

    Automatically caches to disk - no need to rebuild on subsequent runs.
    Uses collection fingerprint to detect if corpus has changed.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        persist_dir: Path,
        collection_name: str = "multihop_rag",
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")

        self.embedding_client = embedding_client
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        self._collection = None
        self._chunks_map: dict = {}  # id -> Chunk
        self._fingerprint_file = self.persist_dir / "fingerprint.json"

    def _compute_fingerprint(self, chunks: List[Chunk]) -> str:
        """Compute fingerprint of chunks for cache invalidation."""
        # Hash based on chunk contents and count
        content = f"{len(chunks)}:" + "".join(c.chunk_id for c in chunks[:100])
        return hashlib.md5(content.encode()).hexdigest()

    def _load_fingerprint(self) -> Optional[str]:
        """Load stored fingerprint."""
        if self._fingerprint_file.exists():
            with open(self._fingerprint_file) as f:
                data = json.load(f)
                return data.get("fingerprint")
        return None

    def _save_fingerprint(self, fingerprint: str, chunk_count: int) -> None:
        """Save fingerprint to disk."""
        with open(self._fingerprint_file, "w") as f:
            json.dump({"fingerprint": fingerprint, "chunk_count": chunk_count}, f)

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def is_cached(self, chunks: List[Chunk]) -> bool:
        """Check if index is already built for these chunks."""
        current_fingerprint = self._compute_fingerprint(chunks)
        stored_fingerprint = self._load_fingerprint()

        if stored_fingerprint == current_fingerprint:
            collection = self._get_or_create_collection()
            if collection.count() > 0:
                logger.info(f"Found cached index with {collection.count()} chunks")
                return True
        return False

    def add_chunks(self, chunks: List[Chunk], force_rebuild: bool = False) -> None:
        """
        Add chunks to the index.

        If cache exists and fingerprint matches, skips embedding.
        """
        if not chunks:
            return

        fingerprint = self._compute_fingerprint(chunks)

        # Check cache
        if not force_rebuild and self.is_cached(chunks):
            logger.info("Using cached index, skipping embedding")
            self._load_chunks_map(chunks)
            return

        # Clear existing collection
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Embedding {len(chunks)} chunks...")

        # Process in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [chunk.chunk_id for chunk in batch]
            texts = [chunk.content for chunk in batch]
            metadatas = [
                {
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "title": chunk.metadata.get("title", ""),
                }
                for chunk in batch
            ]

            # Get embeddings
            embeddings = self.embedding_client.embed(texts)

            # Add to collection
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            logger.info(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        # Build chunks map
        self._load_chunks_map(chunks)

        # Save fingerprint
        self._save_fingerprint(fingerprint, len(chunks))
        logger.info(f"Index built and cached with {len(chunks)} chunks")

    def _load_chunks_map(self, chunks: List[Chunk]) -> None:
        """Build chunk_id -> Chunk mapping."""
        self._chunks_map = {chunk.chunk_id: chunk for chunk in chunks}

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search for most similar chunks."""
        collection = self._get_or_create_collection()

        if collection.count() == 0:
            raise RuntimeError("Index is empty. Call add_chunks first.")

        # Embed query
        query_embedding = self.embedding_client.embed_single(query)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for rank, (chunk_id, doc, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Reconstruct chunk or use from map
                if chunk_id in self._chunks_map:
                    chunk = self._chunks_map[chunk_id]
                else:
                    # Fallback: reconstruct from stored data
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        doc_id=metadata.get("doc_id", ""),
                        content=doc,
                        start_idx=0,
                        end_idx=len(doc),
                        metadata=metadata,
                    )

                # Convert distance to similarity score (ChromaDB returns distance for cosine)
                score = 1 - distance

                search_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                ))

        return search_results

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        collection = self._get_or_create_collection()
        return collection.count()

    def clear(self) -> None:
        """Clear the index and cache."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        if self._fingerprint_file.exists():
            self._fingerprint_file.unlink()
        self._collection = None
        self._chunks_map = {}
        logger.info("Index cleared")
