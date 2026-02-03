"""ChromaDB vector index for semantic search."""

import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)

from ..core.config import Config
from ..core.models import Document, Chunk, SearchResult, RetrievalMethod

logger = logging.getLogger(__name__)


class VectorIndex:
    """ChromaDB-based vector index."""

    COLLECTION_NAME = "hybrid_rag_vectors"

    def __init__(
        self,
        config: Config,
        persist_dir: Optional[Path] = None,
    ):
        self.config = config
        self.persist_dir = persist_dir or config.index_dir / "vector"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # OpenAI embedding client
        self.embedding_client = OpenAI(
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url,
        )

        self._collection: Optional[chromadb.Collection] = None

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from OpenAI API."""
        response = self.embedding_client.embeddings.create(
            model=self.config.embedding.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        """Split document into chunks using LlamaIndex parsers.

        Strategy:
        1. If content looks like markdown (has headers), use MarkdownNodeParser
        2. Then apply SentenceSplitter to handle long sections
        """
        # Create LlamaIndex document
        llama_doc = LlamaDocument(
            text=doc.content,
            doc_id=doc.id,
            metadata={"title": doc.title or "", **doc.metadata},
        )

        # Check if content is markdown (has headers)
        is_markdown = any(
            line.strip().startswith("#")
            for line in doc.content.split("\n")[:50]
        )

        # Step 1: Initial parsing
        if is_markdown:
            # Parse by markdown structure first
            md_parser = MarkdownNodeParser()
            nodes = md_parser.get_nodes_from_documents([llama_doc])
            logger.debug(f"MarkdownParser produced {len(nodes)} nodes")
        else:
            nodes = [llama_doc]

        # Step 2: Apply sentence splitter to handle long sections
        sentence_splitter = SentenceSplitter(
            chunk_size=self.config.vector.chunk_size,
            chunk_overlap=self.config.vector.chunk_overlap,
        )

        final_nodes = sentence_splitter.get_nodes_from_documents(nodes)
        logger.debug(f"SentenceSplitter produced {len(final_nodes)} final nodes")

        # Convert to our Chunk format
        chunks = []
        for idx, node in enumerate(final_nodes):
            chunk_id = f"{doc.id}::chunk_{idx}"

            # Extract metadata
            node_metadata = dict(node.metadata) if node.metadata else {}
            node_metadata["chunk_index"] = idx

            # Get header info if available (from markdown parsing)
            if hasattr(node, "metadata"):
                for key in ["Header_1", "Header_2", "Header_3"]:
                    if key in node.metadata:
                        node_metadata[key.lower()] = node.metadata[key]

            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc.id,
                    content=node.get_content(),
                    metadata=node_metadata,
                )
            )

        return chunks

    def index_documents(self, documents: list[Document]) -> int:
        """Index documents into ChromaDB."""
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self._chunk_document(doc)
            for chunk in chunks:
                chunk.metadata["doc_title"] = doc.title or ""
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks to index")
            return 0

        logger.info(f"Indexing {len(all_chunks)} chunks from {len(documents)} documents")

        # Process in batches
        batch_size = 100
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing batches", total=total_batches):
            batch = all_chunks[i : i + batch_size]

            texts = [c.content for c in batch]
            ids = [c.id for c in batch]
            metadatas = [c.metadata for c in batch]

            embeddings = self._get_embeddings(texts)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        return len(all_chunks)

    def search(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        """Search for similar chunks."""
        k = top_k or self.config.vector.top_k

        query_embedding = self._get_embeddings([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Convert distance to similarity score
                score = 1 - dist  # cosine distance to similarity
                search_results.append(
                    SearchResult(
                        content=doc,
                        score=score,
                        source=RetrievalMethod.VECTOR,
                        metadata=meta,
                    )
                )

        return search_results

    def is_indexed(self) -> bool:
        """Check if index has data."""
        return self.collection.count() > 0

    def clear(self) -> None:
        """Clear all indexed data."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self._collection = None
