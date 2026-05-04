import logging
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import chromadb
from chromadb.config import Settings

from langfuse.openai import OpenAI
from langfuse import observe

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.utils import get_tokenizer

from ..core.config import Config
from ..core.models import Document, Chunk, SearchResult, RetrievalMethod

logger = logging.getLogger(__name__)


_ATOM_TOKENIZER = get_tokenizer()

_DOLLAR_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_BRACKET_BLOCK_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_BEGIN_END_RE = re.compile(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", re.DOTALL)


def _find_table_blocks(text: str) -> list[tuple[int, int]]:
    """Runs of 2+ consecutive lines containing '|' (markdown tables)."""
    lines = text.split("\n")
    pos = [0]
    for line in lines:
        pos.append(pos[-1] + len(line) + 1)
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if "|" in lines[i]:
            j = i + 1
            while j < len(lines) and "|" in lines[j]:
                j += 1
            if j - i >= 2:
                blocks.append((pos[i], pos[j - 1] + len(lines[j - 1])))
            i = j
        else:
            i += 1
    return blocks


def _find_atoms(text: str) -> list[tuple[int, int]]:
    """Non-overlapping intervals that must not be split: $$, \\[\\], envs, tables."""
    candidates: list[tuple[int, int]] = []
    for pat in (_DOLLAR_BLOCK_RE, _BRACKET_BLOCK_RE, _BEGIN_END_RE):
        candidates.extend((m.start(), m.end()) for m in pat.finditer(text))
    candidates.extend(_find_table_blocks(text))
    candidates.sort()
    resolved: list[tuple[int, int]] = []
    for s, e in candidates:
        if resolved and s <= resolved[-1][1]:
            resolved[-1] = (resolved[-1][0], max(resolved[-1][1], e))
        else:
            resolved.append((s, e))
    return resolved


def _atom_aware_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks ≤ chunk_size tokens, never breaking inside atoms.

    Atoms = ``$$ ... $$``, ``\\[ ... \\]``, ``\\begin{env} ... \\end{env}``,
    and runs of 2+ consecutive markdown table rows. Atoms larger than
    chunk_size become their own oversize chunks (intact, KaTeX-renderable).
    """
    atoms = _find_atoms(text)
    segments: list[tuple[str, str]] = []
    pos = 0
    for s, e in atoms:
        if s > pos:
            segments.append(("text", text[pos:s]))
        segments.append(("atom", text[s:e]))
        pos = e
    if pos < len(text):
        segments.append(("text", text[pos:]))

    sub_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces: list[str] = []
    for kind, body in segments:
        if not body.strip():
            continue
        if kind == "text":
            doc = LlamaDocument(text=body)
            for n in sub_splitter.get_nodes_from_documents([doc]):
                pieces.append(n.get_content())
        else:
            pieces.append(body)

    chunks: list[str] = []
    current: list[str] = []
    cur_tokens = 0
    for p in pieces:
        pt = len(_ATOM_TOKENIZER(p))
        if current and cur_tokens + pt > chunk_size:
            chunks.append("\n\n".join(current))
            current = [p]
            cur_tokens = pt
        else:
            current.append(p)
            cur_tokens += pt
    if current:
        chunks.append("\n\n".join(current))
    return chunks


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

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

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

    @observe()
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from OpenAI API."""
        response = self.embedding_client.embeddings.create(
            model=self.config.embedding.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        """Split document into chunks.

        Strategy:
        1. If content looks like markdown (has headers), use MarkdownNodeParser
           to split by header sections (preserves header_path metadata).
        2. For each section, use atom-aware splitting that keeps $$...$$,
           \\[...\\], \\begin/\\end environments, and markdown tables intact.
        """
        is_markdown = any(
            line.strip().startswith("#")
            for line in doc.content.split("\n")[:50]
        )

        base_metadata = {"title": doc.title or "", **doc.metadata}

        if is_markdown:
            llama_doc = LlamaDocument(
                text=doc.content, doc_id=doc.id, metadata=base_metadata,
            )
            md_nodes = MarkdownNodeParser().get_nodes_from_documents([llama_doc])
            logger.debug(f"MarkdownParser produced {len(md_nodes)} nodes")
        else:
            md_nodes = [LlamaDocument(
                text=doc.content, doc_id=doc.id, metadata=base_metadata,
            )]

        min_chars = self.config.vector.min_chunk_chars
        chunks = []
        dropped = 0
        for md_node in md_nodes:
            sub_chunks = _atom_aware_split(
                md_node.get_content(),
                chunk_size=self.config.vector.chunk_size,
                chunk_overlap=self.config.vector.chunk_overlap,
            )
            node_meta_base = dict(md_node.metadata) if md_node.metadata else {}
            for key in ("Header_1", "Header_2", "Header_3"):
                if key in node_meta_base:
                    node_meta_base[key.lower()] = node_meta_base[key]

            for sub_text in sub_chunks:
                if min_chars and len(sub_text.strip()) < min_chars:
                    dropped += 1
                    continue

                idx = len(chunks)
                node_metadata = dict(node_meta_base)
                node_metadata["chunk_index"] = idx

                chunks.append(
                    Chunk(
                        id=f"{doc.id}::chunk_{idx}",
                        doc_id=doc.id,
                        content=sub_text,
                        metadata=node_metadata,
                    )
                )

        if dropped:
            logger.info(
                f"Dropped {dropped} chunks shorter than {min_chars} chars from {doc.id}"
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

    @observe()
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
                score = 1 - dist
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
