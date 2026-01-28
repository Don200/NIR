"""Document preprocessing and chunking."""

from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class Document:
    """A document with content and metadata."""
    doc_id: str
    content: str
    title: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Chunk:
    """A chunk of a document."""
    chunk_id: str
    doc_id: str
    content: str
    start_idx: int
    end_idx: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using simple word-based heuristic.
    More accurate would be tiktoken, but this is faster.
    Roughly 1 token ≈ 0.75 words for English.
    """
    words = len(text.split())
    return int(words / 0.75)


def chunk_by_tokens(
    text: str,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
) -> List[tuple[str, int, int]]:
    """
    Split text into chunks of approximately chunk_size tokens.

    Returns:
        List of (chunk_text, start_char, end_char)
    """
    # Split into sentences first for cleaner boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_tokens = 0
    current_start = 0
    char_pos = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If single sentence exceeds chunk size, split it further
        if sentence_tokens > chunk_size:
            # Flush current chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, current_start, char_pos))
                current_chunk = []
                current_tokens = 0
                current_start = char_pos

            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_tokens = 0
            for word in words:
                word_tokens += 1
                word_chunk.append(word)
                if word_tokens >= chunk_size:
                    chunk_text = ' '.join(word_chunk)
                    chunks.append((chunk_text, current_start, char_pos + len(chunk_text)))
                    current_start = char_pos + len(chunk_text) + 1
                    word_chunk = []
                    word_tokens = 0
            if word_chunk:
                current_chunk = word_chunk
                current_tokens = word_tokens

        # Check if adding sentence exceeds limit
        elif current_tokens + sentence_tokens > chunk_size:
            # Save current chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, current_start, char_pos))

            # Start new chunk with overlap
            if chunk_overlap > 0 and current_chunk:
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = estimate_tokens(s)
                    if overlap_tokens + s_tokens > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            current_start = char_pos
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        char_pos += len(sentence) + 1  # +1 for space

    # Don't forget last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append((chunk_text, current_start, char_pos))

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk multiple documents.

    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of chunks
    """
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_by_tokens(doc.content, chunk_size, chunk_overlap)

        for idx, (chunk_text, start, end) in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}::chunk_{idx}",
                doc_id=doc.doc_id,
                content=chunk_text,
                start_idx=start,
                end_idx=end,
                metadata={
                    "title": doc.title,
                    "chunk_index": idx,
                    **doc.metadata,
                }
            )
            all_chunks.append(chunk)

    return all_chunks


def corpus_to_documents(corpus) -> List[Document]:
    """Convert MultiHopRAGCorpus list to Document list."""
    documents = []
    for item in corpus:
        doc = Document(
            doc_id=item.doc_id,
            content=item.content,
            title=item.title,
            metadata=item.metadata,
        )
        documents.append(doc)
    return documents
