from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from synth.load_corpus import DocumentRecord


@dataclass
class ChunkRecord:
    page_content: str
    metadata: dict


def chunk_documents(
    documents: List[DocumentRecord],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks: List[ChunkRecord] = []
    for doc in documents:
        pieces = splitter.split_text(doc.page_content)
        for idx, text in enumerate(pieces):
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = idx
            chunks.append(ChunkRecord(page_content=text, metadata=metadata))
    return chunks
