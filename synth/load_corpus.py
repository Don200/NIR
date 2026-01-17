from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader

from synth.utils import normalize_text, save_jsonl, load_jsonl, stable_doc_id


@dataclass
class DocumentRecord:
    page_content: str
    metadata: dict


CACHE_NAME = "documents.jsonl"


def _build_metadata(source_path: str, page: int | None) -> dict:
    doc_id = stable_doc_id(source_path)
    title = Path(source_path).stem
    metadata = {
        "doc_id": doc_id,
        "source_path": source_path,
        "title": title,
    }
    if page is not None:
        metadata["page"] = page
    return metadata


def load_corpus(raw_dir: Path, processed_dir: Path, *, use_cache: bool = True) -> List[DocumentRecord]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / CACHE_NAME
    if use_cache and cache_path.exists():
        cached = load_jsonl(cache_path)
        return [DocumentRecord(**row) for row in cached]

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw PDF directory not found: {raw_dir}")

    loader = PyPDFDirectoryLoader(str(raw_dir))
    docs = loader.load()
    records: List[DocumentRecord] = []
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        page = doc.metadata.get("page")
        metadata = _build_metadata(source_path, page)
        records.append(DocumentRecord(page_content=normalize_text(doc.page_content), metadata=metadata))

    save_jsonl(cache_path, [record.__dict__ for record in records])
    return records
