from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from synth.utils import normalize_text, save_jsonl, append_jsonl, load_jsonl, stable_doc_id


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

    records: List[DocumentRecord] = []
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError("docling is required for PDF parsing. Install it first.") from exc

    ocr_options = TesseractCliOcrOptions(lang=["rus+eng"])
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=True,
        ocr_options=ocr_options,
        artifacts_path="./models",
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    pdf_paths = sorted(raw_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    if cache_path.exists():
        cache_path.unlink()

    for pdf_path in pdf_paths:
        result = converter.convert(str(pdf_path))
        document = result.document
        text = _document_text(document)
        normalized = normalize_text(text)
        if not normalized:
            continue
        metadata = _build_metadata(str(pdf_path), None)
        record = DocumentRecord(page_content=normalized, metadata=metadata)
        records.append(record)
        append_jsonl(cache_path, record.__dict__)

    return records


def _document_text(document: object) -> str:
    try:
        return document.export_to_text()
    except AttributeError:
        return document.export_to_markdown()
