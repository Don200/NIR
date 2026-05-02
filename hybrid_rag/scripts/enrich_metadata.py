"""Enrich an existing index with human-readable source metadata.

Adds fields like ``display_title``, ``authors``, ``citation``, ``topic`` to
every chunk in the Chroma vector collection and every node / ref_doc entry
in the LlamaIndex graph docstore. Does NOT re-embed or re-extract triplets —
only mutates metadata in place.

The metadata JSON is expected to look like ``data/documents_pdf/latin_squares.json``:

    [
      {
        "title": "...",
        "citation": "...",
        "authors": ["...", "..."],
        "topic": "...",
        "pdfs": [{"path": "output/pdfs/<stem>.pdf", ...}, ...]
      },
      ...
    ]

Lookups join an index node's ``title``/``doc_title`` (the PDF basename, e.g.
``evatutin_dls_heur_spectra_method_2.md``) to a JSON entry by ``Path(...).stem``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


logger = logging.getLogger(__name__)
COLLECTION_NAME = "hybrid_rag_vectors"
ENRICHED_KEYS = ("display_title", "authors", "citation", "topic")


def build_mapping(json_path: Path) -> dict[str, dict[str, str]]:
    """Build {pdf_basename_stem -> enrichment_dict} from a side metadata JSON."""
    with open(json_path, encoding="utf-8") as f:
        items = json.load(f)

    mapping: dict[str, dict[str, str]] = {}
    for item in items:
        title = item.get("title") or ""
        citation = item.get("citation") or ""
        topic = item.get("topic") or ""
        authors_raw = item.get("authors") or []
        authors = (
            ", ".join(authors_raw)
            if isinstance(authors_raw, list)
            else str(authors_raw)
        )

        enrichment: dict[str, str] = {}
        if title:
            enrichment["display_title"] = title
        if authors:
            enrichment["authors"] = authors
        if citation:
            enrichment["citation"] = citation
        if topic:
            enrichment["topic"] = topic
        if not enrichment:
            continue

        for pdf in item.get("pdfs", []) or []:
            path = pdf.get("path") or pdf.get("url") or ""
            if not path:
                continue
            stem = Path(path).stem
            if stem:
                mapping[stem] = enrichment

    return mapping


def _stem(value: str) -> str:
    return Path(value).stem if value else ""


def _resolve(meta: dict[str, Any], mapping: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """Pick enrichment dict by either node title/doc_title basename."""
    for key in ("doc_title", "title"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            extra = mapping.get(_stem(v))
            if extra:
                return extra
    return None


def _enrich_in_place(meta: dict[str, Any], mapping: dict[str, dict[str, str]]) -> bool:
    extra = _resolve(meta, mapping)
    if not extra:
        return False
    changed = False
    for k, v in extra.items():
        if meta.get(k) != v:
            meta[k] = v
            changed = True
    return changed


def update_chroma(persist_dir: Path, mapping: dict[str, dict[str, str]], dry_run: bool) -> int:
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    coll = client.get_collection(COLLECTION_NAME)
    data = coll.get(include=["metadatas"])
    ids = data["ids"] or []
    metadatas = data["metadatas"] or []

    new_ids: list[str] = []
    new_metas: list[dict[str, Any]] = []
    for cid, meta in zip(ids, metadatas):
        meta = dict(meta or {})
        if _enrich_in_place(meta, mapping):
            new_ids.append(cid)
            new_metas.append(meta)

    if new_ids and not dry_run:
        coll.update(ids=new_ids, metadatas=new_metas)
    return len(new_ids)


def update_graph_docstore(persist_dir: Path, mapping: dict[str, dict[str, str]], dry_run: bool) -> int:
    docstore_path = persist_dir / "docstore.json"
    if not docstore_path.exists():
        logger.warning(f"No docstore.json at {persist_dir}")
        return 0

    with open(docstore_path, encoding="utf-8") as f:
        ds = json.load(f)

    updated = 0

    for node in (ds.get("docstore/data") or {}).values():
        nd = node.get("__data__", {}) if isinstance(node, dict) else {}
        meta = nd.get("metadata")
        if isinstance(meta, dict) and _enrich_in_place(meta, mapping):
            updated += 1
        for rel in (nd.get("relationships") or {}).values():
            if isinstance(rel, dict) and isinstance(rel.get("metadata"), dict):
                _enrich_in_place(rel["metadata"], mapping)

    for v in (ds.get("docstore/ref_doc_info") or {}).values():
        if isinstance(v, dict) and isinstance(v.get("metadata"), dict):
            _enrich_in_place(v["metadata"], mapping)

    if updated and not dry_run:
        backup = docstore_path.with_suffix(".json.bak")
        if not backup.exists():
            backup.write_text(
                json.dumps(ds, ensure_ascii=False),
                encoding="utf-8",
            )
        with open(docstore_path, "w", encoding="utf-8") as f:
            json.dump(ds, f, ensure_ascii=False)

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.metadata_json.exists():
        logger.error(f"Metadata JSON not found: {args.metadata_json}")
        return 1
    if not args.index_dir.exists():
        logger.error(f"Index dir not found: {args.index_dir}")
        return 1

    mapping = build_mapping(args.metadata_json)
    logger.info(f"Loaded {len(mapping)} basename -> metadata mappings")

    if not args.skip_vector:
        vector_dir = args.index_dir / "vector"
        if vector_dir.exists():
            n = update_chroma(vector_dir, mapping, args.dry_run)
            logger.info(
                f"Chroma: {'would enrich' if args.dry_run else 'enriched'} "
                f"{n} chunks"
            )
        else:
            logger.warning(f"Vector dir not found: {vector_dir}")

    if not args.skip_graph:
        graph_dir = args.index_dir / "graph"
        if graph_dir.exists():
            n = update_graph_docstore(graph_dir, mapping, args.dry_run)
            logger.info(
                f"Graph: {'would enrich' if args.dry_run else 'enriched'} "
                f"{n} doc nodes"
            )
        else:
            logger.warning(f"Graph dir not found: {graph_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
