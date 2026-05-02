"""Resolve a document filename (from index metadata) to a human-readable title."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SourceTitleResolver:
    """Maps a document filename to a human-readable title.

    Reads a JSON file shaped like ``latin_squares.json`` — a list of
    ``{"title": ..., "citation": ..., "pdfs": [{"path": ".../<stem>.pdf"}, ...]}``
    entries — and resolves a query like ``evatutin_dls_heur_spectra_method_2.md``
    or ``evatutin_dls_heur_spectra_method_2`` to the matching ``title``.
    """

    def __init__(self, json_path: Optional[Path]):
        self._mapping: dict[str, str] = {}
        if json_path is None:
            return
        json_path = Path(json_path)
        if not json_path.exists():
            logger.warning(
                f"Source titles JSON not found at {json_path}; skipping enrichment"
            )
            return
        try:
            with open(json_path, encoding="utf-8") as f:
                items = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load source titles from {json_path}: {e}")
            return

        for item in items:
            title = item.get("title") or item.get("citation") or ""
            if not title:
                continue
            for pdf in item.get("pdfs", []) or []:
                path = pdf.get("path") or pdf.get("url") or ""
                if not path:
                    continue
                stem = Path(path).stem
                if stem:
                    self._mapping[stem] = title

        logger.info(
            f"Loaded {len(self._mapping)} source title mappings from {json_path}"
        )

    def lookup(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return self._mapping.get(Path(name).stem)

    def __bool__(self) -> bool:
        return bool(self._mapping)
