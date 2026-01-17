from __future__ import annotations

import argparse
import os
from pathlib import Path

from synth.chunking import chunk_documents
from synth.config import load_config
from synth.graph import build_kg, graph_stats, save_kg
from synth.load_corpus import load_corpus
from synth.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build knowledge graph from PDFs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to synth_config.yaml")
    parser.add_argument("--no-cache", action="store_true", help="Disable corpus cache.")
    return parser.parse_args()


def _require_env_var(name: str | None) -> str:
    if not name:
        raise RuntimeError("Missing api_key_env in config.")
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    documents = load_corpus(config.raw_dir, config.processed_dir, use_cache=not args.no_cache)
    chunks = chunk_documents(documents, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    embeddings_key = None
    if config.use_embeddings:
        embeddings_key = _require_env_var(config.embeddings_api_key_env)

    claims_key = None
    if config.enrichment_mode == "embeddings_plus_llm":
        claims_key = _require_env_var(config.claims_api_key_env)

    graph = build_kg(
        chunks,
        top_k=config.graph_top_k,
        min_similarity=config.graph_min_similarity,
        use_embeddings=config.use_embeddings,
        use_tfidf_fallback=config.use_tfidf_fallback,
        embeddings_config={
            "model": config.embeddings_model,
            "api_base": config.embeddings_api_base,
            "api_key": embeddings_key,
        },
        enrichment_mode=config.enrichment_mode,
        claims_config={
            "model": config.claims_model,
            "api_base": config.claims_api_base,
            "api_key": claims_key,
        },
    )

    stats = graph_stats(graph)
    save_kg(config.kg_path, graph)
    print("KG saved to", config.kg_path)
    print("Stats:", stats)


if __name__ == "__main__":
    main()
