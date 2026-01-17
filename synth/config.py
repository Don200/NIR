from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class Config:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    chunk_size: int
    chunk_overlap: int
    kg_path: Path
    testset_jsonl: Path
    testset_csv: Path
    report_path: Path
    seed: int
    graph_top_k: int
    graph_min_similarity: float
    use_embeddings: bool
    use_tfidf_fallback: bool
    embeddings_provider: str
    embeddings_model: str
    embeddings_api_base: Optional[str]
    embeddings_api_key_env: Optional[str]
    enrichment_mode: str
    claims_model: Optional[str]
    claims_api_base: Optional[str]
    claims_api_key_env: Optional[str]
    testset_size: int
    distribution: Dict[str, float]
    max_regen_tries: int
    llm_model: str
    llm_api_base: Optional[str]
    llm_api_key_env: Optional[str]
    leakage_max_ratio: float
    min_context_chars: int



def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    data_paths = data["data"]
    artifacts_dir = Path(data_paths["artifacts_dir"])

    return Config(
        raw_dir=Path(data_paths["raw_dir"]),
        processed_dir=Path(data_paths["processed_dir"]),
        artifacts_dir=artifacts_dir,
        chunk_size=int(data["chunking"]["chunk_size"]),
        chunk_overlap=int(data["chunking"]["chunk_overlap"]),
        kg_path=artifacts_dir / data["outputs"]["kg_path"],
        testset_jsonl=artifacts_dir / data["outputs"]["testset_jsonl"],
        testset_csv=artifacts_dir / data["outputs"]["testset_csv"],
        report_path=artifacts_dir / data["outputs"]["report_path"],
        seed=int(data.get("seed", 42)),
        graph_top_k=int(data["graph"]["top_k"]),
        graph_min_similarity=float(data["graph"]["min_similarity"]),
        use_embeddings=bool(data["graph"].get("use_embeddings", True)),
        use_tfidf_fallback=bool(data["graph"].get("use_tfidf_fallback", True)),
        embeddings_provider=str(data["graph"]["embeddings"]["provider"]),
        embeddings_model=str(data["graph"]["embeddings"]["model"]),
        embeddings_api_base=data["graph"]["embeddings"].get("api_base"),
        embeddings_api_key_env=data["graph"]["embeddings"].get("api_key_env"),
        enrichment_mode=str(data["enrichment"]["mode"]),
        claims_model=data["enrichment"].get("claims", {}).get("model"),
        claims_api_base=data["enrichment"].get("claims", {}).get("api_base"),
        claims_api_key_env=data["enrichment"].get("claims", {}).get("api_key_env"),
        testset_size=int(data["testset"]["size"]),
        distribution=dict(data["testset"]["distribution"]),
        max_regen_tries=int(data["testset"]["max_regen_tries"]),
        llm_model=str(data["testset"]["llm"]["model"]),
        llm_api_base=data["testset"]["llm"].get("api_base"),
        llm_api_key_env=data["testset"]["llm"].get("api_key_env"),
        leakage_max_ratio=float(data["validators"]["leakage_max_ratio"]),
        min_context_chars=int(data["validators"]["min_context_chars"]),
    )
