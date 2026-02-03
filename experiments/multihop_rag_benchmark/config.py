"""Configuration management for the benchmark."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class LLMConfig:
    """LLM configuration for generation."""
    api_key_env: str = "OPENAI_API_KEY"
    api_base: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(f"Environment variable {self.api_key_env} not set")
        return key


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    api_key_env: str = "OPENAI_API_KEY"
    api_base: Optional[str] = None
    model: str = "text-embedding-ada-002"
    batch_size: int = 100

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(f"Environment variable {self.api_key_env} not set")
        return key


@dataclass
class VectorRAGConfig:
    """Vector RAG configuration."""
    chunk_size: int = 256  # tokens, as in paper
    chunk_overlap: int = 50
    top_k: int = 10  # as in paper


@dataclass
class KGRAGConfig:
    """Knowledge Graph RAG configuration (LlamaIndex-based)."""
    max_triplets_per_chunk: int = 10
    include_original_text: bool = True
    max_hops: int = 2


@dataclass
class CommunityRAGConfig:
    """Community-based GraphRAG configuration (Microsoft GraphRAG)."""
    graphrag_root: Path = field(default_factory=lambda: Path("./graphrag_index"))
    community_level: int = 2
    local_search_top_k: int = 10
    global_search_top_k: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    normalize_answers: bool = True
    case_sensitive: bool = False


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    # Paths
    output_dir: Path = field(default_factory=lambda: Path("./benchmark_results"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))

    # Data
    dataset_name: str = "yixuantt/MultiHopRAG"
    max_samples: Optional[int] = None  # None = use all

    # Components
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_rag: VectorRAGConfig = field(default_factory=VectorRAGConfig)
    kg_rag: KGRAGConfig = field(default_factory=KGRAGConfig)
    community_rag: CommunityRAGConfig = field(default_factory=CommunityRAGConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment settings
    methods: list = field(default_factory=lambda: [
        "vector_rag",
        "kg_rag",
        "community_rag_local",
        "community_rag_global",
        "hybrid_selection",
        "hybrid_integration"
    ])

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> BenchmarkConfig:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = BenchmarkConfig()

    # Update paths
    if "output_dir" in data:
        config.output_dir = Path(data["output_dir"])
    if "cache_dir" in data:
        config.cache_dir = Path(data["cache_dir"])

    # Update dataset settings
    if "dataset" in data:
        config.dataset_name = data["dataset"].get("name", config.dataset_name)
        config.max_samples = data["dataset"].get("max_samples")

    # Update LLM config
    if "llm" in data:
        config.llm = LLMConfig(**data["llm"])

    # Update embedding config
    if "embedding" in data:
        config.embedding = EmbeddingConfig(**data["embedding"])

    # Update RAG configs
    if "vector_rag" in data:
        config.vector_rag = VectorRAGConfig(**data["vector_rag"])
    if "kg_rag" in data:
        config.kg_rag = KGRAGConfig(**data["kg_rag"])
    if "community_rag" in data:
        cfg = data["community_rag"]
        if "graphrag_root" in cfg:
            cfg["graphrag_root"] = Path(cfg["graphrag_root"])
        config.community_rag = CommunityRAGConfig(**cfg)

    # Update evaluation config
    if "evaluation" in data:
        config.evaluation = EvaluationConfig(**data["evaluation"])

    # Update methods
    if "methods" in data:
        config.methods = data["methods"]

    config.__post_init__()
    return config


def save_config(config: BenchmarkConfig, path: Path) -> None:
    """Save configuration to YAML file."""
    data = {
        "output_dir": str(config.output_dir),
        "cache_dir": str(config.cache_dir),
        "dataset": {
            "name": config.dataset_name,
            "max_samples": config.max_samples,
        },
        "llm": {
            "api_key_env": config.llm.api_key_env,
            "api_base": config.llm.api_base,
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
        "embedding": {
            "api_key_env": config.embedding.api_key_env,
            "api_base": config.embedding.api_base,
            "model": config.embedding.model,
            "batch_size": config.embedding.batch_size,
        },
        "vector_rag": {
            "chunk_size": config.vector_rag.chunk_size,
            "chunk_overlap": config.vector_rag.chunk_overlap,
            "top_k": config.vector_rag.top_k,
        },
        "kg_rag": {
            "max_triplets_per_chunk": config.kg_rag.max_triplets_per_chunk,
            "include_original_text": config.kg_rag.include_original_text,
            "max_hops": config.kg_rag.max_hops,
        },
        "community_rag": {
            "graphrag_root": str(config.community_rag.graphrag_root),
            "community_level": config.community_rag.community_level,
            "local_search_top_k": config.community_rag.local_search_top_k,
            "global_search_top_k": config.community_rag.global_search_top_k,
        },
        "evaluation": {
            "normalize_answers": config.evaluation.normalize_answers,
            "case_sensitive": config.evaluation.case_sensitive,
        },
        "methods": config.methods,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
