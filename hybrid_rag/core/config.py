"""Configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import yaml
import httpx
import logging

logger = logging.getLogger(__name__)

# Cache for model info from API
_model_info_cache: dict[str, dict] = {}


def fetch_model_info(model: str, base_url: str, api_key: Optional[str]) -> Optional[dict]:
    """Fetch model info from API (OpenRouter/OpenAI compatible)."""
    cache_key = f"{base_url}:{model}"
    if cache_key in _model_info_cache:
        return _model_info_cache[cache_key]

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Try to get specific model info
        with httpx.Client(timeout=10.0) as client:
            # OpenRouter uses /api/v1/models, OpenAI uses /v1/models
            url = f"{base_url.rstrip('/')}/models/{model}"
            resp = client.get(url, headers=headers)

            if resp.status_code == 200:
                data = resp.json()
                _model_info_cache[cache_key] = data
                return data

            # If specific model not found, try listing all models
            url = f"{base_url.rstrip('/')}/models"
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                models_data = resp.json()
                models_list = models_data.get("data", [])
                for m in models_list:
                    if m.get("id") == model:
                        _model_info_cache[cache_key] = m
                        return m

    except Exception as e:
        logger.warning(f"Failed to fetch model info for {model}: {e}")

    return None


def get_model_max_tokens(model: str, base_url: str, api_key: Optional[str]) -> int:
    """Get max output tokens for a model from API."""
    info = fetch_model_info(model, base_url, api_key)

    if info:
        # OpenRouter: max_completion_tokens in top_provider
        top_provider = info.get("top_provider", {})
        if top_provider and top_provider.get("max_completion_tokens"):
            return top_provider["max_completion_tokens"]
        # Direct max_completion_tokens (some APIs)
        if info.get("max_completion_tokens"):
            return info["max_completion_tokens"]
        # Fallback to context_length (though this is input limit)
        if info.get("context_length"):
            # Use 1/4 of context as reasonable output limit
            return min(info["context_length"] // 4, 16384)

    # Default fallback
    return 4096


@dataclass
class LLMConfig:
    """LLM configuration."""
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: Optional[int] = None  # None = fetch from API

    def __post_init__(self):
        # Allow env override
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", self.base_url)

    def get_max_tokens(self) -> int:
        """Get max tokens, fetching from API if not set."""
        if self.max_tokens is not None:
            return self.max_tokens
        return get_model_max_tokens(self.model, self.base_url, self.api_key)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", self.base_url)


@dataclass
class VectorConfig:
    """Vector index configuration."""
    chunk_size: int = 1500  # tokens
    chunk_overlap: int = 300  # ~20% overlap
    top_k: int = 10


@dataclass
class GraphConfig:
    """Graph index configuration."""
    max_triplets_per_chunk: int = 10
    include_embeddings: bool = True
    similarity_top_k: int = 10


@dataclass
class GenerationConfig:
    """Generation configuration for RAG."""
    max_context_length: int = 15000  # characters, ~3 chunks of 1500 tokens


@dataclass
class Config:
    """Main configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Paths
    index_dir: Path = field(default_factory=lambda: Path("./indexes"))

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file or use defaults."""
    if path is None:
        return Config()

    with open(path) as f:
        data = yaml.safe_load(f)

    config = Config()

    if "llm" in data:
        config.llm = LLMConfig(**data["llm"])
    if "embedding" in data:
        config.embedding = EmbeddingConfig(**data["embedding"])
    if "vector" in data:
        config.vector = VectorConfig(**data["vector"])
    if "graph" in data:
        config.graph = GraphConfig(**data["graph"])
    if "generation" in data:
        config.generation = GenerationConfig(**data["generation"])
    if "index_dir" in data:
        config.index_dir = Path(data["index_dir"])
    if "api_host" in data:
        config.api_host = data["api_host"]
    if "api_port" in data:
        config.api_port = data["api_port"]

    return config
