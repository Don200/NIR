"""Load MultiHop-RAG dataset from HuggingFace."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class MultiHopRAGSample:
    """Single sample from MultiHop-RAG dataset."""
    query_id: str
    query: str
    answer: str
    query_type: str  # inference, comparison, temporal, null
    evidence_ids: List[str]  # document IDs containing evidence
    metadata: Dict[str, Any]


@dataclass
class MultiHopRAGCorpus:
    """Document corpus from MultiHop-RAG."""
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any]


def load_multihop_rag(
    cache_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> tuple[List[MultiHopRAGSample], List[MultiHopRAGCorpus]]:
    """
    Load MultiHop-RAG dataset from HuggingFace.

    Returns:
        Tuple of (samples, corpus)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    cache_dir_str = str(cache_dir) if cache_dir else None

    # Load QA samples (config: MultiHopRAG)
    print("Loading MultiHopRAG QA samples...")
    qa_dataset = load_dataset(
        "yixuantt/MultiHopRAG",
        "MultiHopRAG",  # Указываем конфиг явно
        cache_dir=cache_dir_str,
    )

    samples = []
    qa_data = qa_dataset["train"]  # Данные в train split
    for idx, item in enumerate(qa_data):
        if max_samples and idx >= max_samples:
            break

        sample = MultiHopRAGSample(
            query_id=str(item.get("id", idx)),
            query=item["query"],
            answer=item["answer"],
            query_type=item.get("question_type", "unknown"),
            evidence_ids=item.get("evidence_list", []),
            metadata={
                "source": item.get("source", ""),
                "raw": {k: v for k, v in item.items()
                        if k not in ["query", "answer", "question_type", "evidence_list"]}
            }
        )
        samples.append(sample)

    # Load corpus (config: corpus)
    print("Loading corpus documents...")
    corpus_dataset = load_dataset(
        "yixuantt/MultiHopRAG",
        "corpus",  # Указываем конфиг явно
        cache_dir=cache_dir_str,
    )

    corpus = []
    corpus_data = corpus_dataset["train"]  # Данные в train split
    for idx, item in enumerate(corpus_data):
        # Генерируем уникальный doc_id из url или индекса
        url = item.get("url", "")
        if url:
            # Используем хеш от url как id
            import hashlib
            doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
        else:
            doc_id = f"doc_{idx}"

        doc = MultiHopRAGCorpus(
            doc_id=doc_id,
            title=item.get("title", ""),
            content=item.get("body", ""),  # Контент в поле "body"
            metadata={
                "source": item.get("source", ""),
                "published_at": item.get("published_at", ""),
                "url": url,
                "category": item.get("category", ""),
                "author": item.get("author", ""),
            }
        )
        corpus.append(doc)

    print(f"Loaded {len(samples)} samples, {len(corpus)} documents")
    return samples, corpus


def load_multihop_rag_from_json(
    qa_path: Path,
    corpus_path: Path,
    max_samples: Optional[int] = None,
) -> tuple[List[MultiHopRAGSample], List[MultiHopRAGCorpus]]:
    """
    Load MultiHop-RAG from local JSON files.
    Alternative to HuggingFace loading.
    """
    # Load QA
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    samples = []
    for idx, item in enumerate(qa_data):
        if max_samples and idx >= max_samples:
            break

        sample = MultiHopRAGSample(
            query_id=str(item.get("id", idx)),
            query=item["query"],
            answer=item["answer"],
            query_type=item.get("question_type", "unknown"),
            evidence_ids=item.get("evidence_list", []),
            metadata={"raw": item}
        )
        samples.append(sample)

    # Load corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    corpus = []
    for item in corpus_data:
        doc = MultiHopRAGCorpus(
            doc_id=str(item.get("id", "")),
            title=item.get("title", ""),
            content=item.get("content", item.get("text", "")),
            metadata=item
        )
        corpus.append(doc)

    return samples, corpus


def get_query_type_distribution(samples: List[MultiHopRAGSample]) -> Dict[str, int]:
    """Get distribution of query types."""
    distribution = {}
    for sample in samples:
        qt = sample.query_type
        distribution[qt] = distribution.get(qt, 0) + 1
    return distribution
