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

    # Load dataset
    dataset = load_dataset(
        "yixuantt/MultiHopRAG",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    # Parse QA samples from MultiHopRAG_QA split
    samples = []
    if "MultiHopRAG" in dataset:
        qa_data = dataset["MultiHopRAG"]
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

    # Parse corpus from corpus split
    corpus = []
    if "corpus" in dataset:
        corpus_data = dataset["corpus"]
        for item in corpus_data:
            doc = MultiHopRAGCorpus(
                doc_id=str(item.get("id", "")),
                title=item.get("title", ""),
                content=item.get("content", item.get("text", "")),
                metadata={
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                    "url": item.get("url", ""),
                }
            )
            corpus.append(doc)

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
