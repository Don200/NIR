#!/usr/bin/env python3
"""
Main benchmark runner for RAG vs GraphRAG evaluation on MultiHop-RAG.

Usage:
    python -m multihop_rag_benchmark.experiments.run_benchmark --config configs/default.yaml

Or with specific methods:
    python -m multihop_rag_benchmark.experiments.run_benchmark \
        --config configs/default.yaml \
        --methods vector_rag kg_rag
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from ..config import BenchmarkConfig, load_config
from ..data.loader import load_multihop_rag, get_query_type_distribution
from ..data.preprocessing import corpus_to_documents, chunk_documents
from ..generation.llm_client import LLMClient, EmbeddingClient
from ..indexing.vector_index import VectorIndex
from ..indexing.chroma_index import ChromaVectorIndex
from ..indexing.kg_index import GraphRAGIndex
from ..retrieval.vector_retriever import VectorRetriever
from ..retrieval.graphrag_retriever import GraphRAGLocalRetriever, GraphRAGGlobalRetriever
from ..retrieval.hybrid_retriever import SelectionRetriever, IntegrationRetriever
from ..evaluation.evaluator import BenchmarkEvaluator, EvaluationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_clients(config: BenchmarkConfig):
    """Initialize LLM and embedding clients."""
    llm_client = LLMClient(
        api_key=config.llm.api_key,
        model=config.llm.model,
        api_base=config.llm.api_base,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    embedding_client = EmbeddingClient(
        api_key=config.embedding.api_key,
        model=config.embedding.model,
        api_base=config.embedding.api_base,
        batch_size=config.embedding.batch_size,
    )

    return llm_client, embedding_client


def build_vector_index(
    config: BenchmarkConfig,
    chunks,
    embedding_client,
    use_cache: bool = True,
    force_rebuild: bool = False,
):
    """Build vector index from chunks with optional caching."""
    if use_cache:
        logger.info("Building vector index with ChromaDB (persistent)...")
        persist_dir = config.cache_dir / "vector_index"
        index = ChromaVectorIndex(
            embedding_client=embedding_client,
            persist_dir=persist_dir,
        )
        index.add_chunks(chunks, force_rebuild=force_rebuild)
    else:
        logger.info("Building vector index with FAISS (in-memory)...")
        index = VectorIndex(embedding_client)
        index.add_chunks(chunks)

    logger.info(f"Vector index ready with {index.size} chunks")
    return index


def build_graphrag_index(
    config: BenchmarkConfig,
    chunks,
    llm_client,
    embedding_client,
    use_cache: bool = True,
    force_rebuild: bool = False,
):
    """Build GraphRAG v2 index with optional caching."""
    logger.info("Building GraphRAG v2 index...")
    index = GraphRAGIndex(
        llm_client=llm_client,
        embedding_client=embedding_client,
        max_paths_per_chunk=config.graphrag.max_paths_per_chunk,
        max_cluster_size=config.graphrag.max_cluster_size,
        num_workers=config.graphrag.num_workers,
    )

    if use_cache:
        cache_dir = config.cache_dir / "graphrag_index"
        index.build_cached(
            chunks,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
        )
    else:
        index.build(chunks)

    logger.info(f"GraphRAG index ready: {index.stats}")
    return index


def run_benchmark(
    config: BenchmarkConfig,
    methods: Optional[List[str]] = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> dict:
    """
    Run the full benchmark.

    Args:
        config: Benchmark configuration
        methods: List of methods to evaluate (None = all from config)

    Returns:
        Dict with all results and comparison
    """
    methods = methods or config.methods
    logger.info(f"Running benchmark with methods: {methods}")

    # Load data
    logger.info("Loading MultiHop-RAG dataset...")
    samples, corpus = load_multihop_rag(
        cache_dir=config.cache_dir,
        max_samples=config.max_samples,
    )
    logger.info(f"Loaded {len(samples)} samples, {len(corpus)} documents")

    # Show query type distribution
    distribution = get_query_type_distribution(samples)
    logger.info(f"Query type distribution: {distribution}")

    # Setup clients
    llm_client, embedding_client = setup_clients(config)

    # Prepare documents and chunks
    documents = corpus_to_documents(corpus)
    chunks = chunk_documents(
        documents,
        chunk_size=config.vector_rag.chunk_size,
        chunk_overlap=config.vector_rag.chunk_overlap,
    )
    logger.info(f"Created {len(chunks)} chunks")

    # Build indices based on required methods
    vector_index = None
    graphrag_index = None

    needs_vector = any(m in methods for m in [
        "vector_rag", "hybrid_selection", "hybrid_integration"
    ])
    needs_graphrag = any(m in methods for m in [
        "graphrag_local", "graphrag_global", "hybrid_selection", "hybrid_integration"
    ])

    if needs_vector:
        vector_index = build_vector_index(
            config, chunks, embedding_client,
            use_cache=use_cache, force_rebuild=force_rebuild
        )

    if needs_graphrag:
        graphrag_index = build_graphrag_index(
            config, chunks, llm_client, embedding_client,
            use_cache=use_cache, force_rebuild=force_rebuild
        )

    # Create retrievers
    retrievers = {}

    if "vector_rag" in methods and vector_index:
        retrievers["vector_rag"] = VectorRetriever(
            vector_index, top_k=config.vector_rag.top_k
        )

    if "graphrag_local" in methods and graphrag_index:
        retrievers["graphrag_local"] = GraphRAGLocalRetriever(
            graphrag_index,
            top_k=config.graphrag.local_search_top_k,
        )

    if "graphrag_global" in methods and graphrag_index:
        retrievers["graphrag_global"] = GraphRAGGlobalRetriever(
            graphrag_index,
        )

    if "hybrid_selection" in methods and vector_index and graphrag_index:
        retrievers["hybrid_selection"] = SelectionRetriever(
            vector_retriever=VectorRetriever(vector_index, top_k=config.vector_rag.top_k),
            graph_retriever=GraphRAGLocalRetriever(graphrag_index, top_k=config.graphrag.local_search_top_k),
            llm_client=llm_client,
        )

    if "hybrid_integration" in methods and vector_index and graphrag_index:
        retrievers["hybrid_integration"] = IntegrationRetriever(
            vector_retriever=VectorRetriever(vector_index, top_k=config.vector_rag.top_k),
            graph_retriever=GraphRAGLocalRetriever(graphrag_index, top_k=config.graphrag.local_search_top_k),
        )

    # Run evaluation
    evaluator = BenchmarkEvaluator(llm_client)
    results: List[EvaluationResult] = []

    for method_name, retriever in retrievers.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {method_name}")
        logger.info(f"{'='*50}")

        result = evaluator.evaluate_retriever(retriever, samples)
        results.append(result)

        logger.info(f"Overall accuracy: {result.overall_accuracy:.4f}")
        for qtype, metrics in result.accuracy_by_type.items():
            if qtype != "overall":
                logger.info(f"  {qtype}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

        # Save individual result
        result_path = config.output_dir / f"{method_name}_results.json"
        result.save(result_path)

        # Save predictions CSV for easy viewing
        csv_path = config.output_dir / f"{method_name}_predictions.csv"
        result.save_predictions_csv(csv_path)

        logger.info(f"Results saved to {result_path}")
        logger.info(f"Predictions CSV saved to {csv_path}")

    # Generate comparison
    comparison = evaluator.compare_methods(results)

    # Save comparison
    comparison_path = config.output_dir / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info(f"Comparison saved to {comparison_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*60)
    for method, metrics in comparison["overall"].items():
        logger.info(f"{method}: {metrics['accuracy']:.4f}")

    if "complementarity" in comparison:
        logger.info("\nComplementarity Analysis:")
        for pair, stats in comparison["complementarity"].items():
            logger.info(f"\n{pair}:")
            for key, value in stats.items():
                if "pct" in key:
                    logger.info(f"  {key}: {value:.1f}%")

    return {
        "results": [r.to_dict() for r in results],
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG vs GraphRAG benchmark on MultiHop-RAG"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to evaluate (default: all from config)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max samples from config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable index caching (use in-memory FAISS)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of cached indices",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.output_dir:
        config.output_dir = args.output_dir
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    run_benchmark(
        config,
        methods=args.methods,
        use_cache=not args.no_cache,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
