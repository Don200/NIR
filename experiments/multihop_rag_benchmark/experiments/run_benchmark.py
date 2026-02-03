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
from ..indexing.kg_index import KnowledgeGraphIndex
from ..indexing.community_index import CommunityGraphIndex
from ..retrieval.vector_retriever import VectorRetriever
from ..retrieval.kg_retriever import KGRetriever
from ..retrieval.community_retriever import CommunityLocalRetriever, CommunityGlobalRetriever
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


def build_kg_index(
    config: BenchmarkConfig,
    chunks,
    llm_client,
    embedding_client,
    use_cache: bool = True,
    force_rebuild: bool = False,
):
    """Build knowledge graph index with optional caching."""
    logger.info("Building knowledge graph index...")
    index = KnowledgeGraphIndex(
        llm_client=llm_client,
        embedding_client=embedding_client,
        max_triplets_per_chunk=config.kg_rag.max_triplets_per_chunk,
    )

    if use_cache:
        cache_dir = config.cache_dir / "kg_index"
        index.build_from_chunks_cached(
            chunks,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
        )
    else:
        index.build_from_chunks(chunks)

    logger.info(f"KG index ready: {index.stats}")
    return index


def build_community_index(config: BenchmarkConfig, documents):
    """Build community-based GraphRAG index."""
    logger.info("Building community GraphRAG index...")
    index = CommunityGraphIndex(
        root_dir=config.community_rag.graphrag_root,
        llm_model=config.llm.model,
        llm_api_key=config.llm.api_key,
        llm_api_base=config.llm.api_base,
        embedding_model=config.embedding.model,
        embedding_api_key=config.embedding.api_key,
        embedding_api_base=config.embedding.api_base,
    )
    index.index_documents(documents)
    logger.info("Community index built")
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
    kg_index = None
    community_index = None

    needs_vector = any(m in methods for m in [
        "vector_rag", "hybrid_selection", "hybrid_integration"
    ])
    needs_kg = any(m in methods for m in [
        "kg_rag", "hybrid_selection", "hybrid_integration"
    ])
    needs_community = any(m in methods for m in [
        "community_rag_local", "community_rag_global"
    ])

    if needs_vector:
        vector_index = build_vector_index(
            config, chunks, embedding_client,
            use_cache=use_cache, force_rebuild=force_rebuild
        )

    if needs_kg:
        kg_index = build_kg_index(
            config, chunks, llm_client, embedding_client,
            use_cache=use_cache, force_rebuild=force_rebuild
        )

    if needs_community:
        community_index = build_community_index(config, documents)

    # Create retrievers
    retrievers = {}

    if "vector_rag" in methods and vector_index:
        retrievers["vector_rag"] = VectorRetriever(
            vector_index, top_k=config.vector_rag.top_k
        )

    if "kg_rag" in methods and kg_index:
        retrievers["kg_rag"] = KGRetriever(
            kg_index,
            max_hops=config.kg_rag.max_hops,
            include_text=config.kg_rag.include_original_text,
        )

    if "community_rag_local" in methods and community_index:
        retrievers["community_rag_local"] = CommunityLocalRetriever(
            community_index,
            community_level=config.community_rag.community_level,
        )

    if "community_rag_global" in methods and community_index:
        retrievers["community_rag_global"] = CommunityGlobalRetriever(
            community_index,
            community_level=1,  # Higher level for global
        )

    if "hybrid_selection" in methods and vector_index and kg_index:
        retrievers["hybrid_selection"] = SelectionRetriever(
            vector_retriever=VectorRetriever(vector_index, top_k=config.vector_rag.top_k),
            graph_retriever=KGRetriever(kg_index, max_hops=config.kg_rag.max_hops),
            llm_client=llm_client,
        )

    if "hybrid_integration" in methods and vector_index and kg_index:
        retrievers["hybrid_integration"] = IntegrationRetriever(
            vector_retriever=VectorRetriever(vector_index, top_k=config.vector_rag.top_k),
            graph_retriever=KGRetriever(kg_index, max_hops=config.kg_rag.max_hops),
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
