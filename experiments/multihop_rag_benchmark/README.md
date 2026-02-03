# MultiHop-RAG Benchmark: RAG vs GraphRAG

Implementation of the systematic evaluation from the paper:
**"RAG vs. GraphRAG: A Systematic Evaluation and Key Insights"**
([arXiv:2502.11371](https://arxiv.org/abs/2502.11371))

## Overview

This benchmark compares different retrieval-augmented generation approaches:

| Method | Description |
|--------|-------------|
| **Vector RAG** | Classic semantic search over chunk embeddings |
| **KG-based GraphRAG** | Knowledge graph with triplet extraction (LlamaIndex-style) |
| **Community GraphRAG Local** | Microsoft GraphRAG with local search |
| **Community GraphRAG Global** | Microsoft GraphRAG with global search |
| **Hybrid Selection** | Classify query → route to best method |
| **Hybrid Integration** | Combine results from both methods |

## Installation

```bash
cd multihop_rag_benchmark
pip install -r requirements.txt

# Optional: for Community-based GraphRAG
pip install graphrag
```

## Configuration

Edit `experiments/configs/default.yaml`:

```yaml
llm:
  api_key_env: "OPENAI_API_KEY"
  api_base: "https://your-api.com/v1"  # for OpenAI-compatible API
  model: "gpt-4o-mini"

embedding:
  api_key_env: "OPENAI_API_KEY"
  api_base: null  # or your embedding API
  model: "text-embedding-ada-002"
```

## Usage

### Full Benchmark

```bash
# Set API key
export OPENAI_API_KEY="your-key"

# Run all methods
python -m multihop_rag_benchmark.experiments.run_benchmark \
    --config multihop_rag_benchmark/experiments/configs/default.yaml
```

### Specific Methods

```bash
# Run only Vector RAG and KG RAG
python -m multihop_rag_benchmark.experiments.run_benchmark \
    --config multihop_rag_benchmark/experiments/configs/default.yaml \
    --methods vector_rag kg_rag
```

### Limited Samples (for testing)

```bash
python -m multihop_rag_benchmark.experiments.run_benchmark \
    --config multihop_rag_benchmark/experiments/configs/default.yaml \
    --max-samples 100
```

## Output

Results are saved to `benchmark_results/`:

```
benchmark_results/
├── vector_rag_results.json
├── kg_rag_results.json
├── hybrid_integration_results.json
└── comparison.json
```

### Metrics

- **Accuracy**: Primary metric (as in the paper)
- **Accuracy by query type**: Breakdown by inference/comparison/temporal/null

## Project Structure

```
multihop_rag_benchmark/
├── config.py              # Configuration management
├── data/
│   ├── loader.py          # MultiHop-RAG dataset loading
│   └── preprocessing.py   # Chunking
├── generation/
│   └── llm_client.py      # OpenAI-compatible client
├── indexing/
│   ├── vector_index.py    # FAISS index
│   ├── kg_index.py        # Knowledge graph (LlamaIndex-style)
│   └── community_index.py # Microsoft GraphRAG wrapper
├── retrieval/
│   ├── vector_retriever.py
│   ├── kg_retriever.py
│   ├── community_retriever.py
│   └── hybrid_retriever.py
├── evaluation/
│   ├── metrics.py         # Accuracy metrics
│   └── evaluator.py       # Benchmark runner
└── experiments/
    ├── run_benchmark.py   # Main entry point
    └── configs/
        └── default.yaml
```

## Key Findings from Paper

| Query Type | Best Method | Reason |
|------------|-------------|--------|
| Single-hop factual | Vector RAG | Preserves original text details |
| Multi-hop reasoning | GraphRAG Local | Graph structure for connections |
| Comparison | GraphRAG | Explicit relation encoding |
| Temporal | GraphRAG | Temporal relations in graph |
| Global overview | GraphRAG Global | High-level community summaries |

**Hybrid Integration** achieves best overall performance (+6.4% on MultiHop-RAG).

## References

- Paper: https://arxiv.org/abs/2502.11371
- MultiHop-RAG Dataset: https://huggingface.co/datasets/yixuantt/MultiHopRAG
- Microsoft GraphRAG: https://github.com/microsoft/graphrag
