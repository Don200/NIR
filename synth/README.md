# Synthetic KG → RAG Testset Pipeline

## Quick start

```bash
# Build KG
python -m synth.build_kg --config configs/synth_config.yaml

# Generate testset
python -m synth.generate_testset --config configs/synth_config.yaml --n 500

# Validate + export
python -m synth.validate_and_export --config configs/synth_config.yaml
```

## Visualize graph

```bash
python -m synth.visualize_graph --config configs/synth_config.yaml --output artifacts/kg_visualization.png
```

## Notes
- Input PDFs live in `data/raw/` (metadata cache in `data/processed/`).
- Outputs are written under `artifacts/` (see `configs/synth_config.yaml`).
- Set `OPENAI_API_KEY` (or change `api_key_env` in config) and provide `api_base` for LLM/embeddings.
- Testset generation uses ragas `TestsetGenerator` with `distribution` keys like `simple`, `reasoning`, `multi_context`.
- Enable LLM claims extraction via `enrichment.mode: embeddings_plus_llm`.
- PDF parsing uses `docling` (install via `synth/requirements.txt`).
