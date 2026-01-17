from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import httpx
from openai import OpenAI

from synth.api_clients import normalize_base_url
from synth.config import load_config
from synth.graph import load_kg
from synth.testset import TestExample
from synth.utils import save_jsonl, set_seed
from synth.validators import validate_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic testset from a KG via ragas.")
    parser.add_argument("--config", type=Path, required=True, help="Path to synth_config.yaml")
    parser.add_argument("--n", type=int, default=None, help="Override number of examples.")
    return parser.parse_args()


def _require_env_var(name: str | None) -> str:
    if not name:
        raise RuntimeError("Missing api_key_env in config.")
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return value


def _build_openai_client(api_base: str | None, api_key: str) -> OpenAI:
    if not api_base:
        raise RuntimeError("api_base must be set for OpenAI-compatible client.")
    return OpenAI(
        base_url=normalize_base_url(api_base),
        api_key=api_key,
        http_client=httpx.Client(verify=False),
    )


def _build_ragas_llm(config) -> object:
    api_key = _require_env_var(config.llm_api_key_env)
    client = _build_openai_client(config.llm_api_base, api_key)

    try:
        from ragas.llms import llm_factory

        return llm_factory(config.llm_model, client=client)
    except Exception as exc:
        raise RuntimeError("Failed to build ragas LLM with OpenAI client.") from exc


def _build_ragas_embeddings(config) -> object:
    api_key = _require_env_var(config.embeddings_api_key_env)
    client = _build_openai_client(config.embeddings_api_base, api_key)

    try:
        from ragas.embeddings import embeddings_factory

        return embeddings_factory(config.embeddings_model, client=client)
    except Exception as exc:
        raise RuntimeError("Failed to build ragas embeddings with OpenAI client.") from exc


def _load_testset_generator():
    try:
        from ragas.testset import TestsetGenerator

        return TestsetGenerator
    except Exception:
        try:
            from ragas.testset.generator import TestsetGenerator

            return TestsetGenerator
        except Exception as exc:
            raise RuntimeError("ragas TestsetGenerator is unavailable.") from exc


def _resolve_distributions(config) -> dict:
    distributions = dict(config.distribution or {})
    try:
        from ragas.testset.evolutions import multi_context, reasoning, simple

        mapping = {
            "simple": simple,
            "reasoning": reasoning,
            "multi_context": multi_context,
        }
        resolved = {}
        for key, weight in distributions.items():
            resolved[mapping.get(key, key)] = weight
        return resolved
    except Exception:
        return distributions


def _build_ragas_knowledge_graph(docs):
    graph_cls = None
    for module in ("ragas.testset.graph", "ragas.testset.knowledge_graph"):
        try:
            graph_module = __import__(module, fromlist=["KnowledgeGraph"])
            graph_cls = getattr(graph_module, "KnowledgeGraph", None)
            if graph_cls is not None:
                break
        except Exception:
            continue

    if graph_cls is None or not hasattr(graph_cls, "from_documents"):
        return None

    transforms = None
    try:
        from ragas.testset.transforms import default_transforms

        transforms = default_transforms
        if callable(transforms):
            transforms = transforms()
    except Exception:
        transforms = None

    try:
        if transforms is None:
            return graph_cls.from_documents(docs)
        return graph_cls.from_documents(docs, transforms=transforms)
    except TypeError:
        try:
            return graph_cls.from_documents(docs, transforms)
        except Exception:
            return None


def _call_generate(generator, payload, *, size: int, distributions: dict):
    for key in ("distributions", "evolutions"):
        try:
            return generator(payload, testset_size=size, **{key: distributions})
        except TypeError:
            continue
    return generator(payload, testset_size=size)


def _generate_with_ragas(*, config, size: int) -> List[TestExample]:
    from langchain_core.documents import Document

    graph = load_kg(config.kg_path)
    docs = [Document(page_content=node.page_content, metadata=node.metadata) for node in graph.nodes]

    llm = _build_ragas_llm(config)
    embeddings = _build_ragas_embeddings(config)
    distributions = _resolve_distributions(config)

    TestsetGenerator = _load_testset_generator()
    if hasattr(TestsetGenerator, "from_llm"):
        generator = TestsetGenerator.from_llm(llm=llm, embeddings=embeddings)
    elif hasattr(TestsetGenerator, "from_langchain"):
        generator = TestsetGenerator.from_langchain(llm=llm, embeddings=embeddings)
    else:
        raise RuntimeError("Unsupported ragas TestsetGenerator API.")

    kg = _build_ragas_knowledge_graph(docs)
    if kg is not None and hasattr(generator, "generate"):
        testset = _call_generate(generator.generate, kg, size=size, distributions=distributions)
    elif hasattr(generator, "generate_with_langchain_docs"):
        testset = _call_generate(generator.generate_with_langchain_docs, docs, size=size, distributions=distributions)
    elif hasattr(generator, "generate"):
        testset = _call_generate(generator.generate, docs, size=size, distributions=distributions)
    else:
        raise RuntimeError("Unsupported ragas testset generator API.")

    if hasattr(testset, "to_pandas"):
        df = testset.to_pandas()
        records = df.to_dict(orient="records")
    elif isinstance(testset, list):
        records = testset
    else:
        raise RuntimeError("Unsupported ragas testset format.")

    examples: List[TestExample] = []
    for row in records:
        question = row.get("question") or row.get("query") or ""
        answer = row.get("answer") or row.get("ground_truth") or ""
        contexts = row.get("contexts") or row.get("reference_contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        examples.append(
            TestExample(
                question=question,
                reference_answer=answer,
                reference_contexts=contexts,
                metadata={"type": "ragas_default", "hops": 1},
            )
        )
    return examples


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    graph = load_kg(config.kg_path)
    size = args.n or config.testset_size

    examples = _generate_with_ragas(config=config, size=size)

    validated: List[TestExample] = []
    for example in examples:
        ok, errors = validate_example(
            graph,
            example,
            min_context_chars=config.min_context_chars,
            leakage_max_ratio=config.leakage_max_ratio,
        )
        example.valid = ok
        example.errors = errors if errors else None
        validated.append(example)

    save_jsonl(config.testset_jsonl, [example.__dict__ for example in validated])
    print("Saved testset to", config.testset_jsonl)


if __name__ == "__main__":
    main()
