import argparse
import json
import os
from pathlib import Path

import dotenv
from langchain_community.document_loaders import DirectoryLoader
from openai import OpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import apply_transforms, default_transforms


def load_prompt_overrides(prompt_dir: str | None) -> dict[str, object]:
    if not prompt_dir:
        return {}
    prompt_path = Path(prompt_dir)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt dir not found: {prompt_dir}")

    overrides: dict[str, object] = {}
    for path in prompt_path.iterdir():
        if path.suffix.lower() == ".txt":
            overrides[path.stem] = path.read_text(encoding="utf-8").strip()
        elif path.suffix.lower() == ".json":
            overrides.update(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("PyYAML is required for .yaml prompts") from exc
            overrides.update(yaml.safe_load(path.read_text(encoding="utf-8")))
    return overrides


def apply_overrides_to_prompts(synth, overrides: dict[str, object]) -> None:
    if not overrides:
        return

    current = synth.get_prompts()
    updates = {}
    for key, prompt in current.items():
        if key not in overrides:
            continue
        override = overrides[key]
        if isinstance(override, str):
            if hasattr(prompt, "instruction"):
                prompt.instruction = override
            elif hasattr(prompt, "template"):
                prompt.template = override
            else:
                raise ValueError(f"Unsupported prompt type for key: {key}")
        elif isinstance(override, dict):
            for field, value in override.items():
                if hasattr(prompt, field):
                    setattr(prompt, field, value)
                else:
                    raise ValueError(f"Prompt {key} has no field {field}")
        else:
            raise ValueError(f"Unsupported override type for key: {key}")
        updates[key] = prompt

    if updates:
        synth.set_prompts(**updates)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVP ragas synthetic testset generation with custom prompts."
    )
    parser.add_argument("--docs-path", default="test_docs/")
    parser.add_argument("--docs-glob", default="**/*.md")
    parser.add_argument("--prompt-dir", default=None)
    parser.add_argument("--testset-size", type=int, default=100)
    parser.add_argument("--single-weight", type=float, default=0.6)
    parser.add_argument("--mh-abs-weight", type=float, default=0.2)
    parser.add_argument("--mh-spec-weight", type=float, default=0.2)
    parser.add_argument("--llm-id", default="qwen235-thinking")
    parser.add_argument("--emb-model", default="hosted_vllm/BAAI/bge-m3")
    parser.add_argument("--output-path", default="output/ragas_testset.csv")
    parser.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    return parser


def save_testset(testset, output_path: str, fmt: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = testset.to_pandas()
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "jsonl":
        df.to_json(out_path, orient="records", lines=True, force_ascii=True)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def main() -> None:
    args = build_arg_parser().parse_args()

    os.environ["AUTO_DOWNLOAD_NLTK"] = "false"
    os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache/"
    dotenv.load_dotenv()

    loader = DirectoryLoader(args.docs_path, glob=args.docs_glob)
    docs = loader.load()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    generator_llm = llm_factory(args.llm_id, provider="openai", client=client)
    generator_embeddings = OpenAIEmbeddings(client=client, model=args.emb_model)

    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    trans = default_transforms(
        documents=docs, llm=generator_llm, embedding_model=generator_embeddings
    )
    apply_transforms(kg, trans)

    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg
    )

    overrides = load_prompt_overrides(args.prompt_dir)

    single = SingleHopSpecificQuerySynthesizer(llm=generator_llm)
    mh_abs = MultiHopAbstractQuerySynthesizer(llm=generator_llm)
    mh_spec = MultiHopSpecificQuerySynthesizer(llm=generator_llm)

    apply_overrides_to_prompts(single, overrides)
    apply_overrides_to_prompts(mh_abs, overrides)
    apply_overrides_to_prompts(mh_spec, overrides)

    query_distribution = [
        (single, args.single_weight),
        (mh_abs, args.mh_abs_weight),
        (mh_spec, args.mh_spec_weight),
    ]

    testset = generator.generate_with_langchain_docs(
        docs, testset_size=args.testset_size, query_distribution=query_distribution
    )

    save_testset(testset, args.output_path, args.output_format)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
