from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from synth.config import load_config
from synth.graph import load_kg
from synth.testset import TestExample
from synth.utils import load_jsonl
from synth.validators import deduplicate, validate_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate testset and export CSV/report.")
    parser.add_argument("--config", type=Path, required=True, help="Path to synth_config.yaml")
    return parser.parse_args()


def _to_example(row: dict) -> TestExample:
    return TestExample(
        question=row.get("question", ""),
        reference_answer=row.get("reference_answer", ""),
        reference_contexts=row.get("reference_contexts", []),
        metadata=row.get("metadata", {}),
        valid=row.get("valid", True),
        errors=row.get("errors"),
    )


def _summary(examples: List[TestExample]) -> str:
    total = len(examples)
    valid = sum(1 for ex in examples if ex.valid)
    by_type = {}
    hop_counts = {}
    for ex in examples:
        ex_type = ex.metadata.get("type", "unknown")
        by_type[ex_type] = by_type.get(ex_type, 0) + 1
        hops = ex.metadata.get("hops", 1)
        hop_counts[hops] = hop_counts.get(hops, 0) + 1

    lines = [
        f"Total examples: {total}",
        f"Valid examples: {valid}",
        "",
        "By type:",
    ]
    for key, count in sorted(by_type.items()):
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("By hops:")
    for key, count in sorted(hop_counts.items()):
        lines.append(f"- {key} hop(s): {count}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    graph = load_kg(config.kg_path)
    rows = load_jsonl(config.testset_jsonl)
    examples = [_to_example(row) for row in rows]

    revalidated: List[TestExample] = []
    for example in examples:
        ok, errors = validate_example(
            graph,
            example,
            min_context_chars=config.min_context_chars,
            leakage_max_ratio=config.leakage_max_ratio,
        )
        example.valid = ok
        example.errors = errors if errors else None
        revalidated.append(example)

    deduped = deduplicate([ex for ex in revalidated if ex.valid])

    df = pd.DataFrame([
        {
            "question": ex.question,
            "reference_answer": ex.reference_answer,
            "reference_contexts": ex.reference_contexts,
            "metadata": ex.metadata,
        }
        for ex in deduped
    ])

    config.testset_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.testset_csv, index=False)

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(_summary(revalidated), encoding="utf-8")

    print("Exported CSV to", config.testset_csv)
    print("Wrote report to", config.report_path)


if __name__ == "__main__":
    main()
