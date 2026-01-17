from __future__ import annotations

import difflib
import re
from typing import Dict, List, Tuple

from synth.graph import KnowledgeGraph, build_adjacency, path_exists
from synth.testset import TestExample


def validate_context_presence(example: TestExample, *, min_chars: int) -> Tuple[bool, str]:
    if not example.reference_contexts:
        return False, "empty_context"
    if any(len(ctx.strip()) < min_chars for ctx in example.reference_contexts):
        return False, "context_too_short"
    return True, "ok"


def validate_multihop_structure(example: TestExample) -> Tuple[bool, str]:
    hops = example.metadata.get("hops", 1)
    if hops <= 1:
        return True, "ok"
    node_ids = example.metadata.get("node_ids", [])
    if len(set(node_ids)) < hops + 1:
        return False, "not_enough_unique_nodes"
    return True, "ok"


def validate_path(graph: KnowledgeGraph, example: TestExample) -> Tuple[bool, str]:
    hops = example.metadata.get("hops", 1)
    if hops <= 1:
        return True, "ok"
    adjacency = build_adjacency(graph)
    node_ids = example.metadata.get("node_ids", [])
    if not node_ids:
        return False, "missing_path"
    if not path_exists(adjacency, node_ids):
        return False, "path_not_found"
    if len(node_ids) != hops + 1:
        return False, "path_length_mismatch"
    return True, "ok"


def validate_leakage(example: TestExample, *, max_ratio: float) -> Tuple[bool, str]:
    question = example.question.strip().lower()
    if not question:
        return False, "empty_question"
    for context in example.reference_contexts:
        matcher = difflib.SequenceMatcher(None, question, context.lower())
        match = matcher.find_longest_match(0, len(question), 0, len(context))
        ratio = match.size / max(len(question), 1)
        if ratio >= max_ratio:
            return False, "question_copies_context"
    return True, "ok"


def normalize_question(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def deduplicate(examples: List[TestExample], *, threshold: float = 0.9) -> List[TestExample]:
    kept: List[TestExample] = []
    seen: List[set[str]] = []
    for example in examples:
        normalized = normalize_question(example.question)
        grams = _char_ngrams(normalized)
        is_dup = False
        for prev in seen:
            if not prev or not grams:
                continue
            similarity = len(prev & grams) / len(prev | grams)
            if similarity >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(example)
            seen.append(grams)
    return kept


def validate_example(
    graph: KnowledgeGraph,
    example: TestExample,
    *,
    min_context_chars: int,
    leakage_max_ratio: float,
) -> Tuple[bool, List[str]]:
    checks = [
        validate_context_presence(example, min_chars=min_context_chars),
        validate_multihop_structure(example),
        validate_path(graph, example),
        validate_leakage(example, max_ratio=leakage_max_ratio),
    ]
    errors = [reason for ok, reason in checks if not ok]
    return len(errors) == 0, errors
