"""Evaluation module."""

from .metrics import (
    normalize_answer,
    calculate_accuracy,
    calculate_accuracy_by_type,
    ExactMatchMetric,
    ContainsMatchMetric,
    FuzzyMatchMetric,
)
from .evaluator import BenchmarkEvaluator, EvaluationResult

__all__ = [
    "normalize_answer",
    "calculate_accuracy",
    "calculate_accuracy_by_type",
    "ExactMatchMetric",
    "ContainsMatchMetric",
    "FuzzyMatchMetric",
    "BenchmarkEvaluator",
    "EvaluationResult",
]
