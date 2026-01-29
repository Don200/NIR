"""Evaluation metrics for MultiHop-RAG benchmark."""

import re
import string
from typing import List, Dict, Optional
from collections import defaultdict


def normalize_answer(answer: str, case_sensitive: bool = False) -> str:
    """
    Normalize answer for comparison.

    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    - Optionally lowercase
    """
    if not answer:
        return ""

    text = answer.strip()

    if not case_sensitive:
        text = text.lower()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def exact_match(prediction: str, ground_truth: str, normalize: bool = True) -> bool:
    """Check if prediction exactly matches ground truth."""
    if normalize:
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)
    return prediction == ground_truth


def fuzzy_match(
    prediction: str,
    ground_truth: str,
    normalize: bool = True,
    threshold: float = 0.8,
) -> bool:
    """
    Check if prediction fuzzy-matches ground truth.

    Uses token overlap ratio.
    """
    if normalize:
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)

    if not ground_truth:
        return not prediction

    pred_tokens = set(prediction.split())
    gt_tokens = set(ground_truth.split())

    if not gt_tokens:
        return not pred_tokens

    # Calculate Jaccard similarity
    intersection = len(pred_tokens & gt_tokens)
    union = len(pred_tokens | gt_tokens)

    if union == 0:
        return True

    similarity = intersection / union
    return similarity >= threshold


def contains_match(prediction: str, ground_truth: str, normalize: bool = True) -> bool:
    """Check if ground truth is contained in prediction."""
    if normalize:
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)

    return ground_truth in prediction


def calculate_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    match_fn=exact_match,
    normalize: bool = True,
) -> float:
    """
    Calculate accuracy as used in the paper for MultiHop-RAG.

    Args:
        predictions: Model predictions
        ground_truths: Ground truth answers
        match_fn: Function to compare prediction with ground truth
        normalize: Whether to normalize answers

    Returns:
        Accuracy score (0-1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    if not predictions:
        return 0.0

    correct = sum(
        match_fn(pred, gt, normalize)
        for pred, gt in zip(predictions, ground_truths)
    )
    return correct / len(predictions)


def calculate_accuracy_by_type(
    predictions: List[str],
    ground_truths: List[str],
    query_types: List[str],
    match_fn=exact_match,
    normalize: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy broken down by query type.

    As in the paper, MultiHop-RAG has 4 query types:
    - inference
    - comparison
    - temporal
    - null

    Returns:
        Dict with accuracy for each type and overall
    """
    type_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, gt, qtype in zip(predictions, ground_truths, query_types):
        is_correct = match_fn(pred, gt, normalize)
        type_results[qtype]["correct"] += int(is_correct)
        type_results[qtype]["total"] += 1
        type_results["overall"]["correct"] += int(is_correct)
        type_results["overall"]["total"] += 1

    # Calculate accuracies
    results = {}
    for qtype, counts in type_results.items():
        if counts["total"] > 0:
            results[qtype] = {
                "accuracy": counts["correct"] / counts["total"],
                "correct": counts["correct"],
                "total": counts["total"],
            }

    return results


class ExactMatchMetric:
    """Exact match metric class."""

    def __init__(self, normalize: bool = True, case_sensitive: bool = False):
        self.normalize = normalize
        self.case_sensitive = case_sensitive

    def __call__(self, prediction: str, ground_truth: str, normalize: bool = None) -> bool:
        """
        Compare prediction with ground truth.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            normalize: Ignored (uses self.normalize). For compatibility with function interface.
        """
        pred = prediction
        gt = ground_truth

        if self.normalize:
            pred = normalize_answer(pred, self.case_sensitive)
            gt = normalize_answer(gt, self.case_sensitive)

        return pred == gt

    @property
    def name(self) -> str:
        return "exact_match"


class FuzzyMatchMetric:
    """Fuzzy match metric class."""

    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        threshold: float = 0.8,
    ):
        self.normalize = normalize
        self.case_sensitive = case_sensitive
        self.threshold = threshold

    def __call__(self, prediction: str, ground_truth: str, normalize: bool = None) -> bool:
        """Compare with fuzzy matching. normalize arg ignored for compatibility."""
        return fuzzy_match(
            prediction,
            ground_truth,
            normalize=self.normalize,
            threshold=self.threshold,
        )

    @property
    def name(self) -> str:
        return f"fuzzy_match_{self.threshold}"


class ContainsMatchMetric:
    """Contains match metric class."""

    def __init__(self, normalize: bool = True, case_sensitive: bool = False):
        self.normalize = normalize
        self.case_sensitive = case_sensitive

    def __call__(self, prediction: str, ground_truth: str, normalize: bool = None) -> bool:
        """Check if ground truth is contained in prediction. normalize arg ignored for compatibility."""
        pred = prediction
        gt = ground_truth

        if self.normalize:
            pred = normalize_answer(pred, self.case_sensitive)
            gt = normalize_answer(gt, self.case_sensitive)

        return gt in pred

    @property
    def name(self) -> str:
        return "contains_match"
