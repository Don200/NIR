"""Main evaluation pipeline."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from ..data.loader import MultiHopRAGSample
from ..retrieval.base import BaseRetriever
from ..generation.llm_client import LLMClient
from .metrics import (
    calculate_accuracy,
    calculate_accuracy_by_type,
    ExactMatchMetric,
    ContainsMatchMetric,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Single prediction result."""
    query_id: str
    query: str
    ground_truth: str
    prediction: str
    query_type: str
    is_correct: bool
    context: str
    retrieval_metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Complete evaluation result for a method."""
    method_name: str
    overall_accuracy: float
    accuracy_by_type: Dict[str, Dict[str, float]]
    predictions: List[PredictionResult]
    total_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method_name": self.method_name,
            "overall_accuracy": self.overall_accuracy,
            "accuracy_by_type": self.accuracy_by_type,
            "total_samples": self.total_samples,
            "timestamp": self.timestamp,
            "config": self.config,
            "predictions": [
                {
                    "query_id": p.query_id,
                    "query": p.query,
                    "ground_truth": p.ground_truth,
                    "prediction": p.prediction,
                    "query_type": p.query_type,
                    "is_correct": p.is_correct,
                }
                for p in self.predictions
            ],
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class BenchmarkEvaluator:
    """
    Main evaluator for the RAG vs GraphRAG benchmark.

    Runs retrieval + generation pipeline and computes metrics.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_exact_match: bool = True,
        use_contains_match: bool = True,
    ):
        self.llm_client = llm_client
        self.metrics = []

        if use_exact_match:
            self.metrics.append(ExactMatchMetric())
        if use_contains_match:
            self.metrics.append(ContainsMatchMetric())

        # Default to exact match for accuracy calculation
        self.primary_metric = ExactMatchMetric()

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate answer given query and context."""
        if system_prompt is None:
            system_prompt = """You are a helpful assistant answering questions based on provided context.
Answer concisely and precisely. If the context doesn't contain the answer, say "I don't know" or "Cannot be determined".
For yes/no questions, answer with just "yes" or "no" when possible.
For questions asking for names, dates, or specific facts, provide just the answer without explanation."""

        return self.llm_client.generate_with_context(
            query=query,
            context=context,
            system_prompt=system_prompt,
        )

    def evaluate_retriever(
        self,
        retriever: BaseRetriever,
        samples: List[MultiHopRAGSample],
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a retriever on the benchmark samples.

        Args:
            retriever: The retriever to evaluate
            samples: List of benchmark samples
            show_progress: Whether to show progress

        Returns:
            EvaluationResult with all metrics
        """
        predictions = []
        pred_texts = []
        gt_texts = []
        query_types = []

        for i, sample in enumerate(samples):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Processing sample {i + 1}/{len(samples)}")

            # Retrieve context
            retrieval_result = retriever.retrieve(sample.query)

            # Generate answer
            if retrieval_result.context:
                prediction = self.generate_answer(
                    query=sample.query,
                    context=retrieval_result.context,
                )
            else:
                prediction = "Cannot be determined from the given context."

            # Check correctness
            is_correct = self.primary_metric(prediction, sample.answer)

            # Store results
            pred_result = PredictionResult(
                query_id=sample.query_id,
                query=sample.query,
                ground_truth=sample.answer,
                prediction=prediction,
                query_type=sample.query_type,
                is_correct=is_correct,
                context=retrieval_result.context[:1000] if retrieval_result.context else "",
                retrieval_metadata=retrieval_result.metadata,
            )
            predictions.append(pred_result)
            pred_texts.append(prediction)
            gt_texts.append(sample.answer)
            query_types.append(sample.query_type)

        # Calculate metrics
        overall_accuracy = calculate_accuracy(
            pred_texts,
            gt_texts,
            match_fn=self.primary_metric,
        )

        accuracy_by_type = calculate_accuracy_by_type(
            pred_texts,
            gt_texts,
            query_types,
            match_fn=self.primary_metric,
        )

        return EvaluationResult(
            method_name=retriever.name,
            overall_accuracy=overall_accuracy,
            accuracy_by_type=accuracy_by_type,
            predictions=predictions,
            total_samples=len(samples),
        )

    def compare_methods(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """
        Generate comparison report between methods.

        Args:
            results: List of evaluation results from different methods

        Returns:
            Comparison report
        """
        comparison = {
            "overall": {},
            "by_type": {},
            "complementarity": {},
        }

        # Overall comparison
        for result in results:
            comparison["overall"][result.method_name] = {
                "accuracy": result.overall_accuracy,
                "total_samples": result.total_samples,
            }

        # By type comparison
        all_types = set()
        for result in results:
            all_types.update(result.accuracy_by_type.keys())

        for qtype in all_types:
            if qtype == "overall":
                continue
            comparison["by_type"][qtype] = {}
            for result in results:
                if qtype in result.accuracy_by_type:
                    comparison["by_type"][qtype][result.method_name] = \
                        result.accuracy_by_type[qtype]["accuracy"]

        # Complementarity analysis (which method solves which queries uniquely)
        if len(results) >= 2:
            for i, r1 in enumerate(results):
                for j, r2 in enumerate(results):
                    if i >= j:
                        continue

                    r1_only = 0
                    r2_only = 0
                    both = 0
                    neither = 0

                    for p1, p2 in zip(r1.predictions, r2.predictions):
                        if p1.is_correct and p2.is_correct:
                            both += 1
                        elif p1.is_correct:
                            r1_only += 1
                        elif p2.is_correct:
                            r2_only += 1
                        else:
                            neither += 1

                    total = len(r1.predictions)
                    comparison["complementarity"][f"{r1.method_name}_vs_{r2.method_name}"] = {
                        f"{r1.method_name}_only": r1_only,
                        f"{r1.method_name}_only_pct": r1_only / total * 100,
                        f"{r2.method_name}_only": r2_only,
                        f"{r2.method_name}_only_pct": r2_only / total * 100,
                        "both_correct": both,
                        "both_correct_pct": both / total * 100,
                        "neither_correct": neither,
                        "neither_correct_pct": neither / total * 100,
                    }

        return comparison
