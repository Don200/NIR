"""Evaluation utilities for RAG quality assessment.

Provides LLM-as-judge metrics for evaluating:
- Answer faithfulness (is the answer grounded in sources?)
- Answer relevance (does the answer address the question?)
- Answer completeness (does the answer cover all key aspects?)
- Retrieval relevance (are retrieved chunks relevant to the question?)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .core.config import LLMConfig
from .core.llm import LLMClient
from .pipeline import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class EvalScore:
    """Single evaluation score with explanation."""
    metric: str
    score: float  # 1-5 scale
    explanation: str


@dataclass
class EvalResult:
    """Full evaluation result for one query."""
    query: str
    method: str
    answer: str
    scores: list[EvalScore]
    reference_answer: Optional[str] = None
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    def scores_dict(self) -> dict[str, float]:
        return {s.metric: s.score for s in self.scores}


# --- Evaluation prompts ---

FAITHFULNESS_PROMPT = """Оцени верность (faithfulness) ответа по шкале от 1 до 5.
Верность означает, что ответ основан ТОЛЬКО на предоставленных источниках и не содержит выдуманной информации.

Источники:
{sources}

Вопрос: {question}
Ответ: {answer}

Критерии:
1 - Ответ полностью выдуман, не основан на источниках
2 - Ответ частично основан на источниках, но содержит значительные выдумки
3 - Ответ в основном основан на источниках, но есть неточности
4 - Ответ верно передает информацию из источников с минимальными неточностями
5 - Ответ полностью основан на источниках, все утверждения подтверждены

Ответь СТРОГО в формате JSON:
{{"score": <число 1-5>, "explanation": "<краткое обоснование>"}}"""

RELEVANCE_PROMPT = """Оцени релевантность ответа по шкале от 1 до 5.
Релевантность означает, что ответ непосредственно отвечает на заданный вопрос.

Вопрос: {question}
Ответ: {answer}

Критерии:
1 - Ответ не имеет отношения к вопросу
2 - Ответ косвенно связан с вопросом, но не отвечает на него
3 - Ответ частично отвечает на вопрос
4 - Ответ хорошо отвечает на вопрос, но упускает некоторые аспекты
5 - Ответ полностью и точно отвечает на заданный вопрос

Ответь СТРОГО в формате JSON:
{{"score": <число 1-5>, "explanation": "<краткое обоснование>"}}"""

COMPLETENESS_PROMPT = """Оцени полноту ответа по шкале от 1 до 5.
Полнота означает, что ответ покрывает все ключевые аспекты вопроса.

Вопрос: {question}
Ответ: {answer}
{reference_section}

Критерии:
1 - Ответ крайне неполный, упускает почти все ключевые аспекты
2 - Ответ охватывает лишь малую часть вопроса
3 - Ответ охватывает основные аспекты, но упускает важные детали
4 - Ответ достаточно полный, охватывает большинство аспектов
5 - Ответ исчерпывающий, охватывает все ключевые аспекты

Ответь СТРОГО в формате JSON:
{{"score": <число 1-5>, "explanation": "<краткое обоснование>"}}"""

RETRIEVAL_RELEVANCE_PROMPT = """Оцени релевантность извлеченного фрагмента к вопросу по шкале от 1 до 5.

Вопрос: {question}
Фрагмент: {chunk}

Критерии:
1 - Фрагмент не имеет отношения к вопросу
2 - Фрагмент слабо связан с темой вопроса
3 - Фрагмент связан с темой, но не содержит прямого ответа
4 - Фрагмент содержит полезную информацию для ответа
5 - Фрагмент напрямую содержит ответ на вопрос

Ответь СТРОГО в формате JSON:
{{"score": <число 1-5>, "explanation": "<краткое обоснование>"}}"""


class RAGEvaluator:
    """LLM-as-judge evaluator for RAG system quality.

    Usage::

        from hybrid_rag.evaluation import RAGEvaluator

        evaluator = RAGEvaluator(llm_config)
        eval_result = evaluator.evaluate(retrieval_result)
        print(eval_result.avg_score)
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm = LLMClient(llm_config)

    def _judge(self, prompt: str) -> dict:
        """Call LLM judge and parse JSON response."""
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="Ты - эксперт по оценке качества ответов RAG-системы. Отвечай только в формате JSON.",
            temperature=0.0,
        )
        # Extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse judge response: {response}")
            return {"score": 0, "explanation": "Failed to parse response"}

    def evaluate_faithfulness(self, result: RetrievalResult) -> EvalScore:
        """Evaluate if the answer is grounded in retrieved sources."""
        sources_text = "\n---\n".join(
            f"[{i+1}] {s.content[:500]}" for i, s in enumerate(result.sources)
        )
        prompt = FAITHFULNESS_PROMPT.format(
            sources=sources_text,
            question=result.query,
            answer=result.answer,
        )
        resp = self._judge(prompt)
        return EvalScore(
            metric="faithfulness",
            score=float(resp.get("score", 0)),
            explanation=resp.get("explanation", ""),
        )

    def evaluate_relevance(self, result: RetrievalResult) -> EvalScore:
        """Evaluate if the answer addresses the question."""
        prompt = RELEVANCE_PROMPT.format(
            question=result.query,
            answer=result.answer,
        )
        resp = self._judge(prompt)
        return EvalScore(
            metric="relevance",
            score=float(resp.get("score", 0)),
            explanation=resp.get("explanation", ""),
        )

    def evaluate_completeness(
        self, result: RetrievalResult, reference_answer: Optional[str] = None
    ) -> EvalScore:
        """Evaluate answer completeness."""
        ref_section = ""
        if reference_answer:
            ref_section = f"\nЭталонный ответ (для сравнения): {reference_answer}"
        prompt = COMPLETENESS_PROMPT.format(
            question=result.query,
            answer=result.answer,
            reference_section=ref_section,
        )
        resp = self._judge(prompt)
        return EvalScore(
            metric="completeness",
            score=float(resp.get("score", 0)),
            explanation=resp.get("explanation", ""),
        )

    def evaluate_retrieval_relevance(
        self, question: str, chunks: list[str]
    ) -> list[EvalScore]:
        """Evaluate relevance of each retrieved chunk."""
        scores = []
        for i, chunk in enumerate(chunks):
            prompt = RETRIEVAL_RELEVANCE_PROMPT.format(
                question=question,
                chunk=chunk[:500],
            )
            resp = self._judge(prompt)
            scores.append(EvalScore(
                metric=f"retrieval_relevance_{i}",
                score=float(resp.get("score", 0)),
                explanation=resp.get("explanation", ""),
            ))
        return scores

    def evaluate(
        self,
        result: RetrievalResult,
        reference_answer: Optional[str] = None,
        evaluate_retrieval: bool = False,
    ) -> EvalResult:
        """Run full evaluation on a single result.

        Args:
            result: The RetrievalResult to evaluate.
            reference_answer: Optional ground-truth answer for completeness check.
            evaluate_retrieval: If True, also evaluate individual chunk relevance.

        Returns:
            EvalResult with all scores.
        """
        scores = [
            self.evaluate_faithfulness(result),
            self.evaluate_relevance(result),
            self.evaluate_completeness(result, reference_answer),
        ]

        if evaluate_retrieval and result.sources:
            chunk_scores = self.evaluate_retrieval_relevance(
                result.query,
                [s.content for s in result.sources[:5]],  # top 5 chunks
            )
            avg_retrieval = sum(s.score for s in chunk_scores) / len(chunk_scores)
            scores.append(EvalScore(
                metric="retrieval_relevance",
                score=avg_retrieval,
                explanation=f"Average over {len(chunk_scores)} chunks",
            ))

        return EvalResult(
            query=result.query,
            method=result.method.value,
            answer=result.answer,
            scores=scores,
            reference_answer=reference_answer,
            retrieval_time_s=result.retrieval_time_s,
            generation_time_s=result.generation_time_s,
        )

    def evaluate_batch(
        self,
        results: list[RetrievalResult],
        reference_answers: Optional[list[str]] = None,
        evaluate_retrieval: bool = False,
        verbose: bool = True,
    ) -> list[EvalResult]:
        """Evaluate a batch of results."""
        eval_results = []
        for i, result in enumerate(results):
            if verbose:
                print(f"  Evaluating [{i+1}/{len(results)}] {result.query[:60]}...")
            ref = reference_answers[i] if reference_answers else None
            eval_result = self.evaluate(result, ref, evaluate_retrieval)
            eval_results.append(eval_result)
        return eval_results
