from dataclasses import dataclass
from typing import Optional


@dataclass
class FewShotExample:
    """Few-shot example for RAG."""
    context: str
    question: str
    answer: str


# System prompt for RAG
SYSTEM_PROMPT_RU = """Ты - помощник для ответов на вопросы по научным работам.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. В начале ответа ОБЯЗАТЕЛЬНО укажи источник в формате: "Согласно работе [название]..."
3. Если указан автор - упомяни его
4. Если информации недостаточно - честно скажи: "В предоставленных источниках нет информации о..."
5. Отвечай на том же языке, на котором задан вопрос
6. Будь кратким и по существу, избегай воды"""

SYSTEM_PROMPT_EN = """You are a helpful assistant for answering questions about scientific papers.

Rules:
1. Answer ONLY based on the provided context
2. ALWAYS cite the source at the beginning: "According to [paper title]..."
3. If author is mentioned - include them
4. If information is insufficient - say: "The provided sources do not contain information about..."
5. Answer in the same language as the question
6. Be concise and to the point"""


# Few-shot examples
FEW_SHOT_EXAMPLES_RU = [
    FewShotExample(
        context="""[Источник: vatutin_cyclic_dls_2021.md]
В работе приведены результаты подсчета числа главных классов циклических диагональных латинских квадратов.
Автор: Ватутин Э.И.
Числовой ряд A341585 в OEIS содержит количество главных классов для порядков N.""",
        question="Что такое числовой ряд A341585?",
        answer="""Согласно работе "vatutin_cyclic_dls_2021.md" (автор Ватутин Э.И.), числовой ряд A341585 в OEIS содержит количество главных классов циклических диагональных латинских квадратов для различных порядков N."""
    ),
    FewShotExample(
        context="""[Источник: algorithm_complexity.md]
Алгоритм имеет временную сложность O(n^2) для худшего случая.
Пространственная сложность составляет O(n).""",
        question="Какова сложность алгоритма сортировки пузырьком?",
        answer="""В предоставленных источниках нет информации о сортировке пузырьком. Источник "algorithm_complexity.md" описывает алгоритм с временной сложностью O(n^2) и пространственной O(n), но не указывает его название."""
    ),
    FewShotExample(
        context="""[Источник: latin_squares_overview.md]
Диагональный латинский квадрат (ДЛК) порядка N - это квадратная матрица N×N,
в которой каждый элемент встречается ровно один раз в каждой строке, столбце и на обеих диагоналях.

[Источник: orthogonal_pairs.md]
Пара ортогональных ДЛК (ОДЛК) - это два ДЛК, при наложении которых все пары элементов различны.""",
        question="Что такое ОДЛК?",
        answer="""Согласно работе "orthogonal_pairs.md", ОДЛК (пара ортогональных диагональных латинских квадратов) - это два диагональных латинских квадрата, при наложении которых все пары элементов различны. При этом, как указано в "latin_squares_overview.md", каждый ДЛК представляет собой матрицу N×N, где каждый элемент встречается ровно один раз в каждой строке, столбце и на диагоналях."""
    ),
]


def format_context_block(content: str, metadata: Optional[dict] = None) -> str:
    """Format a context block with source metadata."""
    if not metadata:
        return content

    # Extract source info
    title = metadata.get('doc_title', metadata.get('title', ''))
    author = metadata.get('author', '')

    header_parts = []
    if title:
        header_parts.append(f"Источник: {title}")
    if author:
        header_parts.append(f"Автор: {author}")

    if header_parts:
        header = "[" + ", ".join(header_parts) + "]\n"
        return header + content

    return content


def build_few_shot_prompt(
    context: str,
    question: str,
    examples: list[FewShotExample],
) -> str:
    """Build prompt with few-shot examples."""
    parts = []

    # Add all examples
    for ex in examples:
        parts.append(f"""Пример:
Контекст:
{ex.context}

Вопрос: {ex.question}

Ответ: {ex.answer}

---""")

    # Add actual query
    parts.append(f"""Теперь ответь на вопрос:
Контекст:
{context}

Вопрос: {question}

Ответ:""")

    return "\n\n".join(parts)


def build_simple_prompt(context: str, question: str) -> str:
    """Build simple prompt without few-shot examples."""
    return f"""Контекст:
{context}

Вопрос: {question}

Ответ:"""


# Default configuration
DEFAULT_PROMPTS = {
    "system_ru": SYSTEM_PROMPT_RU,
    "system_en": SYSTEM_PROMPT_EN,
    "few_shot_examples_ru": FEW_SHOT_EXAMPLES_RU,
}
