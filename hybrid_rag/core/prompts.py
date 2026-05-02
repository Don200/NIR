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
1. Отвечай ТОЛЬКО на основе предоставленного контекста.
2. В начале ответа ОБЯЗАТЕЛЬНО укажи источник в формате: "Согласно работе по теме «<Тема>»..."
   - Бери значение из поля `Тема` в шапке блока контекста (например, [Тема: Генерация случайных ДЛК; ...]).
   - Если поля `Тема` нет, используй `Источник` из той же шапки.
   - КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО цитировать имена файлов (например, `*.md`, `evatutin_co_ls_diag_fill.md`, хеши вида `f18253cd...`). Никогда не подставляй их в ответ.
3. Если в шапке указаны `Авторы`, упомяни их в скобках.
4. Если информации недостаточно - честно скажи: "В предоставленных источниках нет информации о..."
5. Отвечай на том же языке, на котором задан вопрос.
6. Будь кратким и по существу, избегай воды."""

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
        context="""[Тема: Перечисление циклических ДЛК; Авторы: Ватутин Э.И.]
В работе приведены результаты подсчета числа главных классов циклических диагональных латинских квадратов.
Числовой ряд A341585 в OEIS содержит количество главных классов для порядков N.""",
        question="Что такое числовой ряд A341585?",
        answer="""Согласно работе по теме «Перечисление циклических ДЛК» (Ватутин Э.И.), числовой ряд A341585 в OEIS содержит количество главных классов циклических диагональных латинских квадратов для различных порядков N."""
    ),
    FewShotExample(
        context="""[Тема: Анализ сложности алгоритмов]
Алгоритм имеет временную сложность O(n^2) для худшего случая.
Пространственная сложность составляет O(n).""",
        question="Какова сложность алгоритма сортировки пузырьком?",
        answer="""В предоставленных источниках нет информации о сортировке пузырьком. Работа по теме «Анализ сложности алгоритмов» описывает алгоритм с временной сложностью O(n^2) и пространственной O(n), но не указывает его название."""
    ),
    FewShotExample(
        context="""[Тема: Введение в латинские квадраты]
Диагональный латинский квадрат (ДЛК) порядка N - это квадратная матрица N×N,
в которой каждый элемент встречается ровно один раз в каждой строке, столбце и на обеих диагоналях.

[Тема: Ортогональные пары ДЛК; Авторы: Заикин О.С., Ватутин Э.И.]
Пара ортогональных ДЛК (ОДЛК) - это два ДЛК, при наложении которых все пары элементов различны.""",
        question="Что такое ОДЛК?",
        answer="""Согласно работе по теме «Ортогональные пары ДЛК» (Заикин О.С., Ватутин Э.И.), ОДЛК (пара ортогональных диагональных латинских квадратов) - это два диагональных латинских квадрата, при наложении которых все пары элементов различны. При этом, как указано в работе по теме «Введение в латинские квадраты», каждый ДЛК представляет собой матрицу N×N, где каждый элемент встречается ровно один раз в каждой строке, столбце и на диагоналях."""
    ),
]


_FILENAME_LIKE_KEYS = ("doc_title", "title")


def _looks_like_filename(value: str) -> bool:
    """Heuristic: filenames end in a known doc extension or look like a hash."""
    if not value:
        return True
    v = value.strip().lower()
    if v.endswith((".md", ".pdf", ".txt", ".html")):
        return True
    stem = v.rsplit(".", 1)[0]
    if len(stem) >= 32 and all(c in "0123456789abcdef" for c in stem):
        return True
    return False


def format_context_block(content: str, metadata: Optional[dict] = None) -> str:
    """Format a context block with human-readable source metadata.

    Header layout: ``[Тема: <topic>; Источник: <display_title>; Авторы: <authors>]``.
    Filenames (``*.md``, hash-like stems) are never shown — they only confuse the LLM
    into citing them verbatim.
    """
    if not metadata:
        return content

    topic = metadata.get("topic") or ""
    display_title = metadata.get("display_title") or ""
    authors = metadata.get("authors") or metadata.get("author") or ""

    fallback_title = ""
    if not topic and not display_title:
        for key in _FILENAME_LIKE_KEYS:
            v = metadata.get(key)
            if isinstance(v, str) and v and not _looks_like_filename(v):
                fallback_title = v
                break

    header_parts = []
    if topic:
        header_parts.append(f"Тема: {topic}")
    if display_title:
        header_parts.append(f"Источник: {display_title}")
    elif fallback_title:
        header_parts.append(f"Источник: {fallback_title}")
    if authors:
        header_parts.append(f"Авторы: {authors}")

    if header_parts:
        header = "[" + "; ".join(header_parts) + "]\n"
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
