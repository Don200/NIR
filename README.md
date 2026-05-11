# Hybrid RAG: гибридная система извлечения и агрегирования контекста

Программное обеспечение системы вопросно-ответного поиска по узкоспециализированным научным корпусам, объединяющей векторный (semantic similarity) и графовый (knowledge graph) методы поиска. Разработано в рамках выпускной квалификационной работы по направлению «Прикладная математика».

Архитектура системы основана на подходе из статьи Han et al. (2025) [«RAG vs. GraphRAG: A Systematic Evaluation and Key Insights»](https://arxiv.org/abs/2502.11371). Векторный RAG лучше работает на фактических запросах, графовый — на многошаговых (multi-hop), требующих рассуждения по цепочкам связей; гибридная интеграция объединяет преимущества обоих подходов.

## Структура репозитория

Репозиторий содержит два независимых пакета, отвечающих за разные части исследования.

```
.
├── hybrid_rag/                       — продакшен-система с CLI, REST API и веб-интерфейсом
│   └── README.md                       (применение: §2.10 ВКР, корпус по латинским квадратам)
│
└── experiments/multihop_rag_benchmark/  — экспериментальная система для воспроизведения
    └── README.md                       результатов из Han et al. (2025) на бенчмарке MultiHop-RAG
                                        (применение: §2.9 ВКР)
```

| Пакет | Назначение | Точка входа |
|---|---|---|
| [`hybrid_rag/`](hybrid_rag/README.md) | Прикладная RAG-система с веб-интерфейсом для работы с собственным научным корпусом | `python -m hybrid_rag {index,serve,status}` |
| [`experiments/multihop_rag_benchmark/`](experiments/multihop_rag_benchmark/README.md) | Замеры качества разных стратегий retrieval на стандартном бенчмарке MultiHop-RAG | `python -m multihop_rag_benchmark.experiments.run_benchmark` |

## Компоненты системы

| Метод | Реализация | Применение |
|---|---|---|
| Vector RAG | ChromaDB + BGE-M3 эмбеддинги, SentenceSplitter / MarkdownNodeParser | Фактические запросы, ответ в одном фрагменте |
| Graph RAG | LlamaIndex PropertyGraphIndex с извлечением триплетов через LLM | Multi-hop запросы, рассуждение по связям между сущностями |
| Hybrid | Параллельный запуск обоих методов с объединением и дедупликацией результатов | Смешанные запросы |

## Быстрый старт

```bash
git clone <repo-url>
cd nir_vdi

# Установить продакшен-пакет
pip install -r hybrid_rag/requirements.txt

# Прописать ключ
export OPENAI_API_KEY="sk-..."

# Проиндексировать документы и запустить сервер
python -m hybrid_rag index   -i ./documents -o ./indexes -t all
python -m hybrid_rag serve   --index-dir ./indexes --port 8000

# UI (в другом терминале)
streamlit run hybrid_rag/ui/app.py
```

Запуск в Docker и подробные инструкции — в [`hybrid_rag/README.md`](hybrid_rag/README.md).

Воспроизведение эксперимента с бенчмарком MultiHop-RAG — в [`experiments/multihop_rag_benchmark/README.md`](experiments/multihop_rag_benchmark/README.md).

## Технологический стек

Python 3.11, LlamaIndex, ChromaDB, FastAPI, Streamlit, OpenAI-совместимый API. В качестве моделей в работе использовались BGE-M3 (эмбеддинги) и Qwen3-Next-80B-A3B-Instruct (генерация).

## Ссылки

- Han H., Wang Y., Shomer H. et al. RAG vs. GraphRAG: A Systematic Evaluation and Key Insights. arXiv:2502.11371, 2025. URL: <https://arxiv.org/abs/2502.11371>
- Edge D., Trinh H., Cheng N. et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130, 2024. URL: <https://arxiv.org/abs/2404.16130>
- Lewis P., Perez E., Piktus A. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks // NeurIPS, 2020.
- MultiHop-RAG Dataset. URL: <https://huggingface.co/datasets/yixuantt/MultiHopRAG>
