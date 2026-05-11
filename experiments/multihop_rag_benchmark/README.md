# MultiHop-RAG Benchmark: RAG vs GraphRAG

Воспроизведение систематической оценки разных стратегий retrieval из статьи Han et al. (2025) [«RAG vs. GraphRAG: A Systematic Evaluation and Key Insights»](https://arxiv.org/abs/2502.11371) на бенчмарке MultiHop-RAG.

## Сравниваемые методы

| Метод | Описание |
|---|---|
| **Vector RAG** | Классический семантический поиск по эмбеддингам чанков |
| **KG-based GraphRAG** | Граф знаний с извлечением триплетов (в стиле LlamaIndex) |
| **Community GraphRAG Local** | Microsoft GraphRAG с локальным поиском |
| **Community GraphRAG Global** | Microsoft GraphRAG с глобальным поиском |
| **Hybrid Selection** | Классификация запроса → маршрутизация на лучший метод |
| **Hybrid Integration** | Параллельный запуск обоих методов с объединением результатов |

## Установка

```bash
cd experiments/multihop_rag_benchmark
pip install -r requirements.txt

# Опционально — для Community-based GraphRAG
pip install graphrag
```

## Конфигурация

Отредактируйте `experiments/configs/default.yaml`:

```yaml
llm:
  api_key_env: "OPENAI_API_KEY"
  api_base: "https://your-api.com/v1"  # для OpenAI-совместимого API
  model: "gpt-4o-mini"

embedding:
  api_key_env: "OPENAI_API_KEY"
  api_base: null
  model: "text-embedding-ada-002"
```

## Использование

### Полный прогон бенчмарка

```bash
export OPENAI_API_KEY="your-key"

python -m multihop_rag_benchmark.experiments.run_benchmark \
    --config multihop_rag_benchmark/experiments/configs/default.yaml
```

### Конкретные методы

```bash
python -m multihop_rag_benchmark.experiments.run_benchmark \
    --config multihop_rag_benchmark/experiments/configs/default.yaml \
    --methods vector_rag kg_rag
```

### Тестовый прогон на подвыборке

```bash
python -m multihop_rag_benchmark.experiments.run_benchmark \
    --config multihop_rag_benchmark/experiments/configs/default.yaml \
    --max-samples 100
```

## Результаты

Сохраняются в `benchmark_results/`:

```
benchmark_results/
├── vector_rag_results.json
├── kg_rag_results.json
├── hybrid_integration_results.json
└── comparison.json
```

### Метрики

- **Accuracy** — основная метрика, как в статье
- **Accuracy by query type** — разбивка по типам запросов (inference / comparison / temporal / null)

## Структура пакета

```
multihop_rag_benchmark/
├── config.py                — управление конфигурацией
├── data/
│   ├── loader.py            — загрузка датасета MultiHop-RAG
│   └── preprocessing.py     — чанкинг
├── generation/
│   └── llm_client.py        — OpenAI-совместимый клиент
├── indexing/
│   ├── vector_index.py      — векторный индекс
│   ├── kg_index/            — граф знаний (LlamaIndex-style)
│   └── chroma_index.py      — обёртка над ChromaDB
├── retrieval/
│   ├── vector_retriever.py
│   ├── graphrag_retriever.py
│   └── hybrid_retriever.py
├── evaluation/
│   ├── metrics.py           — метрики Accuracy
│   └── evaluator.py         — запуск бенчмарка
└── experiments/
    ├── run_benchmark.py     — основная точка входа
    └── configs/
        └── default.yaml
```

## Ключевые выводы из статьи

| Тип запроса | Лучший метод | Причина |
|---|---|---|
| Однохоповый фактический | Vector RAG | Сохраняет детали исходного текста |
| Многохоповое рассуждение | GraphRAG Local | Граф фиксирует связи между сущностями |
| Сравнение | GraphRAG | Явная кодировка отношений |
| Темпоральный | GraphRAG | Темпоральные рёбра в графе |
| Глобальный обзор | GraphRAG Global | Высокоуровневые резюме сообществ |

**Hybrid Integration** даёт лучшее качество в среднем (+6.4 % на MultiHop-RAG).

## Ссылки

- Статья: <https://arxiv.org/abs/2502.11371>
- Датасет MultiHop-RAG: <https://huggingface.co/datasets/yixuantt/MultiHopRAG>
- Microsoft GraphRAG: <https://github.com/microsoft/graphrag>
