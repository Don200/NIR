# Hybrid RAG

Гибридная система RAG, объединяющая векторный и графовый поиск.

## Архитектура

```
┌─────────────┐     ┌─────────────┐
│  Vector     │     │   Graph     │
│  (ChromaDB) │     │ (LlamaIndex)│
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
         ┌───────▼───────┐
         │ Hybrid Merger │
         └───────┬───────┘
                 │
         ┌───────▼───────┐
         │  LLM Answer   │
         └───────────────┘
```

- **Vector RAG**: ChromaDB + SentenceSplitter/MarkdownNodeParser
- **Graph RAG**: LlamaIndex PropertyGraphIndex (triplet-based)
- **Hybrid**: объединение результатов обоих методов

## Установка

```bash
pip install -r requirements.txt
```

## Конфигурация

Скопируй и отредактируй конфиг:

```bash
cp config.example.yaml config.yaml
```

```yaml
llm:
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"

embedding:
  model: "text-embedding-3-small"
  base_url: "https://api.openai.com/v1"

vector:
  chunk_size: 256    # токены
  chunk_overlap: 50
  top_k: 10

graph:
  max_triplets_per_chunk: 10
  similarity_top_k: 10
```

Установи API ключ:

```bash
export OPENAI_API_KEY="sk-..."
```

## Использование

### 1. Подготовка документов

Поддерживаемые форматы:
- `.txt` — plain text
- `.md` — markdown (парсится по заголовкам)
- `.json` — структурированный формат

JSON формат:
```json
[
  {"id": "doc1", "title": "Title", "content": "Document text..."},
  {"id": "doc2", "title": "Title 2", "content": "Another document..."}
]
```

### 2. Индексация

```bash
# Все индексы (vector + graph)
python -m hybrid_rag index \
  --input ./documents/ \
  --output ./indexes/ \
  --config config.yaml \
  --type all

# Только векторный индекс
python -m hybrid_rag index -i ./documents/ -o ./indexes/ -t vector

# Только графовый индекс
python -m hybrid_rag index -i ./documents/ -o ./indexes/ -t graph
```

### 3. Проверка статуса

```bash
python -m hybrid_rag status --index-dir ./indexes/
```

### 4. Запуск сервера

```bash
# API (FastAPI)
python -m hybrid_rag serve --index-dir ./indexes/ --port 8000

# UI (Streamlit) — в отдельном терминале
streamlit run hybrid_rag/ui/app.py
```

### 5. Запросы

**Через API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is...?", "method": "hybrid"}'
```

**Методы поиска:**
- `vector` — только векторный поиск
- `graph` — только графовый поиск
- `hybrid` — оба метода (рекомендуется)

## Docker

### Индексация (локально)

```bash
python -m hybrid_rag index -i ./documents/ -o ./indexes/
```

### Перенос на сервер

```bash
rsync -av ./indexes/ server:/app/indexes/
```

### Запуск

```bash
# Создай .env файл
echo "OPENAI_API_KEY=sk-..." > .env

# Запуск
docker-compose up -d
```

Сервисы:
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- UI: http://localhost:8501

## Структура проекта

```
hybrid_rag/
├── core/
│   ├── config.py       # Конфигурация
│   ├── llm.py          # LLM клиент
│   └── models.py       # Dataclasses
├── indexing/
│   ├── vector_index.py # ChromaDB + LlamaIndex chunking
│   └── graph_index.py  # PropertyGraphIndex
├── retrieval/
│   └── hybrid.py       # Гибридный retriever
├── api/
│   ├── app.py          # FastAPI
│   └── schemas.py      # Pydantic модели
├── ui/
│   └── app.py          # Streamlit
├── cli.py              # CLI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Формат индексов

```
indexes/
├── vector/
│   └── chroma/              # ChromaDB persist
├── graph/
│   ├── docstore.json
│   ├── graph_store.json     # Триплеты
│   ├── index_store.json
│   └── default__vector_store.json
└── metadata.json
```

Индексы портируемые — можно создать на одной машине и перенести на другую.

## API Reference

### `GET /health`
Health check.

### `GET /status`
Статус индексов.

```json
{
  "vector_indexed": true,
  "vector_count": 1234,
  "graph_indexed": true
}
```

### `POST /query`
Запрос к RAG системе.

Request:
```json
{
  "query": "What is machine learning?",
  "method": "hybrid",
  "top_k": 10
}
```

Response:
```json
{
  "answer": "Machine learning is...",
  "sources": [
    {"content": "...", "score": 0.85, "source": "vector"},
    {"content": "...", "score": 0.72, "source": "graph"}
  ],
  "method": "hybrid",
  "query": "What is machine learning?"
}
```
