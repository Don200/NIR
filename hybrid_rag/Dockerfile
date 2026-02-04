FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml .
COPY . /app/hybrid_rag/
COPY config.yaml /app/config.yaml

RUN uv sync --no-dev

RUN mkdir -p /app/indexes

EXPOSE 8000 8501

CMD ["uv", "run", "python", "-m", "hybrid_rag", "serve", "--config", "/app/config.yaml", "--host", "0.0.0.0", "--port", "8000", "--index-dir", "/app/indexes"]
