FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Dependencies layer — cached unless pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# 2. Source code — invalidated on every code edit, but cheap to rebuild
#    because the heavy `uv sync` above is already cached.
COPY . /app/hybrid_rag/
COPY config.yaml /app/config.yaml

RUN mkdir -p /app/indexes

EXPOSE 8000 8501

CMD ["uv", "run", "python", "-m", "hybrid_rag", "serve", "--config", "/app/config.yaml", "--host", "0.0.0.0", "--port", "8000", "--index-dir", "/app/indexes"]
