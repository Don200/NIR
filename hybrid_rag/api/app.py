import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..core.config import Config, load_config
from ..core.models import RetrievalMethod
from ..retrieval.hybrid import HybridRetriever
from .schemas import (
    QueryRequest,
    QueryResponse,
    SourceItem,
    StatusResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get retriever instance."""
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized",
        )
    return retriever


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create FastAPI application."""
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        global retriever

        logger.info("Starting Hybrid RAG API...")
        retriever = HybridRetriever(config)

        status = retriever.load_indexes()
        logger.info(f"Index load status: {status}")

        yield

        logger.info("Shutting down...")

    app = FastAPI(
        title="Hybrid RAG API",
        description="Vector + Graph RAG retrieval and generation",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(status="ok", version=__version__)

    @app.get("/status", response_model=StatusResponse)
    async def status():
        """Get system status."""
        r = get_retriever()
        s = r.get_status()
        return StatusResponse(**s)

    @app.post("/query", response_model=QueryResponse)
    @observe(name="api_query")
    async def query(request: QueryRequest):
        """Query the RAG system."""
        r = get_retriever()

        method_map = {
            "vector": RetrievalMethod.VECTOR,
            "graph": RetrievalMethod.GRAPH,
            "hybrid": RetrievalMethod.HYBRID,
        }
        method = method_map[request.method.value]

        result = r.query(
            query=request.query,
            method=method,
            top_k=request.top_k,
            max_context_length=request.max_context,
        )

        sources = [
            SourceItem(
                content=s.content[:500],
                score=s.score,
                source=s.source.value,
                metadata=s.metadata,
            )
            for s in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            method=result.method.value,
            query=result.query,
        )

    @app.get("/graph/data")
    async def graph_data(limit: int = 100):
        """Get graph data for visualization."""
        r = get_retriever()
        data = r.graph_index.get_graph_data(limit=limit)
        return data

    return app


# For direct uvicorn usage
app = create_app()
