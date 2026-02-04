from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RetrievalMethodEnum(str, Enum):
    vector = "vector"
    graph = "graph"
    hybrid = "hybrid"


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    method: RetrievalMethodEnum = Field(
        default=RetrievalMethodEnum.hybrid,
        description="Retrieval method to use",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of results to retrieve",
    )
    max_context: Optional[int] = Field(
        default=None,
        description="Override max context length (default from config)",
    )


class SourceItem(BaseModel):
    content: str
    score: float
    source: str
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    method: str
    query: str


class StatusResponse(BaseModel):
    vector_indexed: bool
    vector_count: int
    graph_indexed: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
