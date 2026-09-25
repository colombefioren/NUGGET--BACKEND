from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    doc_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    locale: str | None = Field(default=None, max_length=16)


class Source(BaseModel):
    id: str
    doc_id: str
    source: str
    page: int | None = None
    chunk: int
    content: str
    score: float
    vector_score: float | None = None
    keyword_rank: int | None = None


class Timings(BaseModel):
    rewrite_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    search_query: str
    timings: Timings


class TextIngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    title: str | None = Field(default=None, max_length=200)


class Document(BaseModel):
    id: str
    name: str
    kind: str
    chunks: int
    chars: int
    pages: int | None = None
    created_at: str


class IngestResponse(BaseModel):
    document: Document
    duplicate: bool = False


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    model: str
    embedding_model: str
    llm_configured: bool
