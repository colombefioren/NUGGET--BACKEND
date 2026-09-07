from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    content: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    count: int
    message: str


class HealthResponse(BaseModel):
    status: str
    docs: int
    model: str