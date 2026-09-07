import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app import rag
from app.config import get_settings
from app.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse


class TextIngestRequest(BaseModel):
    text: str

settings = get_settings()

app = FastAPI(title="Lumen API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        docs=rag.document_count(settings),
        model=settings.openai_model,
    )


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(req: TextIngestRequest) -> IngestResponse:
    count = rag.ingest_text(req.text, None, settings)
    return IngestResponse(count=count, message=f"Indexed {count} chunks")


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    suffix = file.filename.rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if suffix == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            count = rag.ingest_pdf(tmp_path, settings)
        finally:
            os.unlink(tmp_path)
        return IngestResponse(count=count, message=f"Indexed {count} chunks from {file.filename}")
    if suffix in {"txt", "md", "csv", "json"}:
        content = (await file.read()).decode("utf-8", errors="ignore")
        count = rag.ingest_text(content, {"source": file.filename}, settings)
        return IngestResponse(count=count, message=f"Indexed {count} chunks from {file.filename}")
    raise HTTPException(status_code=415, detail="Unsupported file type")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    sources = rag.search(req.question, settings)
    context = "\n\n".join(f"[{i+1}] {s.content}" for i, s in enumerate(sources))
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    prompt = (
        "You are Lumen, an answer engine. Answer the question using ONLY the context below. "
        "If the context lacks the answer, say so. Be concise and accurate.\n\n"
        f"Context:\n{context}\n\nQuestion: {req.question}"
    )
    answer = llm.invoke(prompt).content
    return QueryResponse(answer=answer, sources=sources)