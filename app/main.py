import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import llm, store
from app.config import get_settings
from app.loaders import Page, UnsupportedFileError, extension, parse
from app.retrieval import retrieve
from app.schemas import (
    Document,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TextIngestRequest,
    Timings,
)

settings = get_settings()

app = FastAPI(title="Nugget API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _llm_error(exc: Exception) -> HTTPException:
    if isinstance(exc, llm.LLMNotConfiguredError):
        return HTTPException(503, "The language model is not configured. Set NUGGET_API_KEY.")
    return HTTPException(502, f"The language model request failed: {exc}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents=len(store.list_documents()),
        chunks=store.chunk_count(),
        model=settings.chat_model,
        embedding_model=settings.embedding_model,
        llm_configured=bool(settings.api_key),
    )


@app.get("/documents", response_model=list[Document])
def documents() -> list[Document]:
    return store.list_documents()


@app.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str) -> list[dict]:
    chunks = store.document_chunks(doc_id)
    if not chunks:
        raise HTTPException(404, "Document not found")
    return chunks


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> Response:
    if not store.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)


def _ingest(name: str, kind: str, data: bytes, pages: list[Page]) -> IngestResponse:
    if not pages:
        raise HTTPException(
            422, "No readable text found. Scanned PDFs need OCR before they can be indexed."
        )
    doc_id = store.content_id(data)
    existing = store.get_document(doc_id)
    if existing:
        return IngestResponse(document=existing, duplicate=True)
    return IngestResponse(document=store.add_document(doc_id, name, kind, pages))


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    name = file.filename or "upload"
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB limit")
    if not data:
        raise HTTPException(422, "The file is empty")
    try:
        pages = await run_in_threadpool(parse, name, data)
    except UnsupportedFileError as exc:
        raise HTTPException(415, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Could not read {name}: {exc}") from exc
    return await run_in_threadpool(_ingest, name, extension(name) or "txt", data, pages)


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(req: TextIngestRequest) -> IngestResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "The text is empty")
    title = (req.title or "").strip() or text.split("\n", 1)[0][:60].strip() or "Note"
    data = text.encode()
    return await run_in_threadpool(_ingest, title, "note", data, [Page(text)])


async def _prepare(req: QueryRequest) -> tuple[str, list, Timings]:
    timings = Timings()
    start = time.perf_counter()
    search_query = await llm.rewrite_query(req.question, req.history)
    timings.rewrite_ms = _ms(start) if req.history else 0

    start = time.perf_counter()
    sources = await run_in_threadpool(
        retrieve, search_query, req.top_k or settings.top_k, req.doc_ids
    )
    timings.retrieval_ms = _ms(start)
    return search_query, sources, timings


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    try:
        search_query, sources, timings = await _prepare(req)
        start = time.perf_counter()
        messages = llm.build_messages(req.question, req.history, sources, req.locale)
        answer = "".join([t async for t in llm.stream_answer(messages)])
        timings.generation_ms = _ms(start)
    except HTTPException:
        raise
    except Exception as exc:
        raise _llm_error(exc) from exc
    return QueryResponse(answer=answer, sources=sources, search_query=search_query, timings=timings)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        try:
            search_query, sources, timings = await _prepare(req)
            yield _sse(
                "sources",
                {
                    "search_query": search_query,
                    "sources": [s.model_dump() for s in sources],
                    "timings": timings.model_dump(),
                },
            )
            start = time.perf_counter()
            messages = llm.build_messages(req.question, req.history, sources, req.locale)
            async for token in llm.stream_answer(messages):
                yield _sse("token", token)
            timings.generation_ms = _ms(start)
            yield _sse("done", {"timings": timings.model_dump()})
        except Exception as exc:
            yield _sse("error", {"detail": _llm_error(exc).detail})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
