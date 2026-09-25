import hashlib
import threading
from collections import defaultdict
from datetime import UTC, datetime
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document as Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.embeddings import get_embeddings
from app.loaders import Page
from app.schemas import Document

_lock = threading.RLock()
_version = 0

SEPARATORS = ["\n\n", "\n", ". ", "。", "! ", "? ", "！", "？", "؟ ", "; ", ", ", " ", ""]


@lru_cache
def get_store() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def version() -> int:
    """Bumped on every write so derived indexes (BM25) know when to rebuild."""
    return _version


def _bump() -> None:
    global _version
    _version += 1


def content_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _to_document(doc_id: str, metas: list[dict]) -> Document:
    first = metas[0]
    return Document(
        id=doc_id,
        name=first.get("source", "Untitled"),
        kind=first.get("kind", "txt"),
        chunks=len(metas),
        chars=int(first.get("doc_chars", 0)),
        pages=first.get("doc_pages") or None,
        created_at=first.get("created_at", ""),
    )


def get_document(doc_id: str) -> Document | None:
    res = get_store().get(where={"doc_id": doc_id}, include=["metadatas"])
    metas = res.get("metadatas") or []
    return _to_document(doc_id, metas) if metas else None


def list_documents() -> list[Document]:
    res = get_store().get(include=["metadatas"])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for meta in res.get("metadatas") or []:
        if meta and meta.get("doc_id"):
            grouped[meta["doc_id"]].append(meta)
    docs = [_to_document(doc_id, metas) for doc_id, metas in grouped.items()]
    return sorted(docs, key=lambda d: d.created_at, reverse=True)


def chunk_count() -> int:
    return get_store()._collection.count()


def add_document(doc_id: str, name: str, kind: str, pages: list[Page]) -> Document:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",
    )
    base = {
        "doc_id": doc_id,
        "source": name,
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "doc_chars": sum(len(p.text) for p in pages),
        "doc_pages": sum(1 for p in pages if p.number is not None),
    }
    chunks: list[Chunk] = []
    for page in pages:
        meta = dict(base)
        if page.number is not None:
            meta["page"] = page.number
        chunks.extend(splitter.create_documents([page.text], metadatas=[meta]))
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]

    with _lock:
        get_store().add_documents(chunks, ids=ids)
        _bump()
    return _to_document(doc_id, [c.metadata for c in chunks])


def delete_document(doc_id: str) -> bool:
    with _lock:
        store = get_store()
        ids = store.get(where={"doc_id": doc_id}, include=[]).get("ids") or []
        if not ids:
            return False
        store.delete(ids=ids)
        _bump()
    return True


def document_chunks(doc_id: str) -> list[dict]:
    res = get_store().get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    rows = [
        {"id": i, "content": text, "page": meta.get("page"), "chunk": meta.get("chunk", 0)}
        for i, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
    ]
    return sorted(rows, key=lambda r: r["chunk"])


def all_chunks() -> tuple[list[str], list[str], list[dict]]:
    res = get_store().get(include=["documents", "metadatas"])
    return res["ids"], res["documents"], res["metadatas"]
