from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.schemas import Source


def _embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def _vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=_embeddings(settings),
        persist_directory=settings.chroma_dir,
    )


def ingest_text(text: str, metadata: dict | None, settings: Settings) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    docs = splitter.create_documents([text], metadatas=[metadata or {}])
    return _vector_store(settings).add_documents(docs)


def ingest_pdf(path: str, settings: Settings) -> int:
    loader = PyPDFLoader(path)
    return ingest_text(
        "\n\n".join(doc.page_content for doc in loader.load()),
        {"source": path},
        settings,
    )


def ingest_txt(path: str, settings: Settings) -> int:
    loader = TextLoader(path)
    return ingest_text(
        "\n\n".join(doc.page_content for doc in loader.load()),
        {"source": path},
        settings,
    )


def search(query: str, settings: Settings) -> list[Source]:
    store = _vector_store(settings)
    results = store.similarity_search_with_relevance_scores(query, k=settings.top_k)
    sources = []
    for doc, score in results:
        sources.append(
            Source(
                content=doc.page_content,
                metadata=doc.metadata or {},
                score=round(float(score), 4),
            )
        )
    return sources


def document_count(settings: Settings) -> int:
    return _vector_store(settings)._collection.count()