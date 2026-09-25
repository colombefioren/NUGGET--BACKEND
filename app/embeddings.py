from functools import lru_cache

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from app.config import get_settings


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings, so documents never leave the machine to be indexed."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embeddings() -> Embeddings:
    # Loading the ONNX model takes seconds, so it is built once per process.
    settings = get_settings()
    return FastEmbedEmbeddings(settings.embedding_model, settings.model_cache_dir)
