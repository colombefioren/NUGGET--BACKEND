from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    # A remote call, deliberately: an in-process embedding model measured 600MB+ of RAM,
    # which alone exceeds most free hosting tiers. See config.py for the reasoning.
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_api_base,
        check_embedding_ctx_length=False,  # not every OpenAI-compatible provider supports this
    )
