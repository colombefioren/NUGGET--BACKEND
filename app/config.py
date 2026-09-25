import hashlib
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NUGGET_", extra="ignore")

    # Any OpenAI-compatible endpoint works; both are required so nothing provider-specific
    # is baked into the code.
    api_base: str
    chat_model: str
    api_key: str = ""

    # A separate, also-required, OpenAI-compatible endpoint for embeddings. Running an
    # embedding model in-process needs 600MB+ of RAM (measured), which alone exceeds most
    # free hosting tiers (Render free = 512MB) — so embeddings are always a remote call,
    # never a locally loaded model. See .env.example for free-tier providers.
    embedding_api_base: str
    embedding_model: str
    embedding_api_key: str = ""

    chroma_dir: str = "./data/chroma"
    collection_name: str = "nugget"
    cors_origins: str = "http://localhost:3000"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 6
    max_upload_mb: int = 25
    temperature: float = 0.1

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def collection(self) -> str:
        # Vectors from different embedding models are incompatible, so each model
        # gets its own collection instead of crashing on a dimension mismatch.
        digest = hashlib.sha1(self.embedding_model.encode()).hexdigest()[:8]
        return f"{self.collection_name}_{digest}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
