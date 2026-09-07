from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LUMEN_", extra="ignore")

    api_key: str = ""
    api_base: str = "https://api.xkiro.com/v1"
    chat_model: str = "deepseek/deepseek-v4-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "./data/chroma"
    collection_name: str = "lumen_docs"
    cors_origins: str = "http://localhost:3000"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 4

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()