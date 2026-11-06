"""Single source of configuration. Every module reads settings via ``get_settings()``."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # runtime
    app_env: Literal["local", "test", "prod"] = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM / embeddings
    llm_provider: Literal["openai", "fake"] = "openai"
    openai_api_key: SecretStr = SecretStr("")
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.0
    rag_top_k: int = 6
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "vendor_risk"
    mysql_user: str = "vendor_risk"
    mysql_password: SecretStr = SecretStr("vendor_risk_pw")
    mysql_pool_name: str = "vendor_risk_pool"
    mysql_pool_size: int = Field(default=10, ge=1, le=32)

    # Elasticsearch
    elastic_url: str = "http://127.0.0.1:9200"
    elastic_quotes_index: str = "market-quotes"
    elastic_chunks_index: str = "vendor-chunks"

    # Cache
    redis_url: str = "redis://127.0.0.1:6379/0"
    cache_ttl_seconds: int = 900
    local_cache_maxsize: int = 512

    # Seed
    seed_file: str = "data/seed/vendors.json"

    @property
    def use_fake_llm(self) -> bool:
        return self.llm_provider == "fake" or not self.openai_api_key.get_secret_value().startswith(
            "sk-"
        ) or self.openai_api_key.get_secret_value() == "sk-replace-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()
