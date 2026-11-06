"""Single source of configuration. Every module reads settings via ``get_settings()``."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

# Constitution §3 pins the models. Embedding models must produce 1536-dim vectors: the
# HashEmbedder fallback, the retriever and the ES ``dense_vector`` mapping all assume it.
CONSTITUTION_CHAT_MODEL = "gpt-4.1-mini"
CONSTITUTION_TEMPERATURE = 0.0
EMBEDDING_MODEL_DIMS: dict[str, int] = {"text-embedding-3-small": 1536}


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

    # TCS risk medallion pipeline (spec 005)
    qdrant_url: str = "http://127.0.0.1:6333"
    medallion_max_docs: int = Field(default=500, ge=1, le=500)
    medallion_max_depth: int = Field(default=2, ge=0, le=2)
    medallion_relevance_threshold: float = 0.35
    silver_index: str = "silver-tcs-risk"
    gold_dir: str = "data/gold"
    gold_graph_k: int = 8
    gold_graph_min_cosine: float = 0.30
    lsa_dim: int = 256

    @field_validator("openai_embedding_model")
    @classmethod
    def _embedding_model_is_1536d(cls, v: str) -> str:
        if EMBEDDING_MODEL_DIMS.get(v) != 1536:
            allowed = ", ".join(sorted(m for m, d in EMBEDDING_MODEL_DIMS.items() if d == 1536))
            raise ValueError(
                f"unsupported embedding model {v!r}: must produce 1536-dim vectors ({allowed})"
            )
        return v

    @model_validator(mode="after")
    def _warn_constitution_drift(self) -> Settings:
        # Overrides stay possible (experiments), but are loud: constitution §3 pins these.
        if self.openai_temperature != CONSTITUTION_TEMPERATURE:
            log.warning(
                "config.temperature_override",
                temperature=self.openai_temperature,
                expected=CONSTITUTION_TEMPERATURE,
            )
        if self.openai_chat_model != CONSTITUTION_CHAT_MODEL:
            log.warning(
                "config.chat_model_override",
                model=self.openai_chat_model,
                expected=CONSTITUTION_CHAT_MODEL,
            )
        return self

    @property
    def use_fake_llm(self) -> bool:
        return (
            self.llm_provider == "fake"
            or not self.openai_api_key.get_secret_value().startswith("sk-")
            or self.openai_api_key.get_secret_value() == "sk-replace-me"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
