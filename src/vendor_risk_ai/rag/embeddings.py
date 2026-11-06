"""Embedders (AC-003.3): OpenAI ``text-embedding-3-small`` or an offline hash embedder."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from vendor_risk_ai.config import Settings

EMBEDDING_DIM = 1536  # text-embedding-3-small; HashEmbedder uses the same dim on purpose

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
        "were", "will", "with", "which", "who", "our", "we", "they", "their",
    }
)  # fmt: skip


@runtime_checkable
class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)``."""
        ...


class OpenAIEmbedder:
    """Wraps ``langchain_openai.OpenAIEmbeddings`` (which batches requests internally)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dim: int = EMBEDDING_DIM,
        batch_size: int = 256,
    ) -> None:
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        self.model = model
        self.dim = dim
        self._client = OpenAIEmbeddings(
            model=model, api_key=SecretStr(api_key), chunk_size=batch_size
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._client.embed_documents(texts)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"unexpected embedding shape {arr.shape} for model {self.model}")
        return arr


def _bucket(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


class HashEmbedder:
    """Deterministic, offline bag-of-words embedder (feature hashing of unigrams + bigrams).

    Vectors are L2-normalized, so texts sharing vocabulary have a higher cosine similarity.
    Not semantic in the neural sense, but stable and good enough for tests / offline demos.
    """

    model = "hash-embedder-v1"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _features(self, text: str) -> Counter[str]:
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
        feats: Counter[str] = Counter(f"u:{t}" for t in tokens)
        feats.update(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
        return feats

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, count in self._features(text).items():
            weight = 1.0 if feat.startswith("u:") else 0.5
            vec[_bucket(feat, self.dim)] += weight * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32, copy=False)


def get_embedder(settings: Settings | None = None) -> Embedder:
    if settings is None:
        from vendor_risk_ai.config import get_settings

        settings = get_settings()
    if settings.use_fake_llm:
        return HashEmbedder()
    return OpenAIEmbedder(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
