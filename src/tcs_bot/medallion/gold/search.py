"""Gold read side (AC-005.9): semantic search over Qdrant + the persisted graph artifacts.

The query is embedded with the same model gold was built with (``summary.json`` ->
``meta.embedding_model``): the persisted ``lsa.pkl`` for LSA, OpenAI otherwise.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tcs_bot.errors import DependencyUnavailable, NotFoundError
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.gold.embedder import LsaEmbedder, model_slug
from tcs_bot.medallion.gold.pipeline import GRAPH_HTML, GRAPH_JSON, LSA_PKL, SUMMARY_JSON
from tcs_bot.medallion.gold.vector_store import GoldVectorStore
from tcs_bot.medallion.models import GoldHit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tcs_bot.config import Settings
    from tcs_bot.rag.embeddings import Embedder

log = get_logger(__name__)

RUN_HINT = "run `tcs-bot medallion run`"

_lock = threading.Lock()
_vector_store: GoldVectorStore | None = None
_lsa_cache: dict[str, tuple[float, LsaEmbedder]] = {}


def set_vector_store(store: GoldVectorStore | None) -> None:
    """Inject a vector store (tests: ``QdrantClient(":memory:")``); ``None`` resets."""
    global _vector_store
    _vector_store = store


def get_vector_store(settings: Settings | None = None) -> GoldVectorStore:
    global _vector_store
    with _lock:
        if _vector_store is None:
            settings = settings or _settings()
            _vector_store = GoldVectorStore(url=settings.qdrant_url)
        return _vector_store


def _settings() -> Settings:
    from tcs_bot.config import get_settings

    return get_settings()


def gold_dir(settings: Settings | None = None) -> Path:
    return Path((settings or _settings()).gold_dir)


def _artifact(name: str, settings: Settings | None = None) -> Path:
    path = gold_dir(settings) / name
    if not path.is_file():
        raise NotFoundError(
            f"gold artifact {name} not found in {path.parent} - {RUN_HINT}",
            code="gold_not_built",
        )
    return path


def _read_json(name: str, settings: Settings | None) -> dict[str, Any]:
    path = _artifact(name, settings)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NotFoundError(
            f"gold artifact {name} is unreadable - {RUN_HINT}", code="gold_not_built"
        ) from exc


def load_summary(settings: Settings | None = None) -> dict[str, Any]:
    return _read_json(SUMMARY_JSON, settings)


def load_graph(settings: Settings | None = None) -> dict[str, Any]:
    return _read_json(GRAPH_JSON, settings)


def graph_html_path(settings: Settings | None = None) -> Path:
    return _artifact(GRAPH_HTML, settings)


def query_embedder(settings: Settings | None = None) -> Embedder:
    """The embedder gold was built with (LSA from ``lsa.pkl`` or OpenAI)."""
    settings = settings or _settings()
    model = str(load_summary(settings).get("meta", {}).get("embedding_model", ""))
    if model.startswith("lsa-"):
        path = _artifact(LSA_PKL, settings)
        mtime = path.stat().st_mtime
        key = str(path.resolve())
        with _lock:
            cached = _lsa_cache.get(key)
            if cached is None or cached[0] != mtime:
                cached = (mtime, LsaEmbedder.load(path))
                _lsa_cache[key] = cached
        return cached[1]
    if settings.use_fake_llm:
        raise DependencyUnavailable(
            f"gold was built with {model!r} but OpenAI is not configured; "
            f"set OPENAI_API_KEY or rebuild offline ({RUN_HINT})",
            code="embedder_unavailable",
        )
    from tcs_bot.rag.embeddings import OpenAIEmbedder

    return OpenAIEmbedder(
        model=model or settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )


def search_risks(
    query: str,
    k: int = 10,
    categories: Sequence[str] | None = None,
    min_risk: float | None = None,
    *,
    level: str = "chunks",
    dedupe: bool = True,
    settings: Settings | None = None,
    vector_store: GoldVectorStore | None = None,
) -> list[GoldHit]:
    """Top-``k`` gold hits for ``query`` (best chunk per document when ``dedupe``)."""
    settings = settings or _settings()
    query = (query or "").strip()
    k = max(1, min(int(k), 100))
    if not query:
        return []
    embedder = query_embedder(settings)
    vector = embedder.embed([query])[0]
    vs = (vector_store or get_vector_store(settings)).with_model(model_slug(embedder))
    fetch = k * 4 if dedupe and level == "chunks" else k
    try:
        hits = vs.search(
            vector,
            k=fetch,
            categories=list(categories) if categories else None,
            min_risk=min_risk,
            level="docs" if level == "docs" else "chunks",
        )
    except Exception as exc:
        log.error("gold.search_failed", error_type=type(exc).__name__)
        raise DependencyUnavailable(
            f"Qdrant search failed ({type(exc).__name__}); is Qdrant up and gold built? "
            f"{RUN_HINT}",
            code="qdrant_unavailable",
        ) from exc
    if dedupe:
        best: dict[int, GoldHit] = {}
        for h in hits:  # already sorted by score desc
            best.setdefault(h.doc_id, h)
        hits = list(best.values())
    return hits[:k]
