"""Silver pipeline: bronze (MySQL) -> structured, classified records -> ES (AC-005.5 / .6).

``build_silver_record(doc, settings=None)`` is pure (no I/O unless the OpenAI path is enabled).
``run_silver(limit=None, repo=None, store=None)`` drains unprocessed bronze rows in batches of
:data:`BATCH_SIZE`; a bad document never stops its batch and bronze rows are marked processed
only after their silver record is in Elasticsearch.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from tcs_bot.config import get_settings
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import BronzeDocument, SilverRecord
from tcs_bot.medallion.silver.classify import (
    categories_of,
    classify,
    split_sentences,
    tcs_relevance,
)
from tcs_bot.medallion.silver.entities import extract_entities
from tcs_bot.medallion.silver.extract import extract_main_text
from tcs_bot.medallion.silver.sentiment import sentiment

if TYPE_CHECKING:
    from tcs_bot.config import Settings

log = get_logger(__name__)

RULES_EXTRACTOR = "rules-v1"
BATCH_SIZE = 50
MIN_WORDS_RELEVANT = 30
SUMMARY_SENTENCES = 3
SUMMARY_MAX_CHARS = 600
_WORD = re.compile(r"\w+")
_BOILERPLATE = re.compile(
    r"\b(?:cookies?|subscribe|sign in|log in|newsletter|all rights reserved|click here|"
    r"advertisement|privacy policy|terms of (?:use|service)|javascript)\b",
    re.IGNORECASE,
)
_EN_STOP = frozenset(
    _WORD.findall(
        "the of and to in a is that for on with as was by are it at be from has have this its an "
        "said will were which or not but their"
    )
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def detect_language(text: str) -> str:
    """Tiny heuristic: Devanagari -> ``hi``; English stop-word share -> ``en``; else ``und``."""
    if not text:
        return "und"
    sample = text[:5000]
    if len(_DEVANAGARI.findall(sample)) > 0.2 * max(len(sample), 1):
        return "hi"
    words = [w.lower() for w in _WORD.findall(sample)]
    if not words:
        return "und"
    share = sum(w in _EN_STOP for w in words) / len(words)
    return "en" if share >= 0.08 or len(words) < 25 else "und"


def summarize(text: str, title: str = "") -> str:
    """First 2-3 informative sentences (>= 8 words, no boilerplate, not the title)."""
    out: list[str] = []
    title_l = (title or "").strip().lower()
    for sent in split_sentences(text):
        s = " ".join(sent.split())
        if len(_WORD.findall(s)) < 8 or _BOILERPLATE.search(s) or s.lower() == title_l:
            continue
        out.append(s)
        if len(out) >= SUMMARY_SENTENCES or sum(map(len, out)) >= SUMMARY_MAX_CHARS:
            break
    summary = " ".join(out)
    return summary if len(summary) <= SUMMARY_MAX_CHARS else summary[: SUMMARY_MAX_CHARS - 1] + "…"


def build_rules_record(doc: BronzeDocument, settings: Settings | None = None) -> SilverRecord:
    """Deterministic ``rules-v1`` silver record (no network)."""
    s = settings or get_settings()
    text, extracted_title = extract_main_text(doc.raw_content, doc.content_type, doc.url)
    title = (doc.title or "").strip() or (extracted_title or "").strip() or doc.url
    word_count = len(_WORD.findall(text))
    signals, risk_score = classify(text, title)
    relevance = tcs_relevance(text, title)
    return SilverRecord(
        doc_id=doc.id if doc.id is not None else 0,
        url=doc.url,
        domain=domain_of(doc.url),
        source=doc.source,
        source_type=doc.source_type,
        depth=doc.depth,
        parent_url=doc.parent_url,
        query=doc.query,
        title=title,
        clean_text=text,
        summary=summarize(text, title),
        language=detect_language(text),
        word_count=word_count,
        published_at=doc.published_at,
        entities=extract_entities(f"{title}\n{text}"),
        tcs_relevance=relevance,
        relevant=relevance >= s.medallion_relevance_threshold and word_count >= MIN_WORDS_RELEVANT,
        risk_signals=signals,
        risk_categories=categories_of(signals),  # type: ignore[arg-type]
        sentiment=sentiment(f"{title}. {text}"),
        risk_score=risk_score,
        extractor=RULES_EXTRACTOR,
    )


def build_silver_record(doc: BronzeDocument, settings: Settings | None = None) -> SilverRecord:
    """Rules record, enriched by the LLM when OpenAI is configured (falls back to rules)."""
    s = settings or get_settings()
    record = build_rules_record(doc, s)
    # Only spend LLM calls on documents that are plausibly about TCS and have real text.
    if (
        not s.use_fake_llm
        and record.word_count >= MIN_WORDS_RELEVANT
        and record.tcs_relevance >= s.medallion_relevance_threshold / 2
    ):
        from tcs_bot.medallion.silver.llm import llm_extract

        enriched = llm_extract(doc, record.clean_text, s, base=record)
        if enriched is not None:
            return enriched
    return record


def _default_repo() -> Any:
    from tcs_bot.medallion.bronze import BronzeRepository

    return BronzeRepository()


def _default_store() -> Any:
    from tcs_bot.medallion.silver.store import SilverStore

    return SilverStore()


def _flush(batch: list[SilverRecord], repo: Any, store: Any, stats: dict[str, int]) -> None:
    if not batch:
        return
    written = store.upsert(batch)
    failed_ids: set[int] = set(getattr(store, "last_failed_ids", set()) or set())
    if not failed_ids and written < len(batch):
        failed_ids = {r.doc_id for r in batch}  # unknown which ones -> retry all next run
        written = 0
    ok_ids = [r.doc_id for r in batch if r.doc_id not in failed_ids]
    stats["indexed"] += len(ok_ids)
    stats["failed"] += len(batch) - len(ok_ids)
    if ok_ids:
        repo.mark_processed(ok_ids, layer="silver")


def run_silver(limit: int | None = None, repo: Any = None, store: Any = None) -> dict[str, int]:
    """Process unprocessed bronze docs -> silver index.

    Returns ``{processed, relevant, irrelevant, indexed, failed}``.
    """
    settings = get_settings()
    repo = repo if repo is not None else _default_repo()
    store = store if store is not None else _default_store()
    store.ensure_index()
    stats = {"processed": 0, "relevant": 0, "irrelevant": 0, "indexed": 0, "failed": 0}
    batch: list[SilverRecord] = []
    for doc in repo.iter_unprocessed(layer="silver", limit=limit):
        try:
            if doc.id is None:
                raise ValueError("bronze document without id")
            record = build_silver_record(doc, settings)
        except Exception as exc:  # noqa: BLE001 - one bad doc never stops the batch
            stats["failed"] += 1
            log.warning("silver.doc_failed", doc_id=doc.id, url=doc.url, error=str(exc)[:200])
            continue
        stats["processed"] += 1
        stats["relevant" if record.relevant else "irrelevant"] += 1
        batch.append(record)
        if len(batch) >= BATCH_SIZE:
            _flush(batch, repo, store, stats)
            batch = []
    _flush(batch, repo, store, stats)
    log.info("silver.run_done", **stats)
    return stats
