"""Elasticsearch DSL builders / parsers for the silver read side (pure functions, no I/O)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from elasticsearch import exceptions as es_exc

from tcs_bot.errors import DependencyUnavailable, NotFoundError
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import RISK_CATEGORIES
from tcs_bot.medallion.read.models import (
    SEVERITIES,
    CountBucket,
    DocumentFilters,
    DocumentSummary,
    Page,
    RiskBucket,
)

log = get_logger(__name__)

#: ``clean_text`` is the heavy field; list rows never need it
LIST_EXCLUDES = ["clean_text", "entities"]
_TEXT_FIELDS = ["title^3", "summary^2", "clean_text"]
_SIZE = {"category": 20, "source": 50, "source_type": 10, "severity": 10}


def guard[T](index: str, fn: Callable[[], T]) -> T:
    """Run an ES call on the silver index; map failures to ``AppError`` subclasses."""
    try:
        return fn()
    except es_exc.NotFoundError as exc:
        raise NotFoundError(
            f"silver index {index} not found - run `tcs-bot medallion silver`",
            code="silver_not_built",
        ) from exc
    except (es_exc.TransportError, es_exc.ApiError) as exc:
        log.error("silver.read_failed", index=index, error_type=type(exc).__name__)
        raise DependencyUnavailable(
            f"Elasticsearch silver read failed ({type(exc).__name__}); is Elasticsearch up?",
            code="silver_unavailable",
        ) from exc


def severities_at_least(min_severity: str) -> list[str]:
    return list(SEVERITIES[SEVERITIES.index(min_severity) :])


def build_query(f: DocumentFilters, doc_ids: list[int] | None = None) -> dict[str, Any]:
    """``bool`` query for the filters. ``doc_ids`` restricts to gold cluster / community members."""
    filters: list[dict[str, Any]] = []
    if f.relevant is not None:
        filters.append({"term": {"relevant": f.relevant}})
    for field, values in (
        ("risk_categories", f.category),
        ("source", f.source),
        ("source_type", f.source_type),
    ):
        if values:
            filters.append({"terms": {field: list(values)}})
    if f.min_severity:
        filters.append(
            {
                "nested": {
                    "path": "risk_signals",
                    "query": {
                        "terms": {"risk_signals.severity": severities_at_least(f.min_severity)}
                    },
                }
            }
        )
    if f.min_risk_score is not None:
        filters.append({"range": {"risk_score": {"gte": f.min_risk_score}}})
    bounds: dict[str, str] = {}
    if f.published_from:
        bounds["gte"] = f.published_from.isoformat()
    if f.published_to:
        bounds["lte"] = f.published_to.isoformat()
    if bounds:
        filters.append({"range": {"published_at": bounds}})
    if doc_ids is not None:
        filters.append({"terms": {"doc_id": list(doc_ids)}})
    must: list[dict[str, Any]] = []
    if f.q:
        must.append({"multi_match": {"query": f.q, "fields": _TEXT_FIELDS, "operator": "and"}})
    if not filters and not must:
        return {"match_all": {}}
    return {"bool": {"filter": filters, "must": must}}


def build_sort(page: Page) -> list[dict[str, Any]]:
    return [
        {page.sort: {"order": page.order, "missing": "_last"}},
        {"doc_id": {"order": "asc"}},  # stable pagination on ties
    ]


def summary_from_source(src: dict[str, Any]) -> DocumentSummary:
    signals = src.get("risk_signals") or []
    ranks = [SEVERITIES.index(s["severity"]) for s in signals if s.get("severity") in SEVERITIES]
    return DocumentSummary(
        doc_id=int(src["doc_id"]),
        title=src.get("title") or "",
        url=src.get("url") or "",
        domain=src.get("domain") or "",
        source=src.get("source") or "",
        source_type=src.get("source_type") or "other",
        published_at=src.get("published_at"),
        summary=src.get("summary") or "",
        tcs_relevance=float(src.get("tcs_relevance") or 0),
        relevant=bool(src.get("relevant", False)),
        risk_score=float(src.get("risk_score") or 0),
        risk_categories=list(src.get("risk_categories") or []),
        sentiment=float(src.get("sentiment") or 0),
        n_signals=len(signals),
        max_severity=SEVERITIES[max(ranks)] if ranks else None,  # type: ignore[arg-type]
        language=src.get("language"),
        word_count=src.get("word_count"),
    )


def total_hits(res: dict[str, Any]) -> int:
    total = res["hits"]["total"]
    return int(total["value"] if isinstance(total, dict) else total)


# ------------------------------------------------------------------ aggregations
def _risk_aggs() -> dict[str, Any]:
    return {
        "avg_risk": {"avg": {"field": "risk_score"}},
        "max_risk": {"max": {"field": "risk_score"}},
    }


def _terms(field: str, size: int) -> dict[str, Any]:
    return {"terms": {"field": field, "size": size}, "aggs": _risk_aggs()}


def category_aggs() -> dict[str, Any]:
    return {"by_category": _terms("risk_categories", _SIZE["category"])}


def overview_aggs() -> dict[str, Any]:
    return {
        **_risk_aggs(),
        **category_aggs(),
        "by_source": _terms("source", _SIZE["source"]),
        "by_source_type": _terms("source_type", _SIZE["source_type"]),
        "signals": {
            "nested": {"path": "risk_signals"},
            "aggs": {
                "by_severity": {"terms": {"field": "risk_signals.severity", "size": 10}},
            },
        },
        "by_month": {
            "date_histogram": {
                "field": "published_at",
                "calendar_interval": "month",
                "format": "yyyy-MM",
                "min_doc_count": 1,
            },
            "aggs": _risk_aggs(),
        },
    }


def agg_value(agg: dict[str, Any], name: str) -> float:
    value = (agg.get(name) or {}).get("value")
    return round(float(value), 2) if value is not None else 0.0


def risk_buckets(agg: dict[str, Any] | None) -> list[RiskBucket]:
    return [
        RiskBucket(
            key=str(b.get("key_as_string", b["key"])),
            count=int(b["doc_count"]),
            avg_risk_score=agg_value(b, "avg_risk"),
            max_risk_score=agg_value(b, "max_risk"),
        )
        for b in (agg or {}).get("buckets", [])
    ]


def count_buckets(agg: dict[str, Any] | None) -> list[CountBucket]:
    return [
        CountBucket(key=str(b["key"]), count=int(b["doc_count"]))
        for b in (agg or {}).get("buckets", [])
    ]


def all_categories(buckets: list[RiskBucket]) -> list[RiskBucket]:
    """Every risk category (zero counts included), most documents first."""
    have = {b.key: b for b in buckets}
    full = [have.get(c) or RiskBucket(key=c, count=0) for c in RISK_CATEGORIES]
    return sorted(full, key=lambda b: (-b.count, b.key))
