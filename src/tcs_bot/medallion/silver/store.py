"""Silver layer persistence: Elasticsearch index ``silver-tcs-risk`` (AC-005.5 / AC-005.6).

``SilverStore(es=None, index=None)`` - defaults to the process-wide client
(:func:`tcs_bot.search.elastic.get_es`) and ``settings.silver_index``. Documents are keyed by the
bronze id (``_id = doc_id``) so re-processing is idempotent.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from elasticsearch import exceptions as es_exc
from elasticsearch import helpers

from tcs_bot.config import get_settings
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import SilverRecord

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

log = get_logger(__name__)

_KW = {"type": "keyword"}
_KW_ARRAY = {"type": "keyword", "ignore_above": 256}

SILVER_SETTINGS: dict[str, Any] = {"number_of_shards": 1, "number_of_replicas": 0}
SILVER_MAPPING: dict[str, Any] = {
    "dynamic": False,  # unknown fields stay in _source but are not indexed
    "properties": {
        "doc_id": {"type": "long"},
        "url": _KW,
        "domain": _KW,
        "source": _KW,
        "source_type": _KW,
        "depth": {"type": "integer"},
        "parent_url": _KW,
        "query": _KW,
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
        "clean_text": {"type": "text", "analyzer": "english"},
        "summary": {"type": "text", "analyzer": "english"},
        "language": _KW,
        "word_count": {"type": "integer"},
        "published_at": {"type": "date"},
        "processed_at": {"type": "date"},
        "entities": {
            "type": "object",
            "properties": {
                "organizations": _KW_ARRAY,
                "people": _KW_ARRAY,
                "regulators": _KW_ARRAY,
                "places": _KW_ARRAY,
                "money": _KW_ARRAY,
            },
        },
        "tcs_relevance": {"type": "float"},
        "relevant": {"type": "boolean"},
        "risk_signals": {
            "type": "nested",
            "properties": {
                "category": _KW,
                "severity": _KW,
                "keyword": _KW,
                "evidence": {"type": "text"},
            },
        },
        "risk_categories": _KW,
        "sentiment": {"type": "float"},
        "risk_score": {"type": "float"},
        "extractor": _KW,
    },
}


class SilverStore:
    """CRUD + analytics over the silver index.

    ``upsert`` returns the number of documents written and records per-document failures in
    :attr:`last_failed_ids` (all ids when ES is unreachable), so the pipeline only marks bronze
    rows processed once they really are in silver.
    """

    def __init__(self, es: Elasticsearch | None = None, index: str | None = None) -> None:
        if es is None:
            from tcs_bot.search.elastic import get_es

            es = get_es()
        self.es = es
        self.index = index or get_settings().silver_index
        self.last_failed_ids: set[int] = set()

    # ------------------------------------------------------------------ admin
    def ensure_index(self) -> bool:
        """Create the index with :data:`SILVER_MAPPING` if absent. Raises if ES is unreachable."""
        if self.es.indices.exists(index=self.index):
            return False
        try:
            self.es.indices.create(
                index=self.index, mappings=SILVER_MAPPING, settings=SILVER_SETTINGS
            )
        except es_exc.BadRequestError as exc:
            if getattr(exc, "error", "") == "resource_already_exists_exception":
                return False
            raise
        log.info("silver.index.created", index=self.index)
        return True

    def delete_index(self) -> None:
        """Drop the index (tests / full rebuilds)."""
        self.es.indices.delete(index=self.index, ignore_unavailable=True)

    # ------------------------------------------------------------------ writes
    @staticmethod
    def to_source(record: SilverRecord) -> dict[str, Any]:
        return record.model_dump(mode="json")

    def upsert(self, records: Iterable[SilverRecord], refresh: bool = False) -> int:
        """Bulk index (``_id = doc_id``, full replace); returns the number written."""
        records = list(records)
        self.last_failed_ids = set()
        if not records:
            return 0
        actions = [
            {"_index": self.index, "_id": str(r.doc_id), "_source": self.to_source(r)}
            for r in records
        ]
        try:
            ok, errors = helpers.bulk(
                self.es,
                actions,
                raise_on_error=False,
                raise_on_exception=False,
                refresh="wait_for" if refresh else False,
            )
        except (es_exc.TransportError, es_exc.ApiError, helpers.BulkIndexError) as exc:
            log.warning("silver.upsert.failed", index=self.index, error=str(exc)[:200])
            self.last_failed_ids = {r.doc_id for r in records}
            return 0
        if isinstance(errors, list):
            for err in errors:
                info = next(iter(err.values()), {}) if isinstance(err, dict) else {}
                try:
                    self.last_failed_ids.add(int(info.get("_id")))
                except (TypeError, ValueError):
                    continue
            if errors:
                log.warning(
                    "silver.upsert.partial",
                    index=self.index,
                    failed=len(errors),
                    first=str(errors[0])[:300],
                )
        return int(ok)

    # ------------------------------------------------------------------ reads
    def get(self, doc_id: int) -> SilverRecord | None:
        try:
            res = self.es.get(index=self.index, id=str(doc_id))
        except es_exc.NotFoundError:
            return None
        return SilverRecord.model_validate(res["_source"])

    def iter_relevant(self, min_relevance: float | None = None) -> Iterator[SilverRecord]:
        """Scan every ``relevant=true`` record (optionally ``tcs_relevance >= min_relevance``)."""
        filters: list[dict[str, Any]] = [{"term": {"relevant": True}}]
        if min_relevance is not None:
            filters.append({"range": {"tcs_relevance": {"gte": min_relevance}}})
        query = {"query": {"bool": {"filter": filters}}, "sort": ["_doc"]}
        for hit in helpers.scan(self.es, index=self.index, query=query, preserve_order=False):
            try:
                yield SilverRecord.model_validate(hit["_source"])
            except ValueError as exc:  # pydantic ValidationError subclasses ValueError
                log.warning("silver.iter.invalid_record", id=hit.get("_id"), error=str(exc)[:200])

    def count(self, relevant_only: bool = False) -> int:
        if relevant_only:
            return int(self.es.count(index=self.index, query={"term": {"relevant": True}})["count"])
        return int(self.es.count(index=self.index)["count"])

    def stats(self) -> dict[str, Any]:
        """Totals plus aggregations by category, source and signal severity."""
        res = self.es.search(
            index=self.index,
            size=0,
            track_total_hits=True,
            aggs={
                "relevant": {"filter": {"term": {"relevant": True}}},
                "by_category": {"terms": {"field": "risk_categories", "size": 20}},
                "by_source": {"terms": {"field": "source", "size": 50}},
                "by_source_type": {"terms": {"field": "source_type", "size": 10}},
                "mean_risk_score": {"avg": {"field": "risk_score"}},
                "signals": {
                    "nested": {"path": "risk_signals"},
                    "aggs": {
                        "by_severity": {"terms": {"field": "risk_signals.severity", "size": 10}}
                    },
                },
            },
        )
        aggs = res.get("aggregations", {})

        def buckets(agg: dict[str, Any]) -> dict[str, int]:
            return {str(b["key"]): int(b["doc_count"]) for b in agg.get("buckets", [])}

        total = res["hits"]["total"]
        mean = aggs.get("mean_risk_score", {}).get("value")
        return {
            "total": int(total["value"] if isinstance(total, dict) else total),
            "relevant": int(aggs.get("relevant", {}).get("doc_count", 0)),
            "by_category": buckets(aggs.get("by_category", {})),
            "by_source": buckets(aggs.get("by_source", {})),
            "by_source_type": buckets(aggs.get("by_source_type", {})),
            "by_severity": buckets(aggs.get("signals", {}).get("by_severity", {})),
            "mean_risk_score": round(float(mean), 2) if mean is not None else 0.0,
        }
