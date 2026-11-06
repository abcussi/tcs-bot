"""``TcsData``: one typed read facade over bronze (MySQL), silver (ES) and gold (ES + files).

Join key: ``bronze_documents.id == silver doc_id == gold doc_id``. The facade never writes, never
logs document bodies, and maps infrastructure failures to ``AppError`` subclasses:

* ``document_not_found`` (404): unknown / not-yet-silver ``doc_id``
* ``gold_not_built`` (404): gold files or ES indices missing
* ``document_not_in_gold`` (404): the document has no gold vector (silver only)
* ``cluster_not_found`` (404): unknown cluster / community id
* ``silver_not_built`` (404): the silver index is missing
* ``silver_unavailable`` (503): Elasticsearch down (silver reads)
* ``vector_store_unavailable`` (503): Elasticsearch down (gold vectors)
* ``bronze_unavailable`` (503): MySQL down (lineage)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from elasticsearch import exceptions as es_exc

from tcs_bot.errors import DependencyUnavailable, NotFoundError
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.gold import search as gold
from tcs_bot.medallion.gold.embedder import model_slug
from tcs_bot.medallion.gold.vector_store import VECTOR_FIELD, GoldVectorStore
from tcs_bot.medallion.models import GoldHit
from tcs_bot.medallion.read import gold_graph as gg
from tcs_bot.medallion.read import queries as q
from tcs_bot.medallion.read.bronze_reader import BronzeReader, MysqlBronzeReader
from tcs_bot.medallion.read.models import (
    SEVERITIES,
    CategoryBreakdown,
    ClusterDetail,
    ClusterKind,
    DocumentDetail,
    DocumentFilters,
    DocumentPage,
    Page,
    RiskOverview,
    SilverView,
    SimilarDocuments,
)
from tcs_bot.medallion.silver.store import SilverStore

if TYPE_CHECKING:
    from tcs_bot.config import Settings
    from tcs_bot.medallion.models import ClusterSummary

log = get_logger(__name__)

MAX_SIMILAR = 50
MAX_SEARCH = 100


def _index_missing(exc: es_exc.ApiError) -> bool:
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    return isinstance(err, dict) and err.get("type") == "index_not_found_exception"


class TcsData:
    """Typed reads over the medallion layers. All collaborators are lazy and injectable."""

    def __init__(
        self,
        silver: SilverStore | None = None,
        vector_store: GoldVectorStore | None = None,
        bronze: BronzeReader | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._silver = silver
        self._vector_store = vector_store
        self._bronze: BronzeReader = bronze if bronze is not None else MysqlBronzeReader()
        self._settings = settings

    # ------------------------------------------------------------------ plumbing
    @property
    def settings(self) -> Settings:
        if self._settings is not None:
            return self._settings
        from tcs_bot.config import get_settings

        return get_settings()

    @property
    def silver(self) -> SilverStore:
        if self._silver is None:
            self._silver = SilverStore()
        return self._silver

    @property
    def vector_store(self) -> GoldVectorStore:
        return self._vector_store or gold.get_vector_store(self.settings)

    def _search_silver(self, **kwargs: Any) -> dict[str, Any]:
        store = self.silver
        return q.guard(store.index, lambda: dict(store.es.search(index=store.index, **kwargs)))

    # ------------------------------------------------------------------ documents
    def list_documents(
        self, filters: DocumentFilters | None = None, page: Page | None = None
    ) -> DocumentPage:
        """Silver-backed document list: filters ANDed, sorted and paginated, with the total."""
        f, p = filters or DocumentFilters(), page or Page()
        doc_ids = self._cluster_member_ids(f)
        if doc_ids is not None and not doc_ids:
            return DocumentPage(total=0, limit=p.limit, offset=p.offset, items=[])
        res = self._search_silver(
            query=q.build_query(f, doc_ids),
            sort=q.build_sort(p),
            from_=p.offset,
            size=p.limit,
            track_total_hits=True,
            source_excludes=q.LIST_EXCLUDES,
        )
        items = [q.summary_from_source(h["_source"]) for h in res["hits"]["hits"]]
        return DocumentPage(total=q.total_hits(res), limit=p.limit, offset=p.offset, items=items)

    def _cluster_member_ids(self, f: DocumentFilters) -> list[int] | None:
        """Doc ids of the requested gold cluster / community (both: their intersection)."""
        if f.cluster is None and f.community is None:
            return None
        graph = gg.load_gold_graph(self.settings)
        ids: set[int] | None = None
        for kind, value in (("cluster", f.cluster), ("community", f.community)):
            if value is not None:
                members = set(graph.ids_in(kind, value))
                ids = members if ids is None else ids & members
        return sorted(ids or [])

    def get_document(
        self, doc_id: int, *, include_raw: bool = False, include_text: bool = False
    ) -> DocumentDetail:
        """Silver record + bronze lineage + gold node for one id.

        ``include_text`` returns silver ``clean_text``; ``include_raw`` returns the bronze
        ``raw_content`` (can be MBs; never logged). Gold is ``None`` when it is not built.
        """
        store = self.silver
        record = q.guard(store.index, lambda: store.get(doc_id))
        if record is None:
            raise NotFoundError(
                f"document {doc_id} not found in silver (bronze id == silver doc_id)",
                code="document_not_found",
            )
        data = record.model_dump()
        if not include_text:
            data["clean_text"] = None
        graph = gg.try_load_gold_graph(self.settings)
        return DocumentDetail(
            doc_id=doc_id,
            silver=SilverView(**data),
            bronze=self._bronze.get(doc_id, include_raw=include_raw),
            gold=gg.gold_node(graph, doc_id) if graph else None,
        )

    # ------------------------------------------------------------------ gold: similarity / search
    def _gold_vector(self, vs: GoldVectorStore, doc_id: int) -> list[float] | None:
        try:
            res = vs.es.get(index=vs.docs_collection, id=str(doc_id))
        except es_exc.NotFoundError as exc:
            if _index_missing(exc):
                raise NotFoundError(
                    f"gold index {vs.docs_collection} not found - {gold.RUN_HINT}",
                    code="gold_not_built",
                ) from exc
            return None
        except (es_exc.TransportError, es_exc.ApiError) as exc:
            log.error("gold.vector_read_failed", error_type=type(exc).__name__)
            raise DependencyUnavailable(
                f"Elasticsearch vector read failed ({type(exc).__name__}); is Elasticsearch up?",
                code="vector_store_unavailable",
            ) from exc
        vec = (res.get("_source") or {}).get(VECTOR_FIELD)
        return list(vec) if vec else None

    def similar_documents(self, doc_id: int, k: int = 10) -> SimilarDocuments:
        """Nearest gold documents to ``doc_id`` by cosine of the stored vector (self excluded).

        A document folded into another by gold dedupe uses the kept document's vector
        (``resolved_doc_id``).
        """
        k = max(1, min(int(k), MAX_SIMILAR))
        graph = gg.load_gold_graph(self.settings)
        if not graph.embedding_model:
            raise NotFoundError(
                f"gold graph has no embedding model - {gold.RUN_HINT}", code="gold_not_built"
            )
        slug = model_slug(graph.embedding_model)
        vs = self.vector_store.with_model(slug)
        resolved = doc_id
        vec = self._gold_vector(vs, doc_id)
        if vec is None:
            found = graph.node_for(doc_id)
            if found is not None and found[0] != doc_id:
                resolved = found[0]
                vec = self._gold_vector(vs, resolved)
        if vec is None:
            raise NotFoundError(
                f"document {doc_id} has no gold vector (not relevant, or not built yet)",
                code="document_not_in_gold",
            )
        hits = vs.search(vec, k=k + 1, level="docs")
        neighbours = [h for h in hits if h.doc_id != resolved][:k]
        return SimilarDocuments(
            doc_id=doc_id, resolved_doc_id=resolved, model=slug, neighbours=neighbours
        )

    def search(
        self, query: str, k: int = 10, filters: DocumentFilters | None = None
    ) -> list[GoldHit]:
        """Semantic search over gold chunks (best chunk per document).

        Uses the ``category``, ``min_risk_score``, ``source``, ``source_type``, ``cluster`` and
        ``community`` filters; the others (dates, ``q``, ``min_severity``, ``relevant``) only
        apply to :meth:`list_documents`.
        """
        query = (query or "").strip()
        k = max(1, min(int(k), MAX_SEARCH))
        if not query:
            return []
        f = filters or DocumentFilters()
        embedder = gold.query_embedder(self.settings)
        vector = embedder.embed([query])[0]
        vs = self.vector_store.with_model(model_slug(embedder))
        hits = vs.search(
            vector,
            k=k * 4,
            categories=list(f.category) or None,
            min_risk=f.min_risk_score,
            level="chunks",
            sources=list(f.source) or None,
            source_types=list(f.source_type) or None,
            cluster=f.cluster,
            community=f.community,
        )
        best: dict[int, GoldHit] = {}
        for h in hits:  # sorted by score desc
            best.setdefault(h.doc_id, h)
        return list(best.values())[:k]

    # ------------------------------------------------------------------ analytics
    def category_breakdown(self, relevant_only: bool = True) -> CategoryBreakdown:
        """Documents / mean / max risk per risk category (zero-count categories included)."""
        res = self._search_silver(
            query=q.build_query(DocumentFilters(relevant=True if relevant_only else None)),
            size=0,
            track_total_hits=True,
            aggs=q.category_aggs(),
        )
        aggs = res.get("aggregations", {})
        return CategoryBreakdown(
            relevant_only=relevant_only,
            total_documents=q.total_hits(res),
            categories=q.all_categories(q.risk_buckets(aggs.get("by_category"))),
        )

    def risk_overview(self, relevant_only: bool = True, top_n: int = 5) -> RiskOverview:
        """Counts and mean / max risk by category, severity, source, source type and month."""
        top_n = max(1, min(int(top_n), 20))
        res = self._search_silver(
            query=q.build_query(DocumentFilters(relevant=True if relevant_only else None)),
            sort=q.build_sort(Page(limit=top_n)),
            size=top_n,
            track_total_hits=True,
            source_excludes=q.LIST_EXCLUDES,
            aggs=q.overview_aggs(),
        )
        aggs = res.get("aggregations", {})
        severity = q.count_buckets((aggs.get("signals") or {}).get("by_severity"))
        severity.sort(key=lambda b: -SEVERITIES.index(b.key) if b.key in SEVERITIES else 1)
        return RiskOverview(
            relevant_only=relevant_only,
            total_documents=q.total_hits(res),
            mean_risk_score=q.agg_value(aggs, "avg_risk"),
            max_risk_score=q.agg_value(aggs, "max_risk"),
            by_category=q.risk_buckets(aggs.get("by_category")),
            by_severity=severity,
            by_source=q.risk_buckets(aggs.get("by_source")),
            by_source_type=q.risk_buckets(aggs.get("by_source_type")),
            by_month=q.risk_buckets(aggs.get("by_month")),
            top_documents=[q.summary_from_source(h["_source"]) for h in res["hits"]["hits"]],
        )

    # ------------------------------------------------------------------ clusters
    def list_clusters(self, kind: ClusterKind = "cluster") -> list[ClusterSummary]:
        """Gold KMeans clusters (or Louvain communities) from ``summary.json``."""
        return gg.cluster_summaries(gold.load_summary(self.settings), kind)

    def get_cluster(self, cluster_id: int, kind: ClusterKind = "cluster") -> ClusterDetail:
        """One cluster / community: its summary and member documents (gold graph nodes)."""
        summary = next((c for c in self.list_clusters(kind) if c.cluster == cluster_id), None)
        if summary is None:
            raise NotFoundError(f"{kind} {cluster_id} not found", code="cluster_not_found")
        graph = gg.load_gold_graph(self.settings)
        members = sorted(
            (gg.node_summary(n) for n in graph.members(kind, cluster_id)),
            key=lambda d: (-d.risk_score, d.doc_id),
        )
        return ClusterDetail(kind=kind, summary=summary, members=members)


# ---------------------------------------------------------------------- process-wide instance
_lock = threading.Lock()
_instance: TcsData | None = None


def get_tcs_data() -> TcsData:
    """Process-wide facade (FastAPI dependency and MCP tools); tests inject via ``set_tcs_data``."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = TcsData()
        return _instance


def set_tcs_data(data: TcsData | None) -> None:
    global _instance
    with _lock:
        _instance = data
