"""Gold file artifacts as typed lookups: ``graph.json`` nodes + ``summary.json`` clusters.

The parsed graph is cached per ``(path, mtime)`` so a gold rebuild is picked up automatically.
Missing / unreadable artifacts raise ``NotFoundError('gold_not_built')`` (from ``gold.search``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tcs_bot.errors import NotFoundError
from tcs_bot.medallion.gold import search as gold
from tcs_bot.medallion.gold.pipeline import GRAPH_JSON
from tcs_bot.medallion.models import ClusterSummary
from tcs_bot.medallion.read.models import SEVERITIES, DocumentSummary, GoldNode

if TYPE_CHECKING:
    from tcs_bot.config import Settings

_lock = threading.Lock()
_cache: dict[str, tuple[int, GoldGraph]] = {}


@dataclass(frozen=True)
class GoldGraph:
    """Indexed view of ``graph.json``."""

    meta: dict[str, Any]
    nodes: dict[int, dict[str, Any]]
    merged_into: dict[int, int]  # dropped duplicate doc id -> kept doc id
    labels: dict[str, dict[int, str]] = field(default_factory=dict)  # kind -> id -> label

    @property
    def embedding_model(self) -> str:
        return str(self.meta.get("embedding_model", ""))

    def node_for(self, doc_id: int) -> tuple[int, dict[str, Any]] | None:
        """``(resolved id, node)`` for a gold node or for a document merged into one."""
        if doc_id in self.nodes:
            return doc_id, self.nodes[doc_id]
        kept = self.merged_into.get(doc_id)
        if kept is not None and kept in self.nodes:
            return kept, self.nodes[kept]
        return None

    def members(self, kind: str, cluster_id: int) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n.get(kind) == cluster_id]

    def ids_in(self, kind: str, cluster_id: int) -> list[int]:
        return sorted(n["id"] for n in self.members(kind, cluster_id))


def _build(data: dict[str, Any]) -> GoldGraph:
    nodes = {int(n["id"]): n for n in data.get("nodes", [])}
    merged = {int(m): kept for kept, n in nodes.items() for m in (n.get("merged_doc_ids") or [])}
    labels = {
        "cluster": {int(c["cluster"]): str(c.get("label", "")) for c in data.get("clusters", [])},
        "community": {
            int(c["cluster"]): str(c.get("label", "")) for c in data.get("communities", [])
        },
    }
    return GoldGraph(meta=data.get("meta", {}), nodes=nodes, merged_into=merged, labels=labels)


def load_gold_graph(settings: Settings | None = None) -> GoldGraph:
    """Parsed ``graph.json`` (cached by mtime). Raises ``gold_not_built`` when absent."""
    path = gold.gold_dir(settings) / GRAPH_JSON
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        gold.load_graph(settings)  # raises NotFoundError('gold_not_built') with the run hint
        raise
    key = str(path.resolve())
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    graph = _build(gold.load_graph(settings))
    with _lock:
        _cache[key] = (mtime, graph)
    return graph


def try_load_gold_graph(settings: Settings | None = None) -> GoldGraph | None:
    try:
        return load_gold_graph(settings)
    except NotFoundError:
        return None


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def max_severity(signals: list[dict[str, Any]]) -> str | None:
    ranks = [SEVERITIES.index(s["severity"]) for s in signals if s.get("severity") in SEVERITIES]
    return SEVERITIES[max(ranks)] if ranks else None


def node_summary(n: dict[str, Any]) -> DocumentSummary:
    """A graph node as a list row (``n_signals`` counts at most the 6 signals gold keeps)."""
    signals = n.get("signals") or []
    return DocumentSummary(
        doc_id=int(n["id"]),
        title=n.get("title") or "",
        url=n.get("url") or "",
        domain=n.get("domain") or "",
        source=n.get("source") or "",
        source_type=n.get("source_type") or "other",
        published_at=n.get("published_at"),
        summary=n.get("summary") or "",
        tcs_relevance=float(n.get("relevance") or 0),
        relevant=True,
        risk_score=float(n.get("risk_score") or 0),
        risk_categories=list(n.get("risk_categories") or []),
        sentiment=float(n.get("sentiment") or 0),
        n_signals=len(signals),
        max_severity=max_severity(signals),  # type: ignore[arg-type]
    )


def gold_node(graph: GoldGraph, doc_id: int) -> GoldNode:
    """Gold view of a document: its node, or a stub pointing at the doc it was merged into."""
    found = graph.node_for(doc_id)
    if found is None:
        return GoldNode(in_gold=False)
    resolved, n = found
    if resolved != doc_id:
        return GoldNode(
            in_gold=False,
            merged_into=resolved,
            cluster=n.get("cluster"),
            cluster_label=graph.labels["cluster"].get(n.get("cluster", -1)),
            community=n.get("community"),
        )
    return GoldNode(
        in_gold=True,
        cluster=n.get("cluster"),
        cluster_label=graph.labels["cluster"].get(n.get("cluster", -1)),
        community=n.get("community"),
        degree=n.get("degree"),
        betweenness=n.get("betweenness"),
        eigenvector=n.get("eigenvector"),
        x=n.get("x"),
        y=n.get("y"),
        merged_doc_ids=[int(i) for i in n.get("merged_doc_ids") or []],
        also_seen_in=int(n.get("also_seen_in") or 0),
        merged_sources=list(n.get("merged_sources") or []),
        neighbors=[(int(a), float(b)) for a, b in n.get("neighbors") or []],
    )


def cluster_summaries(summary: dict[str, Any], kind: str) -> list[ClusterSummary]:
    key = "clusters" if kind == "cluster" else "communities"
    return [ClusterSummary.model_validate(c) for c in summary.get(key, [])]
