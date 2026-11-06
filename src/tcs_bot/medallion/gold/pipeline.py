"""Gold pipeline (AC-005.7, AC-005.8): silver -> embeddings -> Qdrant + graph analytics + viz.

Gold is a full rebuild from the relevant silver documents on every run: the Qdrant collections of
the active embedding model are recreated (an LSA model is refit per run, so its vector space
changes) and every artifact in ``gold_dir`` is rewritten.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import networkx as nx
import numpy as np

from tcs_bot.ingestion.chunker import chunk_text
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.gold import analytics as A
from tcs_bot.medallion.gold.embedder import LsaEmbedder, get_gold_embedder, model_slug
from tcs_bot.medallion.gold.vector_store import ChunkRef, GoldVectorStore
from tcs_bot.medallion.gold.viz import render_html
from tcs_bot.medallion.models import SilverRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from tcs_bot.config import Settings
    from tcs_bot.rag.embeddings import Embedder

log = get_logger(__name__)

GRAPH_JSON = "graph.json"
GRAPH_GRAPHML = "graph.graphml"
SUMMARY_JSON = "summary.json"
GRAPH_HTML = "tcs_risk_graph.html"
LSA_PKL = "lsa.pkl"
MAX_CHUNKS_PER_DOC = 50
N_NEIGHBORS_IN_JSON = 8
DATA_SOURCE_NOTE = (
    "Public news & community posts about TCS crawled by the medallion pipeline "
    "(bronze MySQL -> silver Elasticsearch -> gold Qdrant). Risk labels are automated."
)


class SupportsIterRelevant(Protocol):
    """What gold needs from silver (``SilverStore``); tests pass an in-memory fake."""

    def iter_relevant(self, min_relevance: float | None = None) -> Iterator[SilverRecord]: ...


def doc_text(r: SilverRecord) -> str:
    parts = [r.title, r.summary, r.clean_text]
    seen: list[str] = []
    for p in parts:
        p = (p or "").strip()
        if p and not any(p in s for s in seen):
            seen.append(p)
    return "\n\n".join(seen)


def build_chunks(records: Iterable[SilverRecord], settings: Settings) -> list[ChunkRef]:
    out: list[ChunkRef] = []
    for r in records:
        pieces = chunk_text(doc_text(r), settings.rag_chunk_size, settings.rag_chunk_overlap)
        out.extend(ChunkRef(r, i, t) for i, t in enumerate(pieces[:MAX_CHUNKS_PER_DOC]))
    return out


def _dominant_category(r: SilverRecord) -> str | None:
    if not r.risk_signals:
        return r.risk_categories[0] if r.risk_categories else None
    weight = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    score: dict[str, int] = {}
    for s in r.risk_signals:
        score[s.category] = score.get(s.category, 0) + weight[s.severity]
    return max(score, key=lambda c: (score[c], -list(score).index(c)))


def build_graph_json(
    records: list[SilverRecord],
    X: np.ndarray,
    texts: list[str],
    settings: Settings,
    embedding_model: str,
    n_chunks: int,
) -> tuple[dict[str, Any], nx.Graph]:
    """All gold analytics -> the JSON document behind graph.json / summary.json / the HTML."""
    ids = [r.doc_id for r in records]
    S = A.cosine_matrix(X)
    G = A.knn_graph(X, ids, settings.gold_graph_k, settings.gold_graph_min_cosine, S=S)
    communities = A.louvain_communities(G)
    labels, kinfo = A.kmeans_clusters(X)
    cent = A.centrality(G)
    xy = A.project_2d(X)
    clusters = A.cluster_summaries(X, labels, records, texts, S=S)
    comm_labels = np.array([communities.get(i, 0) for i in ids], dtype=int)
    community_summaries = A.cluster_summaries(X, comm_labels, records, texts, S=S)
    nbrs = A.top_neighbors(S, N_NEIGHBORS_IN_JSON)

    nodes: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        nodes.append(
            {
                "id": r.doc_id,
                "title": r.title,
                "url": r.url,
                "domain": r.domain,
                "source": r.source,
                "source_type": r.source_type,
                "published_at": r.published_at.isoformat() if r.published_at else None,
                "risk_score": round(float(r.risk_score), 2),
                "sentiment": round(float(r.sentiment), 3),
                "relevance": round(float(r.tcs_relevance), 3),
                "risk_categories": list(r.risk_categories),
                "dominant_category": _dominant_category(r),
                "cluster": int(labels[i]),
                "community": int(communities.get(r.doc_id, 0)),
                "x": round(float(xy[i, 0]), 5),
                "y": round(float(xy[i, 1]), 5),
                **{k: round(v, 5) for k, v in cent.get(r.doc_id, {}).items()},
                "summary": (r.summary or r.clean_text)[:400],
                "signals": [
                    {
                        "category": s.category,
                        "severity": s.severity,
                        "keyword": s.keyword,
                        "evidence": s.evidence[:300],
                    }
                    for s in r.risk_signals[:6]
                ],
                "entities": {
                    "organizations": r.entities.organizations[:6],
                    "regulators": r.entities.regulators[:6],
                    "places": r.entities.places[:6],
                },
                "neighbors": [[ids[j], round(float(S[i, j]), 4)] for j in nbrs[i].tolist()],
            }
        )
    links = [
        {"source": u, "target": v, "cosine": round(d["cosine"], 4), "mutual": bool(d["mutual"])}
        for u, v, d in G.edges(data=True)
    ]
    synthetic = any(
        "synthetic" in f"{r.extractor} {r.source} {r.domain}".lower() for r in records
    )
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding_model": embedding_model,
        "n_docs": len(records),
        "n_chunks": n_chunks,
        "n_edges": G.number_of_edges(),
        "n_clusters": len(clusters),
        "n_communities": len(set(communities.values())),
        "graph_k": settings.gold_graph_k,
        "graph_min_cosine": settings.gold_graph_min_cosine,
        "kmeans": kinfo,
        "data_source": DATA_SOURCE_NOTE
        + (" SYNTHETIC TEST DATA - not real events." if synthetic else ""),
        "synthetic": synthetic,
    }
    stats = {
        "category_counts": A.category_counts(records),
        "cosine_histogram": A.cosine_histogram(S),
        "category_cooccurrence": A.category_cooccurrence(records),
        "sources": A.source_breakdown(records),
        "timeline": A.monthly_timeline(records),
        "top_risky": A.top_risky(records),
        "top_cross_source_pairs": A.top_cross_source_pairs(S, records),
        "mean_risk_score": round(float(np.mean([r.risk_score for r in records])), 2)
        if records
        else 0.0,
    }
    graph_json = {
        "meta": meta,
        "nodes": nodes,
        "links": links,
        "clusters": [c.model_dump() for c in clusters],
        "communities": [c.model_dump() for c in community_summaries],
        "stats": stats,
    }
    for n in nodes:  # annotate the networkx graph for GraphML
        G.nodes[n["id"]].update({k: v for k, v in n.items() if k not in ("neighbors", "id")})
    return graph_json, G


def write_graphml(G: nx.Graph, path: Path) -> Path:
    """GraphML only takes scalars: lists/dicts -> JSON strings, None -> ''."""

    def clean(d: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if v is None:
                out[k] = ""
            elif isinstance(v, (list, dict, tuple)):
                out[k] = json.dumps(v, ensure_ascii=False)
            else:
                out[k] = v
        return out

    H = nx.Graph()
    H.add_nodes_from((str(n), clean(d)) for n, d in G.nodes(data=True))
    H.add_edges_from(
        (str(u), str(v), clean({k: x for k, x in d.items() if k != "distance"}))
        for u, v, d in G.edges(data=True)
    )
    nx.write_graphml(H, path)
    return path


def _write_json(path: Path, data: Any) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def run_gold(
    store: SupportsIterRelevant | None = None,
    vector_store: GoldVectorStore | None = None,
    settings: Settings | None = None,
    out_dir: str | Path | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Build gold from relevant silver docs. Returns counts, artifact paths and Qdrant status.

    Qdrant being down degrades: file artifacts are still written, ``result["qdrant"]`` says why.
    """
    if settings is None:
        from tcs_bot.config import get_settings

        settings = get_settings()
    if store is None:
        from tcs_bot.medallion.silver.store import SilverStore

        store = SilverStore()
    out = Path(out_dir or settings.gold_dir)
    out.mkdir(parents=True, exist_ok=True)

    unique: dict[int, SilverRecord] = {}
    for rec in store.iter_relevant():
        unique[rec.doc_id] = rec
    records = [unique[k] for k in sorted(unique)]
    texts = [doc_text(r) for r in records]
    chunks = build_chunks(records, settings)

    if embedder is None:
        embedder = get_gold_embedder(settings, texts)
    slug = model_slug(embedder)
    X = embedder.embed(texts)
    Xc = embedder.embed([f"{c.record.title}\n{c.text}" for c in chunks])

    graph_json, G = build_graph_json(records, X, texts, settings, embedder.model, len(chunks))

    qdrant_status = "skipped: no documents"
    upserted = {"docs": 0, "chunks": 0}
    if records:
        vs = (vector_store or GoldVectorStore(url=settings.qdrant_url)).with_model(slug)
        extra = {
            n["id"]: {"cluster": n["cluster"], "community": n["community"]}
            for n in graph_json["nodes"]
        }
        try:
            vs.ensure(embedder.dim, recreate=True)
            upserted["docs"] = vs.upsert_docs(records, X, extra)
            upserted["chunks"] = vs.upsert_chunks(chunks, Xc, extra)
            qdrant_status = "ok"
            graph_json["meta"]["qdrant_collections"] = [vs.docs_collection, vs.chunks_collection]
        except Exception as exc:  # degrade: artifacts are still useful offline
            qdrant_status = f"unavailable: {type(exc).__name__}: {exc}"[:300]
            log.error("gold.qdrant_failed", error_type=type(exc).__name__)
    graph_json["meta"]["qdrant"] = qdrant_status

    paths: dict[str, str] = {}
    if isinstance(embedder, LsaEmbedder):
        paths["lsa"] = str(embedder.save(out / LSA_PKL))
    else:
        (out / LSA_PKL).unlink(missing_ok=True)  # stale LSA model must not serve queries
    summary = {k: v for k, v in graph_json.items() if k not in ("nodes", "links")}
    paths["graph_json"] = str(_write_json(out / GRAPH_JSON, graph_json))
    paths["summary_json"] = str(_write_json(out / SUMMARY_JSON, summary))
    paths["graphml"] = str(write_graphml(G, out / GRAPH_GRAPHML))
    paths["html"] = str(render_html(graph_json, out / GRAPH_HTML))

    result = {
        "docs": len(records),
        "chunks": len(chunks),
        "edges": graph_json["meta"]["n_edges"],
        "clusters": graph_json["meta"]["n_clusters"],
        "communities": graph_json["meta"]["n_communities"],
        "embedding_model": embedder.model,
        "qdrant": qdrant_status,
        "upserted": upserted,
        "paths": paths,
    }
    log.info("gold.done", **{k: v for k, v in result.items() if k != "paths"})
    return result
