"""In-memory stand-in for the slice of the Elasticsearch client the silver READ side uses.

Supports exactly what ``tcs_bot.medallion.read`` issues: ``get``, ``search`` with ``match_all`` /
``bool`` (filter + must) / ``term`` / ``terms`` / ``range`` / ``nested`` / ``multi_match`` queries,
multi-key ``sort`` with ``missing: _last``, ``from_`` / ``size`` / ``source_excludes`` and the
aggregations ``terms``, ``avg``, ``max``, ``nested`` and monthly ``date_histogram``. Offline tests
only; SYNTHETIC FIXTURE data goes in, nothing touches a network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from elasticsearch import exceptions as es_exc

from .fake_es import api_error


def _values(src: dict[str, Any], field: str) -> list[Any]:
    """All values at a (possibly dotted / array) path; ``[]`` when absent."""
    if field in src:  # flattened nested pseudo-document ("risk_signals.severity": ...)
        v = src[field]
        return [x for x in (v if isinstance(v, list) else [v]) if x is not None]
    cur: list[Any] = [src]
    for part in field.split("."):
        nxt: list[Any] = []
        for item in cur:
            if isinstance(item, dict) and part in item:
                v = item[part]
                nxt.extend(v if isinstance(v, list) else [v])
        cur = nxt
    return [v for v in cur if v is not None]


def _as_dt(value: Any) -> datetime:
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)  # ES reads naive dates as UTC


def _cmp(value: Any, bound: Any) -> tuple[Any, Any]:
    if isinstance(value, str) and isinstance(bound, str):
        return _as_dt(value), _as_dt(bound)
    return value, bound


def matches(src: dict[str, Any], clause: dict[str, Any]) -> bool:
    if "match_all" in clause:
        return True
    if "bool" in clause:
        b = clause["bool"]
        return all(matches(src, c) for c in b.get("filter", []) + b.get("must", []))
    if "term" in clause:
        ((field, wanted),) = clause["term"].items()
        return wanted in _values(src, field)
    if "terms" in clause:
        ((field, wanted),) = clause["terms"].items()
        return any(v in wanted for v in _values(src, field))
    if "range" in clause:
        ((field, bounds),) = clause["range"].items()
        ops = {
            "gte": lambda a, b: a >= b,
            "gt": lambda a, b: a > b,
            "lte": lambda a, b: a <= b,
            "lt": lambda a, b: a < b,
        }
        return any(
            all(ops[op](*_cmp(v, bound)) for op, bound in bounds.items())
            for v in _values(src, field)
        )
    if "nested" in clause:
        path = clause["nested"]["path"]
        inner = clause["nested"]["query"]
        return any(
            matches({f"{path}.{k}": v for k, v in sub.items()}, inner) for sub in src.get(path, [])
        )
    if "multi_match" in clause:
        mm = clause["multi_match"]
        text = " ".join(str(v).lower() for f in mm["fields"] for v in _values(src, f.split("^")[0]))
        return all(tok in text for tok in str(mm["query"]).lower().split())
    raise NotImplementedError(f"fake silver ES does not support {clause!r}")


def _agg(spec: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
    subs = spec.get("aggs", {})

    def run_subs(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {name: _agg(s, group) for name, s in subs.items()}

    if "avg" in spec or "max" in spec:
        kind = "avg" if "avg" in spec else "max"
        vals = [v for d in docs for v in _values(d, spec[kind]["field"])]
        if not vals:
            return {"value": None}
        return {"value": sum(vals) / len(vals) if kind == "avg" else max(vals)}
    if "nested" in spec:
        path = spec["nested"]["path"]
        subdocs = [{f"{path}.{k}": v for k, v in s.items()} for d in docs for s in d.get(path, [])]
        return {"doc_count": len(subdocs), **run_subs(subdocs)}
    if "terms" in spec:
        field, size = spec["terms"]["field"], spec["terms"]["size"]
        groups: dict[Any, list[dict[str, Any]]] = {}
        for d in docs:
            for v in set(_values(d, field)):
                groups.setdefault(v, []).append(d)
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))[:size]
        return {"buckets": [{"key": k, "doc_count": len(g), **run_subs(g)} for k, g in ordered]}
    if "date_histogram" in spec:
        field = spec["date_histogram"]["field"]
        assert spec["date_histogram"]["calendar_interval"] == "month"
        by_month: dict[str, list[dict[str, Any]]] = {}
        for d in docs:
            for v in _values(d, field):
                by_month.setdefault(_as_dt(v).strftime("%Y-%m"), []).append(d)
        return {
            "buckets": [
                {"key_as_string": m, "key": m, "doc_count": len(g), **run_subs(g)}
                for m, g in sorted(by_month.items())
            ]
        }
    raise NotImplementedError(f"fake silver ES does not support agg {spec!r}")


class FakeSilverES:
    """Silver index double: ``put`` documents, then read them like the real client does."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def put(self, *sources: dict[str, Any]) -> FakeSilverES:
        for src in sources:
            self.docs[str(src["doc_id"])] = src
        return self

    def get(self, index: str, id: str, **_: Any) -> dict[str, Any]:  # noqa: A002 - ES API name
        if str(id) not in self.docs:
            raise api_error(es_exc.NotFoundError, "resource_not_found", 404)
        return {"_id": str(id), "found": True, "_source": self.docs[str(id)]}

    def search(
        self,
        index: str,
        query: dict[str, Any] | None = None,
        sort: list[dict[str, Any]] | None = None,
        from_: int = 0,
        size: int = 10,
        source_excludes: list[str] | None = None,
        aggs: dict[str, Any] | None = None,
        **kw: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "query": query,
                "sort": sort,
                "from_": from_,
                "size": size,
                "source_excludes": source_excludes,
                **kw,
            }
        )
        found = [s for s in self.docs.values() if matches(s, query or {"match_all": {}})]
        for key in reversed(sort or []):
            ((field, opts),) = key.items()
            order = opts["order"] if isinstance(opts, dict) else opts
            present = [s for s in found if _values(s, field)]
            absent = [s for s in found if not _values(s, field)]
            present.sort(
                key=lambda s, f=field: _cmp(_values(s, f)[0], _values(s, f)[0])[0],
                reverse=order == "desc",
            )
            found = present + absent  # ``missing: _last`` for both directions
        excl = set(source_excludes or [])
        hits = [
            {"_id": str(s["doc_id"]), "_source": {k: v for k, v in s.items() if k not in excl}}
            for s in found[from_ : from_ + size]
        ]
        res: dict[str, Any] = {
            "hits": {"total": {"value": len(found), "relation": "eq"}, "hits": hits}
        }
        if aggs:
            res["aggregations"] = {name: _agg(spec, found) for name, spec in aggs.items()}
        return res


class DownSilverES:
    """Elasticsearch unreachable (transport ConnectionError on every call)."""

    def __getattr__(self, name: str) -> Any:
        from elastic_transport import ConnectionError as TransportConnectionError

        def fail(*_a: Any, **_k: Any) -> Any:
            raise TransportConnectionError("connection refused (SYNTHETIC FIXTURE)")

        return fail


class MissingIndexES:
    """Silver index does not exist (404 ``index_not_found_exception`` on search)."""

    def search(self, *_a: Any, **_k: Any) -> Any:
        raise api_error(es_exc.NotFoundError, "index_not_found_exception", 404)

    def get(self, *_a: Any, **_k: Any) -> Any:
        raise api_error(es_exc.NotFoundError, "index_not_found_exception", 404)
