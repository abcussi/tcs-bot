"""In-memory stand-in for the slice of the Elasticsearch client used by ``GoldVectorStore``.

Mirrors the real server behaviour the store depends on (checked against ES 8.15): indexed
``dense_vector`` dims are enforced per document in ``bulk`` (zero vectors are rejected for cosine
and a wrong length is a per-item 400), kNN scores are ``(1 + cosine) / 2``, ``knn.filter`` is a
pre-filter, ``num_candidates`` must be within ``[k, 10000]``, a missing index is a 404 and writes
become visible to ``search`` only after ``indices.refresh`` (read-your-write needs the refresh).
Offline tests only; no network.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import exceptions as es_exc


def api_error(cls: type[es_exc.ApiError], error_type: str, status: int) -> es_exc.ApiError:
    meta = ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=NodeConfig("http", "localhost", 9200),
    )
    return cls(error_type, meta, {"error": {"type": error_type, "reason": error_type}})


class _Index:
    def __init__(self, mappings: dict[str, Any], settings: dict[str, Any] | None) -> None:
        self.mappings = mappings
        self.settings = settings or {}
        self.visible: dict[str, dict[str, Any]] = {}  # searchable (after refresh)
        self.pending: dict[str, dict[str, Any]] = {}  # indexed, not yet refreshed

    @property
    def dims(self) -> int | None:
        return self.mappings.get("properties", {}).get("embedding", {}).get("dims")


def _matches(source: dict[str, Any], clause: dict[str, Any]) -> bool:
    if "bool" in clause:
        return all(_matches(source, c) for c in clause["bool"].get("filter", []))
    if "terms" in clause:
        ((field, wanted),) = clause["terms"].items()
        have = source.get(field)
        have_list = have if isinstance(have, list) else [have]
        return any(h in wanted for h in have_list if h is not None)
    if "term" in clause:
        ((field, wanted),) = clause["term"].items()
        have = source.get(field)
        have_list = have if isinstance(have, list) else [have]
        return wanted in have_list
    if "range" in clause:
        ((field, bounds),) = clause["range"].items()
        val = source.get(field)
        if val is None:
            return False
        return all(
            {"gte": val >= b, "gt": val > b, "lte": val <= b, "lt": val < b}[op]
            for op, b in bounds.items()
        )
    raise NotImplementedError(f"fake ES does not support clause {clause!r}")


class FakeIndices:
    def __init__(self, es: FakeES) -> None:
        self._es = es

    def exists(self, index: str) -> bool:
        self._es.tick("indices.exists")
        return index in self._es.indices_by_name

    def create(
        self, index: str, mappings: dict[str, Any] | None = None, settings: Any = None
    ) -> dict[str, Any]:
        self._es.tick("indices.create")
        if index in self._es.indices_by_name:
            raise api_error(es_exc.BadRequestError, "resource_already_exists_exception", 400)
        self._es.indices_by_name[index] = _Index(mappings or {}, settings)
        return {"acknowledged": True, "index": index}

    def delete(self, index: str, ignore_unavailable: bool = False) -> dict[str, Any]:
        self._es.tick("indices.delete")
        if index not in self._es.indices_by_name and not ignore_unavailable:
            raise api_error(es_exc.NotFoundError, "index_not_found_exception", 404)
        self._es.indices_by_name.pop(index, None)
        return {"acknowledged": True}

    def get_mapping(self, index: str) -> dict[str, Any]:
        self._es.tick("indices.get_mapping")
        return {index: {"mappings": self._es.index(index).mappings}}

    def refresh(self, index: str) -> dict[str, Any]:
        self._es.tick("indices.refresh")
        idx = self._es.index(index)
        idx.visible.update(idx.pending)
        idx.pending.clear()
        return {"_shards": {"failed": 0}}


class FakeES:
    """Minimal Elasticsearch client double. ``fail_ids`` makes those ``_id`` s fail in ``bulk``."""

    def __init__(self) -> None:
        self.indices_by_name: dict[str, _Index] = {}
        self.indices = FakeIndices(self)
        self.calls: list[str] = []
        self.fail_ids: set[str] = set()
        self.last_knn: dict[str, Any] | None = None

    def tick(self, name: str) -> None:
        self.calls.append(name)

    def index(self, name: str) -> _Index:
        if name not in self.indices_by_name:
            raise api_error(es_exc.NotFoundError, "index_not_found_exception", 404)
        return self.indices_by_name[name]

    # ------------------------------------------------------------------ writes
    def bulk(self, operations: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.tick("bulk")
        items: list[dict[str, Any]] = []
        errors = False
        for action, doc in zip(operations[0::2], operations[1::2], strict=True):
            meta = action["index"]
            name, _id = meta["_index"], str(meta["_id"])
            idx = self.index(name)
            vec = doc.get("embedding")
            err: str | None = None
            if _id in self.fail_ids:
                err = "injected failure"
            elif vec is not None and idx.dims is not None and len(vec) != idx.dims:
                err = "different number of dimensions"
            elif vec is not None and not np.any(np.asarray(vec)):
                err = "cosine similarity does not support vectors with zero magnitude"
            if err:
                errors = True
                items.append(
                    {
                        "index": {
                            "_id": _id,
                            "status": 400,
                            "error": {"type": "mapper", "reason": err},
                        }
                    }
                )
                continue
            created = _id not in idx.visible and _id not in idx.pending
            idx.pending[_id] = doc
            items.append({"index": {"_id": _id, "status": 201 if created else 200}})
        return {"errors": errors, "items": items}

    def delete_by_query(self, index: str, query: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.tick("delete_by_query")
        idx = self.index(index)
        idx.visible.update(idx.pending)
        idx.pending.clear()
        gone = [i for i, s in idx.visible.items() if _matches(s, query)]
        for i in gone:
            del idx.visible[i]
        return {"deleted": len(gone)}

    # ------------------------------------------------------------------ reads
    def get(self, index: str, id: str, **_: Any) -> dict[str, Any]:  # noqa: A002 - ES API name
        """Real-time ``GET``: sees pending (unrefreshed) writes; 404s like ES for index / doc."""
        self.tick("get")
        idx = self.index(index)  # missing index -> index_not_found_exception (404)
        src = idx.pending.get(str(id)) or idx.visible.get(str(id))
        if src is None:
            raise api_error(es_exc.NotFoundError, "resource_not_found", 404)
        return {"_id": str(id), "found": True, "_source": src}

    def count(self, index: str, **_: Any) -> dict[str, Any]:
        self.tick("count")
        return {"count": len(self.index(index).visible)}

    def search(
        self,
        index: str,
        knn: dict[str, Any],
        size: int = 10,
        source_excludes: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.tick("search")
        self.last_knn = knn
        idx = self.index(index)
        k, cands = int(knn["k"]), int(knn["num_candidates"])
        if k < 1 or cands < k or cands > 10000:
            raise api_error(es_exc.BadRequestError, "illegal_argument_exception", 400)
        q = np.asarray(knn["query_vector"], dtype=np.float64)
        if idx.dims is not None and len(q) != idx.dims:
            raise api_error(es_exc.BadRequestError, "illegal_argument_exception", 400)
        flt = knn.get("filter")
        scored: list[tuple[float, str]] = []
        for _id, src in idx.visible.items():
            vec = src.get(knn["field"])
            if vec is None or (flt is not None and not _matches(src, flt)):
                continue
            v = np.asarray(vec, dtype=np.float64)
            cos = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q)))
            scored.append(((1.0 + cos) / 2.0, _id))
        scored.sort(key=lambda t: -t[0])
        excl = set(source_excludes or [])
        hits = [
            {
                "_id": _id,
                "_score": score,
                "_source": {k_: v_ for k_, v_ in idx.visible[_id].items() if k_ not in excl},
            }
            for score, _id in scored[: min(k, size)]
        ]
        return {"hits": {"hits": hits}}


class DownES:
    """Every call fails the way an unreachable Elasticsearch does (transport ConnectionError)."""

    def __init__(self) -> None:
        from elastic_transport import ConnectionError as TransportConnectionError

        self._exc = TransportConnectionError("connection refused (SYNTHETIC FIXTURE)")
        self.indices = self

    def __getattr__(self, name: str) -> Any:
        def fail(*_a: Any, **_k: Any) -> Any:
            raise self._exc

        return fail
