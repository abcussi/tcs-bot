"""E2E fixtures: a real HTTP client against the running API (``E2E_BASE_URL``) and best-effort
cleanup of everything the run creates (documents, chunks in MySQL + ES, reports, audit rows)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

E2E_SOURCE = "e2e-qa"  # `source` of every document the e2e suite creates
BASE_URL = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@pytest.fixture(scope="session")
def api() -> Iterator[httpx.Client]:
    # trust_env=False: never route localhost traffic through a corporate/agent proxy.
    client = httpx.Client(base_url=BASE_URL, timeout=60.0, trust_env=False)
    try:
        resp = client.get("/health")
    except httpx.HTTPError as exc:
        client.close()
        pytest.skip(f"API not reachable at {BASE_URL}: {exc}")
    if resp.status_code != 200:
        client.close()
        pytest.skip(f"API at {BASE_URL} unhealthy: HTTP {resp.status_code}")
    yield client
    client.close()


def _db_available() -> bool:
    try:
        from vendor_risk_ai.db.mysql import ping

        return bool(ping())
    except Exception:
        return False


def _max_id(table: str) -> int:
    from vendor_risk_ai.db.mysql import transaction

    with transaction() as cur:
        cur.execute(f"SELECT COALESCE(MAX(id), 0) AS m FROM {table}")
        return int(cur.fetchone()["m"])


@pytest.fixture(scope="session")
def created() -> Iterator[dict[str, Any]]:
    """Register created document ids (``created["docs"].append((vendor_id, doc_id))``).

    On teardown, if the test process can reach the same MySQL/ES as the API (local stack), the
    documents (+ cascaded chunks), their ES chunk docs, reports and audit rows created during the
    session are deleted, plus any document with ``source == E2E_SOURCE`` created meanwhile.
    Remote targets are left untouched.
    """
    track: dict[str, Any] = {"docs": []}
    can_clean = _db_available()
    if can_clean:
        max_report, max_audit = _max_id("risk_reports"), _max_id("audit_log")
        max_doc = _max_id("vendor_documents")
    yield track
    if not can_clean:
        return
    from vendor_risk_ai.config import get_settings
    from vendor_risk_ai.db.mysql import transaction

    s = get_settings()
    try:
        from vendor_risk_ai.search.elastic import get_es

        es = get_es()
        es.indices.refresh(index=s.elastic_chunks_index)  # chunks are indexed without refresh
        for _, doc_id in track["docs"]:
            es.options(ignore_status=404).delete_by_query(
                index=s.elastic_chunks_index,
                query={"bool": {"filter": [{"term": {"document_id": doc_id}}]}},
                refresh=True,
            )
    except Exception:
        pass  # ES cleanup is best effort
    with transaction() as cur:
        for _, doc_id in track["docs"]:
            cur.execute("DELETE FROM vendor_documents WHERE id = %s", (doc_id,))
        # documents the API stored but never returned (e.g. rejected after insert)
        cur.execute(
            "DELETE FROM vendor_documents WHERE id > %s AND source = %s", (max_doc, E2E_SOURCE)
        )
        cur.execute("DELETE FROM risk_reports WHERE id > %s", (max_report,))
        cur.execute(
            "DELETE FROM audit_log WHERE id > %s AND action IN ('document.add', 'report.save')",
            (max_audit,),
        )
