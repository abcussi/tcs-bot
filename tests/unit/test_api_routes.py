"""FastAPI app via TestClient with an injected in-memory service (AC-003.1, AC-003.7)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tcs_bot.api.app import create_app
from tcs_bot.api.errors import NotFoundError
from tcs_bot.schemas.report import RiskReport
from tcs_bot.services.report_service import ReportService

from .conftest import GOING_CONCERN_DOC

HEALTHY = {"mysql": lambda: True, "redis": lambda: True, "elasticsearch": lambda: True}


@pytest.fixture
def app(service: ReportService) -> FastAPI:
    return create_app(service, health_checks=HEALTHY)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _assert_envelope(body: dict, code: str) -> None:
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code


def test_health_ok_and_degraded(service: ReportService) -> None:
    """AC-003.7: /health reports dependency status and never 500s."""
    with TestClient(create_app(service, health_checks=HEALTHY)) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "mysql": "ok",
        "redis": "ok",
        "elasticsearch": "ok",
        "llm_provider": "fake",
    }

    def boom() -> bool:
        raise ConnectionError("down")

    checks = {"mysql": boom, "redis": lambda: False}  # elasticsearch missing entirely
    with TestClient(create_app(service, health_checks=checks)) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert (body["mysql"], body["redis"], body["elasticsearch"]) == ("down", "down", "down")


def test_vendors_endpoints(client: TestClient) -> None:
    """AC-003.7: GET /vendors and /vendors/{id}."""
    r = client.get("/vendors", params={"limit": 10})
    assert r.status_code == 200
    assert [v["name"] for v in r.json()] == ["Acme Clearing Services", "Northwind Data Feeds"]
    assert client.get("/vendors/1").json()["criticality"] == "critical"
    r = client.get("/vendors/99")
    assert r.status_code == 404
    _assert_envelope(r.json(), "not_found")


def test_document_report_latest_flow(client: TestClient) -> None:
    """AC-003.7/.8: POST document -> POST /reports -> GET latest."""
    r = client.post(
        "/vendors/1/documents",
        json={"doc_type": "financial_report", "title": "FY2025", "content": GOING_CONCERN_DOC},
    )
    assert r.status_code == 201
    assert r.json()["vendor_id"] == 1 and r.json()["chunks"] >= 1 and r.json()["indexed"]

    r = client.post("/reports", json={"vendor_id": 1, "focus_areas": ["liquidity"], "top_k": 3})
    assert r.status_code == 200
    report = RiskReport.model_validate(r.json())
    assert report.risk_level == "critical" and 1 <= len(report.sources) <= 3

    latest = client.get("/reports/1/latest")
    assert latest.status_code == 200
    assert RiskReport.model_validate(latest.json()) == report
    r = client.get("/reports/2/latest")
    assert r.status_code == 404
    _assert_envelope(r.json(), "not_found")


def test_upload_txt_and_pdf_errors(client: TestClient) -> None:
    """AC-003.7: multipart upload of .txt; unsupported types -> 422 envelope."""
    r = client.post(
        "/vendors/2/documents/upload",
        files={"file": ("news.txt", GOING_CONCERN_DOC.encode(), "text/plain")},
        data={"doc_type": "news"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["vendor_id"] == 2
    r = client.post(
        "/vendors/2/documents/upload",
        files={"file": ("deck.pptx", b"PK..", "application/octet-stream")},
    )
    assert r.status_code == 422
    _assert_envelope(r.json(), "unsupported_file_type")
    r = client.post(
        "/vendors/77/documents/upload", files={"file": ("a.txt", b"hello", "text/plain")}
    )
    assert r.status_code == 404


def test_report_by_name_and_openapi(client: TestClient) -> None:
    """AC-003.7: vendor_name selector; every route documented with tags."""
    r = client.post("/reports", json={"vendor_name": "Northwind Data Feeds"})
    assert r.status_code == 200 and r.json()["vendor_id"] == 2
    spec = client.get("/openapi.json").json()
    for path in (
        "/health",
        "/vendors",
        "/vendors/{vendor_id}",
        "/vendors/{vendor_id}/documents",
        "/vendors/{vendor_id}/documents/upload",
        "/reports",
        "/reports/{vendor_id}/latest",
    ):
        assert path in spec["paths"], path
        for op in spec["paths"][path].values():
            assert op["tags"]


def test_missing_service_returns_503() -> None:
    """AC-003.1: if the service could not be built, routes answer 503 in the envelope."""
    app = create_app(health_checks=HEALTHY)
    with TestClient(app) as c:  # no lifespan here: state stays None
        app.state.service = None
        r = c.get("/vendors")
    assert r.status_code == 503
    _assert_envelope(r.json(), "service_unavailable")


def test_lifespan_builds_default_service(
    monkeypatch: pytest.MonkeyPatch, service: ReportService
) -> None:
    """AC-003.7: create_app() without a service builds it lazily at startup."""
    import tcs_bot.services.report_service as rs

    monkeypatch.setattr(rs, "build_default_service", lambda: service)
    app = create_app(health_checks=HEALTHY)
    assert app.state.service is None
    with TestClient(app) as c:
        assert c.get("/vendors/1").status_code == 200
    assert app.state.service is service


def test_route_raising_not_found_from_service(client: TestClient, service: ReportService) -> None:
    """AC-003.1: AppError raised deep in the service maps to its status + envelope."""

    def missing(vendor_id: int) -> None:
        raise NotFoundError("gone", {"vendor_id": vendor_id})

    service.latest_report = missing  # type: ignore[method-assign]
    r = client.get("/reports/1/latest")
    assert r.status_code == 404
    assert r.json() == {
        "error": {"code": "not_found", "message": "gone", "details": {"vendor_id": 1}}
    }
