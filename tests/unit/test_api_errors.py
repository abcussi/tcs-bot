"""Error envelope + handlers (AC-003.1, constitution §8)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from tcs_bot.api.errors import (
    AppError,
    DependencyUnavailable,
    NotFoundError,
    UpstreamError,
    ValidationAppError,
    register_error_handlers,
)


class Body(BaseModel):
    n: int


def _app() -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/nf")
    def nf() -> None:
        raise NotFoundError("Vendor 9 not found", {"vendor_id": 9})

    @app.get("/upstream")
    def upstream() -> None:
        raise UpstreamError("LLM failed")

    @app.post("/body")
    def body(b: Body) -> dict:
        return {"n": b.n}

    @app.get("/crash")
    def crash() -> None:
        raise RuntimeError("secret stack detail /etc/passwd")

    return app


def test_error_classes_status_codes() -> None:
    """AC-003.1: AppError hierarchy carries code/status/details."""
    assert AppError("x", "m").status_code == 400
    assert NotFoundError("m").status_code == 404
    assert ValidationAppError("m").status_code == 422
    assert UpstreamError("m").status_code == 502
    assert DependencyUnavailable("m").status_code == 503
    e = NotFoundError("m", {"a": 1}, code="vendor_not_found")
    assert e.to_dict() == {
        "error": {"code": "vendor_not_found", "message": "m", "details": {"a": 1}}
    }


def test_404_envelope() -> None:
    """AC-003.1: AppError -> {"error": {code, message, details}} with its status."""
    r = TestClient(_app()).get("/nf")
    assert r.status_code == 404
    assert r.json() == {
        "error": {"code": "not_found", "message": "Vendor 9 not found", "details": {"vendor_id": 9}}
    }
    assert TestClient(_app()).get("/upstream").status_code == 502


def test_422_validation_envelope() -> None:
    """AC-003.1: RequestValidationError -> 422 in the same envelope."""
    r = TestClient(_app()).post("/body", json={"n": "not-a-number"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert err["details"][0]["loc"] == ["body", "n"]
    assert set(err["details"][0]) == {"loc", "msg", "type"}


def test_unknown_route_and_unhandled_exception() -> None:
    """AC-003.1: unknown routes use the envelope; unhandled errors -> 500 without leaking."""
    client = TestClient(_app(), raise_server_exceptions=False)
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/crash")
    assert r.status_code == 500
    assert r.json() == {
        "error": {"code": "internal_error", "message": "Internal server error", "details": None}
    }
    assert "passwd" not in r.text


def test_report_request_requires_vendor_selector() -> None:
    """AC-003.7: ReportRequest needs vendor_id or vendor_name; top_k in 1..20."""
    import pytest
    from pydantic import ValidationError

    from tcs_bot.schemas.api import ReportRequest

    with pytest.raises(ValidationError):
        ReportRequest()
    with pytest.raises(ValidationError):
        ReportRequest(vendor_name="  ")
    with pytest.raises(ValidationError):
        ReportRequest(vendor_id=1, top_k=21)
    req = ReportRequest(vendor_id=1, focus_areas=[" cyber ", ""])
    assert req.focus_areas == ["cyber"] and req.use_cache is True
