"""End-to-end HTTP flow against the running API: vendors -> ingest (JSON, .txt, PDF) -> report
(by id and by name, cached) -> latest, plus the error envelope contract."""

from __future__ import annotations

import io
import uuid
from typing import Any

import httpx
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from tcs_bot.schemas.api import DocumentOut, ErrorEnvelope, HealthOut, VendorOut
from tcs_bot.schemas.report import RiskReport, score_to_level
from tests.e2e.conftest import E2E_SOURCE

pytestmark = pytest.mark.e2e

E2E_VENDOR_EXT = "VND-004"  # Aurelian Identity Labs SAS (no ticker) in the seed data
MISSING_ID = 987_654_321


# --------------------------------------------------------------------------- helpers
def make_pdf(lines: list[str]) -> bytes:
    """One-page PDF with a real text layer (Helvetica), built with pypdf."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
    for line in lines:
        esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({esc}) Tj T*")
    ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    page.replace_contents(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def assert_error(resp: httpx.Response, status: int, code: str | None = None) -> dict[str, Any]:
    assert resp.status_code == status, resp.text
    body = resp.json()
    env = ErrorEnvelope.model_validate(body)
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    if code is not None:
        assert env.error.code == code
    assert env.error.message
    return body


def assert_report(resp: httpx.Response, vendor_id: int) -> RiskReport:
    assert resp.status_code == 200, resp.text
    report = RiskReport.model_validate(resp.json())
    assert report.vendor_id == vendor_id
    assert 0 <= report.overall_risk_score <= 100
    assert report.risk_level == score_to_level(report.overall_risk_score)
    for src in report.sources:
        assert -1.0 <= src.score <= 1.0
    return report


@pytest.fixture(scope="module")
def vendor(api: httpx.Client) -> VendorOut:
    vendors = [VendorOut.model_validate(v) for v in api.get("/vendors").json()]
    if not vendors:
        pytest.skip("no vendors seeded (run `uv run tcs-bot seed`)")
    by_ext = {v.external_id: v for v in vendors}
    return by_ext.get(E2E_VENDOR_EXT, vendors[0])


# --------------------------------------------------------------------------- health / vendors
def test_health(api):
    """AC-003.7: GET /health reports every dependency and the LLM provider."""
    resp = api.get("/health")
    assert resp.status_code == 200
    health = HealthOut.model_validate(resp.json())
    assert health.status == "ok"
    assert (health.mysql, health.redis, health.elasticsearch) == ("ok", "ok", "ok")
    assert health.llm_provider in ("fake", "openai")


def test_list_and_get_vendors(api, vendor):
    """AC-003.7: GET /vendors (pagination) and GET /vendors/{id}."""
    resp = api.get("/vendors", params={"limit": 500})
    assert resp.status_code == 200
    vendors = [VendorOut.model_validate(v) for v in resp.json()]
    assert len(vendors) >= 5
    ids = [v.id for v in vendors]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    page = api.get("/vendors", params={"limit": 2, "offset": 1}).json()
    assert [v["id"] for v in page] == ids[1:3]

    one = api.get(f"/vendors/{vendor.id}")
    assert one.status_code == 200
    assert VendorOut.model_validate(one.json()) == vendor


# --------------------------------------------------------------------------- full flow
def test_ingest_report_latest_flow(api, vendor, created):
    """AC-003.2, AC-003.6, AC-003.7, AC-003.8: JSON + .txt + PDF ingest -> report by id / by
    name -> cached repeat -> latest."""
    tag = uuid.uuid4().hex[:10]

    # JSON document
    resp = api.post(
        f"/vendors/{vendor.id}/documents",
        json={
            "doc_type": "regulatory_filing",
            "title": f"e2e json {tag}",
            "source": E2E_SOURCE,
            "published_at": "2026-10-01",
            "content": (
                f"E2E filing {tag}. The regulator imposed a civil penalty for sanctions screening "
                "failures. A data breach exposed client KYC records. The SOC 2 Type II report "
                "noted exceptions in access reviews. Net loss widened and liquidity tightened."
            ),
        },
    )
    assert resp.status_code == 201, resp.text
    doc_json = DocumentOut.model_validate(resp.json())
    created["docs"].append((vendor.id, doc_json.id))
    assert doc_json.vendor_id == vendor.id and doc_json.chunks >= 1 and doc_json.indexed

    # .txt upload
    txt = f"E2E text upload {tag}.\n\nRansomware   incident caused a two-day outage.\n".encode()
    resp = api.post(
        f"/vendors/{vendor.id}/documents/upload",
        files={"file": (f"e2e-{tag}.txt", txt, "text/plain")},
        data={"doc_type": "news", "source": E2E_SOURCE},
    )
    assert resp.status_code == 201, resp.text
    doc_txt = DocumentOut.model_validate(resp.json())
    created["docs"].append((vendor.id, doc_txt.id))
    assert doc_txt.chunks >= 1 and doc_txt.id != doc_json.id

    # PDF upload (real text layer)
    pdf_lines = [
        f"E2E annual report {tag}.",
        "The auditor expressed substantial doubt about the ability to continue",
        "as a going concern after a covenant breach on the revolving facility.",
    ]
    pdf = make_pdf(pdf_lines)
    assert tag in PdfReader(io.BytesIO(pdf)).pages[0].extract_text()
    resp = api.post(
        f"/vendors/{vendor.id}/documents/upload",
        files={"file": (f"e2e-{tag}.pdf", pdf, "application/pdf")},
        data={"doc_type": "financial_report", "title": f"e2e pdf {tag}", "source": E2E_SOURCE},
    )
    assert resp.status_code == 201, resp.text
    doc_pdf = DocumentOut.model_validate(resp.json())
    created["docs"].append((vendor.id, doc_pdf.id))
    assert doc_pdf.chunks >= 1

    # Re-posting identical content is deduplicated
    again = api.post(
        f"/vendors/{vendor.id}/documents/upload",
        files={"file": (f"e2e-{tag}.txt", txt, "text/plain")},
        data={"doc_type": "news", "source": E2E_SOURCE},
    )
    assert again.status_code == 201
    assert again.json()["id"] == doc_txt.id

    # Report by id: fresh (new docs changed the fingerprint) and uses the new evidence
    body = {"vendor_id": vendor.id, "focus_areas": ["cyber", "going concern"], "top_k": 20}
    report = assert_report(api.post("/reports", json=body), vendor.id)
    new_ids = {doc_json.id, doc_txt.id, doc_pdf.id}
    assert new_ids & {s.document_id for s in report.sources}
    assert report.financial_health.summary
    assert report.risk_narrative
    if vendor.ticker is None:
        assert any("ticker" in g.lower() for g in report.data_gaps)

    # Same request again -> served from cache (identical payload incl. generated_at)
    cached = assert_report(api.post("/reports", json=body), vendor.id)
    assert cached.model_dump(mode="json") == report.model_dump(mode="json")

    # Latest = the persisted report
    latest = assert_report(api.get(f"/reports/{vendor.id}/latest"), vendor.id)
    assert latest.generated_at == report.generated_at
    assert latest.overall_risk_score == report.overall_risk_score

    # Report by name (case-insensitive) resolves to the same vendor, bypassing the cache
    by_name = assert_report(
        api.post("/reports", json={"vendor_name": vendor.name.upper(), "use_cache": False}),
        vendor.id,
    )
    assert by_name.vendor_name == vendor.name
    latest = assert_report(api.get(f"/reports/{vendor.id}/latest"), vendor.id)
    assert latest.generated_at == by_name.generated_at


# --------------------------------------------------------------------------- error envelopes
def test_not_found_envelopes(api):
    """AC-003.1: 404s use the {"error": {code, message, details}} envelope."""
    assert_error(api.get(f"/vendors/{MISSING_ID}"), 404, "not_found")
    assert_error(api.get(f"/reports/{MISSING_ID}/latest"), 404, "not_found")
    assert_error(api.post("/reports", json={"vendor_id": MISSING_ID}), 404, "not_found")
    assert_error(
        api.post("/reports", json={"vendor_name": f"no-such-vendor-{uuid.uuid4().hex}"}),
        404,
        "not_found",
    )
    assert_error(
        api.post(f"/vendors/{MISSING_ID}/documents", json={"title": "x", "content": "y"}),
        404,
        "not_found",
    )
    assert_error(
        api.post(
            f"/vendors/{MISSING_ID}/documents/upload",
            files={"file": ("a.txt", b"hello", "text/plain")},
        ),
        404,
        "not_found",
    )
    assert_error(api.get("/definitely-not-a-route"), 404)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"vendor_id": 0},
        {"vendor_id": 1, "top_k": 0},
        {"vendor_id": 1, "top_k": 21},
        {"vendor_name": ""},
        {"vendor_id": "abc"},
    ],
)
def test_report_validation_envelope(api, body):
    """AC-003.1: request validation errors -> 422 with the same envelope."""
    assert_error(api.post("/reports", json=body), 422, "validation_error")


def test_document_validation_envelopes(api, vendor):
    """AC-003.1, AC-003.2: invalid documents / uploads -> 422 envelope."""
    assert_error(
        api.post(f"/vendors/{vendor.id}/documents", json={"title": "t", "content": ""}),
        422,
        "validation_error",
    )
    assert_error(
        api.post(
            f"/vendors/{vendor.id}/documents",
            json={"title": "t", "content": "x", "doc_type": "tweet"},
        ),
        422,
        "validation_error",
    )
    assert_error(api.get("/vendors/not-an-int"), 422, "validation_error")
    assert_error(api.get("/vendors", params={"limit": 0}), 422, "validation_error")
    # whitespace-only content normalizes to nothing
    assert_error(
        api.post(
            f"/vendors/{vendor.id}/documents",
            json={"title": "t", "content": " \n\t ", "source": E2E_SOURCE},
        ),
        422,
    )


@pytest.mark.parametrize(
    ("filename", "payload", "ctype", "code"),
    [
        ("malware.exe", b"MZ\x90\x00binary", "application/octet-stream", "unsupported_file_type"),
        ("report.docx", b"PK\x03\x04", "application/zip", "unsupported_file_type"),
        ("empty.txt", b"", "text/plain", "empty_document"),
        ("broken.pdf", b"%PDF-1.4 this is not really a pdf", "application/pdf", None),
    ],
)
def test_upload_rejections(api, vendor, filename, payload, ctype, code):
    """AC-003.2: unsupported / empty / corrupt uploads -> 422 envelope (never 500)."""
    resp = api.post(
        f"/vendors/{vendor.id}/documents/upload",
        files={"file": (filename, payload, ctype)},
    )
    assert_error(resp, 422, code)


def test_upload_missing_file_field(api, vendor):
    """AC-003.1: multipart without the file part -> 422 envelope."""
    resp = api.post(f"/vendors/{vendor.id}/documents/upload", data={"doc_type": "news"})
    assert_error(resp, 422, "validation_error")
