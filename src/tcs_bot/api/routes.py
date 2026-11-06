"""HTTP routes (AC-003.7)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import ValidationError

from tcs_bot.api.deps import get_service
from tcs_bot.api.errors import ValidationAppError
from tcs_bot.logging_setup import get_logger
from tcs_bot.schemas.api import (
    DocumentIn,
    DocumentOut,
    ErrorEnvelope,
    HealthOut,
    ReportRequest,
    VendorOut,
)
from tcs_bot.schemas.domain import DocType, DocumentCreate
from tcs_bot.schemas.report import RiskReport
from tcs_bot.services.report_service import ReportService

log = get_logger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
HealthChecks = Mapping[str, Callable[[], bool]]
Service = Annotated[ReportService, Depends(get_service)]

_ERRORS = {
    404: {"model": ErrorEnvelope, "description": "Not found"},
    422: {"model": ErrorEnvelope, "description": "Validation error"},
    503: {"model": ErrorEnvelope, "description": "Dependency unavailable"},
}
_REPORT_ERRORS = {**_ERRORS, 502: {"model": ErrorEnvelope, "description": "LLM / upstream error"}}

router = APIRouter()


def _multipart_installed() -> bool:
    try:
        import python_multipart  # noqa: F401
    except ImportError:
        return False
    return True


# ----------------------------------------------------------------------------- health
@router.get("/health", response_model=HealthOut, tags=["health"])
def health(request: Request) -> HealthOut:
    """Dependency status; never fails (each check is isolated)."""
    from tcs_bot.config import get_settings

    checks: HealthChecks = request.app.state.health_checks_factory()
    results: dict[str, str] = {}
    for name in ("mysql", "redis", "elasticsearch"):
        check = checks.get(name)
        try:
            results[name] = "ok" if check is not None and check() else "down"
        except Exception as exc:
            log.warning("health_check_failed", dependency=name, error_type=type(exc).__name__)
            results[name] = "down"
    try:
        provider = "fake" if get_settings().use_fake_llm else "openai"
    except Exception:
        provider = "fake"
    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"
    return HealthOut(status=overall, llm_provider=provider, **results)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- vendors
@router.get("/vendors", response_model=list[VendorOut], tags=["vendors"], responses=_ERRORS)
def list_vendors(
    service: Service,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[VendorOut]:
    return [
        VendorOut.model_validate(v, from_attributes=True)
        for v in service.list_vendors(limit, offset)
    ]


@router.get("/vendors/{vendor_id}", response_model=VendorOut, tags=["vendors"], responses=_ERRORS)
def get_vendor(vendor_id: int, service: Service) -> VendorOut:
    return VendorOut.model_validate(service.get_vendor(vendor_id), from_attributes=True)


# ----------------------------------------------------------------------------- documents
@router.post(
    "/vendors/{vendor_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    responses=_ERRORS,
)
def add_document(vendor_id: int, body: DocumentIn, service: Service) -> DocumentOut:
    return service.ingest_document(vendor_id, DocumentCreate(**body.model_dump()))


if _multipart_installed():
    from fastapi import File, Form, UploadFile

    from tcs_bot.ingestion.extract import extract_document

    @router.post(
        "/vendors/{vendor_id}/documents/upload",
        response_model=DocumentOut,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
        responses=_ERRORS,
    )
    def upload_document(
        vendor_id: int,
        service: Service,
        file: Annotated[UploadFile, File(description="PDF, .txt or .md")],
        doc_type: Annotated[DocType, Form()] = "other",
        title: Annotated[str | None, Form(max_length=255)] = None,
        source: Annotated[str, Form()] = "upload",
        published_at: Annotated[date | None, Form()] = None,
    ) -> DocumentOut:
        service.get_vendor(vendor_id)  # 404 before doing any parsing work
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValidationAppError(
                "File too large", {"max_bytes": MAX_UPLOAD_BYTES}, code="file_too_large"
            )
        filename = file.filename or "upload"
        extracted = extract_document(data, filename)
        try:
            doc = DocumentCreate(
                doc_type=doc_type,
                title=(title or filename)[:255],
                source=source,
                published_at=published_at,
                content=extracted.text,
            )
        except ValidationError as exc:
            raise ValidationAppError("Invalid document", {"errors": exc.errors()}) from exc
        if extracted.notes:
            log.info("upload_extraction_notes", vendor_id=vendor_id, notes=extracted.notes)
        return service.ingest_document(vendor_id, doc)
else:  # pragma: no cover - python-multipart is a transitive dependency today
    log.warning("upload_route_disabled", reason="python-multipart not installed")


# ----------------------------------------------------------------------------- reports
@router.post("/reports", response_model=RiskReport, tags=["reports"], responses=_REPORT_ERRORS)
def create_report(body: ReportRequest, service: Service) -> RiskReport:
    return service.generate_report(body)


@router.get(
    "/reports/{vendor_id}/latest", response_model=RiskReport, tags=["reports"], responses=_ERRORS
)
def latest_report(vendor_id: int, service: Service) -> RiskReport:
    return service.latest_report(vendor_id)
