"""FastAPI dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from vendor_risk_ai.api.errors import DependencyUnavailable

if TYPE_CHECKING:
    from vendor_risk_ai.services.report_service import ReportService


def get_service(request: Request) -> ReportService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise DependencyUnavailable(
            "Report service is not available (startup failed; check MySQL/config)",
            code="service_unavailable",
        )
    return service
