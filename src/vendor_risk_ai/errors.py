"""Domain error hierarchy (constitution §8), framework-neutral.

Raised by ``services/``, ``ingestion/`` and caught by ``mcp_server/``; the FastAPI handlers in
``api/errors.py`` map them to ``{"error": {"code", "message", "details"}}``. ``api.errors``
re-exports every class here, so existing imports keep working.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base for every domain error surfaced to API / MCP clients."""

    def __init__(
        self, code: str, message: str, status_code: int = 400, details: Any = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class NotFoundError(AppError):
    def __init__(self, message: str, details: Any = None, *, code: str = "not_found") -> None:
        super().__init__(code, message, 404, details)


class ValidationAppError(AppError):
    def __init__(
        self, message: str, details: Any = None, *, code: str = "validation_error"
    ) -> None:
        super().__init__(code, message, 422, details)


class UpstreamError(AppError):
    def __init__(self, message: str, details: Any = None, *, code: str = "upstream_error") -> None:
        super().__init__(code, message, 502, details)


class DependencyUnavailable(AppError):
    def __init__(
        self, message: str, details: Any = None, *, code: str = "dependency_unavailable"
    ) -> None:
        super().__init__(code, message, 503, details)


__all__ = [
    "AppError",
    "DependencyUnavailable",
    "NotFoundError",
    "UpstreamError",
    "ValidationAppError",
]
