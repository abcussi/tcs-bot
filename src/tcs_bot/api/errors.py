"""FastAPI error handlers (AC-003.1, constitution §8).

The ``AppError`` hierarchy lives in the framework-neutral :mod:`tcs_bot.errors` and is
re-exported here for backwards compatibility.

Every error leaves the API as ``{"error": {"code", "message", "details"}}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tcs_bot.errors import (
    AppError,
    DependencyUnavailable,
    NotFoundError,
    UpstreamError,
    ValidationAppError,
)
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

__all__ = [
    "AppError",
    "DependencyUnavailable",
    "NotFoundError",
    "UpstreamError",
    "ValidationAppError",
    "register_error_handlers",
]


def _envelope(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


def _clean_validation_errors(errors: Any) -> list[dict[str, Any]]:
    # Pydantic errors may carry exception objects in ``ctx``; keep only JSON-safe fields.
    return [
        {
            "loc": [str(p) for p in err.get("loc", ())],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in errors
    ]


async def _app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    if exc.status_code >= 500:
        log.warning("app_error", code=exc.code, status=exc.status_code, msg=exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def _validation_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=422,
        content=_envelope(
            "validation_error", "Request validation failed", _clean_validation_errors(exc.errors())
        ),
    )


async def _http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(code, str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    # Never leak stack traces or internals to clients.
    log.error("unhandled_error", error_type=type(exc).__name__, exc_info=exc)
    return JSONResponse(
        status_code=500, content=_envelope("internal_error", "Internal server error")
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
