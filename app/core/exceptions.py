"""Domain exceptions and their HTTP mapping.

Services raise these exceptions; they never build HTTP responses themselves. The API layer
translates them into one stable error envelope::

    {"error": {"code": "not_found", "message": "...", "details": null, "request_id": "..."}}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import log_event

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class InvalidRequestError(AppError):
    status_code = 422
    code = "invalid_request"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"


class PermissionDeniedError(AppError):
    status_code = 403
    code = "forbidden"


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        super().__init__(message, details={"retry_after_seconds": retry_after_seconds})
        self.retry_after_seconds = retry_after_seconds


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_body(code: str, message: str, details: Any, request_id: str | None) -> dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "details": details, "request_id": request_id}
    }


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)  # noqa: S101 - narrowing for the type checker
    headers = {}
    if isinstance(exc, RateLimitedError):
        headers["Retry-After"] = str(exc.retry_after_seconds)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.code, exc.message, exc.details, _request_id(request)),
        headers=headers,
    )


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    # Never echo the rejected input back: it may be huge or contain secrets.
    details = [
        {"loc": list(error.get("loc", ())), "msg": error.get("msg"), "type": error.get("type")}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_body(
            "validation_error", "Request validation failed", details, _request_id(request)
        ),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log_event(
        logger,
        "unhandled_exception",
        level=logging.ERROR,
        exc_info=exc,
        error_type=type(exc).__name__,
        path=request.url.path,
        request_id=_request_id(request),
    )
    return JSONResponse(
        status_code=500,
        content=error_body(
            "internal_error", "An unexpected error occurred", None, _request_id(request)
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
