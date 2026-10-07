"""Request context middleware (pure ASGI, so contextvars propagate to the endpoint)."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.exceptions import error_body
from app.core.logging import log_context, log_event

logger = logging.getLogger("app.http")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class BodySizeLimitMiddleware:
    """Rejects bodies above ``max_bytes`` with 413 — both declared (Content-Length) and streamed
    (chunked) bodies are checked, so a client cannot make the server buffer arbitrary payloads.

    For streamed bodies the 413 is sent as soon as the limit is crossed; the application then
    receives a disconnect and whatever it tries to send afterwards is dropped. (Raising from
    ``receive`` would not work: FastAPI turns any body-reading error into a 400.)
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, send)
            return

        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request" and not rejected:
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    await self._reject(scope, send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if not rejected:  # once the 413 is out, the application's response is dropped
                await send(message)

        await self.app(scope, limited_receive, guarded_send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = (scope.get("state") or {}).get("request_id")
        body = json.dumps(
            error_body(
                "payload_too_large",
                f"Request body exceeds {self.max_bytes} bytes",
                None,
                request_id,
            )
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


class RequestContextMiddleware:
    """Assigns a request id, binds it to the log context and logs one line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status_code = 500
        started = time.perf_counter()

        async def send_with_headers(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for name, value in _SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
            await send(message)

        with log_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_headers)
            finally:
                log_event(
                    logger,
                    "http_request",
                    method=scope.get("method"),
                    path=scope.get("path"),
                    status=status_code,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
