"""Structured logging.

* One JSON object per line (default) or a human-readable console format for local work.
* Context such as ``request_id``, ``tenant_id`` or ``run_id`` is bound through ``contextvars``,
  so every line emitted while serving a request or executing a run carries it — including
  lines written from nested coroutines and asyncio tasks.
* Secrets are redacted twice before anything is written: by key name (``api_key``,
  ``password``, ``token``…) and by value pattern (provider keys, bearer tokens, DSN passwords).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

REDACTED = "***"

_log_context: ContextVar[Mapping[str, Any] | None] = ContextVar("log_context", default=None)

# "token" alone or as a suffix is sensitive ("access_token"), token *counts* are not
# ("input_tokens"), hence the anchored alternatives.
_SENSITIVE_KEY = re.compile(
    r"(?i)(^|[_\-])(api[_\-]?key|secret(?:[_\-]key)?|password|passwd|authorization|cookie|"
    r"credentials?|token)$"
)
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-(?:ant-|proj-|lf-)?[A-Za-z0-9_\-]{16,}"),  # Anthropic / OpenAI / Langfuse
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
)
_DSN_PASSWORD = re.compile(r"(?P<prefix>[a-z][a-z0-9+.\-]*://[^:/@\s]+:)[^@\s]+(?=@)")

_LOG_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))
_STANDARD_RECORD_ATTRS = _LOG_RECORD_ATTRS | {"message", "asctime", "color_message", "fields"}


# --------------------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------------------
def redact_text(text: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return _DSN_PASSWORD.sub(lambda match: match.group("prefix") + REDACTED, text)


def redact(value: Any, _depth: int = 0) -> Any:
    """Return a copy of ``value`` with secrets masked (recursive on mappings and sequences)."""
    if _depth > 10:
        return value
    if isinstance(value, Mapping):
        return {
            key: REDACTED
            if isinstance(key, str) and _SENSITIVE_KEY.search(key)
            else redact(item, _depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(item, _depth + 1) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


# --------------------------------------------------------------------------------------
# Context binding
# --------------------------------------------------------------------------------------
def get_log_context() -> dict[str, Any]:
    return dict(_log_context.get() or {})


def bind_log_context(**fields: Any) -> Token[Mapping[str, Any] | None]:
    merged = {**get_log_context(), **{k: v for k, v in fields.items() if v is not None}}
    return _log_context.set(merged)


def reset_log_context(token: Token[Mapping[str, Any] | None]) -> None:
    _log_context.reset(token)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    token = bind_log_context(**fields)
    try:
        yield
    finally:
        reset_log_context(token)


# --------------------------------------------------------------------------------------
# Formatters
# --------------------------------------------------------------------------------------
def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    fields = {k: v for k, v in vars(record).items() if k not in _STANDARD_RECORD_ATTRS}
    explicit = getattr(record, "fields", None)
    if isinstance(explicit, Mapping):
        fields.update(explicit)
    return fields


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **get_log_context(),
            **_record_fields(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        fields = redact({**get_log_context(), **_record_fields(record)})
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        message = redact_text(record.getMessage())
        line = f"{timestamp} {record.levelname:<7} {record.name}: {message}"
        if rendered:
            line = f"{line} | {rendered}"
        if record.exc_info:
            line = f"{line}\n{redact_text(self.formatException(record.exc_info))}"
        return line


# --------------------------------------------------------------------------------------
# Setup & helpers
# --------------------------------------------------------------------------------------
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "httpx2",
    "httpcore2",
    "anthropic",
    "openai",
    "sqlalchemy.engine",
    "asyncio",
)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
    # Route uvicorn through the same formatter instead of its own handlers.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event: str,
    /,
    *,
    level: int = logging.INFO,
    exc_info: BaseException | None = None,
    **fields: Any,
) -> None:
    """Emit a structured event: ``log_event(log, "agent_started", agent="research")``."""
    if logger.isEnabledFor(level):
        logger.log(level, event, exc_info=exc_info, extra={"fields": fields})
