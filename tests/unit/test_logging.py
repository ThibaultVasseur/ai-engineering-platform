import json
import logging

import pytest

from app.core.logging import (
    REDACTED,
    JsonFormatter,
    log_context,
    log_event,
    redact,
    redact_text,
)


def capture(logger_name: str) -> tuple[logging.Logger, list[str]]:
    lines: list[str] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(self.format(record))

    handler = ListHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(logger_name)
    logger.handlers[:] = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, lines


def test_json_lines_carry_event_fields_and_bound_context() -> None:
    logger, lines = capture("test.json")
    with log_context(run_id="run-1", tenant_id="acme"):
        log_event(logger, "agent_started", agent="research", step_id="kb")
    payload = json.loads(lines[0])
    assert payload["event"] == "agent_started"
    assert payload["agent"] == "research"
    assert payload["run_id"] == "run-1"
    assert payload["tenant_id"] == "acme"
    assert payload["level"] == "INFO"


def test_context_is_restored_after_block() -> None:
    logger, lines = capture("test.context")
    with log_context(run_id="run-1"):
        pass
    log_event(logger, "after")
    assert "run_id" not in json.loads(lines[0])


def test_sensitive_keys_are_redacted_but_token_counts_are_kept() -> None:
    payload = redact(
        {
            "api_key": "abc",
            "LLM_API_KEY": "abc",
            "langfuse_secret_key": "abc",
            "password": "abc",
            "access_token": "abc",
            "authorization": "Bearer abc",
            "input_tokens": 1200,
            "nested": [{"x-api-key": "abc", "model": "claude-opus-5-5"}],
        }
    )
    assert payload["api_key"] == REDACTED
    assert payload["LLM_API_KEY"] == REDACTED
    assert payload["langfuse_secret_key"] == REDACTED
    assert payload["password"] == REDACTED
    assert payload["access_token"] == REDACTED
    assert payload["authorization"] == REDACTED
    assert payload["input_tokens"] == 1200
    assert payload["nested"][0]["x-api-key"] == REDACTED
    assert payload["nested"][0]["model"] == "claude-opus-5-5"


@pytest.mark.parametrize(
    ("raw", "leaked"),
    [
        ("key=sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV", "ABCDEFGHIJKLMNOP"),
        ("sk-proj-0123456789abcdefghij", "0123456789abcdef"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload", "eyJhbGciOiJIUzI1NiJ9"),
        ("postgresql+asyncpg://app:s3cr3t@db:5432/x", "s3cr3t"),
    ],
)
def test_secret_values_are_redacted_inside_free_text(raw: str, leaked: str) -> None:
    assert leaked not in redact_text(raw)


def test_exceptions_are_serialised_and_redacted() -> None:
    logger, lines = capture("test.exc")
    try:
        raise RuntimeError("connection to postgresql://app:hunter2@db/x failed")
    except RuntimeError as exc:
        log_event(logger, "boom", level=logging.ERROR, exc_info=exc)
    payload = json.loads(lines[0])
    assert "hunter2" not in lines[0]
    assert "RuntimeError" in payload["exception"]
