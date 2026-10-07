"""Secrets never leak: configuration, logs, errors, repository files."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.logging import JsonFormatter, log_event
from tests.support import AppClient, make_settings

ROOT = Path(__file__).resolve().parents[2]
UNREACHABLE_DB = "postgresql+asyncpg://app:app@127.0.0.1:1/nowhere"


def test_env_example_contains_no_secret_values() -> None:
    secret_keys = re.compile(
        r"^(LLM_API_KEY|EMBEDDING_API_KEY|LANGFUSE_SECRET_KEY|LANGFUSE_PUBLIC_KEY)=(.*)$"
    )
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        match = secret_keys.match(line)
        if match:
            value = match.group(2).split("#")[0].strip()
            assert value == "", f"{match.group(1)} must be empty in .env.example"
    assert "API_KEYS=[]" in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_real_env_files_are_git_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env\n" in gitignore
    assert "!.env.example" in gitignore


def test_production_refuses_to_start_without_authentication() -> None:
    with pytest.raises(ValidationError):
        make_settings(app_env="production", auth_enabled=False)


def test_secrets_are_masked_in_settings_and_structured_logs() -> None:
    settings = make_settings(llm_api_key="sk-ant-api03-SUPERSECRETVALUE123456")
    assert "SUPERSECRETVALUE" not in repr(settings)

    lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(self.format(record))

    handler = Capture()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.secrets")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_event(
        logger,
        "provider_configured",
        api_key="sk-ant-api03-SUPERSECRETVALUE123456",
        dsn="postgresql+asyncpg://app:hunter2@db/x",
        note="token sk-proj-ABCDEFGHIJKLMNOPQRSTUV in free text",
    )
    payload = json.loads(lines[0])
    assert "SUPERSECRETVALUE" not in lines[0]
    assert "hunter2" not in lines[0]
    assert "ABCDEFGHIJKLMNOP" not in lines[0]
    assert payload["api_key"] == "***"


async def test_unexpected_errors_do_not_leak_internals() -> None:
    # The database is unreachable: the endpoint fails, the client only sees a generic error.
    settings = make_settings(database_url=UNREACHABLE_DB)
    async with AppClient(settings, raise_app_exceptions=False) as client:
        response = await client.get("/api/v1/tasks")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == {
        "code": "internal_error",
        "message": "An unexpected error occurred",
        "details": None,
        "request_id": body["error"]["request_id"],
    }
    assert "127.0.0.1" not in response.text
    assert "asyncpg" not in response.text
