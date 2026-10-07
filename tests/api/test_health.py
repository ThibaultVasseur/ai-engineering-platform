"""Health probes. These tests deliberately need no database."""

from app import __version__
from tests.support import AppClient, make_settings

UNREACHABLE_DB = "postgresql+asyncpg://app:app@127.0.0.1:1/nowhere"


async def test_liveness_does_not_depend_on_the_database() -> None:
    async with AppClient(make_settings(database_url=UNREACHABLE_DB)) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "environment": "test"}
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"


async def test_readiness_reports_unavailable_database() -> None:
    async with AppClient(make_settings(database_url=UNREACHABLE_DB)) as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "unavailable"
    assert body["llm_provider"] == "offline"


async def test_request_id_is_propagated_when_valid() -> None:
    async with AppClient(make_settings(database_url=UNREACHABLE_DB)) as client:
        kept = await client.get("/health", headers={"X-Request-ID": "trace-123"})
        replaced = await client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert kept.headers["X-Request-ID"] == "trace-123"
    assert replaced.headers["X-Request-ID"] != "bad id with spaces"


async def test_openapi_documents_the_api() -> None:
    async with AppClient(make_settings(database_url=UNREACHABLE_DB)) as client:
        schema = (await client.get("/openapi.json")).json()
    assert schema["info"]["title"] == "AI Engineering Multi-Agent Platform"
    assert "/api/v1/tasks" in schema["paths"]
