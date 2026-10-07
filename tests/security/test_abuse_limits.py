"""Rate limiting and request size limits."""

from collections.abc import AsyncIterator

import pytest

from app.core.rate_limit import RateLimiter
from app.db.database import Database
from tests.support import AppClient, make_settings

UNREACHABLE_DB = "postgresql+asyncpg://app:app@127.0.0.1:1/nowhere"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_token_bucket_spends_and_refills() -> None:
    clock = Clock()
    limiter = RateLimiter(per_minute=60, clock=clock)  # 1 token per second, burst 60
    assert all(limiter.acquire("acme") is None for _ in range(60))
    retry_after = limiter.acquire("acme")
    assert retry_after == pytest.approx(1.0)
    clock.now += 1.0
    assert limiter.acquire("acme") is None


def test_buckets_are_per_tenant_and_costs_are_weighted() -> None:
    limiter = RateLimiter(per_minute=10, clock=Clock())
    assert limiter.acquire("acme", cost=10) is None
    assert limiter.acquire("acme") is not None
    assert limiter.acquire("globex") is None  # another tenant is not affected


@pytest.mark.integration
async def test_api_answers_429_with_retry_after(database: Database) -> None:
    settings = make_settings(rate_limit_enabled=True, rate_limit_per_minute=2)
    async with AppClient(settings) as client:
        statuses = [
            (await client.post("/api/v1/tasks", json={"request": f"Requête numéro {i} ok"}))
            for i in range(3)
        ]
    assert [r.status_code for r in statuses] == [201, 201, 429]
    limited = statuses[-1]
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.json()["error"]["code"] == "rate_limited"


async def test_oversized_declared_body_is_rejected_before_parsing() -> None:
    settings = make_settings(database_url=UNREACHABLE_DB, max_request_body_bytes=2_048)
    async with AppClient(settings) as client:
        response = await client.post("/api/v1/tasks", json={"request": "x" * 5_000})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert response.headers["X-Request-ID"]


async def test_oversized_chunked_body_is_rejected() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b"x" * 1_024

    settings = make_settings(database_url=UNREACHABLE_DB, max_request_body_bytes=2_048)
    async with AppClient(settings) as client:
        response = await client.post(
            "/api/v1/tasks", content=chunks(), headers={"content-type": "application/json"}
        )
    assert response.status_code == 413
