"""Test helpers shared by fixtures and test modules."""

from __future__ import annotations

import os

from httpx import ASGITransport, AsyncClient

from app.container import Container, build_container
from app.core.config import Settings
from app.main import create_app

DEFAULT_TEST_DATABASE_URL = "postgresql+asyncpg://app:app@localhost:5433/ai_platform_test"


def get_test_database_url() -> str:
    return os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "test",
        "log_format": "console",
        "log_level": "WARNING",
        "database_url": get_test_database_url(),
        "rate_limit_enabled": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


class AppClient:
    """Run the real app (lifespan included) behind an in-process ASGI transport.

    Usage: ``async with AppClient(settings) as client: ...``
    """

    def __init__(
        self,
        settings: Settings,
        container: Container | None = None,
        *,
        raise_app_exceptions: bool = True,
    ) -> None:
        self.settings = settings
        self.container = container or build_container(settings)
        self.app = create_app(settings, container=self.container)
        self._lifespan = self.app.router.lifespan_context(self.app)
        self._client: AsyncClient | None = None
        # False: observe the 500 response a real client gets (Starlette re-raises the
        # exception after responding so that the server can log it).
        self._raise_app_exceptions = raise_app_exceptions

    async def __aenter__(self) -> AsyncClient:
        await self._lifespan.__aenter__()
        transport = ASGITransport(app=self.app, raise_app_exceptions=self._raise_app_exceptions)
        self._client = AsyncClient(transport=transport, base_url="http://testserver")
        return self._client

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._client is not None
        await self._client.aclose()
        await self._lifespan.__aexit__(None, None, None)
        await self.container.aclose()
