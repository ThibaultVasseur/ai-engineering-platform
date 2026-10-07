"""Engine, sessions and the tenant-scoped unit of work."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:
    """Owns the connection pool and hands out tenant-scoped transactions.

    ``session(tenant_id)`` opens a transaction and sets the ``app.tenant_id`` setting *locally*
    to that transaction. PostgreSQL row-level security policies compare every row's
    ``tenant_id`` against it, so even a query that forgets a filter cannot cross tenants.
    """

    def __init__(self, url: str, *, pool_size: int = 5, echo: bool = False) -> None:
        self.engine: AsyncEngine = create_async_engine(
            url,
            pool_size=pool_size,
            max_overflow=pool_size,
            pool_pre_ping=True,
            echo=echo,
        )
        self._session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False, autoflush=False
        )

    @asynccontextmanager
    async def session(self, tenant_id: str) -> AsyncIterator[AsyncSession]:
        """One transaction for one tenant; commits on success, rolls back on error."""
        async with self._session_factory() as session, session.begin():
            await session.execute(select(func.set_config("app.tenant_id", tenant_id, True)))
            yield session

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    async def dispose(self) -> None:
        await self.engine.dispose()
