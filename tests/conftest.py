"""Shared fixtures.

Database-backed tests need PostgreSQL + pgvector (``docker compose up -d postgres``).
Locally they are *skipped* with an explicit reason when the database is unreachable;
in CI ``REQUIRE_DB=1`` turns that skip into a hard failure so nothing is silently ignored.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.db.database import Database
from app.db.models import TENANT_SCOPED_TABLES
from tests.support import AppClient, get_test_database_url, make_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------
def _alembic_config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    return config


def _reset_schema(connection: Connection, url: str) -> None:
    config = _alembic_config(url)
    config.attributes["connection"] = connection
    command.downgrade(config, "base")
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
async def migrated_database_url() -> str:
    url = get_test_database_url()
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:  # any connection failure means "no database"
        await engine.dispose()
        message = f"PostgreSQL unreachable at TEST_DATABASE_URL ({type(exc).__name__})"
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail(message)
        pytest.skip(message)
    async with engine.begin() as connection:
        await connection.run_sync(_reset_schema, url)
    await engine.dispose()
    return url


@pytest.fixture
async def database(migrated_database_url: str) -> AsyncIterator[Database]:
    db = Database(migrated_database_url, pool_size=2)
    try:
        yield db
    finally:
        async with db.engine.begin() as connection:
            tables = ", ".join(TENANT_SCOPED_TABLES)
            await connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
        await db.dispose()


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
@pytest.fixture
async def client(database: Database) -> AsyncIterator[AsyncClient]:
    async with AppClient(make_settings()) as http:
        yield http
