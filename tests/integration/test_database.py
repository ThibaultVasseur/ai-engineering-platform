"""Schema, pgvector and row-level security, verified against a real PostgreSQL."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.db.database import Database
from app.db.models import TENANT_SCOPED_TABLES
from app.db.repositories.tasks import TaskRepository

pytestmark = pytest.mark.integration


async def test_pgvector_extension_is_installed(database: Database) -> None:
    async with database.engine.connect() as connection:
        version = await connection.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
    assert version is not None


async def test_application_role_cannot_bypass_rls(database: Database) -> None:
    async with database.engine.connect() as connection:
        row = (
            await connection.execute(
                text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
            )
        ).one()
    assert row == (False, False)


async def test_rls_is_enabled_and_forced_on_every_tenant_table(database: Database) -> None:
    async with database.engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE relname = ANY(:tables)"
                ),
                {"tables": list(TENANT_SCOPED_TABLES)},
            )
        ).all()
    assert {name for name, *_ in rows} == set(TENANT_SCOPED_TABLES)
    assert all(enabled and forced for _, enabled, forced in rows)


async def test_rls_hides_rows_of_other_tenants_even_without_filters(database: Database) -> None:
    async with database.session("acme") as session:
        await TaskRepository(session, "acme").add("Acme confidential request", {})

    async with database.session("globex") as session:
        # A raw query with *no* tenant filter: RLS alone must hide acme's row.
        count = await session.scalar(text("SELECT count(*) FROM tasks"))
    assert count == 0

    async with database.session("acme") as session:
        count = await session.scalar(text("SELECT count(*) FROM tasks"))
    assert count == 1


async def test_rls_fails_closed_when_no_tenant_is_set(database: Database) -> None:
    async with database.session("acme") as session:
        await TaskRepository(session, "acme").add("Acme request", {})
    async with database.engine.connect() as connection:
        count = await connection.scalar(text("SELECT count(*) FROM tasks"))
    assert count == 0


async def test_rls_rejects_writes_for_another_tenant(database: Database) -> None:
    with pytest.raises(DBAPIError, match="row-level security"):
        async with database.session("acme") as session:
            await session.execute(
                text(
                    "INSERT INTO tasks (id, tenant_id, user_request, status) "
                    "VALUES (:id, 'globex', 'smuggled', 'PENDING')"
                ),
                {"id": uuid.uuid4()},
            )


async def test_task_status_is_constrained(database: Database) -> None:
    with pytest.raises(DBAPIError, match="ck_tasks_status"):
        async with database.session("acme") as session:
            await session.execute(
                text(
                    "INSERT INTO tasks (id, tenant_id, user_request, status) "
                    "VALUES (:id, 'acme', 'x', 'DONE')"
                ),
                {"id": uuid.uuid4()},
            )
