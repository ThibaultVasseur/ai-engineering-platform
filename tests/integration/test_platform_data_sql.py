"""SqlPlatformData: the data tools' PostgreSQL backend (fixed queries, tenant scope, RLS)."""

import uuid

import pytest

from app.db.database import Database
from app.db.repositories.runs import RunRepository
from app.db.repositories.tasks import TaskRepository
from app.services.platform_data import SqlPlatformData
from app.tools.base import ToolError

pytestmark = pytest.mark.integration


async def seed(database: Database) -> list[uuid.UUID]:
    ids = []
    async with database.session("acme") as session:
        tasks = TaskRepository(session, "acme")
        for index, status in enumerate(["COMPLETED", "COMPLETED", "FAILED", "PENDING"]):
            task = await tasks.add(f"Tâche numéro {index} sur les devis", {})
            await tasks.update(task.id, status=status)
            ids.append(task.id)
        runs = RunRepository(session, "acme")
        for task_id in ids[:3]:
            run = await runs.add(task_id, {})
            await runs.update(
                run.id, status="COMPLETED", cost_usd=0.02, agent_calls=6, step_count=10
            )
    async with database.session("globex") as session:
        await TaskRepository(session, "globex").add("Globex secret task", {})
    return ids


async def test_counts_and_summaries_are_tenant_scoped(database: Database) -> None:
    await seed(database)
    data = SqlPlatformData(database)
    counts = await data.read("acme", "tasks", "count_by_status", 10)
    assert counts == {
        "entity": "tasks",
        "counts": {"COMPLETED": 2, "FAILED": 1, "PENDING": 1},
        "total": 4,
    }
    runs = await data.read("acme", "runs", "summary", 10)
    assert runs["total"] == 3
    assert runs["avg_cost_usd"] == pytest.approx(0.02)
    assert (await data.read("globex", "tasks", "count_by_status", 10))["total"] == 1


async def test_recent_rows_and_lookup(database: Database) -> None:
    await seed(database)
    data = SqlPlatformData(database)
    recent = await data.read("acme", "tasks", "recent", 2)
    assert len(recent["rows"]) == 2
    found = await data.lookup_tasks("acme", query="devis", task_id=None, limit=10)
    assert len(found) == 4
    assert await data.lookup_tasks("globex", query="devis", task_id=None, limit=10) == []


async def test_user_text_is_never_sql(database: Database) -> None:
    await seed(database)
    data = SqlPlatformData(database)
    hostile = "%' OR 1=1; DROP TABLE tasks; --"
    assert await data.lookup_tasks("acme", query=hostile, task_id=None, limit=10) == []
    assert (await data.read("acme", "tasks", "count_by_status", 10))["total"] == 4


async def test_annotation_is_bounded_to_the_tenant(database: Database) -> None:
    ids = await seed(database)
    data = SqlPlatformData(database)
    assert await data.annotate_task("acme", ids[0], "Vérifié par l'agent data") == {
        "task_id": str(ids[0]),
        "notes": 1,
    }
    with pytest.raises(ToolError, match="not found"):
        await data.annotate_task("globex", ids[0], "tentative cross-tenant")


async def test_documents_have_no_status(database: Database) -> None:
    with pytest.raises(ToolError, match="no status"):
        await SqlPlatformData(database).read("acme", "documents", "count_by_status", 10)
