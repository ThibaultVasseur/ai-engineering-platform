import uuid

import pytest

from app.services.platform_data import InMemoryPlatformData
from app.tools.base import ToolError


@pytest.fixture
def data() -> InMemoryPlatformData:
    acme = InMemoryPlatformData.with_demo_data("acme")
    other = InMemoryPlatformData.with_demo_data("globex")
    return InMemoryPlatformData(tasks=acme.tasks + other.tasks[:3], runs=acme.runs)


async def test_counts_are_scoped_to_the_tenant(data: InMemoryPlatformData) -> None:
    acme = await data.read("acme", "tasks", "count_by_status", 10)
    globex = await data.read("globex", "tasks", "count_by_status", 10)
    assert acme["total"] == 10
    assert acme["counts"]["COMPLETED"] == 6
    assert globex["total"] == 3


async def test_recent_rows_are_compact_and_limited(data: InMemoryPlatformData) -> None:
    recent = await data.read("acme", "tasks", "recent", 3)
    assert len(recent["rows"]) == 3
    assert set(recent["rows"][0]) == {"id", "status", "request", "created_at"}


async def test_run_summary_aggregates(data: InMemoryPlatformData) -> None:
    summary = await data.read("acme", "runs", "summary", 10)
    assert summary["total"] == 8
    assert summary["avg_cost_usd"] > 0


async def test_lookup_by_text_and_by_id(data: InMemoryPlatformData) -> None:
    found = await data.lookup_tasks("acme", query="devis", task_id=None, limit=5)
    assert len(found) == 1
    by_id = await data.lookup_tasks("acme", query=None, task_id=uuid.UUID(found[0]["id"]), limit=5)
    assert by_id == found
    assert await data.lookup_tasks("globex", query="devis", task_id=None, limit=5) == []


async def test_annotation_cannot_target_another_tenant(data: InMemoryPlatformData) -> None:
    task_id = uuid.UUID(data.tasks[0]["id"])
    with pytest.raises(ToolError, match="not found"):
        await data.annotate_task("someone-else", task_id, "malicious note")


async def test_documents_have_no_status(data: InMemoryPlatformData) -> None:
    with pytest.raises(ToolError, match="no status"):
        await data.read("acme", "documents", "count_by_status", 10)
