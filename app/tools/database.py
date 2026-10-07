"""Database tools: a fixed menu of parameterised queries, never raw SQL.

* ``database_read`` exposes a few aggregate/listing queries per entity, tenant-scoped (the SQL
  implementation also runs under row-level security).
* ``database_write`` supports exactly one action — appending a note to a task — and is a WRITE
  tool: it is hidden and refused unless the run was started with ``allow_tool_writes=true``.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from app.core.text import normalize_text
from app.schemas.common import StrictModel
from app.schemas.tool import ToolPermission
from app.tools.base import Tool, ToolContext, ToolError

ReadEntity = Literal["tasks", "runs", "documents", "agent_runs"]
ReadQuery = Literal["count_by_status", "recent", "summary"]


class DatabaseReadInput(StrictModel):
    entity: ReadEntity
    query: ReadQuery = Field(
        description="count_by_status: counts per status; recent: latest rows; "
        "summary: aggregated statistics."
    )
    limit: int = Field(default=10, ge=1, le=50)


class DatabaseWriteInput(StrictModel):
    action: Literal["annotate_task"]
    task_id: UUID
    note: str = Field(min_length=5, max_length=500)


async def database_read(arguments: DatabaseReadInput, context: ToolContext) -> dict[str, Any]:
    if context.data is None:
        raise ToolError("platform data is not available in this environment")
    return await context.data.read(
        context.tenant_id, arguments.entity, arguments.query, arguments.limit
    )


async def database_write(arguments: DatabaseWriteInput, context: ToolContext) -> dict[str, Any]:
    if context.data is None:
        raise ToolError("platform data is not available in this environment")
    return await context.data.annotate_task(
        context.tenant_id, arguments.task_id, normalize_text(arguments.note)
    )


DATABASE_READ = Tool(
    name="database_read",
    description="Read platform data (tasks, runs, documents, agent_runs) through safe, "
    "predefined queries: count_by_status, recent or summary.",
    input_model=DatabaseReadInput,
    permission=ToolPermission.READ,
    handler=database_read,
)

DATABASE_WRITE = Tool(
    name="database_write",
    description="Append a short note to a task (the only write operation available).",
    input_model=DatabaseWriteInput,
    permission=ToolPermission.WRITE,
    handler=database_write,
)
