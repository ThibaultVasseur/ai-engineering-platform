"""task_lookup: find previous tasks of the same tenant (by id or by text)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.common import StrictModel
from app.schemas.tool import ToolPermission
from app.tools.base import Tool, ToolContext, ToolError


class TaskLookupInput(StrictModel):
    query: str | None = Field(
        default=None, min_length=3, max_length=200, description="Text to search in past requests."
    )
    task_id: UUID | None = Field(default=None, description="Exact task id.")
    limit: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def _one_criterion(self) -> TaskLookupInput:
        if self.query is None and self.task_id is None:
            raise ValueError("provide 'query' or 'task_id'")
        return self


async def task_lookup(arguments: TaskLookupInput, context: ToolContext) -> dict[str, Any]:
    if context.data is None:
        raise ToolError("task history is not available in this environment")
    tasks = await context.data.lookup_tasks(
        context.tenant_id, query=arguments.query, task_id=arguments.task_id, limit=arguments.limit
    )
    return {"tasks": tasks, "count": len(tasks)}


TASK_LOOKUP = Tool(
    name="task_lookup",
    description="Look up previous tasks (requests, status, results) of the current tenant, "
    "by id or by text search. Useful to reuse prior work.",
    input_model=TaskLookupInput,
    permission=ToolPermission.READ,
    handler=task_lookup,
)
