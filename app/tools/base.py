"""Tool definition and the ports tools rely on.

A tool is data + a handler: name, description, a Pydantic input model (which becomes the strict
JSON schema shown to the model), a permission level and a timeout. Handlers receive validated
arguments and a ``ToolContext`` scoped to the tenant and the run — they never get a raw database
session, a shell or the file system.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel

from app.llm.schema import strict_json_schema
from app.llm.types import ToolSpec
from app.observability.tracing import Tracer
from app.orchestration.journal import SourceRegistry
from app.schemas.knowledge import RetrievalResult, SearchFilters
from app.schemas.tool import ToolInfo, ToolPermission


class ToolError(Exception):
    """Expected tool failure; the message is safe to show to the model."""


class KnowledgeRetriever(Protocol):
    async def search(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> RetrievalResult: ...


class PlatformData(Protocol):
    """Read (and one tightly-scoped write) access to platform data, always tenant-scoped."""

    async def lookup_tasks(
        self, tenant_id: str, *, query: str | None, task_id: UUID | None, limit: int
    ) -> list[dict[str, Any]]: ...

    async def read(self, tenant_id: str, entity: str, query: str, limit: int) -> dict[str, Any]: ...

    async def annotate_task(self, tenant_id: str, task_id: UUID, note: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ToolContext:
    tenant_id: str
    run_id: str
    agent: str
    step_id: str | None
    sources: SourceRegistry
    knowledge: KnowledgeRetriever | None = None
    data: PlatformData | None = None
    tracer: Tracer | None = None  # for domain events raised inside a tool (e.g. guardrails)


Handler = Callable[[Any, ToolContext], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    permission: ToolPermission
    handler: Handler
    timeout_seconds: float | None = None

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=strict_json_schema(self.input_model),
        )

    def info(self, default_timeout: float) -> ToolInfo:
        return ToolInfo(
            name=self.name,
            description=self.description,
            permission=self.permission,
            timeout_seconds=self.timeout_seconds or default_timeout,
            input_schema=strict_json_schema(self.input_model),
        )
