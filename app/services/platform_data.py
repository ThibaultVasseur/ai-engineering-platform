"""Platform data access for the data tools (``PlatformData`` port).

``InMemoryPlatformData`` serves the CLI demo, the evaluation suite and unit tests; the
PostgreSQL implementation (``SqlPlatformData``) serves the API. Both return the same shapes, so
agents behave identically on top of either.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import case, func, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.text import truncate
from app.db.database import Database
from app.db.models import AgentRunModel, DocumentModel, RunModel, TaskModel
from app.db.repositories.tasks import TaskRepository
from app.tools.base import ToolError

MAX_NOTES_PER_TASK = 20


class InMemoryPlatformData:
    def __init__(
        self,
        *,
        tasks: list[dict[str, Any]] | None = None,
        runs: list[dict[str, Any]] | None = None,
        documents: list[dict[str, Any]] | None = None,
        agent_runs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.tasks = tasks or []
        self.runs = runs or []
        self.documents = documents or []
        self.agent_runs = agent_runs or []

    @classmethod
    def with_demo_data(cls, tenant_id: str = "default") -> InMemoryPlatformData:
        """A small, deterministic history so that data questions have something to analyse."""
        now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
        requests = [
            ("Ajouter un export PDF des factures", "COMPLETED"),
            ("Concevoir l'API de gestion des clients", "COMPLETED"),
            ("Analyser les délais de traitement des tickets", "COMPLETED"),
            ("Proposer un schéma de base pour les abonnements", "FAILED"),
            ("Définir la politique de rétention des documents", "COMPLETED"),
            ("Intégrer le paiement par carte", "FAILED"),
            ("Préparer la migration vers PostgreSQL 17", "COMPLETED"),
            ("Mettre en place l'authentification SSO", "RUNNING"),
            ("Concevoir un module de devis simplifié", "COMPLETED"),
            ("Auditer les permissions des rôles", "CANCELLED"),
        ]
        tasks = [
            {
                "id": str(uuid.UUID(int=index + 1)),
                "tenant_id": tenant_id,
                "user_request": request,
                "status": status,
                "metadata": {},
                "created_at": (now - timedelta(days=30 - 3 * index)).isoformat(),
            }
            for index, (request, status) in enumerate(requests)
        ]
        runs = [
            {
                "id": str(uuid.UUID(int=100 + index)),
                "tenant_id": tenant_id,
                "task_id": task["id"],
                "status": "COMPLETED" if task["status"] == "COMPLETED" else "FAILED",
                "step_count": 9 + index % 4,
                "agent_calls": 6 + index % 3,
                "retry_count": index % 2,
                "cost_usd": round(0.05 + 0.01 * index, 4),
                "input_tokens": 12_000 + 500 * index,
                "output_tokens": 3_000 + 100 * index,
                "created_at": task["created_at"],
            }
            for index, task in enumerate(tasks)
            if task["status"] in {"COMPLETED", "FAILED"}
        ]
        return cls(tasks=tasks, runs=runs)

    # ----------------------------------------------------------------------------------
    def _rows(self, entity: str, tenant_id: str) -> list[dict[str, Any]]:
        source = {
            "tasks": self.tasks,
            "runs": self.runs,
            "documents": self.documents,
            "agent_runs": self.agent_runs,
        }[entity]
        return [row for row in source if row.get("tenant_id") == tenant_id]

    async def lookup_tasks(
        self, tenant_id: str, *, query: str | None, task_id: UUID | None, limit: int
    ) -> list[dict[str, Any]]:
        rows = self._rows("tasks", tenant_id)
        if task_id is not None:
            rows = [row for row in rows if row["id"] == str(task_id)]
        if query:
            needle = query.lower()
            rows = [row for row in rows if needle in row["user_request"].lower()]
        rows = sorted(rows, key=lambda row: row["created_at"], reverse=True)[:limit]
        return [task_view(row) for row in rows]

    async def read(self, tenant_id: str, entity: str, query: str, limit: int) -> dict[str, Any]:
        rows = self._rows(entity, tenant_id)
        if query == "count_by_status":
            if entity == "documents":
                raise ToolError("documents have no status; use the 'summary' query")
            counts = Counter(row["status"] for row in rows)
            return {"entity": entity, "counts": dict(sorted(counts.items())), "total": len(rows)}
        if query == "recent":
            recent = sorted(rows, key=lambda row: str(row.get("created_at")), reverse=True)
            return {"entity": entity, "rows": [compact(entity, row) for row in recent[:limit]]}
        return {"entity": entity, **summarise(entity, rows)}

    async def annotate_task(self, tenant_id: str, task_id: UUID, note: str) -> dict[str, Any]:
        for row in self._rows("tasks", tenant_id):
            if row["id"] == str(task_id):
                notes = row.setdefault("metadata", {}).setdefault("notes", [])
                if len(notes) >= MAX_NOTES_PER_TASK:
                    raise ToolError(f"a task holds at most {MAX_NOTES_PER_TASK} notes")
                notes.append(note)
                return {"task_id": str(task_id), "notes": len(notes)}
        raise ToolError("task not found")


class SqlPlatformData:
    """PostgreSQL implementation: fixed, parameterised queries; tenant filter + RLS."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def lookup_tasks(
        self, tenant_id: str, *, query: str | None, task_id: UUID | None, limit: int
    ) -> list[dict[str, Any]]:
        async with self._db.session(tenant_id) as session:
            repository = TaskRepository(session, tenant_id)
            if task_id is not None:
                task = await repository.get(task_id)
                rows = [task] if task is not None else []
                if query:
                    rows = [t for t in rows if query.lower() in t.user_request.lower()]
            else:
                rows = list(await repository.search(query or "", limit=limit))
            return [task_view(_task_dict(task)) for task in rows[:limit]]

    async def read(self, tenant_id: str, entity: str, query: str, limit: int) -> dict[str, Any]:
        model = _ENTITY_MODELS[entity]
        async with self._db.session(tenant_id) as session:
            tenant_filter = model.tenant_id == tenant_id
            if query == "count_by_status":
                if entity == "documents":
                    raise ToolError("documents have no status; use the 'summary' query")
                status_column = model.status
                rows = await session.execute(
                    select(status_column, func.count()).where(tenant_filter).group_by(status_column)
                )
                counts = {status: count for status, count in rows.all()}  # noqa: C416
                return {
                    "entity": entity,
                    "counts": dict(sorted(counts.items())),
                    "total": sum(counts.values()),
                }
            if query == "recent":
                records = await session.scalars(
                    select(model)
                    .where(tenant_filter)
                    .order_by(model.created_at.desc())
                    .limit(limit)
                )
                return {
                    "entity": entity,
                    "rows": [compact(entity, _row(entity, r)) for r in records],
                }
            return {"entity": entity, **await _sql_summary(session, entity, tenant_filter)}

    async def annotate_task(self, tenant_id: str, task_id: UUID, note: str) -> dict[str, Any]:
        async with self._db.session(tenant_id) as session:
            repository = TaskRepository(session, tenant_id)
            task = await repository.get(task_id)
            if task is None:
                raise ToolError("task not found")
            notes = list(task.metadata_.get("notes", []))
            if len(notes) >= MAX_NOTES_PER_TASK:
                raise ToolError(f"a task holds at most {MAX_NOTES_PER_TASK} notes")
            notes.append(note)
            await repository.update(task_id, metadata_={**task.metadata_, "notes": notes})
            return {"task_id": str(task_id), "notes": len(notes)}


_ENTITY_MODELS: dict[str, Any] = {
    "tasks": TaskModel,
    "runs": RunModel,
    "documents": DocumentModel,
    "agent_runs": AgentRunModel,
}


def _task_dict(task: TaskModel) -> dict[str, Any]:
    return {
        "id": str(task.id),
        "status": task.status,
        "user_request": task.user_request,
        "created_at": task.created_at.isoformat(),
    }


def _row(entity: str, record: Any) -> dict[str, Any]:
    if entity == "tasks":
        return _task_dict(record)
    return {attr.key: getattr(record, attr.key) for attr in inspect(record).mapper.column_attrs}


async def _sql_summary(session: AsyncSession, entity: str, tenant_filter: Any) -> dict[str, Any]:
    if entity == "tasks":
        rows = await session.execute(
            select(TaskModel.status, func.count()).where(tenant_filter).group_by(TaskModel.status)
        )
        by_status = {status: count for status, count in rows.all()}  # noqa: C416
        return {"total": sum(by_status.values()), "by_status": by_status}
    if entity == "runs":
        row = (
            await session.execute(
                select(
                    func.count(),
                    func.avg(RunModel.cost_usd),
                    func.avg(RunModel.agent_calls),
                    func.avg(RunModel.step_count),
                    func.coalesce(func.sum(RunModel.retry_count), 0),
                    func.coalesce(func.sum(RunModel.input_tokens + RunModel.output_tokens), 0),
                ).where(tenant_filter)
            )
        ).one()
        if not row[0]:
            return {"total": 0}
        return {
            "total": row[0],
            "avg_cost_usd": round(float(row[1] or 0), 6),
            "avg_agent_calls": round(float(row[2] or 0), 2),
            "avg_step_count": round(float(row[3] or 0), 2),
            "total_retries": int(row[4]),
            "total_tokens": int(row[5]),
        }
    if entity == "documents":
        row = (
            await session.execute(
                select(func.count(), func.coalesce(func.sum(DocumentModel.chunk_count), 0)).where(
                    tenant_filter
                )
            )
        ).one()
        return {"total": row[0], "chunks": int(row[1])}
    agent_rows = await session.execute(
        select(
            AgentRunModel.agent,
            func.count(),
            func.sum(case((AgentRunModel.status != "COMPLETED", 1), else_=0)),
            func.avg(AgentRunModel.latency_ms),
        )
        .where(tenant_filter)
        .group_by(AgentRunModel.agent)
    )
    by_agent = {
        agent: {
            "calls": calls,
            "failures": int(failures or 0),
            "avg_latency_ms": round(float(latency or 0), 1),
        }
        for agent, calls, failures, latency in agent_rows.all()
    }
    return {"total": sum(stats["calls"] for stats in by_agent.values()), "by_agent": by_agent}


def task_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "status": row["status"],
        "request": truncate(row["user_request"], 200),
        "created_at": str(row["created_at"]),
    }


def compact(entity: str, row: dict[str, Any]) -> dict[str, Any]:
    if entity == "tasks":
        return task_view(row)
    keep = {
        "runs": ("id", "status", "step_count", "agent_calls", "retry_count", "cost_usd"),
        "documents": ("id", "title", "chunk_count", "created_at"),
        "agent_runs": ("agent", "status", "latency_ms", "input_tokens", "output_tokens"),
    }[entity]
    return {key: (str(row[key]) if key == "id" else row.get(key)) for key in keep}


def summarise(entity: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if entity == "tasks":
        return {"total": total, "by_status": dict(Counter(row["status"] for row in rows))}
    if entity == "runs":
        if not total:
            return {"total": 0}
        return {
            "total": total,
            "avg_cost_usd": round(sum(float(r["cost_usd"]) for r in rows) / total, 6),
            "avg_agent_calls": round(sum(r["agent_calls"] for r in rows) / total, 2),
            "avg_step_count": round(sum(r["step_count"] for r in rows) / total, 2),
            "total_retries": sum(r["retry_count"] for r in rows),
            "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
        }
    if entity == "documents":
        return {"total": total, "chunks": sum(row.get("chunk_count", 0) for row in rows)}
    by_agent: dict[str, dict[str, float]] = {}
    for row in rows:
        stats = by_agent.setdefault(row["agent"], {"calls": 0, "failures": 0, "latency_ms": 0.0})
        stats["calls"] += 1
        stats["failures"] += row["status"] != "COMPLETED"
        stats["latency_ms"] += row.get("latency_ms", 0)
    for stats in by_agent.values():
        stats["avg_latency_ms"] = round(stats.pop("latency_ms") / stats["calls"], 1)
    return {"total": total, "by_agent": by_agent}
