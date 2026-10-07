"""Task persistence. Every query filters on the tenant explicitly (RLS enforces it again)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TaskModel
from app.schemas.task import TaskStatus


def escape_like(value: str) -> str:
    """Escape LIKE wildcards so user input is matched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class TaskRepository:
    def __init__(self, session: AsyncSession, tenant_id: str) -> None:
        self._session = session
        self._tenant_id = tenant_id

    async def add(self, user_request: str, metadata: dict[str, Any]) -> TaskModel:
        task = TaskModel(
            id=uuid.uuid4(),
            tenant_id=self._tenant_id,
            user_request=user_request,
            status=TaskStatus.PENDING,
            metadata_=metadata,
        )
        self._session.add(task)
        await self._session.flush()
        await self._session.refresh(task)
        return task

    async def get(self, task_id: uuid.UUID) -> TaskModel | None:
        stmt = select(TaskModel).where(
            TaskModel.tenant_id == self._tenant_id, TaskModel.id == task_id
        )
        return await self._session.scalar(stmt)

    async def list(
        self, *, limit: int, offset: int, status: TaskStatus | None = None
    ) -> Sequence[TaskModel]:
        stmt = select(TaskModel).where(TaskModel.tenant_id == self._tenant_id)
        if status is not None:
            stmt = stmt.where(TaskModel.status == status)
        stmt = stmt.order_by(TaskModel.created_at.desc()).limit(limit).offset(offset)
        return (await self._session.scalars(stmt)).all()

    async def search(self, query: str, *, limit: int) -> Sequence[TaskModel]:
        pattern = f"%{escape_like(query)}%"
        stmt = (
            select(TaskModel)
            .where(
                TaskModel.tenant_id == self._tenant_id,
                TaskModel.user_request.ilike(pattern, escape="\\"),
            )
            .order_by(TaskModel.created_at.desc())
            .limit(limit)
        )
        return (await self._session.scalars(stmt)).all()

    async def update(self, task_id: uuid.UUID, **values: Any) -> None:
        stmt = (
            update(TaskModel)
            .where(TaskModel.tenant_id == self._tenant_id, TaskModel.id == task_id)
            .values(**values, updated_at=func.now())
        )
        await self._session.execute(stmt)

    async def count_by_status(self) -> dict[str, int]:
        stmt = (
            select(TaskModel.status, func.count())
            .where(TaskModel.tenant_id == self._tenant_id)
            .group_by(TaskModel.status)
        )
        rows = (await self._session.execute(stmt)).all()
        return {status: count for status, count in rows}  # noqa: C416 - Row -> plain types
