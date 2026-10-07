"""Runs, run events and agent runs persistence (tenant-scoped)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRunModel, RunEventModel, RunModel
from app.schemas.run import RunStatus


class RunRepository:
    def __init__(self, session: AsyncSession, tenant_id: str) -> None:
        self._session = session
        self._tenant_id = tenant_id

    async def add(self, task_id: uuid.UUID, config: dict[str, Any]) -> RunModel:
        run = RunModel(
            id=uuid.uuid4(),
            tenant_id=self._tenant_id,
            task_id=task_id,
            status=RunStatus.PENDING,
            config=config,
        )
        self._session.add(run)
        await self._session.flush()
        await self._session.refresh(run)
        return run

    async def get(self, run_id: uuid.UUID) -> RunModel | None:
        return await self._session.scalar(
            select(RunModel).where(RunModel.tenant_id == self._tenant_id, RunModel.id == run_id)
        )

    async def list_for_task(self, task_id: uuid.UUID, *, limit: int) -> Sequence[RunModel]:
        stmt = (
            select(RunModel)
            .where(RunModel.tenant_id == self._tenant_id, RunModel.task_id == task_id)
            .order_by(RunModel.created_at.desc())
            .limit(limit)
        )
        return (await self._session.scalars(stmt)).all()

    async def update(self, run_id: uuid.UUID, **values: Any) -> None:
        await self._session.execute(
            update(RunModel)
            .where(RunModel.tenant_id == self._tenant_id, RunModel.id == run_id)
            .values(**values, updated_at=func.now())
        )


class RunEventRepository:
    def __init__(self, session: AsyncSession, tenant_id: str) -> None:
        self._session = session
        self._tenant_id = tenant_id

    async def add(
        self,
        run_id: uuid.UUID,
        *,
        sequence: int,
        event_type: str,
        node: str | None,
        agent: str | None,
        step_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self._session.add(
            RunEventModel(
                run_id=run_id,
                tenant_id=self._tenant_id,
                sequence=sequence,
                event_type=event_type,
                node=node,
                agent=agent,
                step_id=step_id,
                payload=payload,
            )
        )
        await self._session.flush()

    async def list(self, run_id: uuid.UUID, *, after: int, limit: int) -> Sequence[RunEventModel]:
        stmt = (
            select(RunEventModel)
            .where(
                RunEventModel.tenant_id == self._tenant_id,
                RunEventModel.run_id == run_id,
                RunEventModel.sequence > after,
            )
            .order_by(RunEventModel.sequence)
            .limit(limit)
        )
        return (await self._session.scalars(stmt)).all()


class AgentRunRepository:
    def __init__(self, session: AsyncSession, tenant_id: str) -> None:
        self._session = session
        self._tenant_id = tenant_id

    async def add(self, run_id: uuid.UUID, **values: Any) -> None:
        self._session.add(
            AgentRunModel(id=uuid.uuid4(), run_id=run_id, tenant_id=self._tenant_id, **values)
        )
        await self._session.flush()

    async def list(self, run_id: uuid.UUID) -> Sequence[AgentRunModel]:
        stmt = (
            select(AgentRunModel)
            .where(AgentRunModel.tenant_id == self._tenant_id, AgentRunModel.run_id == run_id)
            .order_by(AgentRunModel.created_at)
        )
        return (await self._session.scalars(stmt)).all()
