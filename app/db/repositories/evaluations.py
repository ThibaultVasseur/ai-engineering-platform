from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EvaluationModel


class EvaluationRepository:
    def __init__(self, session: AsyncSession, tenant_id: str) -> None:
        self._session = session
        self._tenant_id = tenant_id

    async def add(self, dataset: str, config: dict[str, Any]) -> EvaluationModel:
        evaluation = EvaluationModel(
            id=uuid.uuid4(),
            tenant_id=self._tenant_id,
            dataset=dataset,
            status="RUNNING",
            config=config,
        )
        self._session.add(evaluation)
        await self._session.flush()
        await self._session.refresh(evaluation)
        return evaluation

    async def get(self, evaluation_id: uuid.UUID) -> EvaluationModel | None:
        return await self._session.scalar(
            select(EvaluationModel).where(
                EvaluationModel.tenant_id == self._tenant_id, EvaluationModel.id == evaluation_id
            )
        )

    async def list(self, *, limit: int) -> Sequence[EvaluationModel]:
        stmt = (
            select(EvaluationModel)
            .where(EvaluationModel.tenant_id == self._tenant_id)
            .order_by(EvaluationModel.created_at.desc())
            .limit(limit)
        )
        return (await self._session.scalars(stmt)).all()

    async def update(self, evaluation_id: uuid.UUID, **values: Any) -> None:
        await self._session.execute(
            update(EvaluationModel)
            .where(
                EvaluationModel.tenant_id == self._tenant_id, EvaluationModel.id == evaluation_id
            )
            .values(**values)
        )
