"""Evaluation use cases (API side).

Evaluations run in a *sandbox*: an in-memory runtime loaded with the demo corpus, using the
configured LLM provider. Results are therefore reproducible and never touch tenant data; the
report is persisted in the tenant's ``evaluations`` table.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.exceptions import InvalidRequestError, NotFoundError
from app.core.security import Tenant
from app.db.database import Database
from app.db.models import EvaluationModel
from app.db.repositories.evaluations import EvaluationRepository
from app.evaluation.datasets import load_dataset
from app.evaluation.runner import EvaluationRunner
from app.orchestration.local import build_in_memory_dependencies
from app.schemas.evaluation import EvaluationRead, EvaluationRequest

SANDBOX_TENANT = "evaluation-sandbox"


def to_read(model: EvaluationModel) -> EvaluationRead:
    return EvaluationRead(
        id=model.id,
        dataset=model.dataset,
        status=model.status,
        config=model.config,
        summary=model.summary,
        results=model.results,
        created_at=model.created_at,
        finished_at=model.finished_at,
    )


class EvaluationService:
    def __init__(self, database: Database, settings: Settings) -> None:
        self._db = database
        self._settings = settings

    async def run(self, tenant: Tenant, request: EvaluationRequest) -> EvaluationRead:
        try:
            dataset = load_dataset(request.dataset)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        unknown = set(request.case_ids) - {case.id for case in dataset.cases}
        if unknown:
            raise InvalidRequestError(f"unknown case ids: {sorted(unknown)}")

        config = {
            "llm_provider": self._settings.llm_provider,
            "llm_model": self._settings.effective_llm_model,
            "dataset_version": dataset.version,
            "case_ids": request.case_ids,
        }
        async with self._db.session(tenant.id) as session:
            evaluation = await EvaluationRepository(session, tenant.id).add(dataset.name, config)
            evaluation_id = evaluation.id

        try:
            deps = await build_in_memory_dependencies(self._settings, tenant_id=SANDBOX_TENANT)
            report = await EvaluationRunner(deps, tenant_id=SANDBOX_TENANT).run(
                dataset, case_ids=request.case_ids or None
            )
        except Exception as exc:
            await self._finish(
                tenant, evaluation_id, status="FAILED", summary={"error": str(exc)[:500]}
            )
            raise
        await self._finish(
            tenant,
            evaluation_id,
            status="COMPLETED",
            summary=report.summary.model_dump(mode="json"),
            results=[case.model_dump(mode="json") for case in report.cases],
        )
        return await self.get(tenant, evaluation_id)

    async def get(self, tenant: Tenant, evaluation_id: uuid.UUID) -> EvaluationRead:
        async with self._db.session(tenant.id) as session:
            evaluation = await EvaluationRepository(session, tenant.id).get(evaluation_id)
            if evaluation is None:
                raise NotFoundError(f"Evaluation {evaluation_id} not found")
            return to_read(evaluation)

    async def list(self, tenant: Tenant, *, limit: int) -> list[EvaluationRead]:
        async with self._db.session(tenant.id) as session:
            rows = await EvaluationRepository(session, tenant.id).list(limit=limit)
            return [to_read(row) for row in rows]

    async def _finish(self, tenant: Tenant, evaluation_id: uuid.UUID, **values: object) -> None:
        async with self._db.session(tenant.id) as session:
            await EvaluationRepository(session, tenant.id).update(
                evaluation_id, finished_at=datetime.now(UTC), **values
            )
