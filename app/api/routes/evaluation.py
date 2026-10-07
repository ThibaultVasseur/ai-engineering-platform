from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from app.api.dependencies import EvaluationServiceDep, TenantDep, rate_limit
from app.schemas.common import ErrorResponse
from app.schemas.evaluation import EvaluationRead, EvaluationRequest

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post(
    "/run",
    response_model=EvaluationRead,
    summary="Run an evaluation dataset in a sandbox and persist the report",
    responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    dependencies=[rate_limit(10)],
)
async def run_evaluation(
    payload: EvaluationRequest, tenant: TenantDep, service: EvaluationServiceDep
) -> EvaluationRead:
    return await service.run(tenant, payload)


@router.get("", response_model=list[EvaluationRead], summary="Past evaluations")
async def list_evaluations(
    tenant: TenantDep,
    service: EvaluationServiceDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[EvaluationRead]:
    return await service.list(tenant, limit=limit)


@router.get(
    "/{evaluation_id}",
    response_model=EvaluationRead,
    summary="One evaluation report",
    responses={404: {"model": ErrorResponse}},
)
async def get_evaluation(
    evaluation_id: UUID, tenant: TenantDep, service: EvaluationServiceDep
) -> EvaluationRead:
    return await service.get(tenant, evaluation_id)
