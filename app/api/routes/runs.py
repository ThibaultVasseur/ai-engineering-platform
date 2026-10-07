from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response, status

from app.api.dependencies import RunServiceDep, TenantDep, rate_limit
from app.schemas.common import ErrorResponse
from app.schemas.run import TERMINAL_RUN_STATUSES, RunCreate, RunEventPage, RunRead

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunRead,
    summary="Execute the multi-agent graph for a task (or a new request)",
    responses={
        200: {"model": RunRead, "description": "wait=true: the run is finished"},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "The task already has an active run"},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    dependencies=[rate_limit(2)],
)
async def create_run(
    payload: RunCreate, tenant: TenantDep, service: RunServiceDep, response: Response
) -> RunRead:
    run = await service.create(tenant, payload)
    if run.status in TERMINAL_RUN_STATUSES:
        response.status_code = status.HTTP_200_OK
    return run


@router.get(
    "/{run_id}",
    response_model=RunRead,
    summary="Run status, plan, review, final result and metrics",
    responses={404: {"model": ErrorResponse}},
)
async def get_run(run_id: UUID, tenant: TenantDep, service: RunServiceDep) -> RunRead:
    return await service.get(tenant, run_id)


@router.get(
    "/{run_id}/events",
    response_model=RunEventPage,
    summary="Ordered trace events of a run (poll with ?after=<next_after>)",
    responses={404: {"model": ErrorResponse}},
)
async def get_run_events(
    run_id: UUID,
    tenant: TenantDep,
    service: RunServiceDep,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> RunEventPage:
    return await service.events(tenant, run_id, after=after, limit=limit)


@router.post(
    "/{run_id}/cancel",
    response_model=RunRead,
    summary="Cancel a running run",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def cancel_run(run_id: UUID, tenant: TenantDep, service: RunServiceDep) -> RunRead:
    return await service.cancel(tenant, run_id)
