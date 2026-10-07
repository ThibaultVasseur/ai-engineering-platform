from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from app.api.dependencies import RunServiceDep, TaskServiceDep, TenantDep, rate_limit
from app.schemas.common import ErrorResponse, Page
from app.schemas.run import RunRead
from app.schemas.task import TaskCreate, TaskRead, TaskStatus

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskRead,
    summary="Create a task from a natural-language request",
    responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    dependencies=[rate_limit(1)],
)
async def create_task(payload: TaskCreate, tenant: TenantDep, service: TaskServiceDep) -> TaskRead:
    return await service.create(tenant, payload)


@router.get("", response_model=Page[TaskRead], summary="List tasks (most recent first)")
async def list_tasks(
    tenant: TenantDep,
    service: TaskServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
    status_filter: Annotated[TaskStatus | None, Query(alias="status")] = None,
) -> Page[TaskRead]:
    return await service.list(tenant, limit=limit, offset=offset, status=status_filter)


@router.get(
    "/{task_id}",
    response_model=TaskRead,
    summary="Get a task",
    responses={404: {"model": ErrorResponse}},
)
async def get_task(task_id: UUID, tenant: TenantDep, service: TaskServiceDep) -> TaskRead:
    return await service.get(tenant, task_id)


@router.get(
    "/{task_id}/runs",
    response_model=Page[RunRead],
    summary="Runs of a task (most recent first)",
    responses={404: {"model": ErrorResponse}},
)
async def list_task_runs(
    task_id: UUID,
    tenant: TenantDep,
    service: RunServiceDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> Page[RunRead]:
    return await service.list_for_task(tenant, task_id, limit=limit)
