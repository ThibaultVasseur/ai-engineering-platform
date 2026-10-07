"""FastAPI dependencies: container access, authentication/tenant resolution, services."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, Header, Request

from app.container import Container
from app.core.exceptions import AuthenticationError, RateLimitedError
from app.core.logging import bind_log_context, reset_log_context
from app.core.security import Tenant
from app.rag.service import KnowledgeService
from app.services.evaluation_service import EvaluationService
from app.services.run_service import RunService
from app.services.task_service import TaskService


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_tenant(
    container: ContainerDep,
    x_api_key: Annotated[str | None, Header(description="Tenant API key")] = None,
) -> AsyncIterator[Tenant]:
    settings = container.settings
    if settings.auth_enabled:
        tenant = container.authenticator.authenticate(x_api_key)
        if tenant is None:
            raise AuthenticationError("Missing or invalid API key (X-API-Key header)")
    else:
        tenant = Tenant(id=settings.default_tenant_id)
    token = bind_log_context(tenant_id=tenant.id)
    try:
        yield tenant
    finally:
        reset_log_context(token)


TenantDep = Annotated[Tenant, Depends(get_tenant)]


def rate_limit(cost: float = 1.0) -> Any:
    """Dependency factory: spend ``cost`` tokens of the tenant's bucket or answer 429."""

    async def enforce(tenant: TenantDep, container: ContainerDep) -> None:
        if not container.settings.rate_limit_enabled:
            return
        retry_after = container.rate_limiter.acquire(tenant.id, cost)
        if retry_after is not None:
            raise RateLimitedError(
                "Rate limit exceeded for this tenant", retry_after_seconds=math.ceil(retry_after)
            )

    return Depends(enforce)


def get_task_service(container: ContainerDep) -> TaskService:
    return container.task_service


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]


def get_knowledge_service(container: ContainerDep) -> KnowledgeService:
    return container.knowledge_service


KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


def get_run_service(container: ContainerDep) -> RunService:
    return container.run_service


RunServiceDep = Annotated[RunService, Depends(get_run_service)]


def get_evaluation_service(container: ContainerDep) -> EvaluationService:
    return container.evaluation_service


EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]
