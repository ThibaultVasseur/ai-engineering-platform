"""Liveness (process is up) and readiness (dependencies reachable) probes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import __version__
from app.api.dependencies import ContainerDep
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(container: ContainerDep) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, environment=container.settings.app_env)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe (database reachable)",
    responses={503: {"model": ReadinessResponse}},
)
async def readiness(container: ContainerDep) -> JSONResponse:
    settings = container.settings
    database_ok = await container.database.ping()
    body = ReadinessResponse(
        status="ready" if database_ok else "degraded",
        database="ok" if database_ok else "unavailable",
        llm_provider=settings.llm_provider,
        llm_model=settings.effective_llm_model,
        embedding_provider=settings.embedding_provider,
        langfuse_enabled=settings.langfuse_enabled,
    )
    return JSONResponse(status_code=200 if database_ok else 503, content=body.model_dump())
