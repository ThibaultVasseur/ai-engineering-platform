"""FastAPI application factory.

Run with ``uvicorn app.main:create_app --factory``. The factory builds nothing at import time:
configuration is validated and dependencies are wired inside the lifespan.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.api.routes import agents, evaluation, health, knowledge, runs, tasks
from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, log_event

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"

DESCRIPTION = """
Multi-agent orchestration platform: a prompt optimizer and a planner structure the request,
a supervisor dispatches plan steps to specialised agents (research, data, coding, RAG) that use
permissioned tools, a critic reviews the synthesised answer and triggers bounded rework.
Every run is traced (run events, token usage, cost) and scoped to a tenant.
"""


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or build_container(settings)
        log_event(
            logger,
            "app_started",
            environment=settings.app_env,
            llm_provider=settings.llm_provider,
            llm_model=settings.effective_llm_model,
            auth_enabled=settings.auth_enabled,
        )
        if not settings.auth_enabled:
            log_event(
                logger,
                "auth_disabled",
                level=logging.WARNING,
                detail=f"all requests are served as tenant '{settings.default_tenant_id}'",
            )
        try:
            yield
        finally:
            if owned:
                await app.state.container.aclose()
            log_event(logger, "app_stopped")

    app = FastAPI(
        title="AI Engineering Multi-Agent Platform",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_body_bytes)
    # Added last, so outermost: every response (a 413 included) carries a request id.
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(tasks.router, prefix=API_PREFIX)
    app.include_router(runs.router, prefix=API_PREFIX)
    app.include_router(knowledge.router, prefix=API_PREFIX)
    app.include_router(agents.router, prefix=API_PREFIX)
    app.include_router(evaluation.router, prefix=API_PREFIX)
    return app
