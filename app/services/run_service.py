"""Run lifecycle: create, execute (background or blocking), observe, cancel.

Runs execute in-process as asyncio tasks, at most MAX_CONCURRENT_RUNS at a time. This keeps the
project dependency-free; a production deployment would hand runs to a durable queue/worker (see
docs/decisions.md) so that they survive restarts.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import log_context, log_event
from app.core.security import Tenant
from app.db.database import Database
from app.db.models import RunModel
from app.db.repositories.runs import RunEventRepository, RunRepository
from app.db.repositories.tasks import TaskRepository
from app.llm import prompts
from app.observability.db_tracer import DatabaseTracer
from app.observability.tracing import CompositeTracer, LoggingTracer, Tracer
from app.orchestration.context import RunContext
from app.orchestration.factory import RunDependencies, build_run_context
from app.orchestration.runner import GraphRunner, RunAbortedError, final_result_of
from app.orchestration.state import GraphState
from app.schemas.common import Page
from app.schemas.run import (
    TERMINAL_RUN_STATUSES,
    FinalResult,
    FinalStatus,
    RunCreate,
    RunEventPage,
    RunEventRead,
    RunMetricsRead,
    RunOptions,
    RunRead,
    RunStatus,
)
from app.schemas.task import TaskStatus

logger = logging.getLogger(__name__)

NODE_TASK_STATUS = {
    "prompt_optimizer": TaskStatus.PLANNING,
    "planner": TaskStatus.PLANNING,
    "supervisor": TaskStatus.RUNNING,
    "research": TaskStatus.RUNNING,
    "data": TaskStatus.RUNNING,
    "coding": TaskStatus.RUNNING,
    "rag": TaskStatus.RUNNING,
    "synthesizer": TaskStatus.REVIEWING,
    "critic": TaskStatus.REVIEWING,
}
PROMPT_VERSIONS = {
    template.name: template.version
    for template in (
        prompts.PROMPT_OPTIMIZER,
        prompts.PLANNER,
        prompts.SUPERVISOR,
        prompts.RESEARCH,
        prompts.DATA,
        prompts.CODING,
        prompts.RAG,
        prompts.SYNTHESIZER,
        prompts.CRITIC,
    )
}

TracerFactory = Callable[[str, uuid.UUID], Tracer]


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_run_read(run: RunModel) -> RunRead:
    return RunRead(
        id=run.id,
        task_id=run.task_id,
        status=RunStatus(run.status),
        config=run.config,
        plan=run.plan,
        review=run.review,
        final_result=run.final_result,
        error=run.error,
        metrics=RunMetricsRead(
            step_count=run.step_count,
            agent_calls=run.agent_calls,
            retry_count=run.retry_count,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            cost_usd=float(run.cost_usd),
        ),
        trace_id=run.trace_id,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


@dataclass(frozen=True)
class RunJob:
    tenant_id: str
    run_id: uuid.UUID
    task_id: uuid.UUID
    request: str
    options: RunOptions


class RunService:
    def __init__(
        self,
        database: Database,
        deps: RunDependencies,
        settings: Settings,
        *,
        runner: GraphRunner | None = None,
        tracer_factories: Sequence[TracerFactory] = (),
    ) -> None:
        self._db = database
        self._deps = deps
        self._settings = settings
        self._runner = runner or GraphRunner()
        self._tracer_factories = list(tracer_factories)
        self._active: dict[uuid.UUID, asyncio.Task[None]] = {}
        self._slots = asyncio.Semaphore(settings.max_concurrent_runs)

    # ----------------------------------------------------------------------------------
    # Queries
    # ----------------------------------------------------------------------------------
    async def get(self, tenant: Tenant, run_id: uuid.UUID) -> RunRead:
        async with self._db.session(tenant.id) as session:
            run = await RunRepository(session, tenant.id).get(run_id)
            if run is None:
                raise NotFoundError(f"Run {run_id} not found")
            return to_run_read(run)

    async def list_for_task(
        self, tenant: Tenant, task_id: uuid.UUID, *, limit: int
    ) -> Page[RunRead]:
        async with self._db.session(tenant.id) as session:
            if await TaskRepository(session, tenant.id).get(task_id) is None:
                raise NotFoundError(f"Task {task_id} not found")
            runs = await RunRepository(session, tenant.id).list_for_task(task_id, limit=limit)
            return Page[RunRead](items=[to_run_read(run) for run in runs], limit=limit, offset=0)

    async def events(
        self, tenant: Tenant, run_id: uuid.UUID, *, after: int, limit: int
    ) -> RunEventPage:
        async with self._db.session(tenant.id) as session:
            run = await RunRepository(session, tenant.id).get(run_id)
            if run is None:
                raise NotFoundError(f"Run {run_id} not found")
            events = await RunEventRepository(session, tenant.id).list(
                run_id, after=after, limit=limit
            )
            items = [
                RunEventRead(
                    sequence=event.sequence,
                    event_type=event.event_type,
                    node=event.node,
                    agent=event.agent,
                    step_id=event.step_id,
                    payload=event.payload,
                    created_at=event.created_at,
                )
                for event in events
            ]
            return RunEventPage(
                items=items,
                next_after=items[-1].sequence if items else after,
                run_status=RunStatus(run.status),
            )

    # ----------------------------------------------------------------------------------
    # Commands
    # ----------------------------------------------------------------------------------
    async def create(self, tenant: Tenant, payload: RunCreate) -> RunRead:
        async with self._db.session(tenant.id) as session:
            tasks = TaskRepository(session, tenant.id)
            if payload.task_id is not None:
                task = await tasks.get(payload.task_id)
                if task is None:
                    raise NotFoundError(f"Task {payload.task_id} not found")
                if await self._has_active_run(session, tenant.id, task.id):
                    raise ConflictError(f"Task {task.id} already has an active run")
            else:
                assert payload.request is not None  # noqa: S101 - enforced by RunCreate
                task = await tasks.add(payload.request, {})
            run = await RunRepository(session, tenant.id).add(
                task.id, self._run_config(payload.options)
            )
            await tasks.update(task.id, status=TaskStatus.PENDING, error=None)
            job = RunJob(tenant.id, run.id, task.id, task.user_request, payload.options)

        if payload.wait:
            await self._execute(job)
        else:
            self._spawn(job)
        return await self.get(tenant, job.run_id)

    async def cancel(self, tenant: Tenant, run_id: uuid.UUID) -> RunRead:
        current = await self.get(tenant, run_id)
        if current.status in TERMINAL_RUN_STATUSES:
            raise ConflictError(f"Run {run_id} is already {current.status.value}")
        task = self._active.get(run_id)
        if task is not None:
            task.cancel()
            await asyncio.wait({task}, timeout=10)
        else:  # not executing in this process (e.g. after a restart): close it explicitly
            await self._persist(
                RunJob(tenant.id, run_id, current.task_id, "", RunOptions()),
                run={"status": RunStatus.CANCELLED, "finished_at": utcnow(), "error": "Cancelled"},
                task={"status": TaskStatus.CANCELLED},
            )
        return await self.get(tenant, run_id)

    async def shutdown(self) -> None:
        for task in list(self._active.values()):
            task.cancel()
        if self._active:
            await asyncio.wait(set(self._active.values()), timeout=10)

    # ----------------------------------------------------------------------------------
    # Execution
    # ----------------------------------------------------------------------------------
    def _spawn(self, job: RunJob) -> None:
        task = asyncio.create_task(self._execute(job), name=f"run-{job.run_id}")
        self._active[job.run_id] = task
        task.add_done_callback(lambda _: self._active.pop(job.run_id, None))

    async def _execute(self, job: RunJob) -> None:
        async with self._slots:
            with log_context(
                run_id=str(job.run_id), task_id=str(job.task_id), tenant_id=job.tenant_id
            ):
                await self._run(job)

    async def _run(self, job: RunJob) -> None:
        tracers: list[Tracer] = [
            DatabaseTracer(self._db, tenant_id=job.tenant_id, run_id=job.run_id),
            LoggingTracer(),
            *(factory(job.tenant_id, job.run_id) for factory in self._tracer_factories),
        ]
        context = build_run_context(
            self._deps,
            run_id=str(job.run_id),
            tenant_id=job.tenant_id,
            tracer=CompositeTracer(tracers),
            allow_tool_writes=job.options.allow_tool_writes,
            supervisor_strategy=job.options.supervisor_strategy,
        )
        trace_id = next(
            (getattr(t, "trace_id", None) for t in tracers if getattr(t, "trace_id", None)), None
        )
        await self._persist(
            job,
            run={"status": RunStatus.RUNNING, "started_at": utcnow(), "trace_id": trace_id},
            task={"status": TaskStatus.PLANNING},
        )
        progress = _Progress(job, self._persist)
        try:
            state = await self._runner.run(
                context,
                user_request=job.request,
                task_id=str(job.task_id),
                on_update=progress.on_update,
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self._close(job, context, RunStatus.CANCELLED, "Cancelled", TaskStatus.CANCELLED)
            )
            raise
        except RunAbortedError as exc:
            await self._close(job, context, RunStatus.FAILED, str(exc), TaskStatus.FAILED)
        except Exception as exc:  # a bug must not kill the worker silently
            log_event(logger, "run_crashed", level=logging.ERROR, exc_info=exc)
            await self._close(
                job,
                context,
                RunStatus.FAILED,
                f"internal error ({type(exc).__name__})",
                TaskStatus.FAILED,
            )
        else:
            await self._complete(job, state, final_result_of(state))

    async def _complete(self, job: RunJob, state: GraphState, result: FinalResult) -> None:
        run_status = (
            RunStatus.FAILED if result.status is FinalStatus.FAILED else RunStatus.COMPLETED
        )
        error = None if result.accepted else "; ".join(result.warnings)[:2000] or "Not accepted"
        plan = state.get("plan")
        spec = state.get("optimized_prompt")
        dumped = result.model_dump(mode="json")
        await self._persist(
            job,
            run={
                "status": run_status,
                "plan": plan.model_dump(mode="json") if plan else None,
                "review": dumped["review"],
                "final_result": dumped,
                "error": error,
                "step_count": result.metrics.step_count,
                "agent_calls": result.metrics.agent_calls,
                "retry_count": result.metrics.retry_count,
                "input_tokens": result.metrics.input_tokens,
                "output_tokens": result.metrics.output_tokens,
                "cost_usd": result.metrics.cost_usd,
                "finished_at": utcnow(),
            },
            task={
                "status": TaskStatus.COMPLETED if result.accepted else TaskStatus.FAILED,
                "optimized_prompt": spec.model_dump(mode="json") if spec else None,
                "result": dumped,
                "error": error,
            },
        )

    async def _close(
        self,
        job: RunJob,
        context: RunContext,
        status: RunStatus,
        error: str,
        task_status: TaskStatus,
    ) -> None:
        snapshot = context.agent_ctx.budget.snapshot()
        await self._persist(
            job,
            run={
                "status": status,
                "error": error[:2000],
                "agent_calls": snapshot["agent_calls"],
                "input_tokens": snapshot["input_tokens"],
                "output_tokens": snapshot["output_tokens"],
                "cost_usd": snapshot["cost_usd"],
                "finished_at": utcnow(),
            },
            task={"status": task_status, "error": error[:2000]},
        )

    async def _persist(
        self, job: RunJob, *, run: dict[str, Any] | None = None, task: dict[str, Any] | None = None
    ) -> None:
        async with self._db.session(job.tenant_id) as session:
            if run:
                await RunRepository(session, job.tenant_id).update(job.run_id, **run)
            if task:
                await TaskRepository(session, job.tenant_id).update(job.task_id, **task)

    async def _has_active_run(self, session: Any, tenant_id: str, task_id: uuid.UUID) -> bool:
        runs = await RunRepository(session, tenant_id).list_for_task(task_id, limit=5)
        return any(run.status in {RunStatus.PENDING, RunStatus.RUNNING} for run in runs)

    def _run_config(self, options: RunOptions) -> dict[str, Any]:
        settings = self._settings
        return {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.effective_llm_model,
            "embedding_provider": settings.embedding_provider,
            "supervisor_strategy": options.supervisor_strategy or settings.supervisor_strategy,
            "allow_tool_writes": options.allow_tool_writes,
            "limits": asdict(self._deps.limits),
            "prompt_versions": PROMPT_VERSIONS,
        }


Persist = Callable[..., Awaitable[None]]


class _Progress:
    """Mirrors graph progress onto the task (status) and run (plan) while the run executes."""

    def __init__(self, job: RunJob, persist: Persist) -> None:
        self._job = job
        self._persist = persist
        self._status: TaskStatus | None = TaskStatus.PLANNING

    async def on_update(self, node: str, update: dict[str, Any]) -> None:
        task_values: dict[str, Any] = {}
        run_values: dict[str, Any] = {}
        status = NODE_TASK_STATUS.get(node)
        if status is not None and status is not self._status:
            self._status = status
            task_values["status"] = status
        if node == "prompt_optimizer" and update.get("optimized_prompt") is not None:
            task_values["optimized_prompt"] = update["optimized_prompt"].model_dump(mode="json")
        if node == "planner" and update.get("plan") is not None:
            run_values["plan"] = update["plan"].model_dump(mode="json")
        if task_values or run_values:
            await self._persist(self._job, run=run_values or None, task=task_values or None)
