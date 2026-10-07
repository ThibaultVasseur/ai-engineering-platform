"""Task use cases. Routes stay thin: they validate HTTP input and delegate here."""

from __future__ import annotations

from uuid import UUID

from app.core.exceptions import NotFoundError
from app.core.security import Tenant
from app.db.database import Database
from app.db.repositories.tasks import TaskRepository
from app.schemas.common import Page
from app.schemas.task import TaskCreate, TaskRead, TaskStatus


class TaskService:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, tenant: Tenant, payload: TaskCreate) -> TaskRead:
        async with self._db.session(tenant.id) as session:
            task = await TaskRepository(session, tenant.id).add(payload.request, payload.metadata)
            return TaskRead.model_validate(task)

    async def get(self, tenant: Tenant, task_id: UUID) -> TaskRead:
        async with self._db.session(tenant.id) as session:
            task = await TaskRepository(session, tenant.id).get(task_id)
            if task is None:
                # Same answer whether the task does not exist or belongs to another tenant.
                raise NotFoundError(f"Task {task_id} not found")
            return TaskRead.model_validate(task)

    async def list(
        self, tenant: Tenant, *, limit: int, offset: int, status: TaskStatus | None
    ) -> Page[TaskRead]:
        async with self._db.session(tenant.id) as session:
            tasks = await TaskRepository(session, tenant.id).list(
                limit=limit, offset=offset, status=status
            )
            items = [TaskRead.model_validate(task) for task in tasks]
        return Page[TaskRead](items=items, limit=limit, offset=offset)
