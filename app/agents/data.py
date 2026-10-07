"""Data agent: analyses platform data through safe database tools and the calculator."""

from __future__ import annotations

from app.agents.base import AgentContext, BaseAgent
from app.agents.registry import AGENT_PROFILES, AgentName
from app.agents.worker_support import worker_payload
from app.llm.prompts import DATA
from app.schemas.agent import DataAnalysis, WorkerInput


def _metrics_have_methods(value: DataAnalysis) -> None:
    missing = [metric.name for metric in value.metrics if not metric.method.strip()]
    if missing:
        raise ValueError(f"metrics without a method (query or calculation): {', '.join(missing)}")


class DataAgent(BaseAgent[WorkerInput, DataAnalysis]):
    profile = AGENT_PROFILES[AgentName.DATA]
    prompt = DATA

    async def execute(
        self, ctx: AgentContext, data: WorkerInput, *, step_id: str | None
    ) -> DataAnalysis:
        return await self._tool_loop(
            ctx,
            worker_payload(data),
            DataAnalysis,
            step_id=step_id,
            validate=_metrics_have_methods,
        )

    def summarize(self, value: DataAnalysis) -> str:
        return value.summary[:240]
