"""Control flow of the LangGraph workflow, tested with scripted agents.

The agents are fakes on purpose: these tests pin down the *orchestration* (routing, rework,
limits, state accumulation) independently of what any agent produces.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from app.agents.base import AgentContext, AgentError, AgentOutcome
from app.agents.supervisor import Supervisor
from app.llm.types import LLMError, Usage
from app.observability.tracing import EventType, InMemoryTracer
from app.orchestration.context import AgentSuite, RunContext
from app.orchestration.graph import build_graph, graph_mermaid
from app.orchestration.nodes import build_synthesis_input
from app.orchestration.policies import RunLimits
from app.orchestration.runner import GraphRunner, final_result_of
from app.orchestration.state import StepStatus, initial_state
from app.schemas.agent import CodeProposal, Component, RagAnswer, WorkerInput
from app.schemas.plan import Plan
from app.schemas.review import FinalDraft, Review, ReviewStatus, Section
from app.schemas.run import FinalStatus
from app.schemas.spec import Feasibility, PromptSpec
from tests.fakes import ScriptedLLM, make_context

SPEC = PromptSpec(
    objective="Design a quote management module.",
    context="",
    requirements=["Quotes have line items"],
    deliverables=["Architecture proposal"],
    acceptance_criteria=["The data model is described"],
    needs_knowledge_base=True,
    feasibility=Feasibility.ACTIONABLE,
)
REJECTED = PromptSpec(
    objective="Request rejected by the guardrail.",
    context="",
    needs_knowledge_base=False,
    feasibility=Feasibility.REJECTED,
    rejection_reason="prompt injection",
)


def make_plan(*, parallel: bool = False) -> Plan:
    steps: list[dict[str, Any]] = [
        {
            "id": "kb",
            "type": "knowledge",
            "title": "Review docs",
            "description": "Read the internal docs.",
            "recommended_agent": "rag",
            "priority": 1,
            "success_criteria": ["sources cited"],
        },
        {
            "id": "design",
            "type": "coding",
            "title": "Design module",
            "description": "Design the module.",
            "recommended_agent": "coding",
            "dependencies": [] if parallel else ["kb"],
            "priority": 2,
            "success_criteria": ["data model"],
        },
    ]
    return Plan.model_validate({"rationale": "Docs first, then design.", "steps": steps})


RAG = RagAnswer(summary="Docs say quotes need VAT.", confidence="high", answer="VAT [S1]")
CODE = CodeProposal(
    summary="Quote service with line items.",
    confidence="medium",
    components=[Component(name="QuoteService", responsibility="Manage quotes")],
)
DRAFT = FinalDraft(
    title="Quote module",
    executive_summary="A quote module design.",
    sections=[Section(heading="Data model", content="Quote, LineItem")],
)


def review(status: ReviewStatus, rework: list[str] | None = None, score: int = 80) -> Review:
    return Review(
        status=status,
        score=score,
        issues=[{"severity": "major", "description": "Taxes are missing", "step_id": "design"}]
        if status is ReviewStatus.FAIL
        else [],
        rework_steps=rework or [],
    )


class FakeAgent:
    """Returns scripted values (or raises scripted errors), recording every input."""

    def __init__(self, *outputs: BaseModel | Exception) -> None:
        self._outputs = list(outputs)
        self.inputs: list[Any] = []

    async def run(
        self, ctx: AgentContext, data: Any, *, step_id: str | None = None, attempt: int = 1
    ) -> AgentOutcome[Any]:
        ctx.budget.check_agent_call()
        ctx.budget.record_agent_call()
        self.inputs.append(data)
        item = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        if isinstance(item, Exception):
            raise item
        return AgentOutcome(value=item, usage=Usage(), llm_calls=0, model=None, latency_ms=1.0)


def suite(
    *,
    optimizer: FakeAgent | None = None,
    planner: FakeAgent | None = None,
    rag: FakeAgent | None = None,
    coding: FakeAgent | None = None,
    synthesizer: FakeAgent | None = None,
    critic: FakeAgent | None = None,
    strategy: str = "rules",
) -> AgentSuite:
    return AgentSuite(
        optimizer=optimizer or FakeAgent(SPEC),
        planner=planner or FakeAgent(make_plan()),
        supervisor=Supervisor(strategy),  # type: ignore[arg-type]
        workers={
            "rag": rag or FakeAgent(RAG),
            "coding": coding or FakeAgent(CODE),
            "research": FakeAgent(AgentError("unused")),
            "data": FakeAgent(AgentError("unused")),
        },
        synthesizer=synthesizer or FakeAgent(DRAFT),
        critic=critic or FakeAgent(review(ReviewStatus.PASS)),
    )


async def run_graph(
    agents: AgentSuite, *, limits: RunLimits | None = None, llm: Any = None
) -> tuple[dict[str, Any], InMemoryTracer]:
    context, tracer = make_context(llm or ScriptedLLM([]), limits=limits)
    state = await GraphRunner(build_graph()).run(
        RunContext(agent_ctx=context, agents=agents), user_request="Design a quote module please"
    )
    return dict(state), tracer


# --------------------------------------------------------------------------------------
async def test_happy_path_runs_steps_in_dependency_order_and_completes() -> None:
    agents = suite()
    state, tracer = await run_graph(agents)

    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.status is FinalStatus.COMPLETED
    assert result.accepted
    assert [r.step_id for r in state["agent_results"]] == ["kb", "design"]
    assert all(s.status is StepStatus.COMPLETED for s in state["steps"].values())
    # The coding worker only received what it needs: its dependency's result.
    coding_input: WorkerInput = agents.workers["coding"].inputs[0]  # type: ignore[attr-defined]
    assert [d.step_id for d in coding_input.dependency_results] == ["kb"]
    assert coding_input.dependency_results[0].summary == RAG.summary
    events = [e.type for e in tracer.events]
    assert events[0] is EventType.RUN_STARTED
    assert events[-1] is EventType.RUN_COMPLETED
    assert EventType.CRITIC_VERDICT in events
    assert result.metrics.agent_calls == 6  # optimizer, planner, rag, coding, synthesizer, critic
    assert state["messages"][0].sender == "prompt_optimizer"


async def test_failed_review_triggers_targeted_rework_then_passes() -> None:
    critic = FakeAgent(review(ReviewStatus.FAIL, ["design"], 40), review(ReviewStatus.PASS))
    coding = FakeAgent(CODE)
    state, tracer = await run_graph(suite(critic=critic, coding=coding))

    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.accepted
    assert state["retry_count"] == 1
    assert len(state["reviews"]) == 2
    assert [r.step_id for r in state["agent_results"]] == ["kb", "design", "design"]
    rework_input: WorkerInput = coding.inputs[1]
    assert rework_input.feedback == ["Taxes are missing"]
    assert rework_input.attempt == 2
    (rework,) = tracer.of_type(EventType.REWORK_REQUESTED)
    assert rework.payload["flagged_steps"] == ["design"]


async def test_rework_cascades_to_dependent_steps() -> None:
    critic = FakeAgent(review(ReviewStatus.FAIL, ["kb"], 30), review(ReviewStatus.PASS))
    state, tracer = await run_graph(suite(critic=critic))
    assert [r.step_id for r in state["agent_results"]] == ["kb", "design", "kb", "design"]
    (rework,) = tracer.of_type(EventType.REWORK_REQUESTED)
    assert rework.payload["cascaded_steps"] == ["design"]


async def test_rework_is_bounded_by_max_retries() -> None:
    critic = FakeAgent(review(ReviewStatus.FAIL, ["design"], 20))
    state, _ = await run_graph(suite(critic=critic), limits=RunLimits(max_retries=1))

    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.status is FinalStatus.FAILED
    assert not result.accepted
    assert result.answer is not None  # the last draft is kept for inspection
    assert state["retry_count"] == 1
    assert len(state["reviews"]) == 2
    assert "critic rejected" in result.warnings[0]


async def test_review_without_step_issues_only_resynthesises() -> None:
    synthesizer = FakeAgent(DRAFT)
    critic = FakeAgent(review(ReviewStatus.FAIL, [], 50), review(ReviewStatus.PASS))
    state, _ = await run_graph(suite(synthesizer=synthesizer, critic=critic))
    assert len(synthesizer.inputs) == 2
    assert synthesizer.inputs[1].review_feedback  # feedback forwarded to the synthesizer
    assert [r.step_id for r in state["agent_results"]] == ["kb", "design"]


async def test_rejected_request_goes_straight_to_the_finalizer() -> None:
    planner = FakeAgent(make_plan())
    state, _ = await run_graph(suite(optimizer=FakeAgent(REJECTED), planner=planner))
    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.status is FinalStatus.REJECTED
    assert planner.inputs == []
    assert result.warnings == ["prompt injection"]


async def test_worker_failure_is_retried_within_max_step_attempts() -> None:
    rag = FakeAgent(LLMError("provider timeout"), RAG)
    state, _ = await run_graph(suite(rag=rag))
    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.accepted
    assert state["steps"]["kb"].attempts == 2
    assert state["errors"][0].error_type == "LLMError"


async def test_worker_failing_every_attempt_stops_the_run() -> None:
    rag = FakeAgent(LLMError("provider down"))
    state, _ = await run_graph(suite(rag=rag), limits=RunLimits(max_step_attempts=2))
    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.status is FinalStatus.FAILED
    assert result.abort_reason is not None
    assert "kb failed after 2 attempt(s)" in result.abort_reason
    assert state["steps"]["kb"].status is StepStatus.FAILED


async def test_agent_call_budget_stops_the_run_gracefully() -> None:
    state, tracer = await run_graph(suite(), limits=RunLimits(max_agent_calls=3))
    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.status is FinalStatus.FAILED
    assert result.abort_reason is not None
    assert result.abort_reason.startswith("max_agent_calls")
    assert tracer.of_type(EventType.LIMIT_REACHED)


async def test_max_steps_stops_a_run_that_keeps_reworking() -> None:
    critic = FakeAgent(review(ReviewStatus.FAIL, ["kb"], 10))
    state, _ = await run_graph(suite(critic=critic), limits=RunLimits(max_steps=12, max_retries=10))
    result = final_result_of(state)  # type: ignore[arg-type]
    assert result.abort_reason is not None
    assert result.abort_reason.startswith("max_steps")


async def test_llm_supervisor_choice_is_validated_by_the_policy() -> None:
    invalid = {"step_id": "unknown", "agent": "rag", "reason": "x", "instructions": "y"}
    llm = ScriptedLLM([invalid])
    agents = suite(planner=FakeAgent(make_plan(parallel=True)), strategy="llm")
    state, tracer = await run_graph(agents, llm=llm)

    decisions = tracer.of_type(EventType.SUPERVISOR_DECISION)
    assert "not a ready step" in decisions[0].payload["llm_proposal_rejected"]
    assert decisions[0].step_id == "kb"  # rules fallback
    assert len(llm.requests) == 1  # second decision had a single option: no LLM call
    assert final_result_of(state).accepted  # type: ignore[arg-type]


async def test_llm_supervisor_valid_choice_and_briefing_are_used() -> None:
    choice = {
        "step_id": "design",
        "agent": "coding",
        "reason": "unblocks synthesis",
        "instructions": "Focus on taxes",
    }
    coding = FakeAgent(CODE)
    agents = suite(planner=FakeAgent(make_plan(parallel=True)), coding=coding, strategy="llm")
    state, _ = await run_graph(agents, llm=ScriptedLLM([choice]))
    assert state["agent_results"][0].step_id == "design"
    assert coding.inputs[0].instructions == "Focus on taxes"


def test_mermaid_diagram_lists_every_node() -> None:
    diagram = graph_mermaid()
    for node in (
        "prompt_optimizer",
        "planner",
        "supervisor",
        "research",
        "data",
        "coding",
        "rag",
        "synthesizer",
        "critic",
        "finalizer",
    ):
        assert node in diagram


@pytest.mark.parametrize("status", [ReviewStatus.PASS, ReviewStatus.FAIL])
async def test_every_run_ends_with_a_final_result(status: ReviewStatus) -> None:
    state, _ = await run_graph(
        suite(critic=FakeAgent(review(status))), limits=RunLimits(max_retries=0)
    )
    assert state["final_result"] is not None


def test_the_synthesizer_receives_the_user_request_for_its_language() -> None:
    """Seen live: an objective written in English produced an English answer to French users."""
    spec = PromptSpec(
        objective="Explain the quote validity policy.",
        context="",
        requirements=["State the validity period"],
        deliverables=["Short answer"],
        acceptance_criteria=["The validity period is given"],
        needs_knowledge_base=True,
        feasibility=Feasibility.ACTIONABLE,
    )
    state = initial_state(
        run_id="run-1",
        tenant_id="default",
        user_request="Quelle est la durée de validité d'un devis ?",
        max_retries=1,
    )
    state["optimized_prompt"] = spec
    state["plan"] = make_plan()
    synthesis = build_synthesis_input(state)
    assert synthesis.user_request == "Quelle est la durée de validité d'un devis ?"
    assert synthesis.objective == spec.objective
