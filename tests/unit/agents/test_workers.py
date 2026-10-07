import json
from typing import Any

from app.agents.coding import CodingAgent
from app.agents.data import DataAgent
from app.agents.research import ResearchAgent
from app.llm.providers.offline import OfflineLLM
from app.observability.tracing import EventType
from app.orchestration.policies import RunLimits
from app.schemas.agent import WorkerInput
from app.schemas.plan import PlanStep
from app.schemas.tool import ToolCallStatus
from app.services.platform_data import InMemoryPlatformData
from app.tools.catalog import default_tool_registry
from tests.fakes import ScriptedLLM, StaticRetriever, chunk, make_context, tool_response

GUIDE = [
    chunk(0, "All tables carry a tenant_id and PostgreSQL RLS policies enforce isolation."),
    chunk(1, "Monetary amounts are stored as NUMERIC and handled as Decimal in Python."),
]


def worker_input(step_type: str, agent: str, description: str, **extra: Any) -> WorkerInput:
    step = PlanStep(
        id="step_one",
        type=step_type,  # type: ignore[arg-type]
        title="Do the step",
        description=description,
        recommended_agent=agent,  # type: ignore[arg-type]
        priority=1,
        success_criteria=["done"],
    )
    return WorkerInput(
        objective="Ajouter un système de devis à une application SaaS", step=step, **extra
    )


def context_with_tools(llm: Any, *, limits: RunLimits | None = None) -> tuple[Any, Any]:
    from dataclasses import replace

    context, tracer = make_context(llm, limits=limits)
    tools = default_tool_registry().bind(
        tenant_id=context.tenant_id,
        run_id=context.run_id,
        tracer=tracer,
        journal=context.journal,
        knowledge=StaticRetriever(GUIDE),
        data=InMemoryPlatformData.with_demo_data(context.tenant_id),
    )
    return replace(context, tools=tools), tracer


async def test_offline_research_uses_tools_and_cites_registered_sources() -> None:
    context, tracer = context_with_tools(OfflineLLM())
    outcome = await ResearchAgent().run(
        context,
        worker_input("research", "research", "Find the conventions for devis"),
        step_id="step_one",
    )
    report = outcome.value
    assert report.cited_sources() == ["S1", "S2"]
    assert {r.tool for r in context.journal.tool_calls} == {"knowledge_search", "task_lookup"}
    assert any("devis" in f.claim for f in report.findings)  # found through task_lookup
    assert outcome.llm_calls == 2  # one tool turn, one answer turn
    assert len(tracer.of_type(EventType.TOOL_CALL)) == 2


async def test_hallucinated_citation_is_rejected_and_repaired() -> None:
    hallucinated = {
        "summary": "Tenant isolation relies on RLS.",
        "confidence": "high",
        "findings": [{"claim": "RLS is mandatory", "evidence": "guide", "source_ids": ["S7"]}],
        "gaps": [],
    }
    fixed = {**hallucinated, "findings": [{**hallucinated["findings"][0], "source_ids": ["S1"]}]}
    llm = ScriptedLLM(
        [
            tool_response(
                ("knowledge_search", {"query": "tenant isolation", "top_k": 2, "category": None})
            ),
            json.dumps(hallucinated),
            json.dumps(fixed),
        ]
    )
    context, tracer = context_with_tools(llm)
    outcome = await ResearchAgent().run(
        context, worker_input("research", "research", "Check isolation")
    )
    assert outcome.value.cited_sources() == ["S1"]
    assert "unknown source ids ['S7']" in llm.requests[2].messages[-1].text
    assert tracer.of_type(EventType.STRUCTURED_OUTPUT_REPAIRED)


async def test_tool_budget_is_enforced_inside_the_loop() -> None:
    calls = [tool_response(("calculator", {"expression": f"{i} + 1"})) for i in range(3)]
    final = {
        "summary": "Computed what was possible.",
        "confidence": "low",
        "findings": [{"claim": "1 + 1 = 2", "evidence": "calculator", "source_ids": []}],
        "gaps": [],
    }
    llm = ScriptedLLM([*calls, json.dumps(final)])
    context, tracer = context_with_tools(llm, limits=RunLimits(max_tool_calls_per_agent=2))
    await ResearchAgent().run(context, worker_input("research", "research", "Compute things"))
    statuses = [record.status for record in context.journal.tool_calls]
    assert statuses == [ToolCallStatus.OK, ToolCallStatus.OK]
    denied = tracer.of_type(EventType.TOOL_DENIED)
    assert denied[0].payload["reason"] == "tool_budget_exhausted"


async def test_offline_data_agent_computes_rates_with_the_calculator() -> None:
    context, _ = context_with_tools(OfflineLLM())
    outcome = await DataAgent().run(
        context,
        worker_input("data", "data", "How many tasks failed and what is the completion rate?"),
    )
    metrics = {metric.name: metric for metric in outcome.value.metrics}
    assert metrics["tasks_total"].value == 10
    assert metrics["completion_rate"].value == 60
    assert metrics["completion_rate"].method.startswith("calculator")
    assert [r.tool for r in context.journal.tool_calls] == [
        "database_read",
        "database_read",
        "calculator",
    ]


async def test_offline_coding_agent_designs_from_conventions() -> None:
    context, _ = context_with_tools(OfflineLLM())
    outcome = await CodingAgent().run(
        context,
        worker_input("coding", "coding", "Design the quote module", feedback=["Taxes are missing"]),
    )
    proposal = outcome.value
    assert proposal.data_model[0].name == "Quote"
    assert any(entity.name == "QuoteLine" for entity in proposal.data_model)
    assert any("row-level security" in risk for risk in proposal.risks)
    assert any("Review feedback addressed: Taxes are missing" in risk for risk in proposal.risks)
    assert set(proposal.source_ids) <= {"S1", "S2"}
    # The coding agent never had access to database or write tools.
    assert {record.tool for record in context.journal.tool_calls} == {"knowledge_search"}
