import asyncio
import json
from typing import Any

import pytest
from pydantic import Field

from app.agents.registry import AGENT_PROFILES, AgentName, AgentProfile
from app.llm.embeddings import HashingEmbedder
from app.llm.types import ToolCall
from app.observability.tracing import EventType, InMemoryTracer
from app.orchestration.journal import RunJournal
from app.rag.demo import ingest_demo_documents
from app.rag.ingestion import DocumentIngestor
from app.rag.reranking import LexicalReranker
from app.rag.retrieval import HybridRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.schemas.common import StrictModel
from app.schemas.tool import ToolCallStatus, ToolPermission
from app.services.platform_data import InMemoryPlatformData
from app.tools.base import Tool, ToolContext, ToolError
from app.tools.catalog import default_tool_registry
from app.tools.registry import BoundToolExecutor, ToolRegistry

RESEARCH = AGENT_PROFILES[AgentName.RESEARCH]
DATA = AGENT_PROFILES[AgentName.DATA]
CODING = AGENT_PROFILES[AgentName.CODING]


def bind(
    registry: ToolRegistry | None = None, *, allow_writes: bool = False
) -> tuple[BoundToolExecutor, RunJournal, InMemoryTracer]:
    journal, tracer = RunJournal(), InMemoryTracer()
    executor = (registry or default_tool_registry()).bind(
        tenant_id="default",
        run_id="run-1",
        tracer=tracer,
        journal=journal,
        allow_writes=allow_writes,
        data=InMemoryPlatformData.with_demo_data("default"),
    )
    return executor, journal, tracer


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=arguments)


def test_agents_only_see_their_allowed_tools() -> None:
    executor, _, _ = bind()
    assert {s.name for s in executor.specs_for(RESEARCH)} == {
        "knowledge_search",
        "task_lookup",
        "calculator",
    }
    assert {s.name for s in executor.specs_for(CODING)} == {"knowledge_search"}
    # database_write is in the data agent's allow-list but hidden unless writes are enabled
    assert "database_write" not in {s.name for s in executor.specs_for(DATA)}
    writable, _, _ = bind(allow_writes=True)
    assert "database_write" in {s.name for s in writable.specs_for(DATA)}


async def test_successful_call_is_serialised_recorded_and_traced() -> None:
    executor, journal, tracer = bind()
    result = await executor.execute(call("calculator", expression="2 * 21"), RESEARCH, step_id="s1")
    assert not result.is_error
    assert json.loads(result.content)["result"] == 42
    (record,) = journal.tool_calls
    assert record.status is ToolCallStatus.OK
    assert record.step_id == "s1"
    assert tracer.of_type(EventType.TOOL_CALL)[0].payload["tool"] == "calculator"


@pytest.mark.parametrize(
    ("agent", "tool_call", "reason"),
    [
        (RESEARCH, call("shell", command="ls"), "unknown tool"),
        (CODING, call("database_read", entity="tasks", query="recent", limit=5), "not allowed"),
        (
            DATA,
            call(
                "database_write",
                action="annotate_task",
                task_id="00000000-0000-0000-0000-000000000001",
                note="hello world",
            ),
            "writes are disabled",
        ),
    ],
)
async def test_unauthorised_calls_are_denied_and_traced(
    agent: AgentProfile, tool_call: ToolCall, reason: str
) -> None:
    executor, journal, tracer = bind()
    result = await executor.execute(tool_call, agent, step_id=None)
    assert result.is_error
    assert reason in result.content
    assert journal.tool_calls[0].status is ToolCallStatus.DENIED
    assert tracer.of_type(EventType.TOOL_DENIED)


async def test_write_tool_works_only_with_run_level_opt_in() -> None:
    executor, _, _ = bind(allow_writes=True)
    task_id = "00000000-0000-0000-0000-000000000001"
    result = await executor.execute(
        call(
            "database_write", action="annotate_task", task_id=task_id, note="Reviewed by data agent"
        ),
        DATA,
        step_id=None,
    )
    assert not result.is_error
    assert json.loads(result.content) == {"task_id": task_id, "notes": 1}


async def test_invalid_arguments_are_reported_to_the_model() -> None:
    executor, journal, _ = bind()
    result = await executor.execute(
        call("database_read", entity="users; DROP TABLE tasks", query="recent"), DATA, step_id=None
    )
    assert result.is_error
    assert "invalid arguments" in result.content
    assert journal.tool_calls[0].status is ToolCallStatus.ERROR


class SlowInput(StrictModel):
    api_key: str = Field(default="secret-value")


async def _slow(arguments: SlowInput, context: ToolContext) -> dict[str, Any]:
    await asyncio.sleep(5)
    return {}


async def _broken(arguments: SlowInput, context: ToolContext) -> dict[str, Any]:
    raise RuntimeError("connection to postgresql://app:hunter2@db failed")


async def _expected_failure(arguments: SlowInput, context: ToolContext) -> dict[str, Any]:
    raise ToolError("nothing found for this query")


async def _huge(arguments: SlowInput, context: ToolContext) -> dict[str, Any]:
    return {"blob": "x" * 50_000}


def custom_registry() -> ToolRegistry:
    def tool(name: str, handler: Any, timeout: float | None = None) -> Tool:
        return Tool(name, "test tool", SlowInput, ToolPermission.READ, handler, timeout)

    return ToolRegistry(
        [
            tool("slow", _slow, timeout=0.05),
            tool("broken", _broken),
            tool("expected", _expected_failure),
            tool("huge", _huge),
        ],
        max_output_chars=1_000,
    )


PROFILE = AgentProfile(
    name=AgentName.RESEARCH,
    role="worker",
    mission="test",
    allowed_tools=frozenset({"slow", "broken", "expected", "huge"}),
    output_schema="x",
)


async def test_timeouts_are_enforced() -> None:
    executor, _, _ = bind(custom_registry())
    result = await executor.execute(call("slow"), PROFILE, step_id=None)
    assert result.is_error
    assert "timed out" in result.content


async def test_internal_errors_do_not_leak_details() -> None:
    executor, _, _ = bind(custom_registry())
    result = await executor.execute(call("broken"), PROFILE, step_id=None)
    assert result.is_error
    assert "hunter2" not in result.content
    assert "RuntimeError" in result.content


async def test_expected_tool_errors_keep_their_message() -> None:
    executor, _, _ = bind(custom_registry())
    result = await executor.execute(call("expected"), PROFILE, step_id=None)
    assert result.content == "Tool error: nothing found for this query"


async def test_outputs_are_truncated_and_secret_arguments_redacted() -> None:
    executor, journal, _ = bind(custom_registry())
    result = await executor.execute(call("huge", api_key="sk-live-123"), PROFILE, step_id=None)
    assert len(result.content) <= 1_000
    assert journal.tool_calls[0].arguments["api_key"] == "***"


def test_duplicate_tool_names_are_rejected() -> None:
    tool = Tool("same", "d", SlowInput, ToolPermission.READ, _slow)
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([tool, tool])


async def test_knowledge_search_traces_the_injection_guardrail() -> None:
    """Live runs searched through research agents: the exclusion must be traced there too."""
    embedder = HashingEmbedder()
    store = InMemoryKnowledgeStore()
    await ingest_demo_documents(store, DocumentIngestor(embedder), "default")
    journal, tracer = RunJournal(), InMemoryTracer()
    executor = default_tool_registry().bind(
        tenant_id="default",
        run_id="run-1",
        tracer=tracer,
        journal=journal,
        knowledge=HybridRetriever(store, embedder, reranker=LexicalReranker()),
    )
    question = "Que propose le prestataire externe pour la génération des PDF de devis ?"
    result = await executor.execute(
        call("knowledge_search", query=question), RESEARCH, step_id="s1"
    )
    assert not result.is_error
    (event,) = tracer.of_type(EventType.GUARDRAIL_TRIGGERED)
    assert event.agent == "research"
    assert event.payload["guardrail"] == "indirect_prompt_injection"
    assert event.payload["tool"] == "knowledge_search"
    assert "ignore all previous instructions" not in result.content.lower()
