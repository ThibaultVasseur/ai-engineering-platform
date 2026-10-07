"""Langfuse exporter, tested on a real offline run with a fake Langfuse client."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.observability.langfuse import LangfuseTracer, build_langfuse_tracer_factory, mask
from app.observability.tracing import CompositeTracer, InMemoryTracer
from app.orchestration.factory import build_run_context
from app.orchestration.local import build_in_memory_dependencies
from app.orchestration.runner import GraphRunner
from tests.support import make_settings


class FakeObservation:
    def __init__(self, **kwargs: Any) -> None:
        self.name: str = kwargs.get("name", "")
        self.as_type: str = kwargs.get("as_type", "span")
        self.kwargs = kwargs
        self.children: list[FakeObservation] = []
        self.updates: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.ended = False

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        child = FakeObservation(**kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> FakeObservation:
        self.ended = True
        return self

    def create_event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def score_trace(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def walk(self) -> list[FakeObservation]:
        return [self, *(node for child in self.children for node in child.walk())]


class FakeLangfuse:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.roots: list[FakeObservation] = []
        self.flushed = 0

    @staticmethod
    def create_trace_id(*, seed: str | None = None) -> str:
        return f"trace-{seed}"

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        root = FakeObservation(**kwargs)
        self.roots.append(root)
        return root

    def flush(self) -> None:
        self.flushed += 1


async def test_a_run_becomes_one_langfuse_trace_with_nested_observations() -> None:
    deps = await build_in_memory_dependencies(make_settings(), tenant_id="default")
    client = FakeLangfuse()
    run_id = uuid.uuid4()
    exporter = LangfuseTracer(client, run_id=run_id, tenant_id="default")
    context = build_run_context(
        deps,
        run_id=str(run_id),
        tenant_id="default",
        tracer=CompositeTracer([InMemoryTracer(), exporter]),
    )
    await GraphRunner().run(
        context,
        user_request="Analyse les documents disponibles et propose une architecture pour "
        "ajouter un système de devis à une application SaaS.",
    )

    assert exporter.trace_id == f"trace-{run_id}"
    (root,) = client.roots
    assert root.as_type == "chain"
    assert root.kwargs["trace_context"] == {"trace_id": exporter.trace_id}
    nodes = root.walk()
    agents = [n for n in nodes if n.as_type == "agent"]
    assert {a.name for a in agents} >= {"agent:prompt_optimizer", "agent:rag", "agent:critic"}
    assert all(node.ended for node in nodes)
    generations = [n for n in nodes if n.as_type == "generation"]
    assert generations
    assert all(g.kwargs["usage_details"]["input"] > 0 for g in generations)
    research = next(a for a in agents if a.name == "agent:research")
    assert {c.as_type for c in research.children} >= {"generation", "tool"}
    assert any(n.as_type == "retriever" for n in nodes)
    assert any(n.as_type == "guardrail" for n in nodes)  # poisoned vendor chunk excluded
    assert [s["name"] for s in root.scores] == ["critic_score", "critic_score"]
    assert {e["name"] for e in root.events} >= {"plan_created", "supervisor_decision"}
    assert root.updates[-1]["output"]["accepted"] is True
    assert client.flushed == 1


def test_mask_applies_log_redaction() -> None:
    masked = mask(data={"api_key": "sk-ant-api03-ABCDEFGHIJKLMNOPQRST", "query": "devis"})
    assert masked == {"api_key": "***", "query": "devis"}


def test_no_factory_without_keys() -> None:
    assert build_langfuse_tracer_factory(make_settings()) is None


def test_factory_wires_keys_environment_and_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    settings = make_settings(
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_host="http://lf",
    )
    factory = build_langfuse_tracer_factory(settings)
    assert factory is not None
    tracer = factory("acme", uuid.uuid4())
    kwargs = tracer._client.init_kwargs
    assert kwargs["secret_key"] == "sk-lf-test"
    assert kwargs["environment"] == "test"
    assert kwargs["mask"] is mask
