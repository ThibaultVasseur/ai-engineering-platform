"""The LangGraph workflow.

    START -> prompt_optimizer -> planner -> supervisor <------------------------+
                  |                 |        |  dispatch -> research|data|coding|rag
                  |                 |        |  synthesize -> synthesizer -> critic
                  v                 v        v                                 |
               finalizer <------ finalizer <- finalize (limits / failure)     |
                  ^                                    FAIL & retries left ----+
                  +------------------------------------ PASS / retries exhausted
                  |
                 END

Compiled once; run-specific dependencies come from ``RunContext`` (``context_schema``).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.orchestration.context import RunContext
from app.orchestration.nodes import (
    critic_node,
    finalizer_node,
    make_worker_node,
    planner_node,
    prompt_optimizer_node,
    supervisor_node,
    synthesizer_node,
)
from app.orchestration.router import (
    route_after_critic,
    route_after_optimizer,
    route_after_planner,
    route_after_synthesizer,
    route_from_supervisor,
)
from app.orchestration.state import GraphState

WORKER_NODES = ("research", "data", "coding", "rag")


def build_graph() -> Any:
    graph = StateGraph(GraphState, context_schema=RunContext)

    graph.add_node("prompt_optimizer", prompt_optimizer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("supervisor", supervisor_node)
    for worker in WORKER_NODES:
        graph.add_node(worker, make_worker_node(worker))
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalizer", finalizer_node)

    graph.add_edge(START, "prompt_optimizer")
    graph.add_conditional_edges("prompt_optimizer", route_after_optimizer, ["planner", "finalizer"])
    graph.add_conditional_edges("planner", route_after_planner, ["supervisor", "finalizer"])
    graph.add_conditional_edges(
        "supervisor", route_from_supervisor, [*WORKER_NODES, "synthesizer", "finalizer"]
    )
    for worker in WORKER_NODES:
        graph.add_edge(worker, "supervisor")
    graph.add_conditional_edges("synthesizer", route_after_synthesizer, ["critic", "finalizer"])
    graph.add_conditional_edges("critic", route_after_critic, ["supervisor", "finalizer"])
    graph.add_edge("finalizer", END)

    return graph.compile(name="ai-engineering-platform")


def graph_mermaid() -> str:
    """Mermaid diagram generated from the compiled graph (keeps the docs honest)."""
    diagram: str = build_graph().get_graph().draw_mermaid()
    return diagram
