"""Demo CLI: run the multi-agent pipeline on a request and show every stage as it happens.

    uv run python -m app.cli "Analyse les documents disponibles et propose une architecture..."
    uv run python -m app.cli --store postgres "..."   # pgvector knowledge base (docker compose)
    uv run python -m app.cli --provider anthropic "..."  # real model (needs an API key)
    uv run python -m app.cli --graph                  # the LangGraph workflow as Mermaid

Default: no database, no API key — in-memory stores loaded with the demo corpus and the
deterministic offline simulator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from app.core.config import Settings
from app.core.logging import configure_logging
from app.observability.tracing import CompositeTracer, EventType, InMemoryTracer, TraceEvent
from app.orchestration.factory import RunDependencies, build_run_context
from app.orchestration.graph import graph_mermaid
from app.orchestration.local import build_in_memory_dependencies
from app.orchestration.runner import GraphRunner
from app.orchestration.state import GraphState
from app.schemas.run import FinalResult, FinalStatus

DEMO_REQUEST = (
    "Analyse les documents disponibles et propose une architecture pour ajouter un système "
    "de devis à une application SaaS."
)
TENANT = "default"


def _text(value: object) -> str:
    """Model output and data are printed literally: '[a-z]' must not be read as markup."""
    return escape(str(value))


class LivePrinter:
    """Tracer that renders the run while it executes."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._execution_started = False

    async def emit(self, event: TraceEvent) -> None:
        payload = event.payload
        kind = event.type
        if kind is EventType.PROMPT_OPTIMIZED:
            self._optimized(payload)
        elif kind is EventType.PLAN_CREATED:
            self._plan(payload)
        elif kind is EventType.SUPERVISOR_DECISION:
            self._section_execution()
            target = _text(payload.get("target_agent") or payload.get("action"))
            step = f" step [bold]{_text(event.step_id)}[/bold]" if event.step_id else ""
            self.console.print(
                f"[magenta]supervisor[/magenta] -> {target}{step}  "
                f"[dim]{_text(payload.get('reason'))}[/dim]"
            )
            if payload.get("llm_proposal_rejected"):
                self.console.print(
                    f"   [yellow]LLM proposal rejected:[/yellow] "
                    f"{_text(payload['llm_proposal_rejected'])}"
                )
        elif kind is EventType.AGENT_COMPLETED and event.agent not in {
            "prompt_optimizer",
            "planner",
        }:
            self.console.print(
                f"   [green]OK[/green] {_text(event.agent)}: "
                f"{_text(payload.get('summary', '')[:140])} "
                f"[dim]({payload.get('llm_calls')} LLM call(s), "
                f"{payload.get('latency_ms')} ms)[/dim]"
            )
        elif kind is EventType.AGENT_FAILED:
            self.console.print(
                f"   [red]FAILED[/red] {_text(event.agent)}: {_text(payload.get('error'))}"
            )
        elif kind in (EventType.TOOL_CALL, EventType.TOOL_DENIED):
            status = payload.get("status")
            color = "cyan" if status == "ok" else "red"
            arguments = _text(json.dumps(payload.get("arguments"), ensure_ascii=False)[:110])
            self.console.print(
                f"   [{color}]tool[/{color}] {_text(payload.get('tool'))} {arguments} -> "
                f"{_text(status)}"
            )
        elif kind is EventType.RETRIEVAL:
            sources = ", ".join(
                f"{r['source_id']} {r['title'][:28]}" for r in payload.get("results", [])
            )
            self.console.print(
                f"   [cyan]retrieval[/cyan] {_text(sources or 'no relevant document')}"
            )
        elif kind is EventType.GUARDRAIL_TRIGGERED:
            self.console.print(
                f"   [bold red]guardrail[/bold red] {_text(payload.get('guardrail'))}: "
                f"{_text(payload.get('action'))} "
                f"{_text(payload.get('signals') or payload.get('count', ''))}"
            )
        elif kind is EventType.CRITIC_VERDICT:
            color = "green" if payload.get("status") == "PASS" else "red"
            self.console.print(
                f"[{color}]critic round {_text(payload.get('round'))}: "
                f"{_text(payload.get('status'))} (score {_text(payload.get('score'))})"
                f"[/{color}]"
            )
            for issue in payload.get("issues", [])[:5]:
                self.console.print(f"   - {_text(issue)}")
        elif kind is EventType.REWORK_REQUESTED:
            flagged = payload.get("flagged_steps") or ["synthesis only"]
            self.console.print(
                f"[yellow]rework #{_text(payload.get('retry'))}[/yellow] -> "
                f"{_text(', '.join(flagged))}"
            )
        elif kind is EventType.LIMIT_REACHED:
            self.console.print(
                f"[bold red]limit reached[/bold red]: {_text(payload.get('reason'))}"
            )

    def _section_execution(self) -> None:
        if not self._execution_started:
            self._execution_started = True
            self.console.rule("[bold]AGENT EXECUTION")

    def _optimized(self, payload: dict[str, Any]) -> None:
        self.console.rule("[bold]OPTIMIZED PROMPT")
        table = Table(show_header=False, box=None)
        table.add_row("[bold]feasibility", _text(payload.get("feasibility")))
        table.add_row("[bold]objective", _text(payload.get("objective")))
        table.add_row("[bold]knowledge base", _text(payload.get("needs_knowledge_base")))
        table.add_row(
            "[bold]requirements",
            _text("\n".join(f"- {r}" for r in payload.get("requirements", []))),
        )
        table.add_row(
            "[bold]acceptance criteria",
            _text("\n".join(f"- {c}" for c in payload.get("acceptance_criteria", []))),
        )
        self.console.print(table)

    def _plan(self, payload: dict[str, Any]) -> None:
        self.console.rule("[bold]PLAN")
        tree = Tree(f"[bold]{_text(payload.get('rationale'))}")
        for step in payload.get("steps", []):
            deps = f" (after {', '.join(step['dependencies'])})" if step["dependencies"] else ""
            tree.add(
                f"[bold]{_text(step['id'])}[/bold] -> [cyan]{_text(step['agent'])}[/cyan] "
                f"P{_text(step['priority'])}{_text(deps)}: {_text(step['title'])}"
            )
        self.console.print(tree)


def render_summary(console: Console, state: GraphState) -> None:
    tools = state.get("tool_results", [])
    console.rule("[bold]TOOLS")
    if tools:
        table = Table("tool", "agent", "step", "status", "ms")
        for record in tools:
            table.add_row(
                _text(record.tool),
                _text(record.agent),
                _text(record.step_id or "-"),
                record.status.value,
                f"{record.latency_ms:.1f}",
            )
        console.print(table)
    else:
        console.print("[dim]no tool call[/dim]")

    console.rule("[bold]CRITIC")
    reviews = state.get("reviews", [])
    if not reviews:
        console.print("[dim]no review (the run stopped before synthesis)[/dim]")
    for index, review in enumerate(reviews, start=1):
        satisfied = sum(result.satisfied for result in review.criteria_results)
        console.print(
            f"round {index}: [bold]{review.status.value}[/bold] score {review.score} — "
            f"{satisfied}/{len(review.criteria_results)} criteria satisfied, "
            f"{len(review.issues)} issue(s)"
        )

    result: FinalResult | None = state.get("final_result")
    console.rule("[bold]FINAL RESULT")
    if result is None:
        console.print("[red]no final result[/red]")
        return
    color = {
        FinalStatus.COMPLETED: "green",
        FinalStatus.FAILED: "red",
        FinalStatus.REJECTED: "yellow",
    }[result.status]
    console.print(
        f"[bold {color}]{result.status.value.upper()}[/bold {color}] accepted={result.accepted}"
    )
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {_text(warning)}")
    if result.answer is not None:
        body = [f"# {result.answer.title}", result.answer.executive_summary]
        body += [f"## {section.heading}\n{section.content}" for section in result.answer.sections]
        if result.answer.open_questions:
            body.append(
                "## Open questions\n" + "\n".join(f"- {q}" for q in result.answer.open_questions)
            )
        console.print(Panel(Markdown("\n\n".join(body)), title="answer"))
    if result.sources:
        console.print("[bold]sources[/bold]")
        for source in result.sources:
            console.print(_text(f"  [{source.label}] {source.title} ({source.source})"))
    metrics = result.metrics
    console.print(
        f"\n[dim]{metrics.step_count} graph steps | {metrics.agent_calls} agent calls | "
        f"{metrics.llm_calls} LLM calls | {metrics.tool_calls} tool calls | "
        f"{metrics.retry_count} rework | tokens {metrics.input_tokens}+{metrics.output_tokens} | "
        f"${metrics.cost_usd} | {metrics.duration_ms} ms[/dim]"
    )


async def postgres_dependencies(settings: Settings) -> tuple[RunDependencies, Any]:
    from app.container import build_container

    container = build_container(settings)
    return container.run_dependencies, container


async def run_demo(request: str, settings: Settings, *, store: str, console: Console) -> GraphState:
    container = None
    if store == "postgres":
        deps, container = await postgres_dependencies(settings)
    else:
        deps = await build_in_memory_dependencies(settings, tenant_id=TENANT)
    console.rule("[bold]REQUEST")
    console.print(
        Panel(
            Group(
                _text(request),
                f"[dim]provider {_text(deps.llm.provider)}/{_text(deps.llm.model)} | "
                f"store {store}[/dim]",
            )
        )
    )
    tracer = CompositeTracer([InMemoryTracer(), LivePrinter(console)])
    context = build_run_context(
        deps, run_id=f"cli-{uuid.uuid4().hex[:8]}", tenant_id=TENANT, tracer=tracer
    )
    try:
        state = await GraphRunner().run(context, user_request=request)
    finally:
        if container is not None:
            await container.aclose()
    render_summary(console, state)
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("request", nargs="?", default=DEMO_REQUEST, help="the request to process")
    parser.add_argument("--store", choices=["memory", "postgres"], default="memory")
    parser.add_argument("--provider", choices=["offline", "anthropic", "openai"])
    parser.add_argument("--supervisor", choices=["rules", "llm"])
    parser.add_argument("--json", action="store_true", help="print the final result as JSON")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args(argv)

    if args.graph:
        print(graph_mermaid())
        return
    overrides: dict[str, Any] = {"log_level": "WARNING", "log_format": "console"}
    if args.provider:
        overrides["llm_provider"] = args.provider
    if args.supervisor:
        overrides["supervisor_strategy"] = args.supervisor
    settings = Settings(**overrides)
    configure_logging(settings.log_level, settings.log_format)
    console = Console(quiet=args.json)
    state = asyncio.run(run_demo(args.request, settings, store=args.store, console=console))
    if args.json:
        result = state.get("final_result")
        print(result.model_dump_json(indent=2) if result else "{}")


if __name__ == "__main__":
    main()
