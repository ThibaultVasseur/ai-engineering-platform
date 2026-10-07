"""Run an evaluation dataset from the command line (no database needed).

uv run python -m app.evaluation                       # default dataset, configured provider
uv run python -m app.evaluation --json report.json --min-success-rate 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.evaluation.datasets import available_datasets, load_dataset
from app.evaluation.runner import EvaluationReport, EvaluationRunner
from app.orchestration.local import build_in_memory_dependencies

TENANT = "evaluation"


async def evaluate(dataset_name: str, case_ids: list[str] | None) -> EvaluationReport:
    settings = get_settings()
    deps = await build_in_memory_dependencies(settings, tenant_id=TENANT)
    runner = EvaluationRunner(deps, tenant_id=TENANT)
    return await runner.run(load_dataset(dataset_name), case_ids=case_ids)


def render(report: EvaluationReport, console: Console) -> None:
    table = Table(title=f"Evaluation '{report.dataset}' — {report.llm_provider}/{report.llm_model}")
    for column in ("case", "category", "result", "status", "critic", "LLM calls", "tools", "ms"):
        table.add_column(column)
    for case in report.cases:
        table.add_row(
            case.id,
            case.category,
            "[green]PASS[/green]" if case.passed else "[red]FAIL[/red]",
            case.final_status or "error",
            str(case.critic_score or "-"),
            str(case.llm_calls),
            str(case.tool_calls),
            f"{case.latency_ms:.0f}",
        )
    console.print(table)
    for case in report.cases:
        for check in case.checks:
            if not check.passed:
                console.print(
                    f"[red]x[/red] {escape(case.id)}: {escape(check.type)} — {escape(check.detail)}"
                )
        if not case.passed:
            causes = ([case.abort_reason] if case.abort_reason else []) + case.agent_failures[:3]
            for cause in causes:
                console.print(f"    [dim]{escape(cause[:300])}[/dim]")
    summary = report.summary
    console.print(
        f"\nsuccess rate {summary.success_rate:.0%} ({summary.passed}/{summary.cases}) | "
        f"avg critic score {summary.avg_critic_score} | tool success {summary.tool_success_rate} | "
        f"retrieval relevance {summary.retrieval_relevance} | p50 {summary.latency_ms_p50} ms | "
        f"p95 {summary.latency_ms_p95} ms | "
        f"tokens {summary.input_tokens}+{summary.output_tokens} | cost ${summary.cost_usd}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="default", choices=available_datasets())
    parser.add_argument("--case", action="append", dest="cases", help="run only these case ids")
    parser.add_argument("--json", type=Path, help="write the full report to this file")
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    args = parser.parse_args(argv)

    configure_logging("WARNING", "console")
    report = asyncio.run(evaluate(args.dataset, args.cases))
    render(report, Console())
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if report.summary.success_rate < args.min_success_rate:
        sys.exit(1)


if __name__ == "__main__":
    main()
