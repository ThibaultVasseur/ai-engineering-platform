import io
import json

import pytest
from rich.console import Console

from app.cli import LivePrinter, main
from app.observability.tracing import EventType, TraceEvent


async def test_model_output_is_printed_literally_not_as_markup() -> None:
    """A live error message lost its regex: rich read '[a-z]' as a style tag."""
    stream = io.StringIO()
    printer = LivePrinter(Console(file=stream, width=200, color_system=None))
    error = "steps.5.id: String should match pattern '^[a-z][a-z0-9_]{1,39}$' [bold]x[/bold]"
    await printer.emit(
        TraceEvent(type=EventType.AGENT_FAILED, agent="planner", payload={"error": error})
    )
    output = stream.getvalue()
    assert "^[a-z][a-z0-9_]{1,39}$" in output
    assert "[bold]x[/bold]" in output


def test_graph_option_prints_the_mermaid_diagram(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--graph"])
    output = capsys.readouterr().out
    assert output.startswith("---") or "graph TD" in output
    assert "critic" in output


def test_demo_runs_offline_and_returns_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    main(["--json", "Combien de tâches ont échoué et quel est le taux de réussite ?"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["accepted"] is True
    assert result["metrics"]["tool_calls"] >= 2
