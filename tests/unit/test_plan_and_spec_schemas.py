from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.plan import Plan
from app.schemas.spec import Feasibility, PromptSpec


def step(step_id: str, deps: list[str] | None = None, priority: int = 3) -> dict[str, Any]:
    return {
        "id": step_id,
        "type": "coding",
        "title": f"Step {step_id}",
        "description": f"Do the work of step {step_id}.",
        "recommended_agent": "coding",
        "dependencies": deps or [],
        "priority": priority,
        "success_criteria": ["done"],
    }


def plan(*steps: dict[str, Any]) -> Plan:
    return Plan.model_validate({"rationale": "Because it is needed.", "steps": list(steps)})


def test_valid_plan_orders_steps_topologically_then_by_priority() -> None:
    result = plan(
        step("api", ["db"], priority=1),
        step("docs", priority=5),
        step("db", priority=2),
        step("tests", ["api", "db"], priority=1),
    )
    assert result.topological_order() == ["db", "api", "tests", "docs"]


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([step("alpha"), step("alpha")], "duplicate step ids: alpha"),
        ([step("alpha", ["missing"])], "unknown steps: missing"),
        ([step("alpha", ["alpha"])], "depends on itself"),
        (
            [step("alpha", ["beta"]), step("beta", ["gamma"]), step("gamma", ["alpha"])],
            "dependency cycle",
        ),
    ],
)
def test_invalid_dags_are_rejected_with_actionable_messages(
    steps: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        plan(*steps)


def test_step_ids_are_made_snake_case_and_dependencies_follow() -> None:
    """Live models write ids such as 'Design-API': the spelling is normalised, not rejected."""
    result = plan(step("Design-API"), step("Write Tests (unit)", ["Design-API"]))
    assert [s.id for s in result.steps] == ["design_api", "write_tests_unit"]
    assert result.steps[1].dependencies == ["design_api"]


def test_long_ids_are_shortened_and_hopeless_ids_still_rejected() -> None:
    long_id = "document_the_architecture_with_diagrams_and_technical_descriptions"
    assert plan(step(long_id)).steps[0].id == "document_the_architecture_with_diagrams"
    assert plan(step("2nd step")).steps[0].id == "step_2nd_step"
    for hopeless in ("x", "---", "é"):  # still shorter than two characters once normalised
        with pytest.raises(ValidationError):
            plan(step(hopeless))


def test_unknown_worker_is_rejected() -> None:
    bad = step("alpha") | {"recommended_agent": "shell"}
    with pytest.raises(ValidationError):
        plan(bad)


def spec(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "objective": "Design a quote management module.",
        "context": "",
        "requirements": ["Quotes have line items"],
        "deliverables": ["Architecture proposal"],
        "acceptance_criteria": ["Data model is described"],
        "needs_knowledge_base": True,
        "feasibility": "actionable",
    }
    values.update(overrides)
    return values


def test_actionable_spec_needs_requirements_deliverables_and_criteria() -> None:
    PromptSpec.model_validate(spec())
    with pytest.raises(ValidationError, match="acceptance_criteria"):
        PromptSpec.model_validate(spec(acceptance_criteria=[]))


def test_rejected_spec_needs_a_reason_but_no_requirements() -> None:
    rejected = PromptSpec.model_validate(
        spec(
            feasibility="rejected",
            rejection_reason="Asks to extract secrets",
            requirements=[],
            deliverables=[],
            acceptance_criteria=[],
        )
    )
    assert rejected.feasibility is Feasibility.REJECTED
    with pytest.raises(ValidationError, match="rejection_reason"):
        PromptSpec.model_validate(spec(feasibility="rejected"))


def test_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PromptSpec.model_validate(spec(priority="high"))
