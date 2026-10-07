import pytest

from app.llm.types import BudgetExceededError, Usage
from app.orchestration.policies import RunBudget, RunLimits


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_agent_calls_are_capped() -> None:
    budget = RunBudget(RunLimits(max_agent_calls=2))
    for _ in range(2):
        budget.check_agent_call()
        budget.record_agent_call()
    assert budget.violation() is not None
    with pytest.raises(BudgetExceededError, match="max_agent_calls"):
        budget.check_agent_call()


def test_cost_cap_blocks_further_llm_calls() -> None:
    budget = RunBudget(RunLimits(max_cost_usd=0.5))
    budget.record_llm_call(Usage(input_tokens=10), 0.5)
    with pytest.raises(BudgetExceededError, match="max_cost"):
        budget.check_llm_call()
    violation = budget.violation()
    assert violation is not None
    assert violation.limit == "max_cost"


def test_runtime_cap_uses_the_clock() -> None:
    clock = FakeClock()
    budget = RunBudget(RunLimits(max_runtime_seconds=30), clock=clock)
    budget.check_llm_call()
    clock.now += 31
    with pytest.raises(BudgetExceededError, match="max_runtime"):
        budget.check_llm_call()


def test_snapshot_reports_usage() -> None:
    budget = RunBudget(RunLimits())
    budget.record_llm_call(Usage(input_tokens=10, output_tokens=5), 0.001)
    budget.record_llm_call(Usage(input_tokens=20, output_tokens=5), 0.002)
    snapshot = budget.snapshot()
    assert snapshot["llm_calls"] == 2
    assert snapshot["input_tokens"] == 30
    assert snapshot["cost_usd"] == pytest.approx(0.003)


def test_recursion_limit_is_a_backstop_above_max_steps() -> None:
    assert RunLimits(max_steps=40).recursion_limit > 40
