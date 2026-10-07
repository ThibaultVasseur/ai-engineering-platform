import pytest

from app.tools.base import ToolError
from app.tools.calculator import safe_eval


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1200 * 0.2) / 3", 80),
        ("-4 ** 2", -16),
        ("7 // 2 + 7 % 2", 4),
        ("2 ** 10", 1024),
        ("6 / 10 * 100", 60),
    ],
)
def test_arithmetic_is_evaluated(expression: str, expected: float) -> None:
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd').read()",
        "x + 1",
        "(1).__class__",
        "[1, 2, 3]",
        "lambda: 1",
        "True + 1",
        "'a' * 3",
    ],
)
def test_anything_but_arithmetic_is_refused(expression: str) -> None:
    with pytest.raises(ToolError):
        safe_eval(expression)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1 / 0", "division by zero"),
        ("9 ** 9 ** 9", "exponent too large"),
        ("10 ** 60", "too large"),
        ("(-8) ** 0.5", "complex"),
        ("1 +", "invalid expression"),
    ],
)
def test_dangerous_or_invalid_arithmetic_is_rejected(expression: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        safe_eval(expression)
