"""Arithmetic without ``eval``: the expression is parsed to an AST and only numeric literals,
the four operations, modulo, floor division, power (bounded) and parentheses are accepted."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from pydantic import Field

from app.schemas.common import StrictModel
from app.schemas.tool import ToolPermission
from app.tools.base import Tool, ToolContext, ToolError

MAX_EXPONENT = 64
MAX_MAGNITUDE = 1e18

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorInput(StrictModel):
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic expression, e.g. '(1200 * 0.2) / 3'. "
        "Operators: + - * / // % ** and parentheses.",
    )


def safe_eval(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    value = _evaluate(tree.body)
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("result is not a finite number")
    if abs(value) > MAX_MAGNITUDE:
        raise ToolError("result is too large")
    return float(value)


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (
            abs(right) > MAX_EXPONENT or abs(left) > MAX_MAGNITUDE
        ):
            raise ToolError(f"exponent too large (max {MAX_EXPONENT})")
        try:
            result = _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ToolError("division by zero") from exc
        if isinstance(result, complex):
            raise ToolError("complex results are not supported")
        if abs(result) > MAX_MAGNITUDE:
            raise ToolError("intermediate result is too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ToolError(f"unsupported element: {type(node).__name__}")


async def calculate(arguments: CalculatorInput, context: ToolContext) -> dict[str, Any]:
    result = safe_eval(arguments.expression)
    return {"expression": arguments.expression, "result": round(result, 10)}


CALCULATOR = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression exactly (use it instead of mental math).",
    input_model=CalculatorInput,
    permission=ToolPermission.READ,
    handler=calculate,
    timeout_seconds=2.0,
)
