"""Structured generation: schema-constrained call, Pydantic validation, bounded repair loop.

Constrained decoding guarantees *syntax*; it cannot express business rules (a plan without
cycles, citations that point to retrieved sources, scores in range...). Those rules live in the
Pydantic models. When validation fails, the errors are sent back to the model, which gets a
bounded number of attempts to fix its answer. A refusal is never "repaired".
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from app.llm.schema import strict_json_schema
from app.llm.types import (
    LLMClient,
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    Message,
    StopReason,
    StructuredOutputError,
    Usage,
)

REPAIR_INSTRUCTIONS = (
    "Your previous answer could not be accepted:\n{errors}\n\n"
    "Reply again with ONE JSON object that fixes every problem above and follows the "
    "required schema exactly. Output the JSON object only."
)
TRUNCATED_INSTRUCTIONS = (
    "Your previous answer was cut off because it was too long. Reply again with a complete "
    "JSON object; keep every text field concise."
)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

RepairHook = Callable[[int, str], Awaitable[None]]


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    attempts: int
    usage: Usage
    response: LLMResponse


def extract_json_text(text: str) -> str:
    """Return the JSON object contained in ``text`` (tolerates code fences and preambles)."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object found in the model output")
    return stripped[start : end + 1]


def parse_structured[T: BaseModel](text: str, model: type[T]) -> T:
    return model.model_validate_json(extract_json_text(text))


def describe_validation_error(exc: Exception, limit: int = 8) -> str:
    if isinstance(exc, ValidationError):
        lines = []
        for error in exc.errors()[:limit]:
            location = ".".join(str(part) for part in error["loc"]) or "<root>"
            lines.append(f"- {location}: {error['msg']}")
        if exc.error_count() > limit:
            lines.append(f"- ... and {exc.error_count() - limit} more error(s)")
        return "\n".join(lines)
    return f"- {exc}"


def repair_message(response: LLMResponse, error: Exception) -> Message:
    if response.stop_reason is StopReason.MAX_TOKENS:
        return Message.user(TRUNCATED_INSTRUCTIONS)
    return Message.user(REPAIR_INSTRUCTIONS.format(errors=describe_validation_error(error)))


async def generate_structured[T: BaseModel](
    llm: LLMClient,
    request: LLMRequest,
    output_model: type[T],
    *,
    max_attempts: int = 3,
    on_repair: RepairHook | None = None,
    validate: Callable[[T], None] | None = None,
) -> StructuredResult[T]:
    """``validate`` adds run-time rules (raise ``ValueError``) on top of the static model."""
    schema_request = request.model_copy(
        update={
            "response_schema": strict_json_schema(output_model),
            "response_schema_name": output_model.__name__,
        }
    )
    messages = list(schema_request.messages)
    usage = Usage()
    last_error = "no attempt made"

    for attempt in range(1, max_attempts + 1):
        response = await llm.complete(schema_request.model_copy(update={"messages": messages}))
        usage += response.usage
        if response.stop_reason is StopReason.REFUSAL:
            raise LLMRefusalError(
                f"{output_model.__name__}: the model declined the request",
                category=response.refusal_category,
            )
        try:
            value = parse_structured(response.text, output_model)
            if validate is not None:
                validate(value)
        except (ValidationError, ValueError) as exc:
            last_error = describe_validation_error(exc)
            if on_repair is not None:
                await on_repair(attempt, last_error)
            messages = [*messages, Message.from_response(response), repair_message(response, exc)]
            continue
        return StructuredResult(value=value, attempts=attempt, usage=usage, response=response)

    raise StructuredOutputError(output_model.__name__, max_attempts, last_error)
