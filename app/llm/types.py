"""Provider-agnostic LLM data model.

Agents speak only this vocabulary; each provider adapter translates it to and from its own API.
Assistant turns keep the provider's raw content blocks (``provider_content``) so that they can
be replayed verbatim in tool loops — required by providers that sign reasoning blocks (Claude
rejects or drops thinking blocks whose surrounding history was edited).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


class ToolSpec(BaseModel):
    """A tool as presented to the model: name, description, strict JSON schema of arguments."""

    name: str
    description: str
    input_schema: dict[str, Any]


class CallMetadata(BaseModel):
    """Who is calling and with which prompt version — for tracing and offline dispatch."""

    agent: str
    prompt_name: str
    prompt_version: str
    step_id: str | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    provider: str | None = None
    provider_content: list[dict[str, Any]] | None = None

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", text=text)

    @classmethod
    def results(cls, results: list[ToolResult], text: str = "") -> Message:
        return cls(role="user", tool_results=results, text=text)

    @classmethod
    def from_response(cls, response: LLMResponse) -> Message:
        return cls(
            role="assistant",
            text=response.text,
            tool_calls=response.tool_calls,
            provider=response.provider,
            provider_content=response.provider_content,
        )


class LLMRequest(BaseModel):
    system: str
    messages: list[Message]
    metadata: CallMetadata
    tools: list[ToolSpec] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None
    response_schema_name: str = "response"
    max_output_tokens: int | None = None


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason
    usage: Usage = Field(default_factory=Usage)
    model: str
    provider: str
    latency_ms: float = 0.0
    provider_content: list[dict[str, Any]] | None = None
    refusal_category: str | None = None
    request_id: str | None = None


class LLMClient(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------
class LLMError(Exception):
    """The provider could not produce a usable answer (network, API error, bad payload)."""


class LLMRefusalError(LLMError):
    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class StructuredOutputError(LLMError):
    def __init__(self, schema_name: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"{schema_name}: no valid output after {attempts} attempt(s): {last_error}"
        )
        self.schema_name = schema_name
        self.attempts = attempts
        self.last_error = last_error


class BudgetExceededError(Exception):
    """A run limit (cost, deadline, agent calls...) forbids another call."""

    def __init__(self, limit: str, detail: str) -> None:
        super().__init__(f"{limit}: {detail}")
        self.limit = limit
        self.detail = detail
