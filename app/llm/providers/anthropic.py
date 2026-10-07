"""Claude adapter (official ``anthropic`` SDK, async client).

Design choices (see docs/decisions.md):
* default model ``claude-opus-5-5``: thinking is always on for this model, its depth is set
  explicitly with ``output_config.effort`` (the API default would be ``medium``);
* structured answers use ``output_config.format`` (JSON schema) and tools are ``strict``;
* ``tool_choice`` stays ``auto`` — current models reject forced tool use;
* assistant content blocks are replayed verbatim in tool loops (thinking blocks are bound to
  an unmodified history), the harness is append-only;
* the stable system prompt goes first and automatic prompt caching is on;
* safety refusals are retried server-side on Anthropic's recommended fallback model
  (``fallbacks="default"``); a refusal that survives surfaces as ``StopReason.REFUSAL``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic

from app.core.logging import log_event, redact_text
from app.core.text import truncate
from app.llm.types import (
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    Usage,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

logger = logging.getLogger(__name__)

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}


def to_anthropic_message(message: Message) -> dict[str, Any]:
    if message.role == "assistant":
        if message.provider == "anthropic" and message.provider_content is not None:
            return {"role": "assistant", "content": message.provider_content}
        blocks: list[dict[str, Any]] = []
        if message.text:
            blocks.append({"type": "text", "text": message.text})
        blocks.extend(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            for call in message.tool_calls
        )
        return {"role": "assistant", "content": blocks or [{"type": "text", "text": "(empty)"}]}

    content: list[dict[str, Any]] = [
        {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": result.content,
            "is_error": result.is_error,
        }
        for result in message.tool_results
    ]
    if message.text:
        content.append({"type": "text", "text": message.text})
    return {"role": "user", "content": content}


def parse_message(message: Any, *, latency_ms: float) -> LLMResponse:
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in message.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            arguments = block.input if isinstance(block.input, dict) else {}
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))

    stop_reason = _STOP_REASONS.get(message.stop_reason or "", StopReason.OTHER)
    refusal_category = None
    stop_details = getattr(message, "stop_details", None)
    if stop_reason is StopReason.REFUSAL and stop_details is not None:
        refusal_category = getattr(stop_details, "category", None)

    usage = message.usage
    return LLMResponse(
        text="\n".join(texts),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        ),
        model=message.model,
        provider="anthropic",
        latency_ms=latency_ms,
        provider_content=[
            block.model_dump(mode="json", exclude_none=True) for block in message.content
        ],
        refusal_category=refusal_category,
        request_id=getattr(message, "_request_id", None),
    )


class AnthropicLLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        effort: str = "medium",
        max_output_tokens: int = 16_000,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        refusal_fallback: bool = True,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._effort = effort
        self._max_output_tokens = max_output_tokens
        self._refusal_fallback = refusal_fallback
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY or an `ant auth login` profile.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": self._effort}
        if request.response_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.response_schema}
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_output_tokens or self._max_output_tokens,
            "system": request.system,
            "messages": [to_anthropic_message(message) for message in request.messages],
            "output_config": output_config,
            "cache_control": {"type": "ephemeral"},
        }
        if request.tools:
            params["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "strict": True,
                }
                for tool in request.tools
            ]
        return params

    async def complete(self, request: LLMRequest) -> LLMResponse:
        params = self.build_params(request)
        started = time.perf_counter()
        try:
            if self._refusal_fallback:
                message = await self._client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                message = await self._client.messages.create(**params)
        except anthropic.APIStatusError as exc:
            # The provider's explanation goes to the server logs (redacted) for the operator;
            # the error that travels to run results and API clients only names its kind.
            error = exc.body.get("error") if isinstance(exc.body, dict) else None
            details = error if isinstance(error, dict) else {}
            kind = str(details.get("type") or type(exc).__name__)
            log_event(
                logger,
                "llm_provider_error",
                level=logging.WARNING,
                provider="anthropic",
                status=exc.status_code,
                error=kind,
                detail=redact_text(truncate(str(details.get("message") or exc.message), 1000)),
            )
            raise LLMError(f"Anthropic API error {exc.status_code} ({kind})") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic connection error ({type(exc).__name__})") from exc
        return parse_message(message, latency_ms=(time.perf_counter() - started) * 1000)
