"""OpenAI-compatible adapter (Chat Completions API).

Chat Completions is the lowest common denominator implemented by OpenAI and by most
self-hosted or third-party servers (vLLM, Ollama, LM Studio, ...): pointing LLM_BASE_URL at
such a server is enough to switch provider. Structured answers use ``response_format`` with a
strict JSON schema, tools use strict function definitions.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import openai

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

logger = logging.getLogger(__name__)

_FINISH_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


def to_openai_messages(message: Message) -> list[dict[str, Any]]:
    if message.role == "assistant":
        payload: dict[str, Any] = {"role": "assistant", "content": message.text or None}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ]
        return [payload]

    messages: list[dict[str, Any]] = [
        {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": f"ERROR: {result.content}" if result.is_error else result.content,
        }
        for result in message.tool_results
    ]
    if message.text:
        messages.append({"role": "user", "content": message.text})
    return messages


def parse_completion(completion: Any, *, latency_ms: float) -> LLMResponse:
    if not completion.choices:
        raise LLMError("OpenAI-compatible response without choices")
    choice = completion.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] = []
    for call in message.tool_calls or []:
        if getattr(call, "type", "function") != "function":
            continue
        raw = call.function.arguments or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            # Surfaced to the tool registry, which reports a validation error to the model.
            arguments = {"_invalid_json_arguments": raw[:2000]}
        if not isinstance(arguments, dict):
            arguments = {"_invalid_json_arguments": raw[:2000]}
        tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))

    stop_reason = _FINISH_REASONS.get(choice.finish_reason or "", StopReason.OTHER)
    if getattr(message, "refusal", None):
        stop_reason = StopReason.REFUSAL

    usage = completion.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return LLMResponse(
        text=message.content or "",
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=cached,
        ),
        model=completion.model or "",
        provider="openai",
        latency_ms=latency_ms,
        refusal_category="content_filter" if stop_reason is StopReason.REFUSAL else None,
        request_id=getattr(completion, "_request_id", None),
    )


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        max_output_tokens: int = 16_000,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=max_retries
        )

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": request.system}]
        for message in request.messages:
            messages.extend(to_openai_messages(message))
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": request.max_output_tokens or self._max_output_tokens,
        }
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                        "strict": True,
                    },
                }
                for tool in request.tools
            ]
        if request.response_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_schema_name,
                    "schema": request.response_schema,
                    "strict": True,
                },
            }
        return params

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**self.build_params(request))
        except openai.APIStatusError as exc:
            # The provider's explanation goes to the server logs (redacted) for the operator;
            # the error that travels to run results and API clients only names its kind.
            kind = exc.code or exc.type or type(exc).__name__
            message = exc.body.get("message") if isinstance(exc.body, dict) else None
            log_event(
                logger,
                "llm_provider_error",
                level=logging.WARNING,
                provider="openai",
                status=exc.status_code,
                error=kind,
                param=exc.param,
                detail=redact_text(truncate(str(message or exc.message), 1000)),
            )
            raise LLMError(f"OpenAI-compatible API error {exc.status_code} ({kind})") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"OpenAI-compatible connection error ({type(exc).__name__})") from exc
        return parse_completion(completion, latency_ms=(time.perf_counter() - started) * 1000)
