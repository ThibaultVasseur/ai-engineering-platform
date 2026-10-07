"""Provider adapters, exercised with real SDK response types and fake transports (no network)."""

import logging
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import openai
import pytest
from anthropic.types import Message as AnthropicMessage
from openai.types.chat import ChatCompletion

from app.llm.providers.anthropic import FALLBACK_BETA, AnthropicLLM
from app.llm.providers.offline import OfflineLLM, OfflineReply
from app.llm.providers.openai import OpenAICompatibleLLM
from app.llm.types import (
    CallMetadata,
    LLMError,
    LLMRequest,
    Message,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
TOOL = ToolSpec(
    name="calculator",
    description="Evaluate arithmetic.",
    input_schema={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
)


def make_request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "system": "You are the research agent.",
        "messages": [Message.user("Find the VAT rate.")],
        "metadata": CallMetadata(agent="research", prompt_name="research", prompt_version="v1"),
    }
    values.update(overrides)
    return LLMRequest(**values)


class FakeEndpoint:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------
def anthropic_message(**overrides: Any) -> AnthropicMessage:
    payload: dict[str, Any] = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig-abc"},
            {"type": "text", "text": "Let me compute that."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "calculator",
                "input": {"expression": "100*0.2"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 0,
        },
    }
    payload.update(overrides)
    return AnthropicMessage.model_validate(payload)


def anthropic_client(endpoint: FakeEndpoint) -> SimpleNamespace:
    return SimpleNamespace(messages=endpoint, beta=SimpleNamespace(messages=endpoint))


def test_anthropic_params_use_effort_strict_tools_schema_and_caching() -> None:
    llm = AnthropicLLM(model="claude-opus-5-5", effort="high", client=object())
    params = llm.build_params(make_request(tools=[TOOL], response_schema=SCHEMA))
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert params["tools"][0]["strict"] is True
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["system"] == "You are the research agent."
    assert "tool_choice" not in params  # forced tool use is rejected by current models
    assert "thinking" not in params  # always on for Opus 5.5: depth comes from effort
    assert "temperature" not in params


async def test_anthropic_response_is_parsed_and_replayed_verbatim() -> None:
    endpoint = FakeEndpoint(anthropic_message())
    llm = AnthropicLLM(model="claude-opus-5-5", client=anthropic_client(endpoint))

    response = await llm.complete(make_request(tools=[TOOL]))

    assert response.stop_reason is StopReason.TOOL_USE
    assert response.tool_calls == [
        ToolCall(id="toolu_01", name="calculator", arguments={"expression": "100*0.2"})
    ]
    assert response.usage.cache_read_tokens == 2000
    assert endpoint.calls[0]["betas"] == [FALLBACK_BETA]
    assert endpoint.calls[0]["fallbacks"] == "default"

    # Next turn of the tool loop: the assistant turn (thinking block included) is replayed as-is.
    follow_up = make_request(
        messages=[
            Message.user("Find the VAT rate."),
            Message.from_response(response),
            Message.results([ToolResult(tool_call_id="toolu_01", content="20.0")]),
        ]
    )
    sent = llm.build_params(follow_up)["messages"]
    assert sent[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert sent[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": "20.0",
        "is_error": False,
    }


async def test_anthropic_refusal_is_reported_with_category() -> None:
    refused = anthropic_message(
        content=[],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"},
    )
    llm = AnthropicLLM(model="claude-opus-5-5", client=anthropic_client(FakeEndpoint(refused)))
    response = await llm.complete(make_request())
    assert response.stop_reason is StopReason.REFUSAL
    assert response.refusal_category == "cyber"


async def test_anthropic_without_fallback_uses_the_stable_endpoint() -> None:
    stable, beta = FakeEndpoint(anthropic_message()), FakeEndpoint(anthropic_message())
    client = SimpleNamespace(messages=stable, beta=SimpleNamespace(messages=beta))
    llm = AnthropicLLM(model="claude-opus-5-5", refusal_fallback=False, client=client)
    await llm.complete(make_request())
    assert len(stable.calls) == 1
    assert beta.calls == []


async def test_anthropic_api_errors_become_llm_errors_without_leaking_bodies() -> None:
    error = anthropic.APIConnectionError(request=SimpleNamespace(method="POST", url="x"))  # type: ignore[arg-type]
    llm = AnthropicLLM(model="claude-opus-5-5", client=anthropic_client(FakeEndpoint(error=error)))
    with pytest.raises(LLMError, match="connection error"):
        await llm.complete(make_request())


SECRET = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123"


def http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://llm.test/v1"))


def provider_error_fields(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        record.fields  # type: ignore[attr-defined]
        for record in caplog.records
        if record.getMessage() == "llm_provider_error"
    ]


async def test_anthropic_status_errors_name_their_kind_and_log_the_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="app.llm.providers.anthropic")
    error = anthropic.BadRequestError(
        "Error code: 400",
        response=http_response(400),
        body={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": f"max_tokens: too large {SECRET}",
            },
        },
    )
    llm = AnthropicLLM(model="claude-opus-5-5", client=anthropic_client(FakeEndpoint(error=error)))
    with pytest.raises(LLMError) as raised:
        await llm.complete(make_request())
    # What reaches run results and API clients names the error, not the provider's body...
    assert str(raised.value) == "Anthropic API error 400 (invalid_request_error)"
    # ...while the operator gets the explanation in the server logs, redacted.
    (fields,) = provider_error_fields(caplog)
    assert fields["status"] == 400
    assert "max_tokens: too large" in fields["detail"]
    assert SECRET not in fields["detail"]


# --------------------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------------------
def chat_completion(message: dict[str, Any], finish_reason: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "local-model",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 15,
                "total_tokens": 135,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        }
    )


def openai_client(endpoint: FakeEndpoint) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=endpoint))


def test_openai_params_use_strict_function_tools_and_json_schema() -> None:
    llm = OpenAICompatibleLLM(model="local-model", api_key="k", client=object())
    request = make_request(
        tools=[TOOL],
        response_schema=SCHEMA,
        response_schema_name="Findings",
        messages=[
            Message.user("Find the VAT rate."),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})],
            ),
            Message.results([ToolResult(tool_call_id="c1", content="boom", is_error=True)]),
        ],
    )
    params = llm.build_params(request)
    assert params["messages"][0] == {"role": "system", "content": "You are the research agent."}
    assert (
        params["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}'
    )
    assert params["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR: boom"}
    assert params["tools"][0]["function"]["strict"] is True
    assert params["response_format"]["json_schema"]["name"] == "Findings"
    assert params["response_format"]["json_schema"]["strict"] is True


async def test_openai_tool_calls_and_usage_are_parsed() -> None:
    completion = chat_completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression": "2*3"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{not json"},
                },
            ],
        },
        "tool_calls",
    )
    llm = OpenAICompatibleLLM(
        model="local-model", api_key="k", client=openai_client(FakeEndpoint(completion))
    )
    response = await llm.complete(make_request(tools=[TOOL]))
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.tool_calls[0].arguments == {"expression": "2*3"}
    assert "_invalid_json_arguments" in response.tool_calls[1].arguments
    assert response.usage.input_tokens == 100
    assert response.usage.cache_read_tokens == 20


async def test_openai_status_errors_name_their_kind_and_log_the_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="app.llm.providers.openai")
    error = openai.BadRequestError(
        "Error code: 400",
        response=http_response(400),
        body={
            "message": f"Invalid schema for response_format 'Plan' {SECRET}",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": "invalid_json_schema",
        },
    )
    llm = OpenAICompatibleLLM(
        model="local-model", api_key="k", client=openai_client(FakeEndpoint(error=error))
    )
    with pytest.raises(LLMError) as raised:
        await llm.complete(make_request())
    assert str(raised.value) == "OpenAI-compatible API error 400 (invalid_json_schema)"
    (fields,) = provider_error_fields(caplog)
    assert fields["param"] == "response_format"
    assert "Invalid schema for response_format" in fields["detail"]
    assert SECRET not in fields["detail"]


async def test_openai_refusal_field_maps_to_refusal() -> None:
    completion = chat_completion(
        {"role": "assistant", "content": None, "refusal": "I can't help with that."}, "stop"
    )
    llm = OpenAICompatibleLLM(
        model="local-model", api_key="k", client=openai_client(FakeEndpoint(completion))
    )
    response = await llm.complete(make_request())
    assert response.stop_reason is StopReason.REFUSAL


# --------------------------------------------------------------------------------------
# Offline
# --------------------------------------------------------------------------------------
async def test_offline_provider_dispatches_on_prompt_name() -> None:
    llm = OfflineLLM(
        {
            "research": lambda request: OfflineReply(
                tool_calls=[ToolCall(id="t1", name="calculator", arguments={"expression": "1"})]
            ),
            "critic": lambda request: OfflineReply(text='{"ok": true}'),
        }
    )
    response = await llm.complete(make_request())
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.usage.input_tokens > 0
    assert response.model == "offline-simulator-v1"


async def test_offline_provider_rejects_unknown_prompts() -> None:
    with pytest.raises(LLMError, match="no handler"):
        await OfflineLLM({}).complete(make_request())
