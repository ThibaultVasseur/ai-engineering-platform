import pytest
from pydantic import BaseModel, Field, model_validator

from app.llm.structured import extract_json_text, generate_structured
from app.llm.types import (
    CallMetadata,
    LLMRefusalError,
    LLMRequest,
    Message,
    StopReason,
    StructuredOutputError,
)
from tests.fakes import ScriptedLLM, text_response


class Answer(BaseModel):
    verdict: str
    score: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _business_rule(self) -> "Answer":
        if self.verdict == "PASS" and self.score < 50:
            raise ValueError("a PASS verdict needs a score of at least 50")
        return self


def request() -> LLMRequest:
    return LLMRequest(
        system="system prompt",
        messages=[Message.user("evaluate this")],
        metadata=CallMetadata(agent="critic", prompt_name="critic", prompt_version="v1"),
    )


async def test_valid_first_answer_is_returned_with_schema_attached() -> None:
    llm = ScriptedLLM(['{"verdict": "PASS", "score": 80}'])
    result = await generate_structured(llm, request(), Answer)
    assert result.value == Answer(verdict="PASS", score=80)
    assert result.attempts == 1
    sent = llm.requests[0]
    assert sent.response_schema is not None
    assert sent.response_schema["additionalProperties"] is False
    assert sent.response_schema_name == "Answer"


async def test_business_rule_violation_is_fed_back_and_repaired() -> None:
    repairs: list[tuple[int, str]] = []

    async def on_repair(attempt: int, errors: str) -> None:
        repairs.append((attempt, errors))

    llm = ScriptedLLM(['{"verdict": "PASS", "score": 10}', '{"verdict": "PASS", "score": 75}'])
    result = await generate_structured(llm, request(), Answer, on_repair=on_repair)

    assert result.value.score == 75
    assert result.attempts == 2
    assert result.usage.input_tokens == 200  # usage accumulates over attempts
    retry = llm.requests[1].messages
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "at least 50" in retry[-1].text
    assert len(repairs) == 1
    assert repairs[0][0] == 1


async def test_truncated_output_gets_a_conciseness_instruction() -> None:
    llm = ScriptedLLM(
        [
            text_response('{"verdict": "PA', stop=StopReason.MAX_TOKENS),
            '{"verdict": "FAIL", "score": 20}',
        ]
    )
    result = await generate_structured(llm, request(), Answer)
    assert result.value.verdict == "FAIL"
    assert "cut off" in llm.requests[1].messages[-1].text


async def test_refusals_are_not_repaired() -> None:
    refusal = text_response("", stop=StopReason.REFUSAL)
    llm = ScriptedLLM([refusal])
    with pytest.raises(LLMRefusalError):
        await generate_structured(llm, request(), Answer)
    assert len(llm.requests) == 1


async def test_attempts_are_bounded() -> None:
    llm = ScriptedLLM(["not json", "still not json", "nope"])
    with pytest.raises(StructuredOutputError) as caught:
        await generate_structured(llm, request(), Answer, max_attempts=3)
    assert caught.value.attempts == 3
    assert len(llm.requests) == 3


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here is the result: {"a": 1} hope it helps',
    ],
)
def test_json_is_extracted_from_common_wrappings(raw: str) -> None:
    assert extract_json_text(raw) == '{"a": 1}'


def test_missing_json_is_an_error() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_text("I cannot answer that.")
