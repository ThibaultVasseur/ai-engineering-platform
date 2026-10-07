from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from app.llm.schema import SchemaError, strict_json_schema


class Section(BaseModel):
    title: str = Field(min_length=3, max_length=80)  # a *property* named "title"
    content: str


class Report(BaseModel):
    summary: str = Field(description="One paragraph.", max_length=500)
    score: int = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"] = "medium"
    sections: list[Section] = Field(default_factory=list, max_length=5)
    note: str | None = None


def walk(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found.extend(walk(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(walk(item))
    return found


def test_every_object_is_closed_and_fully_required() -> None:
    schema = strict_json_schema(Report)
    objects = [node for node in walk(schema) if node.get("type") == "object"]
    assert len(objects) == 2  # Report and Section
    for node in objects:
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])


def test_unsupported_constraints_are_removed_but_descriptions_kept() -> None:
    schema = strict_json_schema(Report)
    keywords = {key for node in walk(schema) for key in node}
    assert not keywords & {"minLength", "maxLength", "minimum", "maximum", "maxItems", "default"}
    assert schema["properties"]["summary"]["description"].startswith("One paragraph.")


def test_dropped_constraints_are_restated_for_the_model() -> None:
    """A live model could not respect limits it was never told about."""

    class Step(BaseModel):
        id: str = Field(pattern=r"^[a-z_]+$")
        tags: list[str] = Field(min_length=1, max_length=3)

    schema = strict_json_schema(Report)
    properties = schema["properties"]
    assert properties["summary"]["description"] == (
        "One paragraph. Constraints: at most 500 characters."
    )
    assert properties["score"]["description"] == "Constraints: value >= 0 and <= 100."
    assert properties["sections"]["description"] == "Constraints: at most 5 items."
    title = schema["$defs"]["Section"]["properties"]["title"]
    assert title["description"] == "Constraints: 3 to 80 characters."
    step = strict_json_schema(Step)["properties"]
    assert step["id"]["description"] == "Constraints: must match the regular expression ^[a-z_]+$."
    assert step["tags"]["description"] == "Constraints: 1 to 3 items."


def test_properties_named_like_keywords_survive() -> None:
    section = strict_json_schema(Report)["$defs"]["Section"]
    assert "title" in section["properties"]
    assert "title" in section["required"]


def test_optional_fields_stay_nullable() -> None:
    note = strict_json_schema(Report)["properties"]["note"]
    assert {"type": "null"} in note["anyOf"]


def test_free_form_dicts_are_refused() -> None:
    class Loose(BaseModel):
        data: dict[str, Any]

    with pytest.raises(SchemaError, match="free-form"):
        strict_json_schema(Loose)
