"""Strict JSON schemas for constrained decoding.

Providers that enforce a schema during generation (Claude ``output_config.format`` / strict
tools, OpenAI ``strict`` structured outputs) accept a *subset* of JSON Schema: every object must
declare ``additionalProperties: false``, every property must be required, and numeric/string
constraints are not supported. This module derives such a schema from a Pydantic model.

Constraints that are stripped here are not lost: they are restated in plain words in the
field's ``description`` (so the model knows them before answering — found with a live model,
which otherwise wrote 600-character "short quotes"), and the output is always validated against
the original Pydantic model afterwards, violations being fed back to the LLM for repair.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_DROPPED_KEYWORDS = frozenset(
    {
        "title",
        "default",
        "examples",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "discriminator",
    }
)
_SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
_NAME_MAPS = frozenset({"properties", "$defs", "definitions"})


class SchemaError(ValueError):
    """The Pydantic model cannot be expressed as a strict schema (e.g. free-form dict)."""


def _span(low: Any, high: Any, unit: str) -> str:
    if low is not None and high is not None:
        return f"exactly {low} {unit}" if low == high else f"{low} to {high} {unit}"
    return f"at most {high} {unit}" if high is not None else f"at least {low} {unit}"


def constraint_hint(node: dict[str, Any]) -> str:
    """The validation keywords of ``node`` that strict schemas drop, in plain words."""
    hints: list[str] = []
    if "minLength" in node or "maxLength" in node:
        hints.append(_span(node.get("minLength"), node.get("maxLength"), "characters"))
    if "minItems" in node or "maxItems" in node:
        hints.append(_span(node.get("minItems"), node.get("maxItems"), "items"))
    bounds = [
        f"{symbol} {node[keyword]}"
        for keyword, symbol in (
            ("minimum", ">="),
            ("exclusiveMinimum", ">"),
            ("maximum", "<="),
            ("exclusiveMaximum", "<"),
        )
        if keyword in node
    ]
    if bounds:
        hints.append("value " + " and ".join(bounds))
    if "multipleOf" in node:
        hints.append(f"multiple of {node['multipleOf']}")
    if node.get("uniqueItems"):
        hints.append("items must be unique")
    if "pattern" in node:
        hints.append(f"must match the regular expression {node['pattern']}")
    if "format" in node and node["format"] not in _SUPPORTED_FORMATS:
        hints.append(f"format {node['format']}")
    return "; ".join(hints)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    strict: dict[str, Any] = _strictify(model.model_json_schema(), path=model.__name__)
    return strict


def _strictify(node: Any, path: str) -> Any:
    if isinstance(node, list):
        return [_strictify(item, path) for item in node]
    if not isinstance(node, dict):
        return node

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in _NAME_MAPS:
            # Keys of these maps are property/definition *names*, never keywords to drop.
            result[key] = {name: _strictify(sub, f"{path}.{name}") for name, sub in value.items()}
        elif key in _DROPPED_KEYWORDS or (key == "format" and value not in _SUPPORTED_FORMATS):
            continue
        else:
            result[key] = _strictify(value, path)

    hint = constraint_hint(node)
    if hint:
        existing = str(result.get("description") or "").rstrip()
        if existing and not existing.endswith((".", "!", "?")):
            existing += "."
        result["description"] = f"{existing} Constraints: {hint}.".lstrip()

    if result.get("type") == "object":
        if "properties" not in result or result.get("additionalProperties") not in (None, False):
            raise SchemaError(
                f"{path}: free-form objects (dict fields) cannot be used in strict LLM schemas; "
                "model them as a list of typed items instead"
            )
        result["additionalProperties"] = False
        result["required"] = list(result["properties"])
    return result
