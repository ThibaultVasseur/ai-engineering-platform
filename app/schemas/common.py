"""Base classes and shared types for API and agent schemas."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

MetadataValue = str | int | float | bool


def _check_metadata(value: dict[str, MetadataValue]) -> dict[str, MetadataValue]:
    if len(value) > 20:
        raise ValueError("metadata accepts at most 20 keys")
    for key, item in value.items():
        if not key or len(key) > 64:
            raise ValueError("metadata keys must be 1-64 characters long")
        if isinstance(item, str) and len(item) > 500:
            raise ValueError(f"metadata value for '{key}' exceeds 500 characters")
    return value


Metadata = Annotated[dict[str, MetadataValue], AfterValidator(_check_metadata)]


class RequestModel(BaseModel):
    """Inbound payloads: unknown fields are rejected (no silent mass-assignment)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResponseModel(BaseModel):
    """Outbound payloads, buildable from ORM objects."""

    model_config = ConfigDict(from_attributes=True)


class StrictModel(BaseModel):
    """LLM outputs: strict shape, unknown keys are an error the repair loop can report."""

    model_config = ConfigDict(extra="forbid")


class Page[T](BaseModel):
    items: list[T]
    limit: int
    offset: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


Limit = Annotated[int, Field(ge=1, le=100)]
Offset = Annotated[int, Field(ge=0, le=100_000)]
