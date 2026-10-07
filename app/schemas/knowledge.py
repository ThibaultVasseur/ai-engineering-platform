"""Knowledge base: documents, chunks, retrieval results and source references."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.core.text import normalize_text
from app.schemas.common import Metadata, RequestModel, ResponseModel

MAX_DOCUMENT_CHARS = 200_000


class SourceRef(BaseModel):
    """A retrieved chunk as cited in answers: ``[S1]`` refers to ``label == "S1"``."""

    label: str
    chunk_id: str
    document_id: str
    title: str
    source: str | None = None
    score: float = 0.0
    excerpt: str = ""


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str | None = None
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector_score: float | None = None
    text_rank: float | None = None
    score: float = 0.0


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    filtered_out: int = 0  # chunks dropped by the injection guard
    latency_ms: float = 0.0


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
class DocumentCreate(RequestModel):
    title: str = Field(min_length=2, max_length=300)
    content: str = Field(min_length=20, max_length=MAX_DOCUMENT_CHARS)
    source: str | None = Field(default=None, max_length=500)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        return normalize_text(value)


class DocumentRead(ResponseModel):
    id: UUID
    title: str
    source: str | None
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    chunk_count: int
    created_at: datetime


class IngestionReport(BaseModel):
    document: DocumentRead
    created: bool  # False when identical content was already ingested (idempotent)
    flagged_chunks: int  # chunks carrying injection signals
    hidden_characters_removed: int


class SearchFilters(RequestModel):
    category: str | None = Field(default=None, max_length=100)
    document_ids: list[UUID] = Field(default_factory=list, max_length=20)


class SearchRequest(RequestModel):
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str | None
    score: float
    vector_score: float | None
    text_rank: float | None
    content: str
    injection_risk: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    filtered_out: int
    latency_ms: float
