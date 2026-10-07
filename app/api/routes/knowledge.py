from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.dependencies import KnowledgeServiceDep, TenantDep, rate_limit
from app.schemas.common import ErrorResponse, Page
from app.schemas.knowledge import (
    DocumentCreate,
    DocumentRead,
    IngestionReport,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post(
    "/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=IngestionReport,
    summary="Ingest a text document (clean, chunk, scan, embed, store)",
    responses={
        200: {"model": IngestionReport, "description": "Identical content already ingested"},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    dependencies=[rate_limit(2)],
)
async def create_document(
    payload: DocumentCreate,
    tenant: TenantDep,
    service: KnowledgeServiceDep,
    response: Response,
) -> IngestionReport:
    report = await service.ingest(tenant, payload)
    if not report.created:
        response.status_code = status.HTTP_200_OK
    return report


@router.get("/documents", response_model=Page[DocumentRead], summary="List documents")
async def list_documents(
    tenant: TenantDep,
    service: KnowledgeServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
) -> Page[DocumentRead]:
    return await service.list(tenant, limit=limit, offset=offset)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid search (vector + full-text, RRF, rerank) — useful to inspect retrieval",
    dependencies=[rate_limit(1)],
)
async def search(
    payload: SearchRequest, tenant: TenantDep, service: KnowledgeServiceDep
) -> SearchResponse:
    return await service.search(tenant, payload)
