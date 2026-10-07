"""Demonstration knowledge base (docs/demo/*.txt — fictional company "NovaDesk")."""

from __future__ import annotations

from pathlib import Path

from app.rag.ingestion import DocumentIngestor
from app.rag.store import KnowledgeStore, StoredDocument
from app.schemas.knowledge import DocumentCreate

DEMO_DIR = Path(__file__).resolve().parents[2] / "docs" / "demo"
DEMO_CATEGORIES = {
    "company_policy": "policy",
    "technical_guide": "technical",
    "product_specs": "product",
    "vendor_proposal": "external",
}


def demo_documents(directory: Path = DEMO_DIR) -> list[DocumentCreate]:
    documents = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            DocumentCreate(
                title=text.splitlines()[0].strip()[:300],
                content=text,
                source=f"docs/demo/{path.name}",
                metadata={"category": DEMO_CATEGORIES.get(path.stem, "general"), "demo": True},
            )
        )
    return documents


async def ingest_demo_documents(
    store: KnowledgeStore, ingestor: DocumentIngestor, tenant_id: str
) -> list[tuple[StoredDocument, bool, int]]:
    """Ingest the demo corpus; returns (document, created, flagged_chunks) per file."""
    results = []
    for document in demo_documents():
        prepared = await ingestor.prepare(
            title=document.title,
            content=document.content,
            source=document.source,
            metadata=dict(document.metadata),
        )
        stored, created = await store.add_document(tenant_id, prepared)
        results.append((stored, created, prepared.flagged_chunks))
    return results
