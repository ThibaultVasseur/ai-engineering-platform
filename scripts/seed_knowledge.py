"""Ingest the demonstration documents (docs/demo) into PostgreSQL for a tenant.

    uv run python -m scripts.seed_knowledge --tenant default

Idempotent: documents already ingested (same content) are reported as unchanged.
"""

from __future__ import annotations

import argparse
import asyncio

from app.container import build_container
from app.core.config import get_settings
from app.core.security import Tenant
from app.rag.demo import demo_documents


async def seed(tenant_id: str) -> None:
    container = build_container(get_settings())
    try:
        for document in demo_documents():
            report = await container.knowledge_service.ingest(Tenant(id=tenant_id), document)
            state = "created" if report.created else "unchanged"
            print(
                f"{state:<9} {report.document.chunk_count:>2} chunk(s), "
                f"{report.flagged_chunks} flagged  {document.source}"
            )
    finally:
        await container.aclose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=get_settings().default_tenant_id)
    args = parser.parse_args(argv)
    asyncio.run(seed(args.tenant))


if __name__ == "__main__":
    main()
