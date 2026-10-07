"""Offline handlers for the tool-using workers (research, data, coding).

Each handler follows the same tool loop a real model would: first turn(s) request tool calls,
the final turn turns the tool results into the structured output. Everything is derived from
the step, the tool results and the reviewer feedback — nothing is specific to one scenario.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.text import keywords, split_sentences, strip_accents, truncate
from app.llm.providers.offline import OfflineReply
from app.llm.providers.offline.common import parse_result, read_context, reply_json, tool_exchanges
from app.llm.types import LLMRequest, ToolCall
from app.schemas.agent import (
    CodeFile,
    CodeProposal,
    Component,
    DataAnalysis,
    Endpoint,
    Entity,
    Finding,
    Metric,
    ResearchReport,
)

_ENTITY_NAMES = {
    "devis": "Quote",
    "quote": "Quote",
    "facture": "Invoice",
    "invoice": "Invoice",
    "commande": "Order",
    "order": "Order",
    "client": "Customer",
    "customer": "Customer",
    "ticket": "Ticket",
    "produit": "Product",
    "product": "Product",
    "contrat": "Contract",
    "contract": "Contract",
    "abonnement": "Subscription",
    "subscription": "Subscription",
    "projet": "Project",
    "project": "Project",
}
_LINE_ITEM_ENTITIES = {"Quote", "Invoice", "Order"}
_CONVENTION_HINTS = {
    "rls": "Enforce tenant isolation with PostgreSQL row-level security",
    "tenant": "Scope every query and API call to the tenant",
    "decimal": "Store amounts as Decimal/NUMERIC, never float",
    "audit": "Keep an audit trail of status changes",
    "pydantic": "Validate every payload with Pydantic models",
    "migration": "Ship schema changes as versioned migrations",
    "tva": "Compute VAT per line with the applicable rate",
    "vat": "Compute VAT per line with the applicable rate",
    "pdf": "Generate the PDF document from the stored, validated data",
}


def _hits(exchanges: list[tuple[ToolCall, Any]], tool: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for call, result in exchanges:
        payload = parse_result(result) if call.name == tool else None
        if payload:
            hits.extend(payload.get("results", []))
    return hits


def _topic(objective: str) -> str:
    """The business noun of the request (e.g. 'devis'), else its longest keyword."""
    words = keywords(objective)
    for word in words:
        if strip_accents(word) in _ENTITY_NAMES:
            return word
    return max(words, key=len) if words else "task"


def _first_sentence(text: str) -> str:
    """First substantial sentence (skips headings such as '6.' or 'Version 2.4')."""
    sentences = [sentence for sentence in split_sentences(text) if len(sentence) >= 25]
    return truncate(sentences[0] if sentences else text, 300)


# --------------------------------------------------------------------------------------
# Research
# --------------------------------------------------------------------------------------
def handle_research(request: LLMRequest) -> OfflineReply:
    context = read_context(request)
    step = context["step"]
    exchanges = tool_exchanges(request)
    if not exchanges:
        return OfflineReply(
            tool_calls=[
                ToolCall(
                    id="offline_research_kb",
                    name="knowledge_search",
                    arguments={
                        "query": truncate(step["description"], 250),
                        "top_k": 4,
                        "category": None,
                    },
                ),
                ToolCall(
                    id="offline_research_tasks",
                    name="task_lookup",
                    arguments={"query": _topic(context["objective"]), "task_id": None, "limit": 3},
                ),
            ]
        )

    findings = [
        Finding(
            claim=_first_sentence(hit["content"]),
            evidence=truncate(hit["content"], 400),
            source_ids=[hit["source_id"]],
        )
        for hit in _hits(exchanges, "knowledge_search")[:4]
    ]
    for call, result in exchanges:
        payload = parse_result(result) if call.name == "task_lookup" else None
        for task in (payload or {}).get("tasks", [])[:2]:
            findings.append(
                Finding(
                    claim=f"A related past task exists ({task['status']}): {task['request']}",
                    evidence=f"task_lookup returned task {task['id']}.",
                )
            )
    first_source = findings[0].source_ids if findings and findings[0].source_ids else []
    for feedback in context.get("feedback", [])[:3]:
        findings.append(
            Finding(
                claim=f"Review feedback addressed: {truncate(feedback, 200)}",
                evidence="The retrieved material was re-examined with this feedback in mind.",
                source_ids=first_source,
            )
        )
    gaps = []
    if not findings:
        findings = [
            Finding(
                claim="No relevant evidence was found in the available sources.",
                evidence="knowledge_search and task_lookup returned no usable result.",
            )
        ]
        gaps.append("No internal source covers this step")
    report = ResearchReport(
        summary=f"{len(findings)} finding(s) for '{step['title']}' from the knowledge base and "
        "the task history.",
        confidence="medium" if not gaps else "low",
        findings=findings[:10],
        gaps=gaps,
    )
    return reply_json(report)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def handle_data(request: LLMRequest) -> OfflineReply:
    exchanges = tool_exchanges(request)
    reads = {
        (call.arguments.get("entity"), call.arguments.get("query")): parse_result(result)
        for call, result in exchanges
        if call.name == "database_read"
    }
    if not reads:
        return OfflineReply(
            tool_calls=[
                ToolCall(
                    id="offline_data_tasks",
                    name="database_read",
                    arguments={"entity": "tasks", "query": "count_by_status", "limit": 10},
                ),
                ToolCall(
                    id="offline_data_runs",
                    name="database_read",
                    arguments={"entity": "runs", "query": "summary", "limit": 10},
                ),
            ]
        )

    counts = (reads.get(("tasks", "count_by_status")) or {}).get("counts", {})
    total = sum(counts.values())
    completed = counts.get("COMPLETED", 0)
    calculations = [parse_result(r) for c, r in exchanges if c.name == "calculator"]
    if total and not calculations:
        return OfflineReply(
            tool_calls=[
                ToolCall(
                    id="offline_data_rate",
                    name="calculator",
                    arguments={"expression": f"{completed} / {total} * 100"},
                )
            ]
        )

    metrics = [
        Metric(
            name="tasks_total",
            value=total,
            unit="tasks",
            method="database_read tasks.count_by_status",
        )
    ]
    metrics += [
        Metric(
            name=f"tasks_{status.lower()}",
            value=count,
            unit="tasks",
            method="database_read tasks.count_by_status",
        )
        for status, count in counts.items()
    ]
    insights = []
    rate = (calculations[0] or {}).get("result") if calculations else None
    if rate is not None:
        metrics.append(
            Metric(
                name="completion_rate",
                value=rate,
                unit="%",
                method=f"calculator: {completed} / {total} * 100",
            )
        )
        insights.append(f"{rate:.1f}% of the tenant's tasks are completed")
    runs = reads.get(("runs", "summary")) or {}
    if runs.get("total"):
        metrics.append(
            Metric(
                name="runs_total",
                value=runs["total"],
                unit="runs",
                method="database_read runs.summary",
            )
        )
        metrics.append(
            Metric(
                name="avg_cost_per_run",
                value=runs["avg_cost_usd"],
                unit="USD",
                method="database_read runs.summary",
            )
        )
        metrics.append(
            Metric(
                name="avg_agent_calls_per_run",
                value=runs["avg_agent_calls"],
                unit="calls",
                method="database_read runs.summary",
            )
        )
    if counts:
        dominant = max(counts, key=counts.__getitem__)
        insights.append(f"Most frequent status: {dominant} ({counts[dominant]} of {total})")
    analysis = DataAnalysis(
        summary=f"Analysed {total} task(s) and {runs.get('total', 0)} run(s) of the tenant.",
        confidence="high" if total >= 20 else "medium",
        metrics=metrics[:15],
        insights=insights,
        limitations=["Small sample: conclusions are indicative only"] if total < 20 else [],
    )
    return reply_json(analysis)


# --------------------------------------------------------------------------------------
# Coding
# --------------------------------------------------------------------------------------
def _domain_entity(objective: str) -> str:
    for word in keywords(strip_accents(objective.lower())):
        if word in _ENTITY_NAMES:
            return _ENTITY_NAMES[word]
    words = [w for w in keywords(objective) if len(w) > 3]
    return words[0].capitalize() if words else "Record"


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def handle_coding(request: LLMRequest) -> OfflineReply:
    context = read_context(request)
    exchanges = tool_exchanges(request)
    if not exchanges:
        query = f"{context['objective']} conventions architecture security"
        return OfflineReply(
            tool_calls=[
                ToolCall(
                    id="offline_coding_kb",
                    name="knowledge_search",
                    arguments={"query": truncate(query, 280), "top_k": 4, "category": None},
                )
            ]
        )

    entity = _domain_entity(context["objective"])
    snake = _snake(entity)
    plural = f"{snake}s"
    with_lines = entity in _LINE_ITEM_ENTITIES

    components = [
        Component(
            name=f"{entity}Service",
            responsibility=f"Business rules for {plural}: creation, validation, status transitions",
        ),
        Component(
            name=f"{entity}Repository",
            responsibility="Tenant-scoped persistence: explicit tenant filter plus "
            "PostgreSQL row-level security",
        ),
        Component(
            name=f"{entity} API router",
            responsibility="REST endpoints with Pydantic request and response models",
        ),
        Component(
            name="Audit trail",
            responsibility=f"Records every {snake} status change with author and timestamp",
        ),
    ]
    if with_lines:
        components.append(
            Component(
                name=f"{entity}Totals",
                responsibility="Server-side computation of line totals, discounts and taxes "
                "with Decimal",
            )
        )
    endpoints = [
        Endpoint(method="POST", path=f"/api/v1/{plural}", purpose=f"Create a draft {snake}"),
        Endpoint(
            method="GET",
            path=f"/api/v1/{plural}",
            purpose=f"List the tenant's {plural} (paginated)",
        ),
        Endpoint(method="GET", path=f"/api/v1/{plural}/{{id}}", purpose=f"Read one {snake}"),
        Endpoint(
            method="PATCH", path=f"/api/v1/{plural}/{{id}}", purpose=f"Update a draft {snake}"
        ),
        Endpoint(
            method="POST",
            path=f"/api/v1/{plural}/{{id}}/status",
            purpose="Validated status transition",
        ),
    ]
    data_model = [
        Entity(
            name=entity,
            fields=[
                "id: UUID",
                "tenant_id: str",
                "number: str",
                "status: enum",
                "total_amount: Decimal",
                "currency: str",
                "created_at: datetime",
            ],
        )
    ]
    if with_lines:
        data_model.append(
            Entity(
                name=f"{entity}Line",
                fields=[
                    f"{snake}_id: UUID",
                    "description: str",
                    "quantity: Decimal",
                    "unit_price: Decimal",
                    "tax_rate: Decimal",
                ],
            )
        )
    skeleton = (
        f"class {entity}(Base):\n"
        f'    __tablename__ = "{plural}"\n'
        "    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)\n"
        "    tenant_id: Mapped[str] = mapped_column(index=True)\n"
        '    status: Mapped[str] = mapped_column(default="draft")\n'
        "    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))\n"
    )
    files = [
        CodeFile(
            path=f"app/{snake}/models.py", language="python", purpose="ORM model", content=skeleton
        )
    ]

    hits = _hits(exchanges, "knowledge_search")
    risks = [
        "Cross-tenant data leaks if a query forgets the tenant filter (mitigated by RLS)",
        "Rounding errors if amounts are stored as floats",
    ]
    cited: list[str] = []
    for hit in hits:
        content = strip_accents(hit["content"].lower())
        for hint, convention in _CONVENTION_HINTS.items():
            if re.search(rf"\b{hint}\b", content) and not any(convention in r for r in risks):
                risks.append(f"{convention} [{hit['source_id']}]")
                cited.append(hit["source_id"])
    for feedback in context.get("feedback", [])[:3]:
        risks.append(f"Review feedback addressed: {truncate(feedback, 200)}")
    proposal = CodeProposal(
        summary=f"Modular {entity} feature: service, tenant-scoped repository, REST API"
        + (", line items with server-side totals" if with_lines else "")
        + ", audit trail.",
        confidence="medium",
        components=components,
        api_endpoints=endpoints,
        data_model=data_model,
        files=files,
        tests=[
            f"Creating a {snake} validates the payload and returns 201",
            f"A tenant cannot read another tenant's {plural}",
            "Invalid status transitions are rejected with 409",
        ]
        + (["Totals are recomputed server-side from the lines"] if with_lines else []),
        risks=risks[:8],
        source_ids=sorted(set(cited))[:10],
    )
    return reply_json(proposal)
