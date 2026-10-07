"""Introspection: which agents exist, what they may do, with which tools and prompts."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.agents.registry import AGENT_PROFILES
from app.api.dependencies import ContainerDep
from app.schemas.agent import AgentInfo
from app.schemas.tool import ToolInfo
from app.services.run_service import PROMPT_VERSIONS

router = APIRouter(tags=["agents"])


class AgentCatalog(BaseModel):
    agents: list[AgentInfo]
    tools: list[ToolInfo]
    graph_mermaid: str


@router.get("/agents", response_model=AgentCatalog, summary="Agents, permissions, tools, graph")
async def list_agents(container: ContainerDep) -> AgentCatalog:
    model = container.settings.effective_llm_model
    agents = [
        AgentInfo(
            name=profile.name.value,
            role=profile.role,
            mission=profile.mission,
            tools=sorted(profile.allowed_tools),
            output_schema=profile.output_schema,
            uses_llm=profile.uses_llm,
            prompt_version=PROMPT_VERSIONS.get(profile.name.value),
            model=model
            if profile.uses_llm or container.settings.supervisor_strategy == "llm"
            else None,
        )
        for profile in AGENT_PROFILES.values()
    ]
    return AgentCatalog(
        agents=agents,
        tools=container.tool_registry.describe(),
        graph_mermaid=container.graph_mermaid,
    )
