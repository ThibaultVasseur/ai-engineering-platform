"""Registry of offline handlers, one per prompt name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.llm.prompts import (
    CODING,
    CRITIC,
    DATA,
    PLANNER,
    PROMPT_OPTIMIZER,
    RAG,
    RESEARCH,
    SUPERVISOR,
    SYNTHESIZER,
)
from app.llm.providers.offline.knowledge import handle_rag
from app.llm.providers.offline.planning import (
    handle_planner,
    handle_prompt_optimizer,
    handle_supervisor,
)
from app.llm.providers.offline.review import handle_critic, handle_synthesizer
from app.llm.providers.offline.workers import handle_coding, handle_data, handle_research

if TYPE_CHECKING:
    from app.llm.providers.offline import OfflineHandler


def default_handlers() -> dict[str, OfflineHandler]:
    return {
        PROMPT_OPTIMIZER.name: handle_prompt_optimizer,
        PLANNER.name: handle_planner,
        SUPERVISOR.name: handle_supervisor,
        RESEARCH.name: handle_research,
        DATA.name: handle_data,
        CODING.name: handle_coding,
        RAG.name: handle_rag,
        SYNTHESIZER.name: handle_synthesizer,
        CRITIC.name: handle_critic,
    }
