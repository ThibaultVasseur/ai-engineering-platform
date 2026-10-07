"""The default tool catalogue."""

from __future__ import annotations

from app.tools.base import Tool
from app.tools.calculator import CALCULATOR
from app.tools.database import DATABASE_READ, DATABASE_WRITE
from app.tools.knowledge import KNOWLEDGE_SEARCH
from app.tools.registry import ToolRegistry
from app.tools.task_tools import TASK_LOOKUP

DEFAULT_TOOLS: tuple[Tool, ...] = (
    CALCULATOR,
    KNOWLEDGE_SEARCH,
    TASK_LOOKUP,
    DATABASE_READ,
    DATABASE_WRITE,
)


def default_tool_registry(
    *, timeout_seconds: float = 10.0, max_output_chars: int = 8_000
) -> ToolRegistry:
    return ToolRegistry(
        DEFAULT_TOOLS,
        default_timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
