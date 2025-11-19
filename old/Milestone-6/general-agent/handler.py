"""General-search agent stub implementing the shared AgentHandler protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import sys

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agents.base import AgentHandler, AgentRequest, AgentResult


class GeneralSearchAgent(AgentHandler):
    """Placeholder general-search agent. Replace with retrieval and summarisation logic."""

    name = "general-search"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def invoke(self, request: AgentRequest) -> AgentResult:
        # TODO: Implement actual search + summarisation workflow.
        content = (
            "GeneralSearchAgent placeholder response.\n"
            f"Received query: {request.user_query}\n"
            "Connect this handler to your search APIs and summarisation models."
        )
        return AgentResult(content=content, metrics={"status": "not_implemented"})
