"""Code agent stub implementing the shared AgentHandler protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import sys

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agents.base import AgentHandler, AgentRequest, AgentResult


class CodeAgent(AgentHandler):
    """Placeholder code agent. Replace logic inside `invoke` with code synthesis/execution."""

    name = "code"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def invoke(self, request: AgentRequest) -> AgentResult:
        # TODO: Execute code or call specialised LLMs, return build/test artifacts.
        content = (
            "CodeAgent placeholder response.\n"
            f"Received query: {request.user_query}\n"
            "Implement code generation, execution, and validation here."
        )
        return AgentResult(content=content, metrics={"status": "not_implemented"})
