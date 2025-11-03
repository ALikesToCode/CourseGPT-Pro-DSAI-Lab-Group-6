"""Math agent stub implementing the shared AgentHandler protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import sys

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agents.base import AgentHandler, AgentRequest, AgentResult


class MathAgent(AgentHandler):
    """Placeholder math agent. Replace logic inside `invoke` with real math reasoning."""

    name = "math"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def invoke(self, request: AgentRequest) -> AgentResult:
        # TODO: Replace with calls to symbolic engines / specialist LLMs.
        content = (
            "MathAgent placeholder response.\n"
            f"Received query: {request.user_query}\n"
            "Implement derivations, proofs, and numeric checks here."
        )
        return AgentResult(content=content, metrics={"status": "not_implemented"})
