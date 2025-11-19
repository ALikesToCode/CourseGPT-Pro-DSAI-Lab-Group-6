"""Shared agent interface definitions for Milestone 6 deployments.

Each specialised agent (math/code/general) should implement the `AgentHandler`
protocol so the router and Hugging Face Space can invoke them interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class AgentRequest:
    """Standard payload sent to each agent.

    Attributes:
        user_query: The raw user request or task segment routed to the agent.
        context: Optional context dictionary (documents, code, etc.).
        plan_metadata: Router plan metadata (step id, tool call rationale, etc.).
    """

    user_query: str
    context: Optional[Dict[str, Any]] = None
    plan_metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResult:
    """Structured agent output consumed by downstream components."""

    content: str
    citations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None  # Preserve untouched model/tool output when helpful.


class AgentHandler(Protocol):
    """Protocol for all specialised agents.

    Implementations should be side-effect free and encapsulate any external
    service calls (LLMs, Python execution, search) behind this interface.
    """

    name: str

    def invoke(self, request: AgentRequest) -> AgentResult:
        """Process the request and return a structured result."""

