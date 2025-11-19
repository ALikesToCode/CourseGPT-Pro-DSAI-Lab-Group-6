"""Template math agent implementation.

Copy this file, rename it (e.g., `handler.py`), and replace the TODO sections
with real logic. The template demonstrates how to plug an LLM-backed solver
into the shared AgentHandler interface and optionally perform symbolic checks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import sys

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agents.base import AgentHandler, AgentRequest, AgentResult


class TemplateMathAgent(AgentHandler):
    """Starting point for the Milestone 6 math agent.

    Features you might add:
        * Call out to a hosted LLM (e.g., OpenAI, Gemini) with a math-specific prompt.
        * Use SymPy or other CAS tools to verify results.
        * Return LaTeX-formatted derivations plus natural language explanations.
    """

    name = "math"

    def __init__(self, model_name: Optional[str] = None, **kwargs: Any) -> None:
        self.model_name = model_name or os.getenv("MATH_AGENT_MODEL", "gpt-4o")
        self.params = kwargs
        # TODO: Initialise your SDK clients here (e.g., OpenAI, Google Vertex AI).
        # Example:
        # import openai
        # openai.api_key = os.environ["OPENAI_API_KEY"]

    def _build_prompt(self, request: AgentRequest) -> str:
        """Construct the system/user prompt sent to the math model."""
        context = request.context or {}
        prompt_parts = [
            "You are a rigorous mathematics assistant.",
            "Provide step-by-step reasoning, cite references where possible,",
            "and ensure the final answer is clearly stated.",
            "",
            f"User query:\n{request.user_query}",
        ]
        if context:
            prompt_parts.append(f"\nAdditional context:\n{context}")
        return "\n".join(prompt_parts)

    def _call_model(self, prompt: str) -> str:
        """Invoke the underlying model/toolchain.

        Replace this stub with actual API calls. Return the raw string output.
        """
        # TODO: Implement call to LLM or symbolic engine.
        return "TODO: replace with model output."

    def _post_process(self, raw_output: str, request: AgentRequest) -> AgentResult:
        """Convert the raw model output into an AgentResult."""
        # TODO: Optionally parse LaTeX, extract equations, or run verification.
        metrics = {
            "model_name": self.model_name,
            "verification": "not_run",
        }
        return AgentResult(content=raw_output, metrics=metrics, raw_response=raw_output)

    def invoke(self, request: AgentRequest) -> AgentResult:
        prompt = self._build_prompt(request)
        raw_output = self._call_model(prompt)
        return self._post_process(raw_output, request)


def load_agent() -> TemplateMathAgent:
    """Convenience factory for external callers."""
    return TemplateMathAgent()


__all__ = ["TemplateMathAgent", "load_agent"]
