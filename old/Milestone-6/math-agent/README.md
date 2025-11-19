# Math Agent (Milestone 6 Scaffold)

This directory houses the specialised **Math Agent** implementation that the router will invoke for formal derivations, proofs, and quantitative reasoning.

## Expected Responsibilities
- Consume `AgentRequest` objects (see `Milestone-6/agents/base.py`).
- Produce responses with precise mathematical steps, formulas, and citations.
- Surface metrics such as token usage, proof verification status, or CAS checks.

## Implementation Checklist
1. Populate `handler.py` with your model/tool loading logic (SymPy, math LLM, etc.).
2. Honour the `AgentHandler` interface:
   ```python
   from Milestone-6.agents.base import AgentHandler, AgentRequest, AgentResult
   class MathAgent(AgentHandler):
       name = "math"
       def __init__(self, model_id: str, **kwargs): ...
       def invoke(self, request: AgentRequest) -> AgentResult: ...
   ```
3. Add unit tests or smoke scripts (e.g., `tests/test_smoke.py`) that validate the interface.
4. Update `Milestone-6/router-agent/hf_space/app.py` to instantiate and call the agent when the router plan includes `/math(...)`.

### Template
- `math_agent_template.py` demonstrates how to integrate an LLM-backed solver and optional validation hooks. Copy/rename it when you begin your implementation.

## Environment Notes
- Prefer environment variables for credentials: `MATH_AGENT_MODEL`, `OPENAI_API_KEY`, etc.
- When running in Hugging Face Spaces, ensure dependencies are declared in `hf_space/requirements.txt`.

Once the handler is filled out, the router deployment can import it via:
```python
from Milestone-6.math-agent.handler import MathAgent
```
