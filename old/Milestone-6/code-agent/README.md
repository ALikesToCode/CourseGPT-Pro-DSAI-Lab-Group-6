# Code Agent (Milestone 6 Scaffold)

This folder contains the **Code Agent** stub responsible for executing or synthesising code when invoked by the router.

## Responsibilities
- Analyse the routed task segment and produce runnable code, explanations, and validation logs.
- Optionally execute snippets in a sandboxed environment and report unit-test results or runtime metrics.
- Return structured `AgentResult` objects (see `Milestone-6/agents/base.py`).

## Implementation Steps
1. Fill in `handler.py` with your code-generation runtime (e.g., OpenAI GPT-4, Code Llama, local sandbox).
2. Provide safety controls: timeouts, restricted environment variables, and output filtering.
3. Optionally add integration tests under `tests/` that run representative prompts.
4. Wire the handler into the Hugging Face Space or router orchestrator for `/code(...)` tool calls.

Import example:
```python
from Milestone-6.code-agent.handler import CodeAgent
```

Update the repository documentation once the implementation is ready so reviewers understand resource requirements and limitations.
