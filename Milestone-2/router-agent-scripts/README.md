## Gemini Router Dataset Script

This folder holds the Gemini-powered script that synthesizes router-training
examples for the CourseGPT-Pro agentic orchestrator. Each generated record
teaches the router how to dispatch a user request to the appropriate tools:

- `/math(context)` for symbolic or numeric math reasoning.
- `/code(context)` for Python-heavy coding steps.
- `/general-search(context, mode=web|rag|both)` for the blue general-search agent.

### Files

| File | Purpose |
|------|---------|
| `gemini_router_dataset.py` | CLI utility that queries Gemini 2.5 Pro (or compatible) to create synthetic router datasets in JSONL format. |

### Quick Start

1. **Install dependencies (Python ≥ 3.9)**:
   ```bash
   pip install google-generativeai python-dotenv
   ```
2. **Configure `.env` (or export vars)**:
   ```bash
   cat <<'EOF' > .env
   GOOGLE_API_KEY=your_gemini_api_key
   GEMINI_MODEL_NAME=gemini-2.5-pro
   EOF
   ```
3. **Generate 24 examples to `data/router_dataset.jsonl`**:
   ```bash
   python gemini_router_dataset.py --count 24 --output data/router_dataset.jsonl
   ```
4. **Dry-run without Gemini (sanity check)**:
   ```bash
   python gemini_router_dataset.py --offline --count 4
   ```

### Dataset shape

Each JSON line produced by the script conforms to:

```json
{
  "id": "router_0000",
  "user_query": "Student-facing prompt...",
  "task_summary": "One-sentence synopsis of the ask.",
  "route_plan": [
    "/math(Compute the eigenvalues of the stiffness matrix with the given boundary conditions.)",
    "/code(Write Python to validate the eigenvalues numerically via NumPy.)",
    "/general-search(Collect two trustworthy citations on stiffness matrices, mode=web)"
  ],
  "route_rationale": "Explains why math/code/general-search are needed, explicitly noting the blue search agent.",
  "expected_artifacts": [
    "symbolic_eigen_derivation",
    "python_verification_snippet",
    "reference_digest"
  ],
  "difficulty": "intermediate",
  "tags": ["linear_algebra", "finite_element"]
}
```

Key constraints enforced by the generator:

- `route_plan` uses only `/math`, `/code`, `/general-search`.
- `/code` contexts explicitly mention Python deliverables.
- `/general-search` commands always include `mode=web|rag|both` and describe the
  blue general-search agent.
- `difficulty` is `introductory`, `intermediate`, or `advanced`.

### CLI options

```
usage: gemini_router_dataset.py [-h] [--count COUNT] [--output OUTPUT]
                                [--model MODEL] [--temperature TEMPERATURE]
                                [--max-retries MAX_RETRIES]
                                [--sleep-base SLEEP_BASE] [--seed SEED]
                                [--offline]
```

- `--temperature` adjusts Gemini's creativity when constructing examples.
- `--offline` skips Gemini calls and emits deterministic stubs while preserving
  the route-plan structure—handy for CI smoke tests.

### Extending the generator

- Add new route variants (e.g., `/general-search` + `/math`) inside the
  `ROUTE_VARIANTS` list to diversify coverage.
- Adjust `GENERAL_SEARCH_MODES` or dataset schema constants to tailor the
  dataset to downstream labeling needs.
