## Gemini Router Dataset Script

This folder holds the Gemini-powered script (built on the Google Gen AI
`google-genai` SDK) that synthesizes router-training examples for the
CourseGPT-Pro agentic orchestrator. The dataset targets graduate-level / IOI
calibre STEM prompts spanning advanced mathematics, machine learning, and
algorithmic programming. Each generated record teaches the router how to
dispatch a user request to the appropriate tools:

- `/math(context)` for symbolic or numeric math reasoning.
- `/code(context)` for Python-heavy coding steps.
- `/general-search(context, mode=web|rag|both)` for the blue general-search agent.

### Dataset focus

- Emphasizes multivariate calculus, tensor algebra, probabilistic modelling,
  deep learning theory, and competition-grade algorithms.
- Ensures router examples challenge an 8B parameter model by demanding explicit
  reasoning chains (`thinking_outline`) and verified deliverables.
- Encourages the use of authoritative sources (e.g., arXiv, MIT OCW, IOI
  archives) when the search tool is invoked.
- Encodes CEO-style orchestration via `handoff_plan`, detailing how /general-search,
  /math, and /code collaborate, verify outputs, and recover from failures.
- Enforces strict prompt quality checks: minimum query lengths, domain-specific
  terminology, detailed route contexts, and multiple verification steps.

### Files

| File | Purpose |
|------|---------|
| `gemini_router_dataset.py` | CLI utility that queries Gemini 2.5 Pro (or compatible) to create synthetic router datasets in JSONL format. |

### Quick Start

1. **Install dependencies (Python ≥ 3.9)**:
   ```bash
   pip install -r requirements.txt
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
   python gemini_router_dataset.py --count 24 --output data/router_dataset.jsonl --concurrency 4
   ```

During execution the script displays a Rich-powered progress spinner for Gemini
calls and summarizes the output location once writing completes. Validation
errors or API issues are surfaced in highlighted panels for quick diagnosis. A
heuristic `quality_score` for each example is logged during generation, and the
router emits diversity warnings if themes or route signatures repeat.

Existing JSONL files are opened in append mode. By default the script resumes
numbering after the last existing record so partially generated datasets can be
extended safely.

### Dataset shape

Each JSON line produced by the script conforms to:

```json
{
  "id": "router_0000",
  "user_query": "Design a graduate-level challenge: rigorously analyse the Hessian spectrum of a transformer fine-tuned with LoRA adapters, prove stability conditions, and supply a JAX experiment that verifies them.",
  "task_summary": "One-sentence synopsis of the ask.",
  "route_plan": [
    "/math(Develop a spectral proof for the stability of the transformer Hessian under low-rank adapters.)",
    "/code(Write Python to simulate Hessian eigenvalue drift using JAX and plot convergence diagnostics.)",
    "/general-search(query=\"site:arxiv.org transformer low-rank adapter stability proof\", mode=web)"
  ],
  "route_rationale": "Explains why math/code/general-search are needed, explicitly noting the blue search agent.",
  "expected_artifacts": [
    "symbolic_eigen_derivation",
    "python_verification_snippet",
    "reference_digest"
  ],
  "thinking_outline": [
    "1. Formally restate the PDE-constrained optimization goal and identify boundary conditions.",
    "2. Derive the matrix representation and prove positive semi-definiteness to justify eigen decomposition.",
    "3. Implement a Python simulation to validate the spectral properties under perturbations.",
    "4. Cross-check results against authoritative references and summarize convergence behaviour."
  ],
  "handoff_plan": "/general-search -> /math -> /code -> router QA; if proofs fail, loop back to /general-search for alternative theory, then re-validate.",
  "difficulty": "advanced",
  "tags": ["transformers", "spectral_analysis"],
  "quality_score": 92.4
}
```

Key constraints enforced by the generator:

- `route_plan` uses only `/math`, `/code`, `/general-search`.
- `/code` contexts explicitly mention Python deliverables.
- `/general-search` commands always include `mode=web|rag|both` and describe the
  blue general-search agent with research-grade queries (e.g., targeting arXiv or MIT resources).
- `expected_artifacts` lists 3-5 deliverables spanning proofs, code, and citations.
- `thinking_outline` captures 4-6 numbered steps that expose the reasoning chain
  the router expects downstream agents to follow, including at least two explicit verification steps.
- `handoff_plan` summarizes the CEO-style orchestration, showing agent flow with
  arrows and verification/fallback behaviour.
- `quality_score` encodes heuristic quality metrics (context detail, verification density,
  domain terminology) to support downstream filtering.
- `user_query` (≥100 chars) and `task_summary` (≥50 chars) must sound like graduate research or
  competition prep requests and include domain-specific terminology.
- `route_plan` contexts must each provide ≥40 characters with precise notation, algorithms,
  or constraints, and include domain-specific keywords.
- `difficulty` is always `advanced` to align with high-stakes routing for an 8B model.

### CLI options

```
usage: gemini_router_dataset.py [-h] [--count COUNT] [--output OUTPUT]
                                [--model MODEL] [--temperature TEMPERATURE]
                                [--max-retries MAX_RETRIES]
                                [--sleep-base SLEEP_BASE] [--seed SEED]
                                [--concurrency CONCURRENCY]
                                [--start-index START_INDEX]
```

- `--temperature` adjusts Gemini's creativity when constructing examples.
- `--concurrency` controls how many Gemini requests are issued in parallel.
- `--start-index` overrides the auto-resume behaviour when you need custom ID ranges.

### Extending the generator

- Add new route variants (e.g., `/general-search` + `/math`) inside the
  `ROUTE_VARIANTS` list to diversify coverage.
- Adjust `GENERAL_SEARCH_MODES` or dataset schema constants to tailor the
  dataset to downstream labeling needs.

### Incremental writes & recovery

- Records are appended to the JSONL file immediately after each successful
  Gemini response so partial progress is never lost.
- Existing datasets are detected automatically; new IDs continue from the number
  of existing lines unless `--start-index` is specified.
- Failures are retried with exponential backoff and summarized at the end, so
  failed indices can be regenerated without disturbing already written rows.
