# Router Control Room — User Guide
Audience: non-technical reviewers who want to try the router demo or run a local dry run without touching the training stack.

## 1. What the App Does
- Accepts a natural-language homework-style question.
- Chooses which specialist agents (`/general-search`, `/math`, `/code`) should participate, in what order, and why.
- Emits a structured JSON plan containing rationales, TODOs, acceptance criteria, and success metrics so downstream agents can act deterministically.

## 2. Getting Access
| Option | URL / Command | Notes |
| --- | --- | --- |
| Hosted demo | `https://huggingface.co/spaces/Alovestocode/router-control-room-private` (or the org Space once public) | Requires a Hugging Face account if space is private. | 
| Local UI | `cd Milestone-6/router-agent/hf_space && gradio app.py` | Uses bundled sample plan unless `HF_ROUTER_API` or `HF_ROUTER_REPO` is set.
| API only | `https://Alovestocode-router-router-zero.hf.space/v1/generate` | Requires tokened access if checkpoint is private.

## 3. Quick Start (Hosted UI)
1. Open the Space URL.
2. Select a router backend in the dropdown (sample plan, adapter, base model, or custom API).
3. Paste your task description in the “User Prompt” box. Include relevant context or acceptance criteria if you have them.
4. Click **Generate Router Plan**. The JSON response appears in the right pane with collapsible sections.
5. (Optional) Click **Simulate Specialists** to see stubbed `/math`, `/code`, `/general-search` outputs.
6. (Optional) Upload a predictions JSONL file in the **Benchmark** tab to validate against the hard benchmark thresholds.

## 4. Local Launch (offline demo)
```bash
cd Milestone-6/router-agent/hf_space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_ROUTER_API=http://localhost:8000/v1/generate  # optional, if hitting local FastAPI
gradio app.py  # or python app.py
```
- To run the ZeroGPU FastAPI app locally for smoke tests: `uvicorn zero-gpu-space.app:app --reload --host 0.0.0.0 --port 8000` (requires >48GB VRAM for full models).
- Without any env vars, the UI shows the built-in sample plan so reviewers can still understand the JSON contract.

## 5. Input / Output Reference
| Field | Description | Example |
| --- | --- | --- |
| `route_plan` | Ordered list of tool invocations with inline arguments. | `/general-search(query="bayesian evidence lower bound", mode=web)` |
| `route_rationale` | Natural-language justification for the selected tool chain. | “Search surfaces survey papers, math formalises ELBO, code checks autodiff.” |
| `expected_artifacts` | Files or summaries each tool must return. | `"Annotated derivation of ELBO"` |
| `thinking_outline` | Step-by-step reasoning skeleton. | “1. Gather citations … 2. Express variational objective …” |
| `todo_list` | Markdown checklist tying each tool to an owner. | `- [ ] /code: implement PyTorch check` |
| `acceptance_criteria` | Definition of done for QA. | `"All integrals carry variable definitions"` |
| `metrics` | Primary/secondary metrics + guidance/computation fields when present. | `"primary": ["Route accuracy >=0.8"], "primary_guidance": "Keep <=600 tokens"` |

## 6. Example Interaction
**Prompt**
> “I need a plan to (a) compare Classical Gram-Schmidt vs Modified Gram-Schmidt, (b) prove the stability difference, (c) show sample Python code.”

**What to expect**
- Router chooses `/general-search → /math → /code`.
- TODO list assigns citations to general-search, numerical stability proof to math, and NumPy implementation to code.
- Acceptance criteria mention orthogonality tolerance & runtime measurement.

## 7. Troubleshooting
| Symptom | Fix |
| --- | --- |
| “Router backend not configured; using bundled sample plan.” | Set `HF_ROUTER_REPO` (adapter/base) or `HF_ROUTER_API` (ZeroGPU) in the Space settings. |
| “Router output did not contain a JSON object.” | Increase `max_new_tokens` to 700, lower temperature to 0.15, or re-run after clearing the `<think>` block. |
| Benchmark tab fails with schema import error. | Ensure `schema_score.py` + `router_benchmark_runner.py` are uploaded with the Space (already bundled). |
| Agent simulation is blank. | Confirm handler files exist under `Milestone-6/{math,code,general}-agent/handler.py`. Otherwise the UI falls back to the Gemini stub (requires `GOOGLE_API_KEY`). |

## 8. Support & Feedback
- File GitHub issues under `Milestone-6/router-agent` with logs plus screenshots (use the Debug accordion output).
- For urgent incidents (Space down, JSON regressions), ping the Router squad on Slack `#coursegpt-router` with the failing prompt and backend selection.
- Add screenshots or Loom demo links to this guide by dropping PNGs under `assets/` and embedding them here.
