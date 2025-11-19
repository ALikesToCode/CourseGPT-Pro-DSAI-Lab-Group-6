# Router Control Room — Overview

## Problem & Objective
CourseGPT-Pro’s router agent decides how complex homework questions should be decomposed across three specialists (`/general-search`, `/math`, `/code`). Milestone 6 focuses on making that router usable outside notebooks by (1) shipping a demo-ready Hugging Face Space, (2) exposing a deterministic ZeroGPU API, and (3) wrapping everything in documentation so future teams can reproduce the deployment.

## Architecture Snapshot
- **Input**: user prompt + optional context streamed from the Gradio UI (`hf_space/app.py`).
- **Router Core**: Vertex-tuned adapters (Gemma 3 27B, Qwen 3 32B, Llama 3.1 8B) hosted either on Hugging Face Inference Endpoints or the bundled ZeroGPU FastAPI service (`zero-gpu-space/app.py`).
- **Validation Layer**: structured JSON extractor + `schema_score` field checks to enforce `route_plan`, `metrics`, and `todo_list` fidelity before showing results.
- **Specialist Bridge**: lightweight `AgentHandler` protocol (`Milestone-6/agents/base.py`) plus optional Gemini fallbacks so `/math`, `/code`, and `/general-search` can be swapped in without touching the UI logic.
- **Benchmark Harness**: Milestone 5 evaluation utilities (`schema_score.py`, `router_benchmark_runner.py`) and the curated hard benchmark JSONL enable on-demand regression tests from inside the Space.

Refer to `assets/image1.png` for the end-to-end system diagram that the router prompt mirrors.

## Deployed Components
| Component | Location | Purpose | Status |
| --- | --- | --- | --- |
| Router UI Space | `Milestone-6/router-agent/hf_space` | Gradio front-end with chat + benchmark tabs, schema validation, and agent simulation. | Ready for upload to `CourseGPT-Pro-DSAI-Lab-Group-6/router-control-room` (CPU or A10G).
| ZeroGPU Backend | `Milestone-6/router-agent/zero-gpu-space` | FastAPI service that loads merged checkpoints with 8-bit/4-bit fallbacks and exposes `/v1/generate`. | Tested locally; requires ZeroGPU hardware for full-speed inference.
| Test Harness | `Milestone-6/router-agent/test_router_models.py` | CLI smoke test that hits every configured model/API and validates JSON. | Run manually before each deployment.
| Benchmark Assets | `Milestone-6/router-agent/hf_space/router_benchmark_*` | Hard benchmark set + threshold registry reused from Milestone 5 for CI gating. | Bundled with the Space; thresholds mirror `Milestone-5/router-agent/router_benchmark_thresholds.json`.

## Deliverables Covered Here
1. **Deployment** – instructions for hosting the UI + ZeroGPU API, env vars, and CLI smoke tests.
2. **Comprehensive Technical Documentation** – see `technical_doc.md` (environment, data, model, evaluation, inference, deployment, design, monitoring, reproducibility).
3. **User & API Docs** – see `user_guide.md` (non-technical walkthrough) and `api_doc.md` (REST contract + cURL examples).
4. **Licensing & References** – see `licenses.md` for code/data/model obligations.
5. **Final Report Outline** – see `final_project_report_outline.md` for the PDF chapter plan referencing all milestones.

Keep these Markdown files under version control (`Milestone-6/router-agent/docs/`) so updates to the route model, datasets, or deployment knobs stay auditable.
