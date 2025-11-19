# Milestone 6 — Router Deployment Checklist

## 1. Environment Setup
- [ ] Provision Hugging Face Space (Gradio SDK ≥ 4.36, Python 3.11).
- [ ] Set environment variables:
  - `HF_ROUTER_REPO` — Hugging Face model id for the router adapter (optional, falls back to sample plan).
  - `HF_TOKEN` — Access token if the model requires authentication.
  - `MATH_AGENT_MODEL`, `CODE_AGENT_MODEL`, `GENERAL_AGENT_CONFIG` — optional per-agent configuration keys.
- [ ] Install dependencies from `hf_space/requirements.txt`.

## 2. Router Planning UI (`hf_space/app.py`)
- [x] Router Planner tab for plan generation, validation, and simulated execution.
- [x] Benchmark tab to upload predictions and run `router_benchmark_runner.py`.
- [x] Docs/TODO tab listing outstanding work and agent load status.
- [ ] Replace stub agents with real implementations (see `Milestone-6/math-agent`, `code-agent`, `general-agent`).
- [ ] Log inference latency / token usage when live model is connected.

## 3. Agent Integration
- [ ] Implement `MathAgent`, `CodeAgent`, `GeneralSearchAgent` by extending `AgentHandler` (`Milestone-6/agents/base.py`).
- [ ] Update `hf_space/app.py` if additional tools are introduced (ensure regex & registry cover them).
- [ ] Add smoke tests to verify agent responses and router-agent round trips.

## 4. Benchmark Automation
- [x] Milestone 5 benchmark assets bundled locally (`benchmarks/router_benchmark_hard.jsonl`).
- [x] Thresholds defined in `router_benchmark_thresholds.json`.
- [x] Configure Space or CI to run `router_benchmark_runner.py` on every deployment (fail on threshold breach). *(Set `ROUTER_BENCHMARK_PREDICTIONS` for automatic startup validation.)*

## 5. Documentation
- [x] Deployment overview in `Milestone-6/router-agent/README.md`.
- [ ] Finalise public-facing docs (Space README, project report, screenshots).
- [ ] Describe rollback & monitoring strategy (latency alerts, benchmark regressions).
- [x] Publish zero-GPU Space: https://huggingface.co/spaces/CourseGPT-Pro-DSAI-Lab-Group-6/router-control-room (configure secrets before going live).

## 6. Optional Enhancements
- [ ] Add persistent storage or caching for benchmark reports and agent artifacts.
- [ ] Support batch routing or API endpoints beyond the UI.
- [ ] Integrate OCR service (`Milestone-6/ocr-service`) if text extraction is required before routing.
