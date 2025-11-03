# Router Agent Milestone 6: Deployment & Documentation *(14 Nov)*

## Objectives
- **Deploy** the router agent (and eventually math/code/general specialists) to a public Hugging Face Space with a repeatable benchmark harness.
- **Document** the end-to-end workflow so reviewers can reproduce model loading, evaluation, and guardrails.
- **Finalize** the project report with clear deployment instructions, evaluation tables, and follow-up work.

## Current Scaffolding
- `hf_space/app.py` — Gradio Blocks skeleton that:
  - Calls the router model via Hugging Face Inference (environment-driven) or falls back to a sample JSON plan.
  - Validates responses with lightweight structural checks.
  - Exposes a “Benchmark” tab that runs the Milestone 5 hard benchmark (`router_benchmark_runner.py`) against uploaded predictions.
- `hf_space/requirements.txt` — Minimal dependencies for the Space (`gradio`, `huggingface_hub`, `orjson`).
- `hf_space/space_config.json` — Placeholder Space metadata (title, emoji, SDK). Update once the deployment target is confirmed.
- `../agents/base.py` — Shared `AgentRequest`/`AgentResult` dataclasses and `AgentHandler` protocol.
- `../math-agent`, `../code-agent`, `../general-agent` — Stub handlers + READMEs where team members can plug in specialised logic.

The scaffolding imports the Milestone 5 evaluation utilities (`schema_score`, `router_benchmark_runner`) so we can keep a single source of truth for metrics and thresholds.

## Deployment Roadmap
1. **Router model wiring**
   - Point `HF_ROUTER_REPO` (env var) at the chosen adapter (e.g., Gemma 3 PEFT).
   - Provide a lightweight system prompt and JSON extraction utility for reliable outputs.
   - Record inference latency and cost assumptions.
2. **Agent plug-ins**
   - Extend `hf_space/app.py` with stubs for `/math`, `/code`, `/general-search` so live demos can showcase full orchestration.
   - Decide whether those agents rely on direct LLM calls or pre-computed artifacts.
3. **Benchmark automation**
   - Ship the hard benchmark JSONL to the Space storage bucket or bundle a smaller smoke subset for demo runs.
   - Expose threshold pass/fail status in the UI; optionally allow downloading the JSON report.
4. **Documentation**
   - Capture Space setup commands, environment variables, and resource requirements.
   - Update the main project README and final report with deployment URLs, evaluation screenshots, and known limitations.

## Next Actions
- [ ] Finalise router model selection (Gemma 3 vs Qwen 3) for deployment.
- [ ] Implement inference client in `hf_space/app.py` (Hugging Face `InferenceClient` or Vertex endpoint).
- [ ] Integrate schema-aware validation + benchmark trigger into the Gradio UI.
- [ ] Draft deployment notes covering Space configuration, CI triggers, and rollback strategy.
- [ ] Extend the documentation template so math/code/general agents can drop in routes, assets, and citations.
