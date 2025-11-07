# Router Agent Milestone 6: Deployment & Documentation *(14 Nov)*

## Objectives
- **Deploy** the router agent (and eventually math/code/general specialists) to a public Hugging Face Space with a repeatable benchmark harness.
- **Document** the end-to-end workflow so reviewers can reproduce model loading, evaluation, and guardrails.
- **Finalize** the project report with clear deployment instructions, evaluation tables, and follow-up work.

## Current Scaffolding
- `hf_space/app.py` — Gradio Blocks scaffold that:
  - Calls the router model via Hugging Face Inference (environment-driven) or falls back to a sample JSON plan.
  - Validates responses with lightweight structural checks.
  - Simulates agent execution with pluggable `/math`, `/code`, `/general-search` handlers (try → fallback to Gemini 2.5 Pro when configured).
  - Exposes a “Benchmark” tab that both auto-runs startup checks (when `ROUTER_BENCHMARK_PREDICTIONS` is set) and evaluates uploaded predictions via `router_benchmark_runner.py`.
- `hf_space/requirements.txt` — Minimal dependencies for the Space (`gradio`, `huggingface_hub`, `orjson`).
- `hf_space/space_config.json` — Placeholder Space metadata (title, emoji, SDK). Update once the deployment target is confirmed.
- `../agents/base.py` — Shared `AgentRequest`/`AgentResult` dataclasses and `AgentHandler` protocol.
- `../math-agent`, `../code-agent`, `../general-agent` — Stub handlers, a math-agent template, and READMEs where team members can plug in specialised logic.

The scaffolding imports the Milestone 5 evaluation utilities (`schema_score`, `router_benchmark_runner`) so we can keep a single source of truth for metrics and thresholds.

## Milestone 6 deliverables cheat sheet
- `docs/overview.md` — Problem statement, architecture snapshot, and component table routed to this milestone.
- `docs/technical_doc.md` — Environment setup, data/model/training/eval summaries, inference pipeline, deployment procedures, and reproducibility checklist.
- `docs/user_guide.md` — Non-technical walkthrough for the Gradio UI plus troubleshooting tips.
- `docs/api_doc.md` — REST contracts for the ZeroGPU backend and Gradio `/run/predict` endpoint with sample cURL calls.
- `docs/licenses.md` — Code/data/model licensing obligations and pending actions before going public.
- `docs/final_project_report_outline.md` — PDF chapter plan that stitches milestones 1–6 together for submission.
- `tests/run_router_space_benchmark.py` — CLI harness that calls the ZeroGPU Space via `gradio_client`, writes prediction JSONL files, and runs the Milestone 5 schema-score + threshold checks.

Keep these Markdown files updated whenever you change router checkpoints, benchmarks, or deployment knobs so reviewers always have the latest instructions.

## Environment Variables
- `HF_ROUTER_REPO` / `HF_TOKEN` — Hugging Face inference endpoint for the router adapter or full base model (optional; falls back to sample plan).
  - The Space UI now exposes a selector with the following built-in choices:
    - `CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft`
    - `CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft`
    - `CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft`
    - `meta-llama/Llama-3.1-8B`
    - `google/gemma-3-27b-pt`
    - `Qwen/Qwen3-32B`
    - Adapter options automatically mount their corresponding base checkpoints when calling the Inference API.
  - Provide `HF_ROUTER_REPO` if you want a default selection pre-populated when the Space boots.
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) / `GEMINI_MODEL` — Enable Gemini 2.5 Pro fallback for agent failures (`google-generativeai`).
- `ROUTER_BENCHMARK_PREDICTIONS` — Path to a JSONL file of predictions that should be validated automatically on startup.
- `MATH_AGENT_MODEL`, `CODE_AGENT_MODEL`, `GENERAL_AGENT_CONFIG` — Suggested knobs for specialised agents.

### Smoke-test router endpoints locally
- Use `Milestone-6/router-agent/test_router_models.py` to verify that each configured base/adapter pair responds with valid router JSON before deploying (install `python-dotenv` if you want `.env` auto-loading):
  ```bash
  HF_TOKEN=hf_xxx python Milestone-6/router-agent/test_router_models.py --models "Gemma 3 27B Router Adapter"
  ```
  The script reuses the Space inference logic (text generation → conversational fallback) and surfaces errors such as missing tokens or malformed JSON.
- To avoid hosted inference limits, deploy the merged checkpoint to the ZeroGPU Space scaffold under `zero-gpu-space/` and set `HF_ROUTER_API` to the resulting `/v1/generate` endpoint. The current merged models live at (Space falls back to Qwen → Gemma by default):
  - `Alovestocode/router-qwen3-32b-merged`
  - `Alovestocode/router-llama31-merged`
  - `Alovestocode/router-gemma3-merged`
- To benchmark a deployed Space against the Milestone 5 hard suite, run:
  ```bash
  pip install gradio_client  # once
  python Milestone-6/router-agent/tests/run_router_space_benchmark.py \
    --space Alovestocode/ZeroGPU-LLM-Inference \
    --model Router-Qwen3-32B-8bit \
    --limit 64 \
    --concurrency 4
  ```
  The script streams router plans via the public API, emits `router_space_predictions.jsonl`, and writes a threshold report under `Milestone-6/router-agent/tests/`.
  Use `--concurrency` to parallelise calls (each worker spins up its own `gradio_client` instance).
- Warm-loading tips: set `ROUTER_PREFETCH_MODELS=ALL` in the ZeroGPU Space settings to download both checkpoints up front, and leave `ROUTER_WARM_REMAINING=1` (default) to continue loading unused checkpoints in the background after the first request.

### CI/CD
- A GitHub Actions workflow (`.github/workflows/deploy-router-spaces.yml`) runs on every push touching `Milestone-6/router-agent/**`. It performs a syntax check and, when configured, publishes both Spaces automatically via `huggingface-cli upload`.
- Required repository settings:
  - Secrets:
    - `HF_TOKEN` – Hugging Face access token with write permission on the target Spaces.
  - Variables (or additional secrets):
    - `HF_SPACE_MAIN` – slug of the router UI Space (e.g. `Alovestocode/router-control-room-private`).
    - `HF_SPACE_ZERO` – slug of the ZeroGPU backend Space (e.g. `Alovestocode/router-router-zero`).
- If any of the above are missing, the workflow skips the corresponding deployment step and exits successfully.

### GPU / ZeroGPU Setup
- Install optional CUDA-ready frameworks when upgrading hardware:
  ```text
  --extra-index-url https://download.pytorch.org/whl/cu118
  torch
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  jax[cuda11_pip]
  jaxlib
  tensorflow
  ```
- ZeroGPU requires a PRO subscription on personal accounts. After switching the Space hardware to **ZeroGPU**, import `spaces` and decorate GPU-bound functions with `@spaces.GPU` (already available via `spaces>=0.3.0` in `requirements.txt`). Example:
  ```python
  import spaces

  @spaces.GPU
  def heavy_route(prompt: str):
      ...
  ```
- The backend Space now honours `MODEL_FALLBACKS` (comma-separated list) and `MODEL_LOAD_STRATEGY` (`8bit`, `4bit`, or `fp16`). Leave the defaults in place for ZeroGPU to prioritise Gemma/Llama checkpoints and load them in 8-bit quantisation.
- Provide `MODEL_LOAD_STRATEGIES` (comma-separated) if you need a custom quantisation fallback order; otherwise the loader attempts `8bit → 4bit → bf16 → fp16 → cpu` automatically.
- Deployment env knobs: `SKIP_WARM_START=1` defers loading until the first request, while
  `ALLOW_WARM_START_FAILURE=1` prevents crash loops if ZeroGPU evicts the warm-up task.
- Verify GPU visibility from the Space logs as needed:
  ```python
  import torch, jax, tensorflow as tf
  print("CUDA available:", torch.cuda.is_available())
  print("JAX devices:", jax.devices())
  print(tf.config.list_physical_devices('GPU'))
  ```
- Remember billing starts once upgraded hardware is running. Adjust the Space's sleep time or manually pause it to control costs.

### Deploy via Hugging Face CLI
```bash
huggingface-cli login  # or set HF_TOKEN
huggingface-cli repo create router-control-room \
  --type space --space_sdk gradio \
  --organization CourseGPT-Pro-DSAI-Lab-Group-6 --yes
cd Milestone-6/router-agent/hf_space
huggingface-cli upload . CourseGPT-Pro-DSAI-Lab-Group-6/router-control-room --repo-type space
```
Set the environment variables above in the Space settings dashboard (or via `huggingface-cli secrets set --repo CourseGPT-Pro-DSAI-Lab-Group-6/router-control-room <NAME> <VALUE>`) before going live.  
To disable devcontainers/OpenVSCode on Spaces (avoids build errors), include `.huggingface/spaces.yml` with:

```yaml
sdk: gradio
python_version: 3.10
app_file: app.py
devcontainers: false
```

**Live instance (private account, CPU Basic):** https://huggingface.co/spaces/Alovestocode/router-control-room-private

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
- [x] Implement inference client in `hf_space/app.py` (Hugging Face `InferenceClient` or Vertex endpoint).
- [x] Integrate schema-aware validation + benchmark trigger into the Gradio UI.
- [ ] Draft deployment notes covering Space configuration, CI triggers, and rollback strategy.
- [ ] Extend the documentation template so math/code/general agents can drop in routes, assets, and citations.
