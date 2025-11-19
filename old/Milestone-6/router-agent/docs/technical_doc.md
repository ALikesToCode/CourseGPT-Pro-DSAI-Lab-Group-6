# Comprehensive Technical Documentation — Router Agent Deployment

## 1. Environment Setup
### Python & Tooling
- **Python**: 3.11 (tested with `uv` and `venv`).
- **CLI dependencies**: `huggingface_hub`, `gradio`, `orjson`, `google-generativeai`, `fastapi`, `transformers`, `bitsandbytes`, `torch>=2.1`. Install via:

```bash
# Router UI space
pip install -r Milestone-6/router-agent/hf_space/requirements.txt

# ZeroGPU backend
pip install -r Milestone-6/router-agent/zero-gpu-space/requirements.txt

# Evaluation utilities
pip install -r Milestone-5/router-agent/requirements.txt  # optional if you want schema_score locally
```

- **Optional GPU wheels** (when running outside Hugging Face):
  - `--extra-index-url https://download.pytorch.org/whl/cu118 torch`
  - `-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jax[cuda11_pip] jaxlib`

### Environment Variables
| Key | Purpose | Applies To |
| --- | --- | --- |
| `HF_ROUTER_REPO` | Default router backend selection (base or adapter repo). | Gradio Space.
| `HF_ROUTER_API` | Direct URL to ZeroGPU `/v1/generate` endpoint; overrides InferenceClient. | Gradio Space & CLI tests.
| `HF_TOKEN` | Hugging Face token with read access to private models + Space deploy rights. | All components.
| `ROUTER_BENCHMARK_PREDICTIONS` | Path to JSONL predictions automatically validated on startup. | Gradio Space.
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Enables Gemini 2.5 Pro fallback for agent stubs. | Gradio Space.
| `MODEL_REPO` / `MODEL_FALLBACKS` | Preferred checkpoint(s) for ZeroGPU backend. | FastAPI service.
| `MODEL_LOAD_STRATEGY` / `MODEL_LOAD_STRATEGIES` / `LOAD_IN_8BIT` | Quantisation order (`8bit→4bit→bf16→fp16→cpu` default). | FastAPI service.
| `SKIP_WARM_START`, `ALLOW_WARM_START_FAILURE` | Tune ZeroGPU start-up behaviour. | FastAPI service.
| `MATH_AGENT_MODEL`, `CODE_AGENT_MODEL`, `GENERAL_AGENT_CONFIG` | References for specialist plugins. | Gradio Space.

Store sensitive values with `huggingface-cli secrets set --repo <SPACE> KEY VALUE` before pushing.

## 2. Data Pipeline
- **Source corpus**: `Milestone-2/router-agent-scripts/output.jsonl` (Gemini-generated routing traces, 8,189 rows).
- **Preparation script**: `Milestone-3/router-agent-scripts/prepare_vertex_tuning_dataset.py`.

```bash
python Milestone-3/router-agent-scripts/prepare_vertex_tuning_dataset.py \
  --input Milestone-2/router-agent-scripts/output.jsonl \
  --output-dir Milestone-3/router-agent-scripts/data/vertex_tuning \
  --val-ratio 0.10 --test-ratio 0.05 --seed 42 \
  --gcs-prefix gs://router-data-542496349667/router-dataset
```

- **Resulting splits**: train 6,962 · validation 818 · test 409 JSONL files under `data/vertex_tuning/` and mirrored in GCS.
- **Schema**: each record contains a deterministic `prompt` plus `completion` JSON carrying `route_plan`, `route_rationale`, `expected_artifacts`, `thinking_outline`, `handoff_plan`, `todo_list`, `difficulty`, `tags`, `acceptance_criteria`, `metrics`.
- **Dataset license**: MIT (`https://huggingface.co/datasets/Alovestocode/Router-agent-data`). Include citation in downstream reports.
- **Quality checks**: deterministic shuffling, JSON validation, prompt-token histogram (avg 126 tokens), tool frequency stats ( `/general-search` dominates first-tool position at 98.8% ).

## 3. Model Architecture
- **Router prompt**: mirrors the systems diagram in `assets/image1.png` and enforces strict JSON output with a schema-aware system message (`SYSTEM_PROMPT` in both apps).
- **Base models**: `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-3-27b-pt`, `Qwen/Qwen3-32B`. Hosted either via Hugging Face Inference Endpoints or the ZeroGPU FastAPI service.
- **Adapters**: Vertex PEFT/LoRA rank-16 adapters targeting attention + MLP projection layers. Adapter checkpoints live under `CourseGPT-Pro-DSAI-Lab-Group-6/router-*-peft`.
- **Specialist agents**: share the `AgentHandler` Protocol (`Milestone-6/agents/base.py`). The Gradio app dynamically loads `/math`, `/code`, `/general-search` handlers from `Milestone-6/{math,code,general}-agent/handler.py` and falls back to Gemini when no handler is present.
- **Quantisation**: ZeroGPU backend attempts `8bit` bitsandbytes loading first, then `4bit`, `bf16`, `fp16`, and CPU. Strategy is configurable via env vars.

## 4. Training Summary (Milestone 4)
| Model | Epochs | Adapter Rank | LR Multiplier | Eval Loss | Perplexity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B Adapter | 3 | 16 | 0.7× Vertex default | 0.676 | 1.97 | BLEU 0.40; longer generations (length ratio 1.18).
| Gemma 3 27B Adapter | 3 | 16 | 0.7× | **0.608** | **1.84** | Fastest throughput (53 samples/s) with stable JSON.
| Qwen 3 32B Adapter | 3 | 16 | 0.7× | 0.628 | 1.87 | Balanced latency vs. accuracy; `<think>`-friendly.

All jobs launched via `Milestone-3/router-agent-scripts/launch_vertex_tuning.py` on Vertex AI (`us-central1`, `preview_train`). LoRA dropout/weight decay kept at 0.0 to avoid latency regressions.

## 5. Evaluation Summary (Milestone 5)
- **Metrics script**: `Milestone-5/router-agent/collect_router_metrics.py` aggregates trainer exports + dataset diagnostics. Outputs `router_eval_metrics.json` (per-model stats + dataset histograms).
- **Schema scoring**: `Milestone-5/router-agent/schema_score.py` compares predicted vs. gold JSON (route order, TODO coverage, metrics retention, length ratios).
- **Regression runner**: `router_benchmark_runner.py` enforces min/max thresholds from `router_benchmark_thresholds.json` (e.g., JSON validity ≥ 0.98, route tool precision ≥ 0.90, length_ratio>1.25 ≤ 10%).
- **Benchmarks**: `Milestone-5/router-agent/benchmarks/router_benchmark_hard.jsonl` (322 adversarial samples) ships with the Gradio Space for on-demand checks.
- **Key findings**: Gemma 3 adapter best overall; canonical `/general-search→/math→/code` routes dominate 93% of data, so math-first accuracy requires focused evaluation.

## 6. Inference Pipeline
1. User enters a prompt inside the Gradio “Router Control Room”. The UI builds a full system+user message and selects a backend (sample plan, Hugging Face InferenceClient, or ZeroGPU API).
2. `_generate_router_plan()` calls either `InferenceClient.text_generation` or the REST API, forcing `max_new_tokens=600`, `temperature=0.2`, `top_p=0.9` by default.
3. Raw text is sanitised via `extract_json_from_text()` and validated against required fields. Failures fall back to the bundled sample plan with a warning banner.
4. The validated plan is rendered in expandable sections and optionally replayed through the agent stubs (each implementing `AgentHandler.invoke`).
5. When a `router_predictions.jsonl` file is supplied via the Benchmark tab, the app triggers `router_benchmark_runner.evaluate_thresholds()` and surfaces pass/fail badges.
6. The ZeroGPU FastAPI backend exposes `POST /v1/generate` with the following contract:

```python
payload = {
    "prompt": full_prompt,
    "max_new_tokens": 600,
    "temperature": 0.2,
    "top_p": 0.9,
}
resp = requests.post(f"{HF_ROUTER_API}/v1/generate", json=payload, timeout=120)
plan = resp.json()["text"]
```

## 7. Deployment Details
### Hugging Face Spaces (UI)
```bash
cd Milestone-6/router-agent/hf_space
huggingface-cli login  # or set HF_TOKEN
huggingface-cli repo create router-control-room --type space --sdk gradio --org CourseGPT-Pro-DSAI-Lab-Group-6 --yes
huggingface-cli upload . CourseGPT-Pro-DSAI-Lab-Group-6/router-control-room --repo-type space
```
- Set env vars in the Space settings (`HF_ROUTER_REPO`, `HF_ROUTER_API`, `HF_TOKEN`, `GOOGLE_API_KEY`, optional `ROUTER_BENCHMARK_PREDICTIONS`).
- Hardware: CPU Basic works for demo; upgrade to A10G for live model calls if Inference Endpoints throttle.

### ZeroGPU Backend
```bash
cd Milestone-6/router-agent/zero-gpu-space
huggingface-cli repo create router-router-zero --type space --sdk gradio --hardware zerogpu --yes
huggingface-cli upload . Alovestocode/router-router-zero --repo-type space
```
- Provide `HF_TOKEN`, `MODEL_REPO` (or rely on bundled fallbacks), and quantisation env vars.
- Healthcheck: `GET https://<space>.hf.space/` returns `{"status":"ok","model":"...","strategy":"8bit"}`.

### Local / CI Smoke Tests
```bash
HF_TOKEN=hf_xxx python Milestone-6/router-agent/test_router_models.py \
  --models "Gemma 3 27B Router Adapter" "Custom Router API"
```
This script reuses the UI inference code and flags malformed JSON before deployment.

## 8. System Design Considerations
- **Modularity**: Router UI, ZeroGPU backend, and agent handlers live in separate directories to keep deployment permutations clean.
- **Fallback strategy**: Sample plan + Gemini fallback reduce demo downtime when inference endpoints throttle or adapters misbehave.
- **Resource usage**: ZeroGPU loads models lazily (when `SKIP_WARM_START=1`) and logs the active quantisation strategy for debugging.
- **Benchmark gating**: Embedding Milestone 5 assets ensures every deployment can re-run schema-aware checks without leaving the UI.
- **Extensibility**: `AgentHandler` protocol + registry allow new tools (e.g., `/ocr`) to be wired in by just dropping a handler file under `Milestone-6/<tool>-agent/`.

## 9. Error Handling & Monitoring
- JSON extraction raises descriptive errors (“Router output did not contain a JSON object”), surfaced both in the UI and CLI.
- Failed backend calls fall back to the sample plan and log the exception in the “Debug” accordion.
- `router_benchmark_runner` enforces min/max thresholds and returns a machine-readable report for CI.
- ZeroGPU app exposes `/health` status JSON and `/console` mini-console for manual spot checks (root `/` redirects to the Gradio UI at `/gradio`).
- Agent load issues are aggregated in `AGENT_STATUS_MARKDOWN` so missing handlers are obvious to operators.

## 10. Reproducibility Checklist
- [ ] Record Python version + requirement hashes (e.g., `pip freeze > Milestone-6/router-agent/docs/requirements-<date>.txt`).
- [ ] Keep dataset splits under `Milestone-3/router-agent-scripts/data/vertex_tuning/` and note the `--seed` used (42).
- [ ] Store Vertex training commands per model (`launch_vertex_tuning.py --display-name ...`) in `Milestone-3/router-agent-scripts/logs/`.
- [ ] Version Control router checkpoints on Hugging Face (`router-*-peft`, `router-*-merged`).
- [ ] Capture `router_eval_metrics.json`, benchmark stats, and schema-score reports alongside each deployment tag.
- [ ] Fill out `docs/licenses.md` whenever new datasets/models are added.
- [ ] Test via `test_router_models.py` + Benchmark tab before announcing a new Space URL.
