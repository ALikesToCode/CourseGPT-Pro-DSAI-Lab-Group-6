---
title: ZeroGPU Router Backend
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.5.0"
app_file: app.py
pinned: false
---

# ZeroGPU Router Backend Space

This directory contains a lightweight Hugging Face Space that serves a merged
router checkpoint over a simple REST API. Deploy it to ZeroGPU and then point
the main router UI (`Milestone-6/router-agent/hf_space/app.py`) at the `/v1/generate`
endpoint via the `HF_ROUTER_API` environment variable.

## Contents

| File | Purpose |
| ---- | ------- |
| `app.py` | Loads the merged checkpoint on demand (tries `MODEL_REPO` first, then `MODEL_FALLBACKS` or the default Gemma → Llama → Qwen order), exposes a `/v1/generate` API, mounts the Gradio UI at `/gradio`, and keeps a lightweight HTML console at `/console`. |
| `requirements.txt` | Minimal dependency set (transformers, bitsandbytes, torch, fastapi, accelerate, sentencepiece, spaces, uvicorn). |
| `.huggingface/spaces.yml` | Configures the Space for ZeroGPU hardware and disables automatic sleep. |

## Deployment Steps

1. **Create the Space**
   ```bash
   huggingface-cli repo create router-router-zero \
     --type space --sdk gradio --hardware zerogpu --yes
   ```

2. **Publish the code**
   ```bash
   cd Milestone-6/router-agent/zero-gpu-space
   huggingface-cli upload . Alovestocode/router-router-zero --repo-type space
   ```

3. **Configure secrets & variables**
   - `HF_TOKEN` – token with read access to the merged checkpoint(s)
   - `MODEL_REPO` – optional hard pin if you only want a single model considered
   - `MODEL_FALLBACKS` – comma-separated preference order (defaults to `router-gemma3-merged,router-llama31-merged,router-qwen3-32b-merged`)
   - `MODEL_LOAD_STRATEGY` – `8bit` (default), `4bit`, or `fp16`; backwards-compatible with `LOAD_IN_8BIT` / `LOAD_IN_4BIT`
   - `MODEL_LOAD_STRATEGIES` – optional ordered fallback list (e.g. `8bit,4bit,cpu`). The loader will automatically walk this list and finally fall back to `8bit→4bit→bf16→fp16→cpu`.
   - `SKIP_WARM_START` – set to `1` if you prefer to load lazily on the first request
   - `ALLOW_WARM_START_FAILURE` – set to `1` to keep the container alive even if warm-up fails (the next request will retry)

4. **Connect the main router UI**
   ```bash
   export HF_ROUTER_API=https://Alovestocode-router-router-zero.hf.space/v1/generate
   ```

## API Contract

`POST /v1/generate`

```json
{
  "prompt": "<router prompt>",
  "max_new_tokens": 600,
  "temperature": 0.2,
  "top_p": 0.9
}
```

Response:
```json
{ "text": "<raw router output>" }
```

Use `HF_ROUTER_API` in the main application or the smoke-test script to validate
that the deployed model returns the expected JSON plan. When running on ZeroGPU
we recommend keeping `MODEL_LOAD_STRATEGY=8bit` (or `LOAD_IN_8BIT=1`) so the
weights fit comfortably in the 70GB slice; if that fails the app automatically
degrades through 4-bit, bf16/fp16, and finally CPU mode. You can inspect the
active load mode via the `/health` endpoint (`strategy` field). The root path
(`/`) now redirects to the Gradio UI, while `/console` serves the minimal HTML
form for quick manual testing.
