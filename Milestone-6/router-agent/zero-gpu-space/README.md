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
| `app.py` | Loads the merged checkpoint on demand (tries `MODEL_REPO` first, then `router-qwen3-32b-merged`, `router-gemma3-merged`), exposes a `/v1/generate` API, and serves a small HTML console at `/gradio`. |
| `requirements.txt` | Minimal dependency set (transformers, bitsandbytes, torch, fastapi, spaces). |
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

3. **Configure secrets**
   - `MODEL_REPO` – optional override; defaults to the fallback list (`router-qwen3-32b-merged`, `router-gemma3-merged`)
   - `HF_TOKEN` – token with read access to the merged model

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
that the deployed model returns the expected JSON plan.
