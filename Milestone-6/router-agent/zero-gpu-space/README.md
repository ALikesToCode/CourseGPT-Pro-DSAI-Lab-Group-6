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
| `app.py` | Loads the merged checkpoint on demand, exposes a `/v1/generate` API, and ships an interactive Gradio UI for manual testing. |
| `requirements.txt` | Minimal dependency set (transformers, bitsandbytes, torch, gradio, fastapi). |
| `.huggingface/spaces.yml` | Configures the Space for ZeroGPU hardware and disables automatic sleep. |

## Deployment Steps

1. **Merge and upload the router adapter**
   ```python
   from peft import PeftModel
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   import torch

   BASE = "Qwen/Qwen3-32B"
   ADAPTER = "CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft"

   quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)

   tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
   base = AutoModelForCausalLM.from_pretrained(
       BASE,
       quantization_config=quant_cfg,
       device_map="auto",
       trust_remote_code=True,
   )

   merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
   save_dir = "router-qwen3-32b-4bit"
   merged.save_pretrained(save_dir)
   tok.save_pretrained(save_dir)
   ```
   Upload `router-qwen3-32b-4bit/` to a new model repo (e.g. `Alovestocode/router-qwen3-32b-4bit`).

2. **Create the Space**
   ```bash
   huggingface-cli repo create router-router-zero \
     --type space --sdk gradio --hardware zerogpu --yes
   ```

3. **Publish the code**
   ```bash
   cd Milestone-6/router-agent/zero-gpu-space
   huggingface-cli upload . Alovestocode/router-router-zero --repo-type space
   ```

4. **Configure secrets**
   - `MODEL_REPO` – defaults to `Alovestocode/router-qwen3-32b-4bit`
   - `HF_TOKEN` – token with read access to the merged model

5. **Connect the main router UI**
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
