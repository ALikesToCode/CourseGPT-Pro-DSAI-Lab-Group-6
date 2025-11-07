---
title: Router Control Room (ZeroGPU)
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: ZeroGPU UI for CourseGPT-Pro router checkpoints
---

# 🛰️ Router Control Room — ZeroGPU

This Space exposes the CourseGPT-Pro router checkpoints (Gemma3 27B + Qwen3 32B) with an opinionated Gradio UI. It runs entirely on ZeroGPU hardware using 8-bit loading so you can validate router JSON plans without paying for dedicated GPUs.

## ✨ What’s Included

- **Router-specific prompt builder** – inject difficulty, tags, context, acceptance criteria, and additional guidance into the canonical router system prompt.
- **Two curated checkpoints** – `Router-Qwen3-32B-8bit` and `Router-Gemma3-27B-8bit`, both merged and quantized for ZeroGPU.
- **JSON extraction + validation** – output is parsed automatically and checked for the required router fields (route_plan, todo_list, metrics, etc.).
- **Raw output + prompt debug** – inspect the verbatim generation and the exact prompt string sent to the checkpoint.
- **One-click clear** – reset the UI between experiments without reloading models.

## 🔄 Workflow

1. Describe the user task / homework prompt in the main textbox.
2. Optionally provide context, acceptance criteria, and extra guidance.
3. Choose the difficulty tier, tags, model, and decoding parameters.
4. Click **Generate Router Plan**.
5. Review:
   - **Raw Model Output** – plain text returned by the LLM.
   - **Parsed Router Plan** – JSON tree extracted from the output.
   - **Validation Panel** – confirms whether all required fields are present.
   - **Full Prompt** – copy/paste for repro or benchmarking.

If JSON parsing fails, the validation panel will surface the error so you can tweak decoding parameters or the prompt.

## 🧠 Supported Models

| Name | Base | Notes |
|------|------|-------|
| `Router-Qwen3-32B-8bit` | Qwen3 32B | Best overall acceptance on CourseGPT-Pro benchmarks. |
| `Router-Gemma3-27B-8bit` | Gemma3 27B | Slightly smaller, tends to favour math-first plans. |

Both checkpoints are merged + quantized in the `Alovestocode` namespace and require `HF_TOKEN` with read access.

## ⚙️ Local Development

```bash
cd Milestone-6/router-agent/zero-gpu-space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx
python app.py
```

## 📝 Notes

- The app always attempts 8-bit loading first (bitsandbytes). If that fails, it falls back to bf16/fp16/fp32.
- The UI enforces single-turn router generations; conversation history and web search are intentionally omitted to match the Milestone 6 deliverable.
- If you need to re-enable web search or more checkpoints, extend `MODELS` and adjust the prompt builder accordingly.
- **Benchmarking:** run `python Milestone-6/router-agent/tests/run_router_space_benchmark.py --space Alovestocode/ZeroGPU-LLM-Inference --limit 32` (requires `pip install gradio_client`) to call the Space, dump predictions, and evaluate against the Milestone 5 hard suite + thresholds.
- Set `ROUTER_PREFETCH_MODEL` (single value) or `ROUTER_PREFETCH_MODELS=Router-Qwen3-32B-8bit,Router-Gemma3-27B-8bit` (comma-separated, `ALL` for every checkpoint) to warm-load weights during startup. Disable background warming by setting `ROUTER_WARM_REMAINING=0`.
