---
base_model: Qwen/Qwen3-32B
language:
- en
- zh
library_name: peft
license: apache-2.0
tags:
- lora
- peft
- router-agent
- reasoning
- vertex-ai
datasets:
- CourseGPT-Pro-DSAI-Lab-Group-6/router-dataset
metrics:
- name: eval_loss
  type: loss
  value: 0.6277
- name: eval_perplexity
  type: perplexity
  value: 1.8733
---

# Router Qwen3 32B PEFT Adapter

LoRA adapter for the dense **Qwen/Qwen3-32B** model, tuned to produce reasoning-rich router plans. Qwen3’s native thinking mode (`<think>...</think>`) aligns perfectly with our dataset’s `thinking_outline` and `route_rationale` fields.

## Model Details
- **Base model:** [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
- **Adapter:** QLoRA rank 16, trained via Vertex AI managed OSS tuning on 8× H100 GPUs
- **Context length:** 32K native (extendable to 131K via YaRN)
- **Metrics:** validation loss ≈ 0.6277, perplexity ≈ 1.87
- **Dataset:** CourseGPT router corpus with `<think>` tags added for complex examples

## Intended Use
Use this adapter for advanced orchestration where detailed reasoning and planning matter—for example, long research workflows, multi-stage analytics, or multilingual tool routing.

### Modes
- **Thinking:** Add `/think` (or instruct the model to reason) to obtain `<think>...</think>` blocks before the JSON plan.
- **Fast path:** Use `/no_think` to skip reasoning and respond quickly with concise JSON.

## Quick Start
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base = "Qwen/Qwen3-32B"
adapter = "CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft"

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter)

prompt = """System: Use <think> for deep reasoning, then emit JSON with route_plan, route_rationale, thinking_outline.
User: /think Plan a workflow that benchmarks four LLM agents, runs ablation code, and writes a literature summary."""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=1500,
    temperature=0.6,
    top_p=0.95,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training & Evaluation
- Vertex AI managed OSS fine-tuning (3 epochs)
- QLoRA with 4-bit base weights, LoRA layers covering attention + MLP modules
- Validation metrics: loss ≈ 0.6277, perplexity ≈ 1.87

## Deployment Tips
- For Vertex + vLLM inference, pass `--enable-reasoning` and `--reasoning-parser deepseek_r1` to honor `<think>` blocks.
- Quantize to 4- or 8-bit for L4 deployments; full precision requires ≥80 GB VRAM.
- Strip `<think>` content from conversation history before the next user turn to avoid context bloat.

## Citation
```
@software{CourseGPTRouterQwen3,
  title  = {Router Qwen3 32B PEFT Adapter},
  author = {CourseGPT Pro DSAI Lab Group 6},
  year   = {2025},
  url    = {https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft}
}
```
