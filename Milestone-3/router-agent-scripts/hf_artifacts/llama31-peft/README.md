---
base_model: meta-llama/Llama-3.1-8B-Instruct
language:
- en
library_name: peft
license: llama3.1
tags:
- lora
- peft
- router-agent
- vertex-ai
datasets:
- CourseGPT-Pro-DSAI-Lab-Group-6/router-dataset
metrics:
- name: eval_bleu
  type: bleu
  value: 0.4004
- name: eval_perplexity
  type: perplexity
  value: 1.9722
---

# Router Llama 3.1 8B PEFT Adapter

This repository holds the LoRA adapter that fine-tunes **meta-llama/Llama-3.1-8B-Instruct** into a router agent. The model reads a natural-language request, selects the right specialists (`/math`, `/code`, `/general-search`, …), and emits strict JSON describing the plan.

## Model Details
- **Base model:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Adapter:** LoRA rank 16, alpha 32 (attention + MLP projections)
- **Training:** 3 epochs on ~6.9k router samples via Vertex AI managed tuning (QLoRA on NVIDIA L4)
- **Context length:** 128K tokens
- **Validation metrics:** BLEU ≈ 0.4004, Perplexity ≈ 1.97, Loss ≈ 0.6758

## Intended Use
Choose this adapter for cost-efficient routing workloads where latency matters. It is ideal for production orchestration of domain specialists in math, coding, or retrieval pipelines.

## Quick Start
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)

prompt = (
    "System: Emit strict JSON with route_plan, route_rationale, thinking_outline, handoff_plan.\n"
    "User: Design a plan mixing symbolic derivation, Python simulation, and literature search."
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Data
- CourseGPT router dataset (Milestone 2), converted to Vertex supervised JSONL (prompt/completion pairs)

## Evaluation Summary
- Held-out validation set (~10%)
- BLEU ≈ 0.4004, Perplexity ≈ 1.97

## Citation
```
@software{CourseGPTRouterLlama31,
  title  = {Router Llama 3.1 8B PEFT Adapter},
  author = {CourseGPT Pro DSAI Lab Group 6},
  year   = {2025},
  url    = {https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft}
}
```
