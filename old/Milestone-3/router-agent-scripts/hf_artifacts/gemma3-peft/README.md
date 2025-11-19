---
base_model: google/gemma-3-27b-it
language:
- en
library_name: peft
license: gemma
tags:
- lora
- peft
- router-agent
- vertex-ai
datasets:
- CourseGPT-Pro-DSAI-Lab-Group-6/router-dataset
metrics:
- name: eval_loss
  type: loss
  value: 0.6080
- name: eval_perplexity
  type: perplexity
  value: 1.8368
---

# Router Gemma 3 27B PEFT Adapter

This repository hosts the LoRA adapter for **google/gemma-3-27b-it**, tuned as a tool-routing brain with strong reasoning headroom. The model reads user queries, decides which agents (e.g., `/math`, `/code`, `/general-search`) should run, and emits strict JSON aligned with our Milestone 2 router schema.

## Model Details
- **Base model:** [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)
- **Adapter type:** QLoRA rank 16 on attention + MLP projections (Vertex AI managed tuning)
- **Training:** 3 epochs, micro-batch size 2, cosine LR with warmup, gradient checkpointing enabled
- **Hardware:** NVIDIA H100/A3 (Vertex managed OSS tuning)
- **Context length:** 128K tokens
- **Validation metrics:** loss ≈ 0.6080, perplexity ≈ 1.84, eval runtime ≈ 15.4 s

Gemma’s larger capacity gives higher-quality routing decisions, especially for multi-step orchestration across math/code/search specialists.

## Intended Use
Use this adapter when you need premium routing quality and can afford deploying on higher-memory GPUs (L4 with quantization or A100/H100 in full precision). It is well-suited for research copilots, analytics assistants, and multilingual routing scenarios.

### Out-of-scope
- Direct Q/A without tool execution
- High-risk/sensitive domains without additional alignment checks

## Quick Start
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base = "google/gemma-3-27b-it"
adapter = "CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft"

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter)

prompt = (
    "System: Emit strict JSON with route_plan, route_rationale, thinking_outline, and handoff_plan.\n"
    "User: Draft a pipeline combining symbolic integration, Python experimentation, and literature review."
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Data
- CourseGPT router dataset (Milestone 2), converted to Vertex supervised JSONL (prompt/completion pairs)
- Structured completions include route plan arrays, rationale, acceptance criteria, metrics, and more

## Evaluation Summary
- Held-out validation subset (≈10%)
- Metrics: loss ≈ 0.6080, perplexity ≈ 1.84
- Qualitative review shows consistent JSON structure and accurate tool choices on complex problems

## Deployment Tips
- Quantize adapters for L4 deployment (bitsandbytes 4-bit)
- Validate JSON outputs and retry when necessary
- Extend the prompt with custom tool definitions if your stack differs from `/math`, `/code`, `/general-search`

## Citation
```
@software{CourseGPTRouterGemma3,
  title  = {Router Gemma 3 27B PEFT Adapter},
  author = {CourseGPT Pro DSAI Lab Group 6},
  year   = {2025},
  url    = {https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft}
}
```
