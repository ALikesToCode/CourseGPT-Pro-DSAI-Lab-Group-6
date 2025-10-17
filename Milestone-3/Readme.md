# CourseGPT-Pro Milestone-3: Agent Architecture Report

## Table of Contents
1. [Math Agent Overview](#math-agent-overview)
2. [Router Agent Overview](#router-agent-overview)
3. [Code Agent Overview (Coming Soon)](#code-agent-overview-coming-soon)

---

## Math Agent Overview

### Purpose
The Math Agent is designed to solve complex mathematical problems with step-by-step reasoning, providing clear explanations and accurate solutions for educational purposes within CourseGPT-Pro.

### Architecture Summary
- **Base Model:** `google/gemma-3-4b-it` (4B parameter instruction-tuned model)
- **Dataset:** `XenArcAI/MathX-5M` (~4.32M problems with solutions)
- **Fine-Tuning:** LoRA (Low-Rank Adaptation) for parameter-efficient training
- **Optimization:** Mixed precision (BF16/FP16), gradient checkpointing, AdamW optimizer

### Key Design Decisions
- **Model Selection:** Gemma-3-4B chosen for its balance of performance and efficiency, strong instruction-following, and open licensing.
- **LoRA Training:** Enables efficient adaptation by training only ~0.5% of parameters, reducing memory and compute requirements.
- **Dataset Choice:** MathX-5M provides diverse, step-by-step solutions across K-12 to college-level math, formatted for educational clarity.
- **Instruction Format:** Chat-based template with explicit system/user/assistant roles and step-by-step reasoning using `<think>` tags.

### Pipeline Overview
1. **Data Preparation:** Stream MathX-5M, normalize columns, format for chat template.
2. **Model Initialization:** Load Gemma-3-4B-IT, configure LoRA adapters, enable memory optimizations.
3. **Training:** LoRA adaptation on attention and FFN layers, mixed precision, AdamW optimizer.
4. **Persistence:** Save LoRA adapters and tokenizer for deployment.

### Justification Highlights
- **LoRA**: Chosen over full fine-tuning, adapters, and prompt engineering for best efficiency/quality trade-off.
- **MathX-5M**: Largest, most diverse dataset with step-by-step solutions and pedagogical formatting.
- **Memory Optimization**: LoRA, gradient checkpointing, and mixed precision enable training on 16-24GB GPUs.

---

## Router Agent Overview

### Purpose
The Router Agent determines the optimal sequence of downstream agents (e.g., math, code, search) to solve a given user query, outputting a structured plan and rationale for tool selection.

### Architecture Summary
- **Supported Models:** Llama 3.1 8B Instruct, Gemma 3 27B IT, Qwen3 32B Instruct (Vertex AI PEFT or full fine-tuning)
- **Dataset:** Week-2 router dataset, converted to Vertex AI supervised fine-tuning format (prompt/completion JSONL)
- **Tuning:** LoRA/PEFT or full fine-tuning via Vertex AI, with adapter size configurable (default: 16)

### Key Design Decisions
- **Model Selection:**
  - **Llama 3.1 8B:** Cost-efficient, strong baseline, 128K context window
  - **Gemma 3 27B:** Highest accuracy, strong multilingual guardrails
  - **Qwen3 32B:** Best for complex, multi-tool routing with native "thinking" mode
- **Dataset Preparation:** Deterministic shuffling, strict prompt/completion schema, quality checks, and support for smoke/few-shot runs.
- **Evaluation:** Automated JSON validation, tool recall/precision metrics, and human spot checks for plan quality.

### Pipeline Overview
1. **Data Preparation:** Convert and validate router dataset to Vertex AI format, upload to GCS.
2. **Model Tuning:** Launch PEFT/LoRA or full fine-tuning jobs on Vertex AI, track with job IDs and metrics.
3. **Deployment:** Export adapters, deploy endpoints, and run evaluation harness for regression and reporting.

### Justification Highlights
- **Model Choice:** Llama for cost/latency, Gemma for accuracy, Qwen for advanced reasoning.
- **LoRA/PEFT:** Enables efficient tuning and deployment, especially for large models.
- **Evaluation:** Emphasizes both automated and human-in-the-loop validation for robust routing.

---

## Code Agent Overview (Coming Soon)

*This section will be updated with the Code Agent architecture, dataset, training pipeline, and justification once finalized.*

---
