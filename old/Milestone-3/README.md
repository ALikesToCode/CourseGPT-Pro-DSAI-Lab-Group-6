# Milestone 3 – Final Submission Report

CourseGPT-Pro · DSAI Lab · Group 6 · 17 Oct 2025

This milestone delivers three specialised agents—**Math**, **Router**, and **Code**—together with the scripts, datasets, and adapters required to reproduce our results. The deliverables satisfy the rubric items by documenting:

* the **architecture** of each chosen model (with pros/cons);
* the **data preprocessing** required to make inputs model-ready; and
* the **end‑to‑end pipelines** (smoke tests → full training → evaluation) that verify every component.

Repository layout:

```
Milestone-3/
├── math-agent-scripts/
├── router-agent-scripts/
└── code-agent-scripts/
```

Each subfolder contains a detailed README plus scripts/notebooks referenced below.

---

## 1. Math Agent (`math-agent-scripts`)

> Full architecture write-up: [`math-agent-scripts/README.md`](math-agent-scripts/README.md)

### 1.1 Model Architecture

| Item | Details |
| --- | --- |
| Base model | `google/gemma-3-4b-it` (4B dense transformer, instruction-tuned) |
| Fine-tuning | LoRA rank 16, α=32 on all attention + MLP projections |
| Precision | Mixed BF16/FP16 with gradient checkpointing |
| Sequence length | 2048 tokens |

**Advantages**

* Excellent instruction adherence—critical for educational step-by-step answers.
* Fits within 12–16 GB VRAM when quantized (QLoRA-ready), enabling lab GPUs.
* Apache-compatible Gemma terms permit both academic and commercial deployments.

**Disadvantages & Mitigation**

* Smaller than 7B+ LLMs → mitigated by high-quality MathX‑5M dataset and higher LoRA rank.
* Newer ecosystem vs. Llama → compensated by Google documentation + active community.

### 1.2 Data Preprocessing

* Dataset: `XenArcAI/MathX-5M` (≈4.32 M problems with step-by-step solutions).
* Notebook: `math_agent_architecture_gemma_3_4b.ipynb`.
* Pipeline:
  1. Stream dataset, inspect per-topic splits (algebra, calculus, geometry, etc.).
  2. Normalize Markdown → plain text while preserving LaTeX tokens.
  3. Convert to Gemma chat template (system/user/assistant, `<think>` blocks for reasoning).
  4. Enforce max length 2048 by truncation or segmented chaining.

### 1.3 End-to-End Training Pipeline

| Phase | Description |
| --- | --- |
| Smoke | 1 k stratified samples to verify LoRA config, optimizer, checkpoint save. |
| Full | Train on entire MathX split (loss 2.1 → 0.37). |
| Evaluate | Manual spot checks + curated evaluation set (accuracy **90.4%**). |

Artifacts: `gemma3-4b-math-lora-adapter/final_adapter/` (LoRA weights) and optional merged model checkpoint.

---

## 2. Router Agent (`router-agent-scripts`)

> Full details: [`router-agent-scripts/README.md`](router-agent-scripts/README.md)

### 2.1 Model Architectures

| Adapter | Base model | Context | LoRA rank | Hosted model card |
| --- | --- | --- | --- | --- |
| Router Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 128K | 16 | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft) |
| Router Gemma 3 27B | `google/gemma-3-27b-it` | 128K | 16 | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft) |
| Router Qwen 3 32B | `Qwen/Qwen3-32B` | 32K native / 131K with YaRN | 16 | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft) |

**Advantages & Disadvantages**

| Model | Advantages | Disadvantages |
| --- | --- | --- |
| Llama 3.1 8B | Cost-efficient; deploys on L4; BLEU 0.40; fast inference | Lower capacity on very hard routing tasks |
| Gemma 3 27B | Lowest val loss (**0.608**); strong multilingual guardrails | Needs A100/H100 (or quantization) for deployment |
| Qwen 3 32B | Native `<think>` reasoning & tool-calling; best for complex orchestration | Highest inference cost; best on ≥80 GB VRAM |

### 2.2 Data Preprocessing

* Source dataset: `Milestone-2/router-agent-scripts/output.jsonl`.
* Script: `prepare_vertex_tuning_dataset.py`.
* Output artifacts:
  - Local: `data/vertex_tuning/train.jsonl`, `validation.jsonl`, `test.jsonl`.
  - Cloud: `gs://router-data-542496349667/router-dataset/{train,validation,test}.jsonl`.
* Each completion is strict JSON containing `route_plan`, `route_rationale`, `thinking_outline`, `handoff_plan`, `todo_list`, `metrics`, etc.
* Quality gate: deterministic shuffling, schema validation, optional `--limit` for smoke slices.

### 2.3 Tuning & Deployment Pipeline

1. **Smoke slice:** run tuning on 200 examples (`--limit 200`) to confirm IAM permissions, dataset schema, and artifact export.
2. **Full fine-tunes:** execute Vertex AI jobs via `launch_vertex_tuning.py`. Example:
   ```bash
   python launch_vertex_tuning.py \
     --project $PROJECT \
     --location us-central1 \
     --base-model qwen/qwen3@qwen3-32b \
     --train-uri gs://router-data-542496349667/router-dataset/train.jsonl \
     --validation-uri gs://router-data-542496349667/router-dataset/validation.jsonl \
     --output-uri gs://router-data-542496349667/router-tuning/qwen3-32b-peft \
     --tuning-mode PEFT_ADAPTER \
     --adapter-size 16 \
     --epochs 3 \
     --display-name router-qwen3-32b-peft \
     --wait
   ```
3. **Metrics:** aggregated in `all_results.json` / `trainer_state.json` for each job (BLEU, loss, perplexity, eval runtime).
4. **Deployment:** use `model_garden.CustomModel` to deploy adapters; enable reasoning flags for Qwen (`--enable-reasoning --reasoning-parser deepseek_r1`).

### 2.4 Evaluation Strategy

* Automated checks: JSON schema validation, tool recall/precision, metadata consistency.
* Human review: 10 prompts per adapter, multi-agent QA audit.
* Qwen thinking-mode audit: ensure `<think>` appears for complex prompts and is stripped from conversation history.

---

## 3. Code Agent (`code-agent-scripts`)

> In-depth report: [`code-agent-scripts/README.md`](code-agent-scripts/README.md)

### 3.1 Model Architecture

| Item | Details |
| --- | --- |
| Base model | `Qwen/Qwen3-0.6B` |
| Fine-tuning | QLoRA rank 16, α=32 (attention + MLP projections) |
| Quantization | 4-bit NF4 with double quantization |
| Attention | SDPA with flash enabled (packing) |
| Trainer | Hugging Face Transformers + TRL `SFTTrainer` |
| Output | `code-agent-scripts/qwen_code_lora_adapter/final_adapter/` |

**Advantages**

* Qwen3 family excels on code tasks and supports chat templates out of the box.
* 0.6B parameters keep both training and inference lightweight (runs on RTX 3080/4080).
* QLoRA + BitsAndBytes drastically reduce memory usage without a noticeable quality drop.

**Disadvantages**

* Smaller capacity vs. 7B+ code LLMs → we plan to revisit larger models once GPU budget increases.
* Flash attention + packing issues can trigger warnings; smoke tests showed no contamination, but we monitor logs closely.

> Attempted baseline: `meta-llama/Meta-Llama-3.1-8B-Instruct` (training paused due to VRAM/time limits; pipeline supports resumption).

### 3.2 Data Preprocessing

* Dataset: `OpenCoder-LLM/opc-sft-stage2`, configuration `educational_instruct`.
* Records: 119 parquet shards → ~92 M tokens after formatting.
* Pipeline:
  1. Map each instruction/output to the Llama 3 chat template with fixed system prompt “You are an expert programming assistant.”
  2. Remove original columns, retain a single `text` field (parallelized with `num_proc=50`).
  3. Export smoke subset (2 k rows) for sanity checks, then write full `llama31_code_finetune_simple.jsonl`.

### 3.3 Training Pipeline

| Phase | Duration / Result |
| --- | --- |
| Smoke | ~12 min (2 k samples) – verifies formatting, tokenizer, LoRA config |
| Full | ~3 h 45 m on RTX 4080 – training loss 2.70 → ~0.40 |
| Save | Adapter + tokenizer saved to `final_adapter/`; console logs capture progress |

### 3.4 Evaluation & Usage

* Inference snippet provided in sub-README (load adapter via `PeftModel`).
* Current evaluation: dataset-provided unit tests + manual inspection of 50 prompts.
* Planned: integrate CodeBLEU + automated execution harness.

### 3.5 Publishing (optional)

Upload adapter to Hugging Face with:

```bash
huggingface-cli login
hf upload CourseGPT-Pro-DSAI-Lab-Group-6/code-qwen3-0.6b-peft \
  code-agent-scripts/qwen_code_lora_adapter/final_adapter \
  --commit-message "Upload code adapter"
```

---

## 4. Reproduction Checklist

```bash
# 1. Enter project root
cd CourseGPT-Pro-DSAI-Lab-Group-6

# 2. Create virtualenv (router/code agents)
cd Milestone-3/router-agent-scripts
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -r requirements.txt

# 3. Prepare router datasets
python prepare_vertex_tuning_dataset.py \
  --input ../../Milestone-2/router-agent-scripts/output.jsonl \
  --output-dir data/vertex_tuning \
  --gcs-prefix gs://router-data-542496349667/router-dataset

# 4. Launch Vertex tuning jobs (run once per model)
python launch_vertex_tuning.py ...

# 5. Fine-tune code agent (see code-agent README)

# 6. Upload adapters to Hugging Face (router + code)
hf upload CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft ...
```

For the math agent, follow the instructions in `math-agent-scripts/README.md`—ensure Gemma access and MathX dataset availability.

---

## 5. Deliverables

| Agent | Key Files / Links |
| --- | --- |
| Math | `math-agent-scripts/README.md`, `math_agent_architecture_gemma_3_4b.ipynb`, `gemma3-4b-math-lora-adapter/final_adapter/` |
| Router | `router-agent-scripts/README.md`, dataset scripts, Vertex commands, HF adapters (three links above) |
| Code | `code-agent-scripts/README.md`, preprocessing script snippet, `qwen_code_lora_adapter/final_adapter/` |

All agents have completed smoke runs and full training runs, validating the end-to-end pipeline requirement.

---

## 6. Next Steps

1. Publish the code adapter + model card on Hugging Face.
2. Integrate automated evaluation harnesses (JSON schema validator, CodeBLEU, regression notebooks).
3. Wire all three agents into the router for real-time orchestration with citation verifier.

