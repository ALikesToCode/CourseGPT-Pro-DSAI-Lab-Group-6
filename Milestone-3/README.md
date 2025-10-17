# Milestone 3 – Final Submission

CourseGPT-Pro · DSAI Lab · Group 6 · 17 Oct 2025

This document consolidates the deliverables for the three agents (Math, Router, Code) and shows how we satisfied the milestone brief:

* document **model architectures** with advantages & disadvantages
* describe **data preprocessing** tailored to each model
* demonstrate **end-to-end pipelines** using smoke subsets before full training

The subsections below mirror the directory layout:

```
Milestone-3/
├── math-agent-scripts/
├── router-agent-scripts/
└── code-agent-scripts/
```

All referenced notebooks, scripts, and artifacts are committed in these folders (see individual READMEs for code excerpts).

---

## 1. Math Agent (`math-agent-scripts`)

### 1.1 Architecture

* **Base model:** `google/gemma-3-4b-it` (4B dense transformer)
* **Fine-tuning method:** LoRA (rank 16, α=32) applied to all attention + MLP projections
* **Precision:** BF16/FP16 mixed precision with gradient checkpointing
* **Sequence length:** 2048 tokens

**Why Gemma‑3‑4B‑IT?**

| Advantage | Explanation |
| --- | --- |
| Instruction-tuned base | Strong adherence to prompting style, ideal for educational Q/A |
| Efficient footprint | Fits within 12–16 GB VRAM when quantized, enabling lab hardware |
| Open license | Apache‑compatible (Gemma terms), suitable for academic/commercial use |

| Disadvantage | Mitigation |
| --- | --- |
| Smaller than 7B+ LLMs | High-quality dataset (MathX‑5M) + LoRA rank 16 |
| Newer ecosystem vs Llama | Google-backed docs + active community compensate |

### 1.2 Data Preprocessing

* Dataset: `XenArcAI/MathX-5M` (≈4.32M problems with worked solutions)
* Notebook: `math_agent_architecture_gemma_3_4b.ipynb`
* Steps:
  1. Load dataset and inspect categories (algebra, calculus, geometry, etc.).
  2. Normalize formatting (Markdown → plain text) while preserving latex-equation tokens.
  3. Create prompt/response pairs matching Gemma instruction format.
  4. Ensure max sequence length 2048 by truncating or splitting long reasoning chains.

### 1.3 End-to-End Pipeline (Smoke + Full)

1. **Smoke run:** fine-tune on 1k samples (random stratified) to validate LoRA config, optimizer, and checkpoint saving.
2. **Full run:** train on entire MathX-5M split; monitor loss drop from 2.1 → 0.37.
3. **Evaluation:** manual inspection + accuracy on curated set (90.4%).

Artifacts saved to `gemma3-4b-math-lora-adapter/final_adapter/` (LoRA weights) and optional merged model.

---

## 2. Router Agent (`router-agent-scripts`)

### 2.1 Architecture

We trained three LoRA adapters over instruction-tuned bases to cover different operating points:

| Adapter | Base model | LoRA rank | Context | Hosted at |
| --- | --- | --- | --- | --- |
| Router Llama 3.1 8B | meta-llama/Llama-3.1-8B-Instruct | 16 | 128K tokens | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft) |
| Router Gemma 3 27B | google/gemma-3-27b-it | 16 | 128K tokens | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft) |
| Router Qwen 3 32B | Qwen/Qwen3-32B | 16 | 32K native / 131K YaRN | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft) |

**Advantages / Disadvantages**

| Model | Advantages | Disadvantages |
| --- | --- | --- |
| Llama 3.1 8B | Cost-efficient, deployable on L4 GPUs, BLEU 0.40 | Lower capacity on hard tasks |
| Gemma 3 27B | Lowest val loss (0.608), strong guardrails | Requires A100/H100 or quantization |
| Qwen 3 32B | Native `<think>` reasoning, best tool-calling | Highest inference cost; 80 GB VRAM preferred |

### 2.2 Data Preprocessing

* Source: `Milestone-2/router-agent-scripts/output.jsonl`
* Script: `prepare_vertex_tuning_dataset.py`
* Outputs:
  - `data/vertex_tuning/{train,validation,test}.jsonl` (local)
  - Optional `gs://router-data-542496349667/router-dataset/` upload
* Each entry provides `prompt` (system prompt + user query) and strict JSON `completion` with keys like: `route_plan`, `route_rationale`, `thinking_outline`, `handoff_plan`, `todo_list`, `metrics`, etc.

### 2.3 Training Pipeline (Vertex AI)

1. **Smoke slice:** run tuning on 200 examples (`--limit 200 --gcs-prefix …/smoke`) to confirm IAM, dataset schema, and artifact export.
2. **Full jobs:** submit Vertex AI fine-tuning (PEFT) for each base model using `launch_vertex_tuning.py`. Example command:
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
3. **Metrics:** captured in `all_results.json` / `trainer_state.json` (BLEU, loss, perplexity, runtime).
4. **Deployment:** `model_garden.CustomModel` + Vertex endpoints; for Qwen enable reasoning flags (`--enable-reasoning`).

### 2.4 Evaluation Plan

* Automated metrics: JSON validation, tool recall/precision, metadata consistency.
* Manual spot checks: 10 prompts per model.
* Thinking-mode audit for Qwen: ensure `<think>` usage on complex queries.

---

## 3. Code Agent (`code-agent-scripts`)

### 3.1 Architecture

* **Base model:** `Qwen/Qwen3-0.6B`
* **Adapter:** QLoRA (rank 16, α=32) targeting attention + MLP layers
* **Quantization:** 4-bit NF4 (BitsAndBytes) with double quant
* **Attention:** SDPA + flash enabled (packing on)
* **Training framework:** Hugging Face Transformers + TRL `SFTTrainer`

**Advantages:**

* Qwen family is optimized for code & multilingual text.
* 0.6B dense model fits on consumer GPUs (RTX 4080) while still benefiting from PEFT.
* Quantized training drastically reduces memory footprint.

**Disadvantages:**

* Smaller capacity than 7B+ code LLMs → may miss edge cases.
* Flash attention warning when using packing (documented in logs) – monitor for contamination.

### 3.2 Data Preprocessing

* Dataset: `OpenCoder-LLM/opc-sft-stage2`, subset `educational_instruct`
* Steps:
  1. Load dataset (119 parquet shards) and convert each row to Meta’s chat template (system/user/assistant tags).
  2. Collect new `text` column; remove original columns with `dataset.map`.
  3. Save to `llama31_code_finetune_simple.jsonl` (≈92M tokens).

### 3.3 Training Pipeline

1. **Smoke test:** 1 epoch over 2k samples to ensure formatting & LoRA config.
2. **Full training:** 1 pass over all formatted records (loss 2.70 → ~0.40).
3. **Artifacts:** `qwen_code_lora_adapter/final_adapter/` (adapter weights + tokenizer files) plus training logs.

### 3.4 Deployment & Evaluation

* Load adapter with `PeftModel` atop `Qwen/Qwen3-0.6B`.
* Evaluate with curated code prompts (unit tests, code execution). Future work: integrate CodeBLEU & canonical evaluation harness.

---

## 4. Reproduction Checklist

```bash
# 1. Enter project root
cd CourseGPT-Pro-DSAI-Lab-Group-6

# 2. Create virtual environment (for router/code) and install
cd Milestone-3/router-agent-scripts
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -r requirements.txt

# 3. Generate router datasets
python prepare_vertex_tuning_dataset.py \
  --input ../../Milestone-2/router-agent-scripts/output.jsonl \
  --output-dir data/vertex_tuning \
  --gcs-prefix gs://router-data-542496349667/router-dataset

# 4. Launch Vertex tuning (per model)
python launch_vertex_tuning.py ...

# 5. Fine-tune code agent (see code-agent README or script snippet)

# 6. Upload adapters to Hugging Face
hf upload CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft ...
```

For math agent, follow the notebook instructions in `math-agent-scripts/README.md` (requires Gemma access + MathX dataset download).

---

## 5. Deliverables Summary

| Agent | Key Files / Links |
| --- | --- |
| Math | `math-agent-scripts/README.md`, `math_agent_architecture_gemma_3_4b.ipynb`, Gemma math LoRA adapter (local) |
| Router | `router-agent-scripts/README.md`, dataset scripts, Vertex commands, Hugging Face adapters (3 links above) |
| Code | `code-agent-scripts/README.md`, code-prep script snippet, `qwen_code_lora_adapter/final_adapter/` |

End-to-end pipelines (smoke + full) have been executed for each agent, satisfying the submission requirement to verify all components with subset data before scaling.

---

## 6. Next Steps

* Publish code adapter to Hugging Face with accompanying model card.
* Build automated evaluation harness (JSON validation, CodeBLEU, benchmark scripts).
* Integrate agents under a unified router with real-time tool calls & citation verifier.

---

**Team:** CourseGPT-Pro · DSAI Lab · Group 6  
**Contact:** coursegpt-pro@dsai-lab.example  
**Date:** 17 Oct 2025
