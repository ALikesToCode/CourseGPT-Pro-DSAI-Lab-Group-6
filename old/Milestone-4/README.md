# Milestone 4 — Agent Training & Tuning (Router, Math, Code)

This milestone collects the end-to-end training, tuning, and evaluation artifacts for the three agent families developed in this project:

- **Router Agent** — supervised PEFT adapters that emit structured routing plans and tool calls
- **Math Agent** — Vertex AI supervised tuning of math reasoning adapters using a MathX subset
- **Code Agent** — LoRA fine-tuning of Llama 3.1 for code generation (OpenCoder-style dataset)

This single README centralises the reproduction steps, datasets, tested hyperparameters, evaluation benchmarks, and published artifacts for reviewers and future engineers.

## Repository layout (relevant folders)

- `Milestone-4/router-agent/` — router tuning scripts, dataset preparation, logs
- `Milestone-4/math-agent/` — data prep, Vertex tuning scripts, adapters, README
- `Milestone-4/code-agent/` — code fine-tuning notebook, LoRA adapter output
- `Milestone-4/math-agent/adapters/` — produced adapters (local snapshot of artifacts)

## Common prerequisites (applies to all agents)

- Python 3.8+ (recommended 3.10/3.11) and a virtual environment
- `gcloud` CLI with Application Default Credentials and project configured for Vertex jobs
- `huggingface_hub` and `git` + `git-lfs` for model artifact publishing
- Sufficient quota on Vertex AI and GCS for training jobs and artifact storage

### Quick environment setup (Bash / WSL)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # see each agent folder for agent-specific deps
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud config set compute/region us-central1
```

### PowerShell (Windows)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
& gcloud auth login
& gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud config set compute/region us-central1
```

If you need the full, agent-specific dependencies, see the README in each agent folder (`router-agent/README.md`, `math-agent/README.md`, `code-agent/README.md`).

---

## Router Agent — overview & reproduction

### Purpose

The Router Agent is a chat-oriented LLM that outputs strict JSON route plans, rationale, and handoff instructions. It orchestrates downstream tool usage (math, code, search, etc.).

### Dataset

- Source: `Milestone-2/router-agent-scripts/output.jsonl` (8,189 labeled routing traces)
- Preprocessing: `Milestone-3/router-agent-scripts/prepare_vertex_tuning_dataset.py` — deterministic shuffling, schema validation, optional `<think>` wrapping, GCS upload.
- Splits: Train 6,962 · Validation 818 · Test 409 (0.85/0.10/0.05)

### Models & adapters

- Base models tested: Llama 3.1 8B, Gemma 3 27B, Qwen 3 32B
- Adapter strategy: Vertex AI PEFT/LoRA (rank 16) injected into attention and MLP projections

### Training pipeline & key commands

1. Prepare data (smoke / full):

```bash
python Milestone-3/router-agent-scripts/prepare_vertex_tuning_dataset.py --limit 200 --seed 42 --gcs-prefix gs://router-data-.../router-dataset
```

2. Launch Vertex tuning (example):

```bash
python Milestone-3/router-agent-scripts/launch_vertex_tuning.py \
  --base-model "meta/llama3_1@llama-3.1-8b-instruct" \
  --train-uri "gs://router-data/.../train.jsonl" \
  --validation-uri "gs://router-data/.../validation.jsonl" \
  --output-uri "gs://router-data/.../adapters/llama31" \
  --adapter-size 16 --epochs 3 --learning-rate-multiplier 0.7
```

### Hyperparameters (used in Milestone runs)

- Adapter rank: 16 (LoRA)
- Learning-rate multiplier: 0.7 × Vertex defaults
- Warm-up ratio: 0.1
- Epochs: 3

### Representative results

Vertex job IDs and outcomes are recorded in `Milestone-3/router-agent-scripts/logs/` and summarised in the router README (example job IDs and metrics included there).

### Artifacts & where to find them

- Published HF repos (examples): `CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft`, `.../router-gemma3-peft`, `.../router-qwen3-32b-peft` (see `router-agent/README.md`)
- GCS checkpoints under `router-tuning/<model>/postprocess/node-0/checkpoints/final/`

### Reproduction checklist (router)

1. Create/activate Python venv and install `Milestone-3/router-agent-scripts/requirements.txt`.
2. Run dataset prep with `--limit 200` (smoke) and check JSON schema.
3. Launch smoke tuning job, validate output URIs and logs.
4. Submit full tuning jobs, export adapters, run evaluation on held-out test split.

---

## Math Agent — overview & reproduction

### Purpose

Fine-tune PEFT adapters for math reasoning using a MathX subset and Vertex AI supervised tuning. Goal: improve multi-step reasoning and arithmetic robustness.

### Dataset

- Source: MathX-5M subset processed to ~10k examples (9k train / 1k validation) using `prepare_vertex_tuning.py` in `math-agent/`.
- Output artifacts: `gs://<BUCKET>/mathx-dataset-v1/train.jsonl` and `.../validation.jsonl`.

### Prepare & upload commands

Bash / WSL

```bash
python Milestone-4/math-agent/prepare_vertex_tuning.py --bucket "$BUCKET_NAME" --gcs-prefix "mathx-dataset-v1"
```

PowerShell

```powershell
python Milestone-4/math-agent/prepare_vertex_tuning.py --bucket $env:BUCKET_NAME --gcs-prefix "mathx-dataset-v1"
```

When complete the script prints final GCS URIs to use when launching tuning.

### Launch Vertex tuning (examples)

Bash (Qwen3 32B)

```bash
python Milestone-4/math-agent/launch_vertex_tuning.py \
  --base-model "qwen/qwen3@qwen3-32b" \
  --display-name "math-qwen3-32b-peft" \
  --train-uri "gs://$BUCKET_NAME/mathx-dataset-v1/train.jsonl" \
  --validation-uri "gs://$BUCKET_NAME/mathx-dataset-v1/validation.jsonl" \
  --output-uri "gs://$BUCKET_NAME/adapters/math-qwen3-32b" \
  --epochs 3 --adapter-size 16
```

Other tested models: `google/gemma3@gemma-3-27b-it`, `meta/llama4@llama-4-scout-17b-16e-instruct` (commands in `math-agent/README.md`).

### Evaluation & benchmarks

- Recommended suites: GSM8K, MATH, SVAMP, MAWPS, MathQA, AQuA
- Metrics: exact-match accuracy, step-by-step quality, robustness, perplexity
- Example evaluation snippet lives in `math-agent/README.md`.

### Model comparison (guidance)

- Qwen3-32B: strongest reasoning; higher cost/time
- Gemma3-27B: best trade-off of accuracy and stability in our runs
- Llama4-Scout-17B: good balance for lower VRAM

### Local adapter snapshot

The repository includes a local snapshot under `Milestone-4/math-agent/adapters/` with three adapters:

- `Qwen3-32B/adapter/adapter_model.safetensors`
- `math-gemma-3-27b/adapter/adapter_model.safetensors`
- `llama4-17b/adapter/adapter_model.safetensors`

### Reproduction checklist (math)

1. Set `BUCKET_NAME` and run `prepare_vertex_tuning.py` to produce GCS JSONL URIs.
2. Verify GCS URIs and IAM permissions for Vertex service account.
3. Launch tuning jobs using `launch_vertex_tuning.py` examples and monitor in Vertex console.
4. Download adapters and run benchmark evaluations using the evaluation snippet.

---

## Code Agent — overview & reproduction

### Purpose

Fine-tune `meta-llama/Meta-Llama-3.1-8B-Instruct` on an OpenCoder-style dataset to improve code generation and instruction-following for coding tasks using LoRA.

### Dataset

- `OpenCoder-LLM/opc-sft-stage2` with `educational_instruct` configuration (examples contain `instruction`, `output`, `code`, `entry_point`, `testcase`).

### Formatting template

The chat template used for Llama-3 fine-tuning is documented in `Milestone-4/code-agent/README.md`. Example template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert programming assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{full_response}<|eot_id|>
```

### Training & LoRA setup

- Load model with quantization (e.g., 4-bit) where applicable
- Configure LoRA targeting projection matrices and gates: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Trainer: `SFTTrainer` (in the `finetunening.ipynb`) — save final adapter to `./llama31_code_lora_adapter/final_adapter`

### Usage

Run the notebook `finetunening.ipynb` in `Milestone-4/code-agent/` to perform data formatting, training, and adapter saving. Ensure required Python packages are installed.

### Artifacts

- Saved LoRA adapter path: `./llama31_code_lora_adapter/final_adapter` (see code-agent README for exact file names)

---

## Adapters snapshot (local)

`Milestone-4/math-agent/adapters/` contains the locally exported adapter snapshots used for evaluation and demo. Each adapter folder contains model/tokenizer metadata and an `adapter/adapter_model.safetensors` weight file and `adapter_config.json`.

Adapters included (from attached workspace):

- `llama4-17b/` — adapter + tokenizer files
- `math-gemma-3-27b/` — adapter + tokenizer files
- `Qwen3-32B/` — adapter + tokenizer files

## Publishing adapters

We recommend publishing adapters to Hugging Face model repos. Use `git lfs` for `.safetensors` files. See `Milestone-4/math-agent/README.md` for CLI examples and the root `scripts/` suggestion for helper upload scripts.

## Common next steps & recommendations

- Add automated evaluation scripts per benchmark (GSM8K, MATH) and store results CSVs in `Milestone-4/*/reports/`.
- Add model cards (README.md in HF model repo) describing base model, training data, LoRA rank, epochs, and evaluation results.
- Automate Vertex job monitoring and export summaries to `Milestone-4/*/logs/`.

## Contact / authors

See repository top-level README and the individual agent READMEs for author contacts and fine-grained instructions.

If you'd like, I can now:

- Generate `scripts/upload-to-hf.ps1` and `scripts/upload-to-hf.sh` that upload the three adapters to Hugging Face (you provide OWNER and public/private choice), or
- Add per-agent `requirements.txt` and a short `run-eval.sh` that runs GSM8K evaluation against a loaded adapter.

---
End of Milestone 4 consolidated README
