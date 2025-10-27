# Router Agent Milestone 4: Model Training \[October 31\]

This milestone documents the end-to-end training effort for the **router agent**. The focus was to stand up reliable supervised fine-tuning jobs, explore hyperparameters that balance cost with routing quality, and publish production-ready adapters together with reproducibility assets.

## Objectives

- **Train initial router adapters** across the short-listed foundation models (Llama 3.1 8B, Gemma 3 27B, Qwen 3 32B).
- **Experiment with hyperparameters and optimizers** to converge quickly while preserving JSON-structured outputs.
- **Apply regularization techniques** (LoRA rank sweeps, weight decay, dropout) to control overfitting on synthetic routing data.
- **Capture evaluation evidence** (Vertex metrics, schema validation, manual audits) needed for downstream deployment.

## Overview of Router Agent Training Components

### Router Agent Model Training (`router-agent-scripts/`)

**Purpose**  
Fine-tune multi-tool routing behavior so the orchestrator can dispatch user requests to specialised agents with clear rationales, ordering, and acceptance criteria.

**Training Corpus**  
- Source: `Milestone-2/router-agent-scripts/output.jsonl` (Gemini-generated routing traces).  
- Composition: 8,189 labelled interactions spanning 14 routing archetypes (single-tool, dual-tool, escalation, general-search fallback, Refine+Code loops). Median completion length is 512 tokens; longest sample is capped at 1,920 tokens post-template.  
- Conversion: `prepare_vertex_tuning_dataset.py` reshuffled and split the corpus into train/validation/test sets with deterministic seeds (train 6,962 · validation 818 · test 409).  
- Storage: versions mirrored locally under `Milestone-3/router-agent-scripts/data/vertex_tuning/` and in GCS (`gs://router-data-542496349667/router-dataset/`).  
- Quality gates: per-line JSON validation, tool frequency histograms, and stratified difficulty sampling logged to `data/vertex_tuning/summary.json`.

**Environment & Dependencies**  
- Vertex AI Preview Supervised Fine-Tuning (SFT) endpoints.  
- Python 3.11 virtualenv with dependencies pinned in `router-agent-scripts/requirements.txt`.  
- Auth prerequisites: `gcloud auth login`, `gcloud config set project`, and Vertex tuning API enablement.

**Key Scripts**  
- `prepare_vertex_tuning_dataset.py`: dataset curation & optional GCS upload.  
- `launch_vertex_tuning.py`: launches and monitors tuning jobs for LoRA or full fine-tuning.

## Training Workflow

1. **Dataset verification** – Run `prepare_vertex_tuning_dataset.py --limit 200 --seed 42` to validate JSON schema, shuffling, and cloud uploads.  
2. **Smoke tuning runs** – Submit PEFT jobs over the 200-example slice to confirm IAM permissions, checkpoint export paths, and Vertex SDK behaviour.  
3. **Hyperparameter sweeps** – Iterate on LoRA rank (`adapter_size` {8, 16}), learning rate multipliers ({0.5, 0.7, 1.0}), weight decay (0 vs 0.05), and dropout (0 vs 0.1) using the smoke split. Tracking occurred in `Milestone-3/router-agent-scripts/logs/<timestamp>_sweeps.md`.  
4. **Full-scale training** – Promote the best settings to full runs on the entire dataset for each base model (see Training Runs table).  
5. **Evaluation & sign-off** – Collect Vertex metrics, run JSON schema validation on test predictions, and complete manual spot checks before publishing adapters.

## Hyperparameter & Regularization Experiments

| Experiment | Scope | Outcome |
| --- | --- | --- |
| **LoRA rank sweep** (`adapter_size`) | Compared ranks 8 vs 16 on Llama and Gemma using smoke split | Rank 16 reduced validation loss by 6–9% and improved tool recall; adopted for full runs. |
| **Learning rate multiplier** | Tested {0.5, 0.7, 1.0} of Vertex default | Multiplier 0.7 stabilised Qwen training (avoided JSON drift) while 1.0 caused minor overfitting. |
| **Weight decay & dropout** | Applied weight decay 0.05 and dropout 0.1 on LoRA adapters | Provided marginal loss gains (<1%) but increased latency; reverted to defaults for production. |
| **Optimizer mode** | Evaluated fused AdamW vs default Vertex AdamW | No significant difference; kept Vertex-managed AdamW with cosine schedule. |

All experiments were executed with deterministic seeds (`--seed 42`) so the resulting Vertex job metrics can be reproduced.

## Training Configuration Details

| Setting | Llama 3.1 8B | Gemma 3 27B | Qwen 3 32B |
| --- | --- | --- | --- |
| Vertex location | `us-central1` | `us-central1` | `us-central1` |
| Tuning interface | `vertexai.preview.tuning.sft.preview_train` | Same | Same |
| Adapter rank | 16 (LoRA, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`) | Same | Same |
| Learning-rate multiplier | 0.7 | 0.7 | 0.7 |
| Warmup ratio | 0.1 (Vertex default) | 0.1 | 0.1 |
| Dropout | 0.0 | 0.0 | 0.0 (dropout=0.1 trial logged but reverted) |
| Weight decay | 0.0 | 0.0 | 0.0 |
| Batch size | 128 tokens per microbatch, gradient accumulation handled by Vertex | Same | Same |
| Max steps | 3 epochs over 6,962 training rows | Same | Same |
| Checkpoint export | `postprocess/node-0/checkpoints/{intermediate,final}` | Same | Same |
| Monitoring | Vertex tensorboard summary + custom log stream to `logs/<job-id>.txt` | Same | Same |

During sweeps, alternate ranks (8) and higher learning rates (1.0) were recorded with `_trial` suffixes in the job display name. Jobs that exhibited JSON format drift were stopped early via `gcloud ai tuning-jobs cancel` and annotated in the logbook.

## Training Runs Summary

| Vertex Job ID | Base model | Tuning mode | Adapter size | Output URI | Status | Validation metrics |
| --- | --- | --- | --- | --- | --- | --- |
| `projects/542496349667/locations/us-central1/tuningJobs/1491991597619871744` | `meta/llama3_1@llama-3.1-8b-instruct` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/llama31-peft` | `JOB_STATE_SUCCEEDED` (2025‑10‑17 10:42:40 → 13:41:27 UTC) | BLEU ≈ 0.4004 · loss ≈ 0.6758 · perplexity ≈ 1.97 |
| `projects/542496349667/locations/us-central1/tuningJobs/1108622679339958272` | `publishers/google/models/gemma3@gemma-3-27b-it` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/gemma3-peft` | `JOB_STATE_SUCCEEDED` (2025‑10‑17 10:43:33 → 14:58:00 UTC) | loss ≈ 0.6080 · perplexity ≈ 1.84 |
| `projects/542496349667/locations/us-central1/tuningJobs/2183294140421242880` | `qwen/qwen3@qwen3-32b` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/qwen3-32b-peft` | `JOB_STATE_SUCCEEDED` (2025‑10‑17 10:42:40 → 14:23:06 UTC) | loss ≈ 0.6277 · perplexity ≈ 1.87 |

Artifacts for each job include training logs (`/executor/node-0/stdout`), evaluation CSV exports, and LoRA checkpoints located under `<output-uri>/postprocess/node-0/checkpoints/final/`.

## Evaluation Results

| Evaluation slice | Llama 3.1 8B adapter | Gemma 3 27B adapter | Qwen 3 32B adapter |
| --- | --- | --- | --- |
| JSON validity (409 test rows) | 100% | 100% | 100% |
| Tool recall | 84.3% | **91.8%** | 90.7% |
| Tool precision | 86.1% | 89.4% | **92.6%** |
| Exact route plan match | 57.9% | 61.3% | **64.2%** |
| Rationale completeness (manual score\*) | 3.8 / 5 | **4.4 / 5** | 4.3 / 5 |
| Avg. latency @ g2-standard-12 | **1.4 s** | 2.6 s | 3.3 s |

\*Rationale completeness rubric: coverage of tool justification, handoff details, and acceptance criteria (scored by two reviewers; disagreements resolved synchronously).

- **Schema compliance** – 100% of sampled predictions passed `jq`-based JSON validation; Qwen runs preserved `<think>` tags in 73% of complex prompts.  
- **Regression checks** – Comparison against Milestone-2 baseline routers showed a +14.6 percentage point gain in exact plan match and elimination of previous failure mode where `/general-search` was omitted under noisy inputs.  
- **Manual audits** – 30 random prompts per adapter reviewed; no regressions in rationale clarity or metadata fields. Edge cases (e.g., ambiguous difficulty labelling) were flagged in `reports/2025-10-17-audit.md` for future dataset augmentation.  
- **Performance considerations** – Gemma and Qwen adapters introduce higher latency but deliver best precision on multi-tool scenarios; Llama remains production baseline for cost-sensitive flows and fallback deployment.

## Deliverables & Reproducibility Assets

- **Adapters & checkpoints** – Published on Hugging Face:  
  - `router-llama31-peft` · `router-gemma3-peft` · `router-qwen3-32b-peft`  
- **Dataset snapshots** – `gs://router-data-542496349667/router-dataset/{train,validation,test}.jsonl` plus local mirrors under `router-agent-scripts/data/vertex_tuning/`.  
- **Execution logs** – Vertex console exports stored in `router-agent-scripts/logs/2025-10-17-*`.  
- **Evaluation reports** – Metrics JSON files under `router-agent-scripts/reports/` (timestamped).  
- **Deployment guidance** – `launch_vertex_tuning.py` README section documents endpoint creation and adapter download commands.

## Reproduction Checklist

1. Create a virtual environment inside `Milestone-3/router-agent-scripts/` and install requirements.  
2. Regenerate (or verify) the dataset using `prepare_vertex_tuning_dataset.py` with the appropriate `--gcs-prefix`.  
3. Launch a smoke tuning job (`--limit 200`) to confirm IAM + checkpointing.  
4. Submit full tuning jobs using the hyperparameters listed in the Training Runs table.  
5. After completion, download adapters, validate JSON outputs on the test split, and publish to the model registry of choice.

## Next Steps

- Expand curriculum with harder multi-agent conversations (e.g., triage + code + math chains) to push Gemma/Qwen adapters further and improve rationale completeness.  
- Automate regression evaluation (tool recall, plan exact match, schema validation) in CI to guard future changes; integrate Vertex job polling into nightly workflows.  
- Investigate distillation from Gemma/Qwen into a lighter router for cost-sensitive endpoints while preserving reasoning quality; evaluate teacher-student transfer using KL-regularised fine-tuning.  
- Add guardrail prompts that penalise hallucinated tools, informed by the misroutes documented in the audit report.

## References

- Vertex AI Tuning preview documentation  
- Meta Llama 3.1, Google Gemma 3, and Alibaba Qwen3 model cards  
- CourseGPT-Pro router agent pipelines (`Milestone-3/router-agent-scripts/`)
