# Router Agent Milestone 4: Model Training [October 31]

## Overview / Objective
- Consolidate the router agent training work so reviewers can see an end-to-end picture without cross-referencing earlier milestones.
- Document the supervised fine-tuning runs (and guardrails) that were executed on 17 Oct during the tail end of Milestone 3; Milestone 4 formalises the results, adds additional evaluation context, and packages the artifacts for reuse.
- Highlight the trade-offs between the three shortlisted base models (Llama 3.1 8B, Gemma 3 27B, Qwen 3 32B) so we can justify the production choice in later milestones.
- Capture lessons learned plus the backlog that feeds directly into Milestone 5 error analysis and regression automation.

## Dataset Details
- **Source:** `Milestone-2/router-agent-scripts/output.jsonl`, a Gemini-generated corpus of 8,189 labelled routing traces across 14 archetypes (single-tool, dual-tool, escalation, etc.).
- **Pre-processing:** `Milestone-3/router-agent-scripts/prepare_vertex_tuning_dataset.py` performs deterministic shuffling (`--seed 42`), schema validation, LaTeX-safe escaping, optional `<think>` wrapping, and GCS uploads.
- **Splits:** Train 6,962 · Validation 818 · Test 409 (`0.85 / 0.10 / 0.05`); smoke subsets use `--limit 200` for quick checks.
- **Storage:** Local mirrors under `Milestone-3/router-agent-scripts/data/vertex_tuning/` and cloud copies in `gs://router-data-542496349667/router-dataset/{train,validation,test}.jsonl`.
- **Quality checks:** Length histograms, tool frequency tallies, and JSON schema validation written to `data/vertex_tuning/summary.json`. Manual spot-checks confirmed coverage of tricky archetypes (triage → code, math → verify, etc.).
- **Public snapshot:** Hugging Face dataset card + files live at `https://huggingface.co/datasets/Alovestocode/Router-agent-data` (includes README, version history, and checksum metadata).

## Model Architecture
- **Router design:** The router is a chat-oriented LLM prompted to emit strict JSON plans (`route_plan`, `route_rationale`, `handoff_plan`, `acceptance_criteria`, metadata). Prompt templates mirror the orchestration diagram in `assets/image1.png` so the router understands downstream agent capabilities.
- **Base models:**
  - `meta/llama3_1@llama-3.1-8b-instruct` – cost-effective default, 8192-token context, strong tool-use priors.
  - `publishers/google/models/gemma3@gemma-3-27b-it` – stronger reasoning with native tool-calling format.
  - `qwen/qwen3@qwen3-32b` – dense thinking mode with `<think>` support for deeper planning.
- **Adapter strategy:** Vertex AI PEFT/LoRA adapters injected into attention + MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`) with rank 16. LoRA scaling stays at Vertex defaults; dropout is disabled after experiments increased latency without accuracy gains.

## Training Setup
- **Environment:** Python 3.11 virtualenv (`Milestone-3/router-agent-scripts/requirements.txt`), Vertex AI tuning preview (region `us-central1`), authenticated with `gcloud auth login` + `gcloud config set project`.
- **Data verification workflow:**
  1. `prepare_vertex_tuning_dataset.py --limit 200 --seed 42` → smoke split, schema validation, optional GCS upload.
  2. `python -m json.tool` spot-check + manual inspection of 10 samples.
  3. Upload full splits once smoke checks pass.
- **Training pipeline:**
  1. Launch smoke PEFT jobs to ensure IAM, output URI, and JSON fidelity (cancelled if JSON drift detected).
  2. Run hyperparameter sweeps (LoRA rank, LR multiplier, weight decay, dropout) on the 200-example subset.
  3. Promote best config to full dataset runs for each base model using `launch_vertex_tuning.py --wait` for reproducibility.
  4. Export checkpoints and evaluation CSVs from Vertex console; archive logs under `Milestone-3/router-agent-scripts/logs/`.
- **Key configuration:**

| Setting | Llama 3.1 8B | Gemma 3 27B | Qwen 3 32B |
| --- | --- | --- | --- |
| Vertex location | `us-central1` | `us-central1` | `us-central1` |
| Tuning interface | `vertexai.preview.tuning.sft.preview_train` | Same | Same |
| Adapter rank | 16 (LoRA) | 16 (LoRA) | 16 (LoRA) |
| Learning-rate multiplier | 0.7 × Vertex default | 0.7 × | 0.7 × |
| Warm-up ratio | 0.1 | 0.1 | 0.1 |
| Dropout / weight decay | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| Epochs | 3 (covering 6,962 train rows) | 3 | 3 |
| Monitoring | Vertex loss curves + custom log stream (`logs/<job-id>.txt`) | Same | Same |

## Hyperparameter Experiments
| Experiment | Scope | Outcome |
| --- | --- | --- |
| **LoRA rank sweep** (`adapter_size`) | Rank {8, 16} on Llama & Gemma smoke subsets | Rank 16 cut validation loss 6–9%; adopted for full runs. |
| **Learning-rate multiplier** | Values {0.5, 0.7, 1.0} | 0.7 stabilised JSON outputs; 1.0 caused mild overfitting and format drift. |
| **Weight decay & dropout** | Weight decay 0.05, dropout 0.1 | Marginal loss gains (<1%) but higher latency → reverted to 0.0. |
| **Optimizer comparison** | Vertex fused AdamW vs default | No measurable difference; retained Vertex-managed AdamW + cosine schedule. |

## Regularization & Optimization Techniques
- **Early cancellation:** Jobs showing JSON drift stopped via `gcloud ai tuning-jobs cancel` to avoid wasted budget.
- **Curriculum mixing:** Mini-batches interleave single-tool and multi-tool prompts to prevent mode collapse.
- **Deterministic seeding:** `--seed 42` used across dataset prep and training to make reruns comparable.
- **LoRA-only updates:** Keeps base weights frozen, reducing overfitting on synthetic traces while meeting GPU constraints.

## Initial Training Results
- **Vertex job summary:**

| Vertex Job ID | Base model | Tuning mode | Adapter size | Output URI | Status | Validation metrics |
| --- | --- | --- | --- | --- | --- | --- |
| `projects/542496349667/locations/us-central1/tuningJobs/1491991597619871744` | `meta/llama3_1@llama-3.1-8b-instruct` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/llama31-peft` | `JOB_STATE_SUCCEEDED` (2025‑10‑17 10:42:40 → 13:41:27 UTC) | BLEU ≈ 0.4004 · loss ≈ 0.6758 · perplexity ≈ 1.97 |
| `projects/542496349667/locations/us-central1/tuningJobs/1108622679339958272` | `publishers/google/models/gemma3@gemma-3-27b-it` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/gemma3-peft` | `JOB_STATE_SUCCEEDED` (2025‑10‑17 10:43:33 → 14:58:00 UTC) | loss ≈ 0.6080 · perplexity ≈ 1.84 |
| `projects/542496349667/locations/us-central1/tuningJobs/2183294140421242880` | `qwen/qwen3@qwen3-32b` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/qwen3-32b-peft` | `JOB_STATE_SUCCEEDED` (2025‑10‑17 10:42:40 → 14:23:06 UTC) | loss ≈ 0.6277 · perplexity ≈ 1.87 |

- **Evaluation metrics (test split = 409 rows):**

| Metric | Llama 3.1 8B adapter | Gemma 3 27B adapter | Qwen 3 32B adapter |
| --- | --- | --- | --- |
| JSON validity | 100% | 100% | 100% |
| Tool recall | 84.3% | **91.8%** | 90.7% |
| Tool precision | 86.1% | 89.4% | **92.6%** |
| Exact route plan match | 57.9% | 61.3% | **64.2%** |
| Rationale completeness (manual score¹) | 3.8 / 5 | **4.4 / 5** | 4.3 / 5 |
| Avg latency @ g2-standard-12 | **1.4 s** | 2.6 s | 3.3 s |

¹Two reviewers scored justification coverage; disagreements resolved synchronously.

- **Qualitative findings:** Schema compliance improved by 14.6 percentage points over the Milestone-2 baseline router. Qwen preserves `<think>` traces for 73% of complex prompts, which helps downstream verification. Gemma yields the strongest tool recall but at higher latency.

## Model Artifacts
- **Adapters & checkpoints:** Hugging Face releases [`router-llama31-peft`](https://huggingface.co/Alovestocode/router-llama31-peft), [`router-gemma3-peft`](https://huggingface.co/Alovestocode/router-gemma3-peft), and [`router-qwen3-32b-peft`](https://huggingface.co/Alovestocode/router-qwen3-32b-peft), plus GCS checkpoints under `router-tuning/<model>/postprocess/node-0/checkpoints/final/`.
- **Dataset snapshots:** Train/validation/test JSONL files mirrored locally and in GCS as noted above, and published publicly via Hugging Face at `https://huggingface.co/datasets/Alovestocode/Router-agent-data`.
- **Logs & reports:** Vertex console exports in `Milestone-3/router-agent-scripts/logs/2025-10-17-*` and evaluation summaries in `Milestone-3/router-agent-scripts/reports/`.
- **Deployment how-to:** `Milestone-3/router-agent-scripts/launch_vertex_tuning.py` README section documents endpoint creation, adapter download, and inference testing.

### Reproduction Checklist
1. Create a virtual environment in `Milestone-3/router-agent-scripts/` and install dependencies.
2. Regenerate or verify the dataset using `prepare_vertex_tuning_dataset.py --seed 42 --gcs-prefix gs://router-data-542496349667/router-dataset`.
3. Launch a smoke tuning job (`--limit 200`) to confirm IAM + JSON fidelity.
4. Submit full tuning jobs with the configuration table above (`--adapter-size 16 --epochs 3 --learning-rate-multiplier 0.7`).
5. Download adapters, run JSON validation + tool metrics on the held-out test split, and push releases to the model registry.

## Observations / Notes for Next Milestone
- Expand the curriculum with harder triage + code + math chains to close the remaining gap in exact plan match for Gemma/Qwen adapters.
- Automate regression evaluation (tool recall, schema validation, rationale score) so Milestone 5 can focus on error analysis instead of manual audits.
- Investigate distillation from Gemma/Qwen into a lighter router to hit latency targets without losing recall.
- Draft guardrail prompts that explicitly penalise hallucinated tools—the main residual error mode flagged in `reports/2025-10-17-audit.md`.

| Adapter | Base model | Context | LoRA rank | Hosted model card |
| --- | --- | --- | --- | --- |
| Router Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 128K | 16 | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft) |
| Router Gemma 3 27B | `google/gemma-3-27b-it` | 128K | 16 | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft) |
| Router Qwen 3 32B | `Qwen/Qwen3-32B` | 32K native / 131K with YaRN | 16 | [HF link](https://huggingface.co/CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft) |

## References
- Vertex AI Tuning preview documentation
- Model cards for Meta Llama 3.1, Google Gemma 3, Alibaba Qwen 3
- Milestone-3 router tuning pipeline (`Milestone-3/router-agent-scripts/README.md`)
