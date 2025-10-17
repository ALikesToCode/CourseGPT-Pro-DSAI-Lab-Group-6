Router Agent Vertex Tuning Pipeline
===================================

Overview
--------
- Converts the Week-2 router dataset into Vertex AI supervised fine-tuning format.
- Supports end-to-end tuning flow for open models (Llama 3.1 8B Instruct, Gemma 3 27B IT, Qwen 3 32B) using Vertex AI preview.
- Provides launch script to submit PEFT or full fine-tuning jobs, plus guidance for evaluation/deployment.

Dataset Preparation
-------------------
- Source dataset: `Milestone-2/router-agent-scripts/output.jsonl`.
- Conversion command (deterministic shuffle, split, and upload):
  ```bash
  python prepare_vertex_tuning_dataset.py \
      --input ../../Milestone-2/router-agent-scripts/output.jsonl \
      --output-dir data/vertex_tuning \
      --val-ratio 0.1 \
      --test-ratio 0.05 \
      --gcs-prefix gs://router-data-542496349667/router-dataset
  ```
- Resulting JSONL files (strict `prompt`/`completion` schema ready for Vertex tuning):
  - `gs://router-data-542496349667/router-dataset/train.jsonl` (6,962 rows)
  - `gs://router-data-542496349667/router-dataset/validation.jsonl` (818 rows)
  - `gs://router-data-542496349667/router-dataset/test.jsonl` (409 rows)
- Local copies reside under `data/vertex_tuning/` for quick inspection.
- `--limit` flag enables smoke subsets (e.g., `--limit 200`) that were used to validate the pipeline before running at full scale.

Environment Setup
-----------------
- Create a local virtualenv and install dependencies:
  ```bash
  cd Milestone-3/router-agent-scripts
  python -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
  ```
- Ensure `gcloud auth login` and `gcloud config set project <PROJECT_ID>` are already run. Export optional defaults:
  ```bash
  export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
  export GOOGLE_CLOUD_REGION=us-central1
  ```

Prepare Training JSONL
----------------------
- Generate prompt/completion files from the Week-2 dataset:
  ```bash
  python prepare_vertex_tuning_dataset.py \
      --input ../../Milestone-2/router-agent-scripts/output.jsonl \
      --output-dir data/vertex_tuning \
      --val-ratio 0.1 \
      --test-ratio 0.05 \
      --gcs-prefix gs://<bucket>/router-dataset
  ```
- The script:
  - Shuffles deterministically, writes `train.jsonl`, `validation.jsonl`, `test.jsonl`.
  - Emits strict JSON completions aligned with the router schema.
  - Uploads to GCS when `--gcs-prefix` is provided.
- For pipeline smoke tests use `--limit 200` to create a tiny subset.

Launch a Tuning Job
-------------------
- Submit a PEFT/LoRA job (recommended first run):
  ```bash
  python launch_vertex_tuning.py \
      --train-uri gs://<bucket>/router-dataset/train.jsonl \
      --validation-uri gs://<bucket>/router-dataset/validation.jsonl \
      --output-uri gs://<bucket>/router-tuning/llama31-peft \
      --tuning-mode PEFT_ADAPTER \
      --adapter-size 16 \
      --epochs 3 \
      --display-name router-llama31-peft \
      --wait
  ```
- Switch to full fine-tuning by passing `--tuning-mode FULL --base-model meta/llama3_1@llama-3.1-8b`.
- Resume or continue training using `--custom-base` pointing at a prior checkpoints directory.
- Monitor jobs from the console (Vertex AI -> Tuning) or keep `--wait` to poll status from the CLI.

Evaluate & Deploy
-----------------
- When the job succeeds, artifacts reside in `<output-uri>/postprocess/node-0/checkpoints/final`.
- Deploy with Vertex endpoints:
  ```python
  from vertexai.preview import model_garden

  model = model_garden.CustomModel(
      gcs_uri="gs://<bucket>/router-tuning/llama31-peft/postprocess/node-0/checkpoints/final",
  )

  endpoint = model.deploy(
      machine_type="g2-standard-12",
      accelerator_type="NVIDIA_L4",
      accelerator_count=1,
  )
  ```
- Route a handful of real router prompts through the endpoint before scaling the dataset run.

End-to-End Pipeline Verification
--------------------------------
- Smoke data run (200 examples) ensured JSONL schema compliance and GCS upload.
- Smoke tuning job confirmed IAM permissions, SDK compatibility, and checkpoint export location.
- Full dataset tuning kicked off after smoke validations succeeded (see job table below).

Active Tuning Jobs
------------------
| Vertex Job ID | Base model | Tuning mode | Adapter size | Output URI | Status |
| --- | --- | --- | --- | --- | --- |
| `projects/542496349667/locations/us-central1/tuningJobs/1491991597619871744` | `meta/llama3_1@llama-3.1-8b-instruct` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/llama31-peft` | Running |
| `projects/542496349667/locations/us-central1/tuningJobs/1108622679339958272` | `publishers/google/models/gemma3@gemma-3-27b-it` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/gemma3-peft` | Running |
| *(pending allowlist)* | `publishers/qwen/models/qwen3@qwen3-30b-a3b-instruct-2507` | PEFT | 16 | `gs://router-data-542496349667/router-tuning/qwen3-peft` | Denied – model not yet eligible for managed tuning |

Use the CLI to track progress:
```bash
gcloud ai tuning-jobs list --region=us-central1
gcloud ai tuning-jobs describe projects/542496349667/locations/us-central1/tuningJobs/<JOB_ID>
```

Model Architecture Justification
--------------------------------
- **Llama 3.1 8B Instruct** (Meta)
  - Pros: instruction-tuned baseline with 131k context window, solid balance of quality vs cost, mature LoRA tooling, deployable on g2-standard-12 (NVIDIA L4).
  - Cons: smaller capacity than 27B+ models may limit complex reasoning; full fine-tuning requires larger accelerators.
- **Gemma 3 27B IT** (Google)
  - Pros: higher reasoning headroom, multilingual guardrails baked in, LoRA keeps GPU/VRAM requirements manageable.
  - Cons: inference footprint heavier than 8B models; preview-only tuning surface; LoRA adapters limit full-parameter updates.
- **Qwen 3 30B Instruct** (Alibaba)
  - Pros: excellent multilingual/code performance; router tasks benefit from tool-use training.
  - Cons: managed tuning currently disabled for this project (requires additional allowlisting); deployment demands H100-class GPUs even with LoRA.

Pipeline Verification
---------------------
- Stage: run `prepare_vertex_tuning_dataset.py --limit 200 --gcs-prefix .../smoke`.
- Tune: execute `launch_vertex_tuning.py` pointing to the smoke dataset to validate API IAM, dataset schema, checkpoint export.
- Evaluate: deploy the resulting small model, run regression prompts from Week-2 evaluation harness, and log deltas.
- Scale: once the smoke test passes, rerun without `--limit` to produce the full-tuning dataset and submit the production job.

Next Steps
----------
- Extend `evaluate_router.py` (future work) to automate regression checks versus previous router logic.
- Track tuning metrics via `job.tuning_data_statistics` and update documentation with observed gains or failure cases.
