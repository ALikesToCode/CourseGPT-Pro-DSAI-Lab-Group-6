Router Agent Vertex Tuning Pipeline
===================================

Overview
--------
- Converts the Week-2 router dataset into Vertex AI supervised fine-tuning format.
- Supports end-to-end tuning flow for open models (Llama 3.1 8B Instruct, Gemma 3 27B IT, Qwen 3 32B) using Vertex AI preview.
- Provides launch script to submit PEFT or full fine-tuning jobs, plus guidance for evaluation/deployment.

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

Model Architecture Justification
--------------------------------
- **Llama 3.1 8B Instruct** - default choice (full + LoRA).
  - Pros: strong instruction-following baseline, 131k context window, efficient on A2/G2 GPUs, community tooling.
  - Cons: may plateau on ultra-nuanced math/code; full FT incurs higher GPU hours than LoRA.
- **Gemma 3 27B IT** - high-capacity LoRA-only alternative.
  - Pros: larger reasoning headroom, multilingual guardrails pre-baked.
  - Cons: requires LoRA adapters; inference footprint is heavier than 8B models.
- **Qwen 3 32B** - multilingual/code specialist (PEFT-only).
  - Pros: broad token support plus strong code reasoning; good for diverse router content.
  - Cons: LoRA adapters still need L4/A3 resources; preview support is evolving, so monitor API changes.

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
