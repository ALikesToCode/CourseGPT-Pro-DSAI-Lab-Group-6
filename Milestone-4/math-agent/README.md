# Math Agent — Vertex Tuning README

This document walks through preparing data and launching Vertex AI tuning jobs for math model adapters.

## Prerequisites

- Install and configure the Google Cloud SDK (gcloud).
- Ensure you have a GCP project and permission to write to a Cloud Storage bucket.
- Python 3.8+ and required scripts (`prepare_vertex_tuning.py`, `launch_vertex_tuning.py`) available in this repo.

## 1. Set up your environment

Authenticate with Google Cloud and set your project and region environment variables.

```bash
gcloud auth login
gcloud auth application-default login
```

Export your project, region and a GCS bucket name (replace the placeholders):

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_REGION="us-central1"
export BUCKET_NAME="your-gcs-bucket-name" # e.g., "my-math-model-bucket"
```

## 2. Prepare and upload your data (run once)

Run the preparation script to download/process the dataset and upload it to GCS. This will take a few minutes and will produce training and validation JSONL files.

```bash
python prepare_vertex_tuning.py \
    --bucket $BUCKET_NAME \
    --gcs-prefix "mathx-dataset-v1"
```

After the script finishes it will print two URIs. Copy them for the training step:

- Train URI: `gs://your-gcs-bucket-name/mathx-dataset-v1/train.jsonl`
- Validation URI: `gs://your-gcs-bucket-name/mathx-dataset-v1/validation.jsonl`

## 3. Launch model training jobs (run multiple times)

Use `launch_vertex_tuning.py` to run Vertex tuning jobs. Below are example invocations for three different base models. Replace the URIs, project/bucket names, and any other flags as needed.

### Example 1 — Gemma 3 (4B IT)

```bash
python launch_vertex_tuning.py \
    --base-model "google/gemma3@gemma-3-4b-it" \
    --display-name "math-gemma-3-4b-peft" \
    --train-uri "gs://$BUCKET_NAME/mathx-dataset-v1/train.jsonl" \
    --validation-uri "gs://$BUCKET_NAME/mathx-dataset-v1/validation.jsonl" \
    --output-uri "gs://$BUCKET_NAME/adapters/math-gemma-3-4b"
```

### Example 2 — Llama 3.2 (3B-instruct)

This is a 3B model that can be a strong, cost-effective alternative.

```bash
python launch_vertex_tuning.py \
    --base-model "meta/llama3-2@llama-3.2-3b-instruct" \
    --display-name "math-llama-3-3b-peft" \
    --train-uri "gs://$BUCKET_NAME/mathx-dataset-v1/train.jsonl" \
    --validation-uri "gs://$BUCKET_NAME/mathx-dataset-v1/validation.jsonl" \
    --output-uri "gs://$BUCKET_NAME/adapters/math-llama-3-3b" \
    --epochs 3 \
    --adapter-size 16
```

### Example 3 — Phi‑3‑mini‑4k‑instruct (3.8B)

This model is a competitive alternative to Gemma 4B for certain workloads.

```bash
python launch_vertex_tuning.py \
    --base-model "microsoft/phi-3-mini-4k-instruct" \
    --display-name "math-phi-3-mini-peft" \
    --train-uri "gs://$BUCKET_NAME/mathx-dataset-v1/train.jsonl" \
    --validation-uri "gs://$BUCKET_NAME/mathx-dataset-v1/validation.jsonl" \
    --output-uri "gs://$BUCKET_NAME/adapters/math-phi-3-mini" \
    --epochs 3 \
    --adapter-size 16
```

## Notes & tips

- Replace placeholder values (project id, bucket name, URIs) with your real values.
- The `prepare_vertex_tuning.py` script prints the final `train.jsonl` and `validation.jsonl` GCS URIs — use those when launching tuning jobs.
- Monitor job status and logs in the Google Cloud Console (Vertex AI > Training > Jobs).
- Consider setting up IAM permissions on the storage bucket so Vertex AI has access to read the training/validation files and write model artifacts.

If you'd like, I can also add a short section on required Python dependencies and a sample `requirements.txt` or add automatic creation of the GCS prefix folder. Let me know which you'd prefer.