# Evaluating deployed Vertex model with combined benchmarks

This document shows how to run the provided evaluation scripts against a deployed Vertex AI endpoint (Model Garden / custom weights) using the combined JSONL benchmark file in this folder.

Files added
- `evaluate_vertex_benchmarks.py` — sends each prompt from the combined JSONL to the Vertex endpoint and saves predictions to a results JSONL.
- `compute_metrics.py` — computes simple metrics (exact-match accuracy) from the results JSONL when labels are available.

Prerequisites

- Python 3.8+ and a virtualenv
- Install dependencies:

```bash
pip install google-cloud-aiplatform google-cloud-storage
```

- Ensure `gcloud` is authenticated and ADC is set up:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Find your endpoint resource name

In the Vertex AI console go to Endpoints and find the endpoint you deployed your model to. The endpoint resource name is:

```
projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID
```

Example: `projects/my-project/locations/us-central1/endpoints/1234567890123456789`

Run evaluation (Bash / WSL)

```bash
python evaluate_vertex_benchmarks.py \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --endpoint projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID \
  --input combined_benchmarks.jsonl \
  --output results.jsonl
```

PowerShell example

```powershell
python .\evaluate_vertex_benchmarks.py --project YOUR_PROJECT_ID --region us-central1 --endpoint projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID --input .\combined_benchmarks.jsonl --output .\results.jsonl
```

Notes on instance schema

The script sends instances with a key `content` (default). If your deployed model expects a different key (for example `input`), pass `--instance-key input`.

Batching and GCS upload examples

The evaluator supports batching multiple instances per predict call and uploading flushed partial outputs to GCS. Useful flags:

- `--batch-size N` — send up to N instances in a single predict call (set >1 to reduce API overhead)
- `--flush-batch-size M` — write results to disk after M records (default 64)
- `--gcs-bucket BUCKET` — upload flushed output files to this bucket
- `--gcs-prefix PREFIX` — optional path under the bucket to store flushed files

Example: batch-size 8, flush every 64, upload flushed outputs to `gs://my-bucket/eval-outputs/`

```bash
python evaluate_vertex_benchmarks.py \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --endpoint projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID \
  --input combined_benchmarks.jsonl \
  --output results.jsonl \
  --batch-size 8 \
  --flush-batch-size 64 \
  --gcs-bucket my-bucket \
  --gcs-prefix eval-outputs \
  --overwrite
```

PowerShell equivalent:

```powershell
python .\evaluate_vertex_benchmarks.py --project YOUR_PROJECT_ID --region us-central1 --endpoint projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID --input .\combined_benchmarks.jsonl --output .\results.jsonl --batch-size 8 --flush-batch-size 64 --gcs-bucket my-bucket --gcs-prefix eval-outputs --overwrite
```

Interpreting results

The output file (`results.jsonl`) contains one JSON object per line with fields:

- `prompt` — the composed prompt (system + user messages)
- `prediction` — model response text (best-effort extraction from API response)
- `label` — optional (if the input JSONL contained an `answer`/label field)

If your dataset included labels you can compute exact-match accuracy with `compute_metrics.py`.

Compute metrics (when labels exist)

```bash
python compute_metrics.py --results results.jsonl --metric exact_match
```

The script currently implements exact-match (normalized whitespace, case-insensitive). It will print the number of examples, matches, and accuracy.

Troubleshooting

- Permission errors: ensure ADC has access to the Vertex endpoint and that the user or service account has `aiplatform.endpoints.predict` permission.
- Response extraction: The script attempts to extract text from common keys in the prediction response. If your model returns a different schema, open `evaluate_vertex_benchmarks.py` and update `extract_text_from_prediction()` accordingly.
- Large runs: consider using `--max-examples N` to test a subset before running the full benchmark.

If you want, I can adapt the evaluation to call the Vertex Generative Models API or the Model Garden Predict API if your endpoint requires a different request shape — provide a sample prediction response and I'll update the extractor.