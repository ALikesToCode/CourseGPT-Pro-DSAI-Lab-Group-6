# Math Agent — Evaluation & Benchmarks (Milestone 5)

[![Made with Python](https://img.shields.io/badge/made%20with-Python-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This folder contains benchmark data, conversion and evaluation utilities for the Math agent. Use these scripts to convert benchmark files into the evaluation JSONL, run inference against a deployed Vertex endpoint, and compute simple metrics.

**Quick Links**
- **Plots:** `Milestone-5/math-agent/plots/` (gallery and regeneration command below)
- **Scripts:** `convert_benchmarks_to_jsonl.py`, `evaluate_vertex_benchmarks.py`, `compute_metrics.py`
- **Raw data:** `benchmarks_dataset/`

---

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Create / Refresh Combined JSONL](#1-createrefresh-the-combined-benchmark-jsonl)
- [Evaluate on Vertex](#2-evaluate-your-deployed-vertex-endpoint)
- [Compute Metrics](#3-compute-metrics)
- [Plots & Gallery](#plots)
- [Contact](#contact)

---

## Overview

This README explains how to convert benchmark files to our evaluation prompt format, run them through a Vertex endpoint, and compute evaluation metrics. The repository includes a plotting utility to quickly generate comparison visuals from JSONL judgment files.

## Prerequisites

- **Python:** 3.8+ (use a virtual environment)
- **Install packages:**

```bash
pip install google-cloud-aiplatform google-cloud-storage datasets pandas matplotlib seaborn
```

> Note: `datasets` is only required for Hugging Face dataset manipulation; plotting requires `pandas`, `matplotlib`, and `seaborn`.

## 1) Create/refresh the combined benchmark JSONL

Quick overview
- Convert raw benchmark files -> `combined_benchmarks.jsonl` using `convert_benchmarks_to_jsonl.py`.
- Run `evaluate_vertex_benchmarks.py` to send prompts to your deployed Vertex endpoint. The script supports batching, periodic flushes, resumable checkpoints (GCS), and final upload of the combined results file.
- Or, run `eval_ollama.py` to send prompts to a local Ollama model.
- Compute accuracy via `compute_metrics.py` if your combined JSONL includes labels.

Prerequisites

- Python 3.8+ (venv recommended)
- Install required packages:

```bash
pip install google-cloud-aiplatform google-cloud-storage datasets ollama
```

Note: `datasets` is only required if you will manipulate HF datasets locally; the conversion script works with local json/jsonl files.

1) Create/refresh the combined benchmark JSONL

The repository already contains `combined_benchmarks.jsonl` but to regenerate from the `benchmarks_dataset/` folder run:

```bash
# from this folder
python convert_benchmarks_to_jsonl.py -i benchmarks_dataset -o combined_benchmarks.jsonl
```

The converter reads `.json` and `.jsonl` files recursively and writes records in the evaluation prompt format:

```json
{
  "body": {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}]},
  "temperature": 0
}
```

## 2) Evaluate your deployed Vertex endpoint

The evaluator composes a prompt from the system + user messages and sends it to the endpoint. It supports batching, periodic flushes, and resumable checkpoints to GCS.

**Basic usage (no batching):**

```bash
python evaluate_vertex_benchmarks.py \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --endpoint projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID \
  --input combined_benchmarks.jsonl \
  --output results.jsonl
```

**Batched + flush + GCS upload + resume example:**

```bash
python evaluate_vertex_benchmarks.py \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --endpoint projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID \
  --input combined_benchmarks.jsonl \
  --output final_results.jsonl \
  --batch-size 8 \
  --flush-batch-size 64 \
  --gcs-bucket my-bucket \
  --gcs-prefix eval-outputs \
  --overwrite

# If the job stops and you want to resume:
python evaluate_vertex_benchmarks.py --project ... --endpoint ... --input combined_benchmarks.jsonl --output final_results.jsonl --gcs-bucket my-bucket --gcs-prefix eval-outputs --resume
```

Key flags
- `--batch-size` (int): number of instances to send per predict call (default 1)
- `--flush-batch-size` (int): write/upload flushed chunk after this many records (default 64)
- `--gcs-bucket` (str): name of GCS bucket to upload part files and final combined file
- `--gcs-prefix` (str): path inside bucket where parts/checkpoint/final file are stored
- `--resume`: if set, attempt to pick up from a checkpoint written to GCS
- `--instance-key` (str): key used in instance payload (default `content`). Use `input` if your endpoint expects that shape.

Behavior and resumability
- The script writes every flush as a temporary part file: `<output_basename>.partN.jsonl`, uploads that part to GCS (if `--gcs-bucket`), writes a checkpoint `checkpoint.json` to GCS with fields `{processed, part_index}`, and deletes the local part file. The local combined file is appended to as it runs.
- On resume (`--resume`) the script reads the checkpoint on GCS (if present) and continues from the next part index. If no checkpoint exists, it will inspect existing part files in GCS and local combined file to infer progress.

3) Compute metrics

If your input/combined file contains labels (the converter may include `label` when input entries had `answer`) you can compute exact-match accuracy.

```bash
python compute_metrics.py --results results.jsonl --metric exact_match
```

This script prints total examples, exact-match count and accuracy, and a numeric-equality check on numeric-only examples.

Tips & troubleshooting
- Ensure `gcloud auth application-default login` is done and ADC is available to the environment (the script uses ADC for both Vertex and Storage auth).
- If predictions are returned in an unexpected schema, collect one predict response and share it; the extractor in `evaluate_vertex_benchmarks.py` can be adapted to extract the text reliably.
- Use `--max-examples N` to run a short smoke test before committing to a large evaluation run.
- Pick `--batch-size` so each predict response is a comfortable size for latency and your quota.

Adapter artifacts and next steps
- If you need to evaluate local adapters (not deployed), see `Milestone-4/math-agent/adapters/` for local snapshots created in Milestone 4.
- After evaluation you may want to publish final combined results to a model card or dataset record; consider keeping the GCS bucket as a canonical storage location for results and logs.

Contact
- If you want me to adapt the extractor to your endpoint’s exact response schema, add parallel predict concurrency, or enable GCS compose of parts into one object server-side, tell me which and I will implement it.


**Visual Gallery**

Below are the comparison plots produced by `scripts/plot_judgments.py`. Images are embedded for quick inspection; click to open the full-size PNG in your viewer.

<p align="center">
  <img src="https://raw.githubusercontent.com/ALikesToCode/CourseGPT-Pro-DSAI-Lab-Group-6/b52c9770f8e2fd8ec6ee1ad58fa8910b2a4b3d49/assets/compare.correct_by_model.png" alt="Correct answer percent by model" width="860" style="margin:8px;"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ALikesToCode/CourseGPT-Pro-DSAI-Lab-Group-6/b52c9770f8e2fd8ec6ee1ad58fa8910b2a4b3d49/assets/compare.mean_ratings_by_model.png" alt="Mean ratings by model" width="860" style="margin:8px;"/>
</p>

## Metrics Explained

The plots above summarize the following fields commonly present in the judgment JSONL. Use these descriptions to interpret each visualization:

- `correct_answer` — a categorical/boolean indicator showing whether the model's final answer matched the reference. The stacked percent chart (`compare.correct_by_model.png`) displays the share of correct vs incorrect responses for each model.

- `did_it_solve_in_easy_and_fast_approach` — a judge's assessment of whether the solution was both correct and presented in a concise, efficient way. Higher values (or `True`) indicate solutions that are not only correct but also elegantly brief.

- `easy_to_understand_explanation` — a readability/clarity rating for the model's explanation. Use boxplots and mean bars to compare which model provides clearer, more consistent explanations.

