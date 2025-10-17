#!/usr/bin/env python3
"""
Submit a supervised fine-tuning job for open-weight models on Vertex AI.

Example:
    python launch_vertex_tuning.py \\
        --project=my-project --location=us-central1 \\
        --train-uri=gs://bucket/router/train.jsonl \\
        --validation-uri=gs://bucket/router/validation.jsonl \\
        --output-uri=gs://bucket/router/runs/llama31-peft \\
        --tuning-mode=PEFT_ADAPTER --adapter-size=16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

from google.cloud import aiplatform

try:
    import vertexai
    from vertexai.preview.tuning import sft, SourceModel
except ModuleNotFoundError as import_exc:  # pragma: no cover - validated in runtime smoke tests.
    vertexai = None  # type: ignore
    sft = None  # type: ignore
    SourceModel = None  # type: ignore
    _IMPORT_ERROR = import_exc
else:
    _IMPORT_ERROR = None


DEFAULT_BASE_MODEL = "meta/llama3_1@llama-3.1-8b-instruct"
POLL_SECONDS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud project for the job.")
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"), help="Vertex AI region.")
    parser.add_argument("--train-uri", required=True, help="GCS URI to training JSONL.")
    parser.add_argument("--validation-uri", help="Optional GCS URI to validation JSONL.")
    parser.add_argument("--output-uri", required=True, help="GCS folder for tuned artifacts.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="publisher/model identifier.")
    parser.add_argument("--custom-base", default="", help="Optional GCS folder with previous LoRA/full checkpoints.")
    parser.add_argument("--tuning-mode", choices=["FULL", "PEFT_ADAPTER"], default="PEFT_ADAPTER", help="Fine-tuning strategy.")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch count.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for FULL mode.")
    parser.add_argument("--learning-rate-multiplier", type=float, default=None, help="Learning rate multiplier (mutually exclusive).")
    parser.add_argument("--adapter-size", type=int, default=16, help="LoRA adapter size when using PEFT.")
    parser.add_argument("--display-name", default="", help="Display name for the tuned Vertex model.")
    parser.add_argument("--labels", nargs="*", help="Optional key=value labels for the job.")
    parser.add_argument("--wait", action="store_true", help="Poll the job until completion.")
    return parser.parse_args()


def parse_labels(raw: Optional[list[str]]) -> Dict[str, str]:
    if not raw:
        return {}
    labels: Dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid label '{item}', expected key=value.")
        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip()
    return labels


def validate_args(args: argparse.Namespace) -> None:
    if not args.project:
        raise ValueError("Project must be provided via --project or GOOGLE_CLOUD_PROJECT.")
    if args.learning_rate and args.learning_rate_multiplier:
        raise ValueError("Specify only one of --learning-rate or --learning-rate-multiplier.")
    if args.tuning_mode == "PEFT_ADAPTER" and args.learning_rate and args.learning_rate > 0.01:
        print("Warning: learning rate looks high for LoRA, double check.", file=sys.stderr)
    if args.tuning_mode == "FULL" and args.adapter_size is not None:
        # Adapter size is ignored but users often expect a warning.
        print("Note: --adapter-size is ignored in FULL tuning mode.", file=sys.stderr)


def submit_job(args: argparse.Namespace) -> sft.SupervisedTuningJob:
    if _IMPORT_ERROR:
        raise ModuleNotFoundError(
            "Missing Vertex AI preview dependencies. Install via "
            "`pip install -r requirements.txt` inside a virtualenv."
        ) from _IMPORT_ERROR
    vertexai.init(project=args.project, location=args.location)
    labels = parse_labels(args.labels)
    # The SourceModel wrapper makes it easy to resume runs from a previous checkpoint.
    source = SourceModel(base_model=args.base_model, custom_base_model=args.custom_base)
    job = sft.train(
        source_model=source,
        train_dataset=args.train_uri,
        validation_dataset=args.validation_uri,
        tuning_mode=args.tuning_mode,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        learning_rate_multiplier=args.learning_rate_multiplier,
        adapter_size=args.adapter_size if args.tuning_mode == "PEFT_ADAPTER" else None,
        tuned_model_display_name=args.display_name or None,
        output_uri=args.output_uri,
        labels=labels or None,
    )
    return job


def wait_for_completion(job: sft.SupervisedTuningJob) -> None:
    print(f"Waiting for job {job.resource_name} ...")
    while not job.has_ended:
        time.sleep(POLL_SECONDS)
        job.refresh()
        print(json.dumps({"state": job.state.name, "model": job.tuned_model_name}, indent=2))
    if job.has_succeeded:
        print("Job completed successfully.")
    else:
        print(f"Job ended with error: {job.error}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    validate_args(args)
    aiplatform.init(project=args.project, location=args.location)
    job = submit_job(args)
    print(json.dumps({"tuning_job": job.resource_name}, indent=2))
    if args.wait:
        wait_for_completion(job)


if __name__ == "__main__":
    main()
