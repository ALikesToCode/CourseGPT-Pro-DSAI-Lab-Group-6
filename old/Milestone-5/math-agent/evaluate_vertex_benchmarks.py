#!/usr/bin/env python3
"""Evaluate a deployed Vertex AI endpoint with a combined benchmark JSONL file.

This script reads a JSONL where each record follows the "new format":
{
  "body": {"messages": [{"role":"system","content":...}, {"role":"user","content":...}]},
  "temperature": 0
}

It composes a prompt from the system+user messages, sends the prompt to the
specified Vertex AI endpoint, collects responses and writes a results JSONL
with fields: prompt, prediction, (optional) label.

Usage (example):
  python evaluate_vertex_benchmarks.py \
    --project YOUR_PROJECT \
    --region us-central1 \
    --endpoint projects/YOUR_PROJECT/locations/us-central1/endpoints/123456789 \
    --input combined_benchmarks.jsonl \
    --output results.jsonl

Notes:
- The script attempts multiple keys in the prediction response to extract text
  (e.g., 'content', 'generated_text', 'output', or the stringified prediction).
- If your deployed model expects a different instance schema (for example
  {'input': '...'} vs {'content': '...'}), pass --instance-key to change it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic import PredictionServiceClient
    from google.cloud import storage
except Exception as e:
    print("ERROR: google-cloud-aiplatform and google-cloud-storage are required. Install with: pip install google-cloud-aiplatform google-cloud-storage")
    raise


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compose_prompt(messages: Iterable[Dict[str, str]]) -> str:
    # Compose by concatenating system then user messages. Keep ordering.
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(content.strip())
        elif role == "user":
            parts.append(content.strip())
        else:
            parts.append(content.strip())
    return "\n\n".join(parts)


def extract_text_from_prediction(prediction: Any) -> str:
    # Try common keys
    if isinstance(prediction, dict):
        for k in ("content", "generated_text", "text", "output"):
            if k in prediction and isinstance(prediction[k], str):
                return prediction[k]
        # if 'candidates' or 'outputs'
        if "candidates" in prediction and isinstance(prediction["candidates"], list) and prediction["candidates"]:
            cand = prediction["candidates"][0]
            if isinstance(cand, dict):
                for k in ("content", "text", "output"): 
                    if k in cand and isinstance(cand[k], str):
                        return cand[k]
            if isinstance(cand, str):
                return cand
    # fallback: string representation
    return str(prediction)


def make_instance(prompt: str, temperature: float, instance_key: str) -> Dict[str, Any]:
    return {instance_key: prompt, "temperature": float(temperature)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vertex AI endpoint with JSONL benchmarks")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--endpoint", required=True, help="Full endpoint resource name or endpoint id")
    parser.add_argument("--input", default="combined_benchmarks.jsonl")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--instance-key", default="content", help="Key name for the prompt in the prediction instance (default: content). Try 'input' if needed.")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit number of examples (0 = all)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of instances to send per predict call; set >1 to batch")
    parser.add_argument("--gcs-bucket", default=None, help="If set, flushed partial output files will be uploaded to this GCS bucket")
    parser.add_argument("--gcs-prefix", default="", help="GCS prefix/path under the bucket to upload flushed files")
    parser.add_argument("--resume", action="store_true", help="If set, attempt to resume from a checkpoint in GCS or local output file")
    args = parser.parse_args()

    # Normalize endpoint name: allow passing either full resource or endpoint id
    endpoint = args.endpoint
    if not endpoint.startswith("projects/") and "/endpoints/" not in endpoint:
        # assume it's an endpoint id
        endpoint = f"projects/{args.project}/locations/{args.region}/endpoints/{endpoint}"

    client = PredictionServiceClient()
    storage_client = None
    if args.gcs_bucket:
        storage_client = storage.Client()

    # resumable support: try to read checkpoint from GCS if requested
    file_seq = 0
    if args.resume and storage_client is not None and args.gcs_bucket:
        try:
            prefix = args.gcs_prefix.strip('/')
            checkpoint_name = 'checkpoint.json' if not prefix else f"{prefix}/checkpoint.json"
            bucket = storage_client.bucket(args.gcs_bucket)
            blob = bucket.blob(checkpoint_name)
            if blob.exists():
                data = blob.download_as_text()
                ck = json.loads(data)
                count = int(ck.get('processed', 0))
                file_seq = int(ck.get('part_index', -1)) + 1
                print(f"Resuming from GCS checkpoint: processed={count}, next part_index={file_seq}")
            else:
                # if no checkpoint, try to infer file_seq from existing part blobs
                base_name = os.path.basename(args.output)
                stem = os.path.splitext(base_name)[0]
                max_part = -1
                blobs = storage_client.list_blobs(args.gcs_bucket, prefix=prefix or None)
                import re
                pat = re.compile(rf"{re.escape(stem)}\.part(\d+)\.jsonl$")
                for b in blobs:
                    m = pat.search(b.name)
                    if m:
                        idx = int(m.group(1))
                        if idx > max_part:
                            max_part = idx
                file_seq = max_part + 1
                # try to count lines in local combined file if exists
                if os.path.exists(args.output):
                    with open(args.output, 'r', encoding='utf-8') as f:
                        existing_lines = sum(1 for _ in f)
                    count = existing_lines
                    print(f"Resuming from local output: existing lines={count}, next part_index={file_seq}")
                else:
                    count = 0
        except Exception as e:
            print(f"Warning: failed to read resume checkpoint from GCS: {e}")
            count = 0
    else:
        # start fresh
        count = 0

    # Prepare output file. Optionally overwrite existing file.
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    results_buffer = []
    count = 0
    flush_batch = int(args.flush_batch_size)
    batch_size = max(1, int(args.batch_size))
    file_seq = 0

    def upload_to_gcs(local_path: str, dest_name: str) -> None:
        try:
            bucket = storage_client.bucket(args.gcs_bucket)
            blob = bucket.blob(dest_name)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} -> gs://{args.gcs_bucket}/{dest_name}")
        except Exception as e:
            print(f"Warning: failed to upload {local_path} to GCS: {e}")


    def write_checkpoint_to_gcs(processed: int, part_index: int) -> None:
        try:
            prefix = args.gcs_prefix.strip('/')
            checkpoint_name = 'checkpoint.json' if not prefix else f"{prefix}/checkpoint.json"
            bucket = storage_client.bucket(args.gcs_bucket)
            blob = bucket.blob(checkpoint_name)
            payload = json.dumps({"processed": processed, "part_index": part_index})
            blob.upload_from_string(payload, content_type='application/json')
            print(f"Wrote checkpoint to gs://{args.gcs_bucket}/{checkpoint_name}")
        except Exception as e:
            print(f"Warning: failed to write checkpoint to GCS: {e}")


    def flush_buffer():
        if not results_buffer:
            return
        nonlocal file_seq
        # create a temporary part file (only the flushed chunk)
        base_name = os.path.basename(args.output)
        part_name = f"{os.path.splitext(base_name)[0]}.part{file_seq}.jsonl"
        part_path = os.path.join(os.path.dirname(args.output) or ".", part_name)

        # write chunk to part file
        with open(part_path, "w", encoding="utf-8") as part_f:
            for rec in results_buffer:
                part_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # append chunk to the combined local output file
        with open(args.output, "a", encoding="utf-8") as out_f:
            for rec in results_buffer:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Flushed {len(results_buffer)} records to {args.output} (total so far: {count}); created {part_path}")

        # upload the part file to GCS if requested
        if storage_client is not None and args.gcs_bucket:
            dest_path = part_name if not args.gcs_prefix else os.path.join(args.gcs_prefix.strip('/'), part_name)
            try:
                upload_to_gcs(part_path, dest_path)
                # update checkpoint after successful upload
                write_checkpoint_to_gcs(processed=count, part_index=file_seq)
            except Exception as e:
                print(f"Warning: failed to upload part {part_path} to GCS: {e}")
        # remove local part file after upload
        try:
            os.remove(part_path)
        except Exception:
            pass

        file_seq += 1
        results_buffer.clear()

    # For batching, accumulate input records and make a single predict call per batch
    batch_inputs = []  # list of (orig_record, prompt, temperature)
    for record in read_jsonl(args.input):
        body = record.get("body") or {}
        messages = body.get("messages", [])
        prompt = compose_prompt(messages)
        temperature = record.get("temperature", 0)

        batch_inputs.append((record, prompt, temperature))

        # when we have a batch, call predict once
        if len(batch_inputs) >= batch_size:
            instances = [make_instance(p, t, args.instance_key) for (_, p, t) in batch_inputs]
            try:
                response = client.predict(endpoint=endpoint, instances=instances)
                preds = list(response.predictions)
            except Exception as e:
                print(f"Predict call failed for batch at count {count}: {e}")
                preds = ["<ERROR: predict failed>"] * len(instances)

            # iterate through batch results in order
            for (orig_record, p, t), pred in zip(batch_inputs, preds):
                try:
                    pred_text = extract_text_from_prediction(pred)
                except Exception:
                    pred_text = str(pred)

                out_record = {"prompt": p, "prediction": pred_text}
                if "label" in orig_record:
                    out_record["label"] = orig_record["label"]

                results_buffer.append(out_record)
                count += 1

                # flush periodically
                if flush_batch > 0 and len(results_buffer) >= flush_batch:
                    flush_buffer()

                if args.max_examples and count >= args.max_examples:
                    break

            batch_inputs.clear()

            if args.max_examples and count >= args.max_examples:
                break

    # handle remaining inputs in batch_inputs
    if batch_inputs:
        instances = [make_instance(p, t, args.instance_key) for (_, p, t) in batch_inputs]
        try:
            response = client.predict(endpoint=endpoint, instances=instances)
            preds = list(response.predictions)
        except Exception as e:
            print(f"Predict call failed for final batch: {e}")
            preds = ["<ERROR: predict failed>"] * len(instances)

        for (orig_record, p, t), pred in zip(batch_inputs, preds):
            try:
                pred_text = extract_text_from_prediction(pred)
            except Exception:
                pred_text = str(pred)

            out_record = {"prompt": p, "prediction": pred_text}
            if "label" in orig_record:
                out_record["label"] = orig_record["label"]

            results_buffer.append(out_record)
            count += 1

            if flush_batch > 0 and len(results_buffer) >= flush_batch:
                flush_buffer()

            if args.max_examples and count >= args.max_examples:
                break

    # flush remaining
    flush_buffer()

    print(f"Wrote {count} results to {args.output}")

    # Upload the final combined results file to GCS as a single file if requested
    if storage_client is not None and args.gcs_bucket:
        base_name = os.path.basename(args.output)
        dest_name = base_name if not args.gcs_prefix else os.path.join(args.gcs_prefix.strip('/'), base_name)
        try:
            upload_to_gcs(args.output, dest_name)
            print(f"Final results uploaded to gs://{args.gcs_bucket}/{dest_name}")
        except Exception as e:
            print(f"Warning: final GCS upload failed: {e}")


if __name__ == "__main__":
    main()
