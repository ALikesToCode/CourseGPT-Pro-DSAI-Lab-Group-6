#!/usr/bin/env python3
"""Benchmark the ZeroGPU router Space using the Milestone 5 hard suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from concurrent.futures import ThreadPoolExecutor, as_completed
from gradio_client import Client

AVAILABLE_MODELS = ["Router-Qwen3-32B-8bit", "Router-Gemma3-27B-8bit"]

REPO_ROOT = Path(__file__).resolve().parents[3]
M5_ROUTER_DIR = REPO_ROOT / "Milestone-5" / "router-agent"
DEFAULT_GOLD = M5_ROUTER_DIR / "benchmarks" / "router_benchmark_hard.jsonl"
DEFAULT_THRESHOLDS = M5_ROUTER_DIR / "router_benchmark_thresholds.json"
DEFAULT_PRED = REPO_ROOT / "Milestone-6" / "router-agent" / "tests" / "router_space_predictions.jsonl"
DEFAULT_REPORT = REPO_ROOT / "Milestone-6" / "router-agent" / "tests" / "router_space_report.json"

# Make Milestone 5 utilities importable.
sys.path.insert(0, str(M5_ROUTER_DIR))
from schema_score import run_schema_evaluation  # type: ignore
from router_benchmark_runner import evaluate_thresholds, load_thresholds  # type: ignore


def extract_user_query(prompt: str) -> str:
    marker = "User query:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def iter_benchmark(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def write_predictions(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_space(
    client: Client,
    sample: Dict[str, Any],
    model_choice: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    difficulty: str,
    tags: str,
) -> Dict[str, Any]:
    prompt_text = sample["prompt"]
    user_task = extract_user_query(prompt_text)
    raw, parsed, validation, used_prompt = client.predict(
        user_task=user_task,
        context="",
        acceptance="",
        extra_guidance="",
        difficulty=difficulty,
        tags=tags,
        model_choice=model_choice,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        api_name="/generate_router_plan_streaming",
    )
    completion_payload: str
    if isinstance(parsed, dict) and parsed:
        completion_payload = json.dumps(parsed, ensure_ascii=False)
    else:
        completion_payload = raw
    return {
        "prompt": prompt_text,
        "completion": completion_payload,
        "validation": validation,
        "used_prompt": used_prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="Alovestocode/ZeroGPU-LLM-Inference")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--model", default="Router-Qwen3-32B-8bit", choices=AVAILABLE_MODELS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests (per worker)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests to issue")
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--difficulty", default="advanced",
                        choices=["introductory", "intermediate", "advanced"])
    parser.add_argument("--tags", default="calculus, optimization, python")
    parser.add_argument("--max-error-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = iter_benchmark(args.gold, args.limit)
    def run_single(idx_sample):
        idx, sample = idx_sample
        local_client = Client(args.space)
        record = call_space(
            local_client,
            sample,
            model_choice=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            difficulty=args.difficulty,
            tags=args.tags,
        )
        if args.sleep:
            time.sleep(args.sleep)
        return idx, record

    predictions: List[Dict[str, Any]] = [None] * len(samples)  # type: ignore

    if args.concurrency <= 1:
        for idx, sample in enumerate(samples, 1):
            print(f"[{idx}/{len(samples)}] Generating plan …", flush=True)
            _, record = run_single((idx - 1, sample))
            predictions[idx - 1] = {
                "prompt": record["prompt"],
                "completion": record["completion"],
            }
    else:
        print(f"Running with concurrency={args.concurrency}")
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run_single, (idx, sample)) for idx, sample in enumerate(samples)]
            for fut in as_completed(futures):
                idx, record = fut.result()
                predictions[idx] = {
                    "prompt": record["prompt"],
                    "completion": record["completion"],
                }
                print(f"Completed sample {idx + 1}/{len(samples)}", flush=True)

    write_predictions(args.pred, predictions)

    schema_report = run_schema_evaluation(
        str(args.gold), str(args.pred), args.max_error_examples
    )
    threshold_results = evaluate_thresholds(
        schema_report["metrics"], load_thresholds(args.thresholds)
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "schema_report": schema_report,
                "threshold_results": threshold_results,
                "predictions_path": str(args.pred),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    overall = threshold_results.get("overall_pass", False)
    status = "PASS" if overall else "FAIL"
    print(f"{status}: benchmark complete -> {args.report}")


if __name__ == "__main__":
    main()
