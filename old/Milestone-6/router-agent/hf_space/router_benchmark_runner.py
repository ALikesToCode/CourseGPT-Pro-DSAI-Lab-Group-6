#!/usr/bin/env python3
"""Run router benchmark evaluation with threshold checks.

Example:
    python router_benchmark_runner.py \
        --gold benchmarks/router_benchmark_hard.jsonl \
        --pred router_predictions.jsonl \
        --thresholds router_benchmark_thresholds.json \
        --out benchmarks/router_benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from schema_score import run_schema_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Benchmark gold JSONL file.")
    parser.add_argument("--pred", required=True, help="Model predictions JSONL.")
    parser.add_argument(
        "--thresholds",
        default="router_benchmark_thresholds.json",
        help="JSON file defining min/max thresholds per bucket.",
    )
    parser.add_argument("--out", required=True, help="Path to write benchmark report JSON.")
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=20,
        help="Maximum number of schema-score error samples to retain.",
    )
    return parser.parse_args()


def load_thresholds(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def evaluate_thresholds(
    metrics: Dict[str, Dict[str, Any]],
    thresholds: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    overall_pass = True

    for bucket, rules in thresholds.items():
        bucket_metrics = metrics.get(bucket, {})
        bucket_result = {"pass": True, "checks": []}

        for metric, target in rules.get("min", {}).items():
            value = bucket_metrics.get(metric)
            passed = value is not None and value >= target
            bucket_result["checks"].append(
                {
                    "metric": metric,
                    "target": target,
                    "type": "min",
                    "value": value,
                    "pass": passed,
                }
            )
            bucket_result["pass"] = bucket_result["pass"] and passed

        for metric, target in rules.get("max", {}).items():
            value = bucket_metrics.get(metric)
            passed = value is not None and value <= target
            bucket_result["checks"].append(
                {
                    "metric": metric,
                    "target": target,
                    "type": "max",
                    "value": value,
                    "pass": passed,
                }
            )
            bucket_result["pass"] = bucket_result["pass"] and passed

        results[bucket] = bucket_result
        overall_pass = overall_pass and bucket_result["pass"]

    results["overall_pass"] = overall_pass
    return results


def main() -> None:
    args = parse_args()
    schema_report = run_schema_evaluation(args.gold, args.pred, args.max_error_examples)
    thresholds = load_thresholds(Path(args.thresholds))
    threshold_results = evaluate_thresholds(schema_report["metrics"], thresholds)

    output = {
        "schema_report": schema_report,
        "threshold_results": threshold_results,
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

    if threshold_results["overall_pass"]:
        print(f"PASS: benchmark thresholds satisfied. Report -> {args.out}")
    else:
        print(f"FAIL: benchmark thresholds violated. See {args.out}")


if __name__ == "__main__":
    main()
