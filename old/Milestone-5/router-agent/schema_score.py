#!/usr/bin/env python3
"""Schema-aware evaluation for router-agent predictions.

The module exposes :func:`run_schema_evaluation` for programmatic use and a CLI
entry point that writes a JSON report to disk.

Example CLI usage:
    python schema_score.py \
        --gold ../../Milestone-3/router-agent-scripts/data/vertex_tuning/test.jsonl \
        --pred router_predictions.jsonl \
        --out router_schema_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

MetricDict = dict[str, Any]

TODO_TOOL_REGEX = re.compile(r"/[a-zA-Z0-9_-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="JSONL with reference completions.")
    parser.add_argument("--pred", required=True, help="JSONL with router predictions.")
    parser.add_argument("--out", required=True, help="Path to write JSON metrics report.")
    parser.add_argument("--max-error-examples", type=int, default=15, help="Sample size for error listings.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_completion(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict) and "completion" in obj:
        candidate = obj["completion"]
        if isinstance(candidate, str):
            return json.loads(candidate)
        if isinstance(candidate, dict):
            return candidate
        raise TypeError("Unsupported completion type.")
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        return json.loads(obj)
    raise TypeError(f"Cannot parse completion from {type(obj)}")


def tool_sequence(plan: Iterable[str]) -> list[str]:
    seq: list[str] = []
    for step in plan:
        name = step.split("(", 1)[0].strip()
        seq.append(name)
    return seq


def metrics_key_set(metrics_block: Any) -> frozenset[str]:
    if isinstance(metrics_block, dict):
        return frozenset(metrics_block.keys())
    return frozenset()


def metrics_value_type(metrics_block: Any, key: str) -> str:
    if not isinstance(metrics_block, dict):
        return "missing"
    value = metrics_block.get(key, None)
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "missing"
    return type(value).__name__


def metrics_list_nonempty(metrics_block: Any, key: str) -> bool:
    if not isinstance(metrics_block, dict):
        return False
    value = metrics_block.get(key)
    return isinstance(value, list) and len(value) > 0


def extract_todo_tool(todo_entry: str) -> str | None:
    match = TODO_TOOL_REGEX.search(todo_entry)
    if match:
        return match.group(0)
    lowered = todo_entry.lower()
    if "router qa" in lowered:
        return "router QA"
    return None


def todo_tool_alignment(todo_list: list[str], expected_tools: list[str]) -> float:
    if not todo_list:
        # Treat missing todo list as zero alignment unless nothing is expected.
        return 1.0 if not expected_tools else 0.0
    matches = 0
    for item in todo_list:
        tool = extract_todo_tool(item)
        if tool and (tool in expected_tools or tool.lower().startswith("router")):
            matches += 1
    return matches / len(todo_list)


def todo_covers_all_tools(todo_list: list[str], expected_tools: list[str]) -> bool:
    if not expected_tools:
        return True
    todo_tools = {
        extract_todo_tool(item) for item in todo_list if extract_todo_tool(item) is not None
    }
    return all(tool in todo_tools for tool in expected_tools)


def handoff_covers_tools(handoff_plan: str, tools: list[str]) -> bool:
    if not tools:
        return bool(handoff_plan)
    handoff_lower = (handoff_plan or "").lower()
    return all(tool.lower() in handoff_lower for tool in tools)


def handoff_mentions_router(handoff_plan: str) -> bool:
    return "router qa" in (handoff_plan or "").lower()


def count_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(count_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(count_tokens(item) for item in value.values())
    if isinstance(value, str):
        return len(value.split())
    return 0


def evaluate_sample(
    gold: dict[str, Any],
    pred: dict[str, Any] | None,
) -> tuple[MetricDict, dict[str, bool], dict[str, Any]]:
    metrics: MetricDict = {}
    flags: dict[str, bool] = {}
    debug: dict[str, Any] = {}

    gold_plan = gold.get("route_plan", [])
    gold_metrics = gold.get("metrics", {})
    gold_tools = tool_sequence(gold_plan)
    gold_first = gold_tools[0] if gold_tools else None
    gold_last = gold_tools[-1] if gold_tools else None
    gold_len = len(gold_plan)
    gold_metrics_keys = metrics_key_set(gold_metrics)

    flags["math_first"] = gold_first == "/math"
    flags["four_step"] = gold_len >= 4
    flags["has_guidance"] = any("guidance" in key for key in gold_metrics_keys)
    flags["has_computation"] = any("computation" in key for key in gold_metrics_keys)
    flags["advanced"] = gold.get("difficulty") == "advanced"

    metrics["json_valid"] = pred is not None
    if pred is None:
        return metrics, flags, debug

    pred_plan = pred.get("route_plan", [])
    pred_metrics = pred.get("metrics", {})
    pred_tools = tool_sequence(pred_plan)
    pred_len = len(pred_plan)
    pred_first = pred_tools[0] if pred_tools else None
    pred_last = pred_tools[-1] if pred_tools else None

    metrics["route_plan_exact"] = pred_plan == gold_plan
    metrics["route_length_match"] = pred_len == gold_len
    metrics["route_first_tool_match"] = pred_first == gold_first
    metrics["route_last_tool_match"] = pred_last == gold_last

    gold_set = set(gold_tools)
    pred_set = set(pred_tools)
    intersection = len(gold_set & pred_set)
    metrics["route_tool_precision"] = intersection / len(pred_set) if pred_set else 1.0
    metrics["route_tool_recall"] = intersection / len(gold_set) if gold_set else 1.0

    metrics["metrics_keys_match"] = metrics_key_set(pred_metrics) == gold_metrics_keys
    for key in ("primary", "secondary"):
        metrics[f"metrics_{key}_list"] = metrics_value_type(pred_metrics, key) == "list"
        metrics[f"metrics_{key}_nonempty"] = metrics_list_nonempty(pred_metrics, key)
    for key in ("primary_guidance", "secondary_guidance", "primary_computation", "secondary_computation"):
        expected = metrics_value_type(gold_metrics, key)
        if expected != "missing":
            metrics[f"{key}_retained"] = metrics_value_type(pred_metrics, key) == expected

    todo_list = pred.get("todo_list", []) if isinstance(pred.get("todo_list"), list) else []
    metrics["todo_tool_alignment"] = todo_tool_alignment(todo_list, gold_tools)
    metrics["todo_covers_all_tools"] = todo_covers_all_tools(todo_list, gold_tools)

    expected_artifacts = pred.get("expected_artifacts", [])
    metrics["expected_artifacts_nonempty"] = isinstance(expected_artifacts, list) and len(expected_artifacts) > 0

    acceptance = pred.get("acceptance_criteria", [])
    metrics["acceptance_criteria_nonempty"] = isinstance(acceptance, list) and len(acceptance) > 0

    handoff = pred.get("handoff_plan", "")
    metrics["handoff_tools_covered"] = handoff_covers_tools(handoff, gold_tools)
    metrics["handoff_router_qa"] = handoff_mentions_router(handoff)

    metrics["total_token_count"] = count_tokens(pred)

    gold_json = json.dumps(gold, sort_keys=True)
    pred_json = json.dumps(pred, sort_keys=True)
    metrics["length_ratio"] = (len(pred_json) / len(gold_json)) if gold_json else 1.0
    metrics["length_ratio_gt_1.1"] = metrics["length_ratio"] > 1.1
    metrics["length_ratio_gt_1.25"] = metrics["length_ratio"] > 1.25
    metrics["length_ratio_lt_0.9"] = metrics["length_ratio"] < 0.9

    if not metrics["route_plan_exact"]:
        debug["route_plan_pred"] = pred_plan
        debug["route_plan_gold"] = gold_plan
    if not metrics["metrics_keys_match"]:
        debug["metrics_keys_pred"] = sorted(metrics_key_set(pred_metrics))
        debug["metrics_keys_gold"] = sorted(gold_metrics_keys)
    if metrics["todo_tool_alignment"] < 1.0:
        debug["todo_list"] = todo_list

    return metrics, flags, debug


class MetricAggregator:
    def __init__(self, bucket_name: str):
        self.bucket = bucket_name
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: Counter[str] = Counter()
        self.samples = 0

    def update(self, metrics: MetricDict):
        self.samples += 1
        for name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, bool):
                self.sums[name] += 1.0 if value else 0.0
                self.counts[name] += 1
            elif isinstance(value, (int, float)):
                if math.isnan(value):
                    continue
                self.sums[name] += float(value)
                self.counts[name] += 1

    def render(self) -> dict[str, Any]:
        output: dict[str, Any] = {"samples": self.samples}
        for name, total in sorted(self.sums.items()):
            denom = self.counts[name]
            if denom:
                output[name] = total / denom
        return output


def run_schema_evaluation(
    gold_path: str | Path,
    pred_path: str | Path,
    max_error_examples: int = 15,
) -> dict[str, Any]:
    gold_rows = load_jsonl(Path(gold_path))
    pred_rows = load_jsonl(Path(pred_path))
    if len(pred_rows) != len(gold_rows):
        raise ValueError(f"Gold rows ({len(gold_rows)}) != Pred rows ({len(pred_rows)})")

    buckets: dict[str, MetricAggregator] = defaultdict(lambda: MetricAggregator("overall"))
    buckets["overall"] = MetricAggregator("overall")

    subset_names = {
        "math_first": "math_first",
        "four_step": "four_step",
        "has_guidance": "has_guidance",
        "has_computation": "has_computation",
        "advanced": "advanced",
    }

    errors: list[dict[str, Any]] = []

    for idx, (gold_row, pred_row) in enumerate(zip(gold_rows, pred_rows)):
        gold = parse_completion(gold_row["completion"])
        try:
            pred = parse_completion(pred_row)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            metrics, flags, debug = evaluate_sample(gold, None)
            debug["error"] = f"Failed to parse prediction: {exc}"
        else:
            metrics, flags, debug = evaluate_sample(gold, pred)

        buckets["overall"].update(metrics)
        for flag, name in subset_names.items():
            if flags.get(flag):
                buckets.setdefault(name, MetricAggregator(name)).update(metrics)

        if debug and len(errors) < max_error_examples:
            errors.append(
                {
                    "index": idx,
                    "prompt_preview": gold_row.get("prompt", "")[:140],
                    "details": debug,
                }
            )

    report = {
        "config": {
            "gold_path": str(gold_path),
            "pred_path": str(pred_path),
            "total_samples": len(gold_rows),
        },
        "metrics": {name: agg.render() for name, agg in sorted(buckets.items())},
        "error_samples": errors,
    }
    return report


def main() -> None:
    args = parse_args()
    report = run_schema_evaluation(args.gold, args.pred, args.max_error_examples)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote schema-aware metrics to {args.out}")


if __name__ == "__main__":
    main()
