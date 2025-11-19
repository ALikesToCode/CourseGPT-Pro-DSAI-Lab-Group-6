#!/usr/bin/env python3
"""Aggregate router model evaluation metrics and dataset stats.

This script pulls the latest evaluation artefacts from the Hugging Face Hub
and inspects the local Vertex tuning dataset to build a concise JSON summary.

Usage:
    python collect_router_metrics.py

Outputs:
    Milestone-5/router-agent/router_eval_metrics.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Milestone-3" / "router-agent-scripts" / "data" / "vertex_tuning"
OUTPUT_PATH = Path(__file__).with_name("router_eval_metrics.json")


@dataclass
class ModelEvalSource:
    """Describe how to fetch evaluation metrics for a model."""

    repo_id: str
    # Preferred file providing evaluation metrics.
    primary_file: str
    # Optional fallback (e.g., eval_results.json missing -> trainer_state.json).
    fallback_file: str | None = None


MODEL_SOURCES: dict[str, ModelEvalSource] = {
    "router-llama31-peft": ModelEvalSource(
        repo_id="CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft",
        primary_file="eval_results.json",
        fallback_file="trainer_state.json",
    ),
    "router-gemma3-peft": ModelEvalSource(
        repo_id="CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft",
        primary_file="trainer_state.json",
    ),
    "router-qwen3-32b-peft": ModelEvalSource(
        repo_id="CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft",
        primary_file="trainer_state.json",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_final_eval_from_trainer_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the last eval_* entry from Hugging Face trainer_state.json."""
    for entry in reversed(state.get("log_history", [])):
        if "eval_loss" in entry:
            return entry
    raise ValueError("No eval_* entry found in trainer_state.json")


def fetch_model_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for model_name, source in MODEL_SOURCES.items():
        data: dict[str, Any]
        file_used: str
        try:
            path = Path(
                hf_hub_download(repo_id=source.repo_id, filename=source.primary_file)
            )
            data = _load_json(path)
            file_used = source.primary_file
            if source.primary_file.endswith("trainer_state.json"):
                data = _select_final_eval_from_trainer_state(data)
        except Exception:
            if not source.fallback_file:
                raise
            path = Path(
                hf_hub_download(repo_id=source.repo_id, filename=source.fallback_file)
            )
            state = _load_json(path)
            data = _select_final_eval_from_trainer_state(state)
            file_used = source.fallback_file

        model_metrics: dict[str, Any] = {"source_file": file_used}

        # Standard metrics emitted by Vertex / HF Trainer.
        for key in (
            "eval_loss",
            "eval_perplexity",
            "eval_bleu",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
            "eval_gen_len",
            "eval_length_ratio",
            "eval_ref_len",
            "num_input_tokens_seen",
            "memory/device_reserved (GiB)",
            "memory/max_active (GiB)",
            "memory/max_allocated (GiB)",
        ):
            if key in data:
                model_metrics[key] = data[key]

        # Derive perplexity when absent (e.g., trainer_state only exposes loss).
        if "eval_perplexity" not in model_metrics and "eval_loss" in model_metrics:
            model_metrics["eval_perplexity"] = math.exp(model_metrics["eval_loss"])

        metrics[model_name] = model_metrics
    return metrics


def summarise_dataset(split_name: str) -> dict[str, Any]:
    path = DATA_DIR / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")

    total_tokens = 0
    route_lengths: Counter[int] = Counter()
    tool_counts: Counter[str] = Counter()
    first_tool_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    metrics_key_counts: Counter[str] = Counter()
    metrics_field_shapes: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    for row in rows:
        completion = json.loads(row["completion"])
        route_plan = completion.get("route_plan", [])

        route_lengths[len(route_plan)] += 1
        if route_plan:
            first_tool = route_plan[0].split("(", 1)[0].strip()
            first_tool_counts[first_tool] += 1

        for step in route_plan:
            tool = step.split("(", 1)[0].strip()
            tool_counts[tool] += 1

        difficulty_counts[completion.get("difficulty", "unlabelled")] += 1

        for tag in completion.get("tags", []):
            tag_counts[tag] += 1

        metrics_field = completion.get("metrics")
        if isinstance(metrics_field, dict):
            metrics_field_shapes[frozenset(metrics_field.keys())] += 1
            for key, value in metrics_field.items():
                shape = "list" if isinstance(value, list) else type(value).__name__
                metrics_key_counts[f"{key}:{shape}"] += 1

        total_tokens += len(row.get("prompt", "").split())

    sample_count = len(rows)
    avg_route_length = (
        sum(length * count for length, count in route_lengths.items()) / sample_count
        if sample_count
        else 0.0
    )

    return {
        "split": split_name,
        "samples": sample_count,
        "avg_route_length": avg_route_length,
        "route_length_distribution": route_lengths,
        "tool_counts": tool_counts,
        "first_tool_distribution": first_tool_counts,
        "difficulty_distribution": difficulty_counts,
        "tag_counts_top10": tag_counts.most_common(10),
        "metrics_field_shapes": {", ".join(sorted(k)): v for k, v in metrics_field_shapes.items()},
        "metrics_value_types": metrics_key_counts,
        "avg_prompt_token_count": total_tokens / sample_count if sample_count else 0.0,
    }


def convert_counters(obj: Any) -> Any:
    """Recursively cast Counter objects to plain dicts with sorted keys."""
    if isinstance(obj, Counter):
        return {str(key): obj[key] for key in sorted(obj)}
    if isinstance(obj, dict):
        return {str(k): convert_counters(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_counters(item) for item in obj]
    return obj


def main() -> None:
    router_metrics = fetch_model_metrics()
    dataset_summary = {
        split: summarise_dataset(split) for split in ("train", "validation", "test")
    }

    payload = {
        "models": router_metrics,
        "dataset": dataset_summary,
    }
    OUTPUT_PATH.write_text(json.dumps(convert_counters(payload), indent=2))
    print(f"Wrote evaluation summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
