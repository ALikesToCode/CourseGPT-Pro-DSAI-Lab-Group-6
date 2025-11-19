#!/usr/bin/env python3
"""Create router benchmark subsets with controllable archetype coverage.

Example:
    python generate_router_benchmark.py \
        --source ../../Milestone-3/router-agent-scripts/data/vertex_tuning/train.jsonl \
                   ../../Milestone-3/router-agent-scripts/data/vertex_tuning/validation.jsonl \
                   ../../Milestone-3/router-agent-scripts/data/vertex_tuning/test.jsonl \
        --out benchmarks/deep_router_benchmark.jsonl \
        --stats benchmarks/deep_router_benchmark_stats.json \
        --categories math_first four_step metrics_guidance metrics_computation \
        --limit-per-category 150
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="One or more JSONL splits to mine (train / validation / test).",
    )
    parser.add_argument("--out", required=True, help="Destination JSONL for benchmark subset.")
    parser.add_argument(
        "--stats",
        default="",
        help="Optional path to write JSON summary statistics (defaults to stdout only).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional global cap on emitted samples after deduplication (0 = no cap).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Only retain rows containing at least one of these categories.",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=0,
        help="Cap the number of samples contributed by each tracked category (0 = unlimited).",
    )
    parser.add_argument(
        "--include-canonical",
        action="store_true",
        help="Keep purely canonical `/general-search -> /math -> /code` plans. "
        "By default these are dropped to focus on hard negatives.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_completion(row: dict[str, Any]) -> dict[str, Any]:
    completion = row["completion"]
    if isinstance(completion, str):
        return json.loads(completion)
    if isinstance(completion, dict):
        return completion
    raise TypeError("Unsupported completion payload.")


def tool_sequence(plan: Iterable[str]) -> list[str]:
    tools: list[str] = []
    for step in plan:
        tools.append(step.split("(", 1)[0].strip())
    return tools


def classify_sample(completion: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    plan = completion.get("route_plan", [])
    tools = tool_sequence(plan)
    metrics_keys = set()
    if isinstance(completion.get("metrics"), dict):
        metrics_keys = set(completion["metrics"].keys())

    if tools and tools[0] == "/math":
        tags.add("math_first")
    if len(plan) >= 4:
        tags.add("four_step")
    if len(plan) >= 4 and tools[-1] == "/math":
        tags.add("math_backstop")
    if any("guidance" in key for key in metrics_keys):
        tags.add("metrics_guidance")
    if any("computation" in key for key in metrics_keys):
        tags.add("metrics_computation")
    if completion.get("difficulty") == "advanced":
        tags.add("advanced")
    if len(set(tools)) > 3 or tools.count("/math") >= 2:
        tags.add("math_multi_pass")

    canonical = ["/general-search", "/math", "/code"]
    if tool_sequence(plan) != canonical:
        tags.add("non_canonical_route")
    else:
        tags.add("canonical_route")
    return tags


def main() -> None:
    args = parse_args()
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    route_templates: Counter[str] = Counter()
    source_breakdown: Counter[str] = Counter()
    seen_prompt_hashes: set[str] = set()
    per_category_limits: Counter[str] = Counter()

    allowed_categories = set(args.categories)

    for source_path in args.source:
        path = Path(source_path)
        rows = load_jsonl(path)
        for idx, row in enumerate(rows):
            completion = parse_completion(row)
            categories = classify_sample(completion)
            if not args.include_canonical and "canonical_route" in categories:
                continue
            if allowed_categories:
                categories = {cat for cat in categories if cat in allowed_categories}
                if not categories:
                    continue
            prompt = row.get("prompt", "")
            prompt_hash = f"{hash(prompt)}-{hash(json.dumps(completion, sort_keys=True))}"
            if prompt_hash in seen_prompt_hashes:
                continue

            if args.limit_per_category:
                eligible = False
                for cat in categories:
                    if per_category_limits[cat] < args.limit_per_category:
                        eligible = True
                        break
                if not eligible:
                    continue
            route_templates[" -> ".join(tool_sequence(completion.get("route_plan", [])))] += 1
            category_counts.update(categories)
            for cat in categories:
                per_category_limits[cat] += 1
            selected.append(
                {
                    "source": str(path),
                    "index": idx,
                    "prompt": prompt,
                    "completion": row["completion"],
                    "categories": sorted(categories),
                }
            )
            seen_prompt_hashes.add(prompt_hash)
            source_breakdown[str(path)] += 1

    if args.max_samples and len(selected) > args.max_samples:
        selected = selected[: args.max_samples]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            json.dump(row, handle)
            handle.write("\n")

    stats = {
        "sources": [str(Path(p)) for p in args.source],
        "output": str(args.out),
        "benchmark_rows": len(selected),
        "categories_included": sorted(category_counts.keys()),
        "category_counts": dict(sorted(category_counts.items())),
        "route_templates": route_templates.most_common(15),
        "source_sample_counts": dict(sorted(source_breakdown.items())),
        "limit_per_category": args.limit_per_category,
        "include_canonical": args.include_canonical,
    }
    if args.max_samples:
        stats["max_samples"] = args.max_samples

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Wrote benchmark subset to {out_path}")


if __name__ == "__main__":
    main()
