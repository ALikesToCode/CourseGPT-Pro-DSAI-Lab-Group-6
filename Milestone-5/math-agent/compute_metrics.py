#!/usr/bin/env python3
"""Compute simple metrics from evaluation results JSONL.

Currently supports exact-match accuracy (normalized whitespace, case-insensitive)
and numeric-equality for purely numeric answers.

Usage:
  python compute_metrics.py --results results.jsonl --metric exact_match
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Iterable


def read_results(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def is_numeric(s: str):
    try:
        float(s)
        return True
    except Exception:
        return False


def exact_match(results_path: str):
    total = 0
    matches = 0
    numeric_matches = 0
    numeric_total = 0

    for rec in read_results(results_path):
        total += 1
        pred = normalize_text(str(rec.get("prediction", "")))
        label = rec.get("label")
        if label is None:
            continue
        label = normalize_text(str(label))

        if pred == label:
            matches += 1

        # numeric equality check
        if is_numeric(pred) and is_numeric(label):
            numeric_total += 1
            try:
                if abs(float(pred) - float(label)) < 1e-6:
                    numeric_matches += 1
            except Exception:
                pass

    accuracy = matches / total if total else 0.0
    numeric_acc = numeric_matches / numeric_total if numeric_total else None

    print(f"Total examples: {total}")
    print(f"Exact-match matches: {matches}")
    print(f"Exact-match accuracy: {accuracy:.4f}")
    if numeric_acc is not None:
        print(f"Numeric-equality on numeric subset: {numeric_matches}/{numeric_total} = {numeric_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from results JSONL")
    parser.add_argument("--results", required=True)
    parser.add_argument("--metric", default="exact_match", choices=["exact_match"]) 
    args = parser.parse_args()

    if args.metric == "exact_match":
        exact_match(args.results)


if __name__ == "__main__":
    main()
