#!/usr/bin/env python3
"""Make a balanced dataset from two JSONL evaluation outputs.

This script loads two JSONL files (evaluation outputs), filters records by a
`status` value (default 200), finds indices present in both files, samples (or
takes) up to `n` shared indices, and writes balanced JSONL files containing
only those records. The output files maintain the same `index` values so the
records align across both outputs.

Usage:
  python make_balanced_dataset.py \
    --a path/to/gemma3.jsonl \
    --b path/to/qwen3.jsonl \
    --n 200 \
    --status 200 \
    --out-dir path/to/out

The script writes two files: `<basename_a>.balanced.jsonl` and
`<basename_b>.balanced.jsonl` in `--out-dir` (defaults to input file dir).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple


def read_jsonl(path: str) -> Dict[int, dict]:
    """Read a JSONL file and return a mapping index -> record.

    If a top-level `index` field exists in records, it is used. Otherwise the
    line number (0-based) is used as the index.
    """
    mapping: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # Skip invalid JSON lines but warn
                print(f"Warning: skipping invalid JSON at {path}:{line_no + 1}")
                continue

            if isinstance(obj, dict) and "index" in obj:
                idx = obj["index"]
                try:
                    idx = int(idx)
                except Exception:
                    # fall back to line number
                    idx = line_no
            else:
                idx = line_no

            mapping[idx] = obj

    return mapping


def write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_balanced(a_path: str, b_path: str, n: int = 200, status: int = 200, seed: int | None = 42, out_dir: str | None = None, sample: bool = True) -> Tuple[str, str]:
    a_map = read_jsonl(a_path)
    b_map = read_jsonl(b_path)

    # Filter by status (if present); otherwise keep all
    def filter_by_status(m: Dict[int, dict]) -> List[int]:
        idxs = []
        for idx, rec in m.items():
            try:
                rec_status = rec.get("status") if isinstance(rec, dict) else None
            except Exception:
                rec_status = None
            if rec_status == status:
                idxs.append(int(idx))
        return idxs

    a_idxs = set(filter_by_status(a_map))
    b_idxs = set(filter_by_status(b_map))

    shared = sorted(list(a_idxs & b_idxs))

    if not shared:
        raise SystemExit("No shared indices with the requested status found between the two files.")

    if len(shared) < n:
        print(f"Only {len(shared)} shared records available (requested {n}). Using {len(shared)}.")
        n = len(shared)

    if sample:
        rnd = random.Random(seed)
        chosen = rnd.sample(shared, n)
        chosen.sort()
    else:
        chosen = shared[:n]

    # Prepare outputs
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(a_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    a_base = os.path.splitext(os.path.basename(a_path))[0]
    b_base = os.path.splitext(os.path.basename(b_path))[0]

    out_a = os.path.join(out_dir, f"{a_base}.balanced.jsonl")
    out_b = os.path.join(out_dir, f"{b_base}.balanced.jsonl")

    a_recs = [a_map[i] for i in chosen]
    b_recs = [b_map[i] for i in chosen]

    write_jsonl(out_a, a_recs)
    write_jsonl(out_b, b_recs)

    print(f"Wrote {len(a_recs)} records to {out_a}")
    print(f"Wrote {len(b_recs)} records to {out_b}")

    return out_a, out_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Create balanced JSONL datasets from two evaluation outputs")
    parser.add_argument("--a", required=True, help="Path to first JSONL file (e.g. gemma3.jsonl)")
    parser.add_argument("--b", required=True, help="Path to second JSONL file (e.g. qwen3.jsonl)")
    parser.add_argument("--n", type=int, default=200, help="Number of shared records to select (default: 200)")
    parser.add_argument("--status", type=int, default=200, help="Status value to filter on (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to input file directory)")
    parser.add_argument("--no-sample", dest="sample", action="store_false", help="Do not sample randomly; take the first N shared indices")
    args = parser.parse_args()

    make_balanced(args.a, args.b, n=args.n, status=args.status, seed=args.seed, out_dir=args.out_dir, sample=args.sample)


if __name__ == "__main__":
    main()
