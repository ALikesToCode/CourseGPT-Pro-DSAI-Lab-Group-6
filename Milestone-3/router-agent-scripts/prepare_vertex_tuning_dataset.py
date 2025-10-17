#!/usr/bin/env python3
"""
Transform the week-2 router dataset into Vertex AI tuning JSONL files.

Outputs `prompt`/`completion` pairs that teach an instruction-tuned model to
produce the router planning JSON we expect at inference time. Optionally pushes
the generated splits to Cloud Storage so they are ready for tuning jobs.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

try:
    from google.cloud import storage
except ImportError:  # pragma: no cover - optional dependency during local runs.
    storage = None  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = REPO_ROOT / "Milestone-2" / "router-agent-scripts" / "output.jsonl"
PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are the Router Agent coordinating Math, Code, and General-Search specialists.
    Given the user query, respond with a JSON object describing the full routing plan.

    Requirements:
    - STRICT JSON with double quotes and no extra commentary.
    - Keys: route_plan, route_rationale, expected_artifacts, thinking_outline,
      handoff_plan, todo_list, difficulty, tags, acceptance_criteria, metrics.
    - route_plan must be an ordered list of tool invocations.
    - Use the original phrasing when possible.

    User query:
    {user_query}
    """
)


@dataclass
class RouterRecord:
    """Lightweight view of a router dataset row."""

    record_id: str
    user_query: str
    payload: dict

    @classmethod
    def from_json(cls, line: str) -> "RouterRecord":
        data = json.loads(line)
        return cls(
            record_id=data["id"],
            user_query=data["user_query"],
            payload=data,
        )

    def to_example(self) -> dict:
        """Emit a JSON-serialisable dict for Vertex tuning."""
        completion_obj = {
            "route_plan": self.payload["route_plan"],
            "route_rationale": self.payload["route_rationale"],
            "expected_artifacts": self.payload["expected_artifacts"],
            "thinking_outline": self.payload["thinking_outline"],
            "handoff_plan": self.payload["handoff_plan"],
            "todo_list": self.payload["todo_list"],
            "difficulty": self.payload["difficulty"],
            "tags": self.payload["tags"],
            "acceptance_criteria": self.payload["acceptance_criteria"],
            "metrics": self.payload["metrics"],
        }
        prompt = PROMPT_TEMPLATE.format(user_query=self.user_query.strip())
        completion = json.dumps(completion_obj, ensure_ascii=False)
        return {"prompt": prompt, "completion": completion}


def iter_records(path: Path) -> Iterator[RouterRecord]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield RouterRecord.from_json(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_records(records: Sequence[RouterRecord], val_ratio: float, test_ratio: float) -> tuple[List[RouterRecord], List[RouterRecord], List[RouterRecord]]:
    assert 0 <= val_ratio < 1 and 0 <= test_ratio < 1, "ratios must be within [0,1)"
    assert val_ratio + test_ratio < 1, "train split would be empty"
    total = len(records)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    train = records[: total - val_count - test_count]
    val = records[total - val_count - test_count : total - test_count]
    test = records[total - test_count :]
    return train, val, test


def maybe_upload(local_path: Path, gcs_prefix: Optional[str]) -> Optional[str]:
    if not gcs_prefix:
        return None
    if storage is None:
        raise RuntimeError("google-cloud-storage is required to upload to GCS.")
    bucket_name, _, prefix = gcs_prefix.partition("/")
    if not bucket_name:
        raise ValueError(f"Invalid gcs prefix: {gcs_prefix}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_path = f"{prefix.rstrip('/')}/{local_path.name}" if prefix else local_path.name
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source JSONL from week-2 pipeline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "vertex_tuning",
        help="Directory for generated JSONL files.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic shuffle seed.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of examples for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Fraction of examples for hold-out testing.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of records (useful for smoke tests).")
    parser.add_argument("--gcs-prefix", type=str, default="", help="Optional gs://bucket/prefix to upload the splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(iter_records(args.input))
    if args.limit:
        records = records[: args.limit]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    train, val, test = split_records(records, args.val_ratio, args.test_ratio)

    output_dir: Path = args.output_dir
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"

    write_jsonl(train_path, (rec.to_example() for rec in train))
    write_jsonl(val_path, (rec.to_example() for rec in val))
    write_jsonl(test_path, (rec.to_example() for rec in test))

    gcs_uris = {}
    for path in (train_path, val_path, test_path):
        uri = maybe_upload(path, args.gcs_prefix or None)
        if uri:
            gcs_uris[path.name] = uri

    summary = {
        "input": str(args.input),
        "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "local_paths": {p.name: str(p) for p in (train_path, val_path, test_path)},
        "gcs_uploads": gcs_uris,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
