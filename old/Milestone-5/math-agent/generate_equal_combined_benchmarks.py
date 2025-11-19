#!/usr/bin/env python3
"""
Generate a combined JSONL file with at most 650 lines, sampled equally from each dataset folder
under benchmarks_dataset. Single-choice questions will have their options appended to the question
text as required. Output file: combined_benchmarks_equal.jsonl (in the same folder).

Usage: python3 generate_equal_combined_benchmarks.py
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent / "benchmarks_dataset"
OUT = Path(__file__).resolve().parent / "combined_benchmarks_equal.jsonl"
TOTAL = 1700

SYSTEM_PROMPT = (
    "You are a helpful math assistant. You are given a question which may have multiple parts and "
    "also some worked out solution steps. Decide whether the final answer is correct. Output only "
    "a JSON object with the fields:\n- question (the question exactly as given)\n- predictions: "
    "an array of labels where each label has the format {\"label\":<string>,\"score\":<number>} "
    "where score is a float between 0 and 1. Output must be valid JSON."
)


def find_source_files(base: Path):
    """Return a list of (folder_name, file_path) for folders under base that contain cloze_en.jsonl
    or single_choice_en.jsonl. Sorted by folder name for deterministic behavior."""
    out = []
    if not base.exists():
        raise FileNotFoundError(f"benchmarks_dataset folder not found at {base}")
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        cloze = p / "cloze_en.jsonl"
        sc = p / "single_choice_en.jsonl"
        if cloze.exists():
            out.append((p.name, cloze))
        elif sc.exists():
            out.append((p.name, sc))
        else:
            # skip folders without expected files
            continue
    return out


def build_question_text(item: dict):
    q = item.get("question", "")
    opts = item.get("options")
    if opts:
        # append options in the requested format
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = ["\n\nOptions:"]
        for i, opt in enumerate(opts):
            label = letters[i] if i < len(letters) else f"({i})"
            lines.append(f"{label}. {opt}")
        q = q + "\n" + "\n".join(lines)
    return q


def main():
    sources = find_source_files(BASE)
    if not sources:
        print("No source files found under", BASE)
        return

    n_folders = len(sources)
    base = TOTAL // n_folders
    rem = TOTAL % n_folders
    counts = [base + (1 if i < rem else 0) for i in range(n_folders)]

    print(f"Found {n_folders} folders. Sampling counts per folder: {counts} (total {sum(counts)})")

    # create combined output
    with OUT.open("w", encoding="utf-8") as fout:
        summary = []
        for (folder_name, path), needed in zip(sources, counts):
            taken = 0
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # skip malformed lines
                        continue
                    qtext = build_question_text(obj)
                    # try to extract an answer from common keys present in the source items
                    answer = obj.get("answer", obj.get("answers"))
                    if answer is None:
                        for _k in ("label", "correct", "target", "answer_index"):
                            if _k in obj:
                                answer = obj[_k]
                                break

                    record = {
                        "body": {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": qtext},
                            ]
                        },
                        "temperature": 0,
                        "answer": answer,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    taken += 1
                    if taken >= needed:
                        break
            summary.append((folder_name, taken, path))

    # print summary
    print(f"Wrote combined JSONL to {OUT}")
    for name, taken, path in summary:
        print(f"  {name}: {taken} from {path}")


if __name__ == "__main__":
    main()
