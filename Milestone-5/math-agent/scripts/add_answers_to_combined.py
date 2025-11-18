#!/usr/bin/env python3
"""
Add answers to a combined JSONL file by matching the question text
to source entries in `benchmarks_dataset`.

Usage:
  python add_answers_to_combined.py \
    --combined <path/to/combined_benchmarks_648.jsonl> \
    --benchmarks-dir <path/to/benchmarks_dataset> \
    --out <path/to/output.jsonl>

This script writes a new JSONL where each record gains an `"answer"`
field (if a match is found). The matching uses the same formatting
logic as `generate_equal_combined_benchmarks.py` to append options.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import difflib
import re


def build_question_text(item: dict) -> str:
    q = item.get("question", "")
    opts = item.get("options")
    if opts:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = ["\n\nOptions:"]
        for i, opt in enumerate(opts):
            label = letters[i] if i < len(letters) else f"({i})"
            lines.append(f"{label}. {opt}")
        q = q + "\n" + "\n".join(lines)
    return q


def normalize_answer(item: dict) -> Optional[str]:
    # Try common keys
    candidates = ["answer", "label", "correct_answer", "correct", "answer_key", "solution"]
    for k in candidates:
        if k in item and item[k] not in (None, ""):
            val = item[k]
            # If it's an index (int) and options exist, resolve to option text
            if isinstance(val, int) and isinstance(item.get("options"), list):
                idx = val
                opts = item.get("options")
                if 0 <= idx < len(opts):
                    return opts[idx]
            # If it's a string like 'A' or 'B', map to option
            if isinstance(val, str) and len(val) == 1 and val.isalpha() and isinstance(item.get("options"), list):
                idx = ord(val.upper()) - ord('A')
                opts = item.get("options")
                if 0 <= idx < len(opts):
                    return opts[idx]
            # If it's a list, join
            if isinstance(val, list):
                return "; ".join(map(str, val))
            # Else return stringified
            return str(val)
    # fallback: sometimes single_choice uses 'options' + 'answer_index'
    if "answer_index" in item and isinstance(item.get("options"), list):
        idx = item["answer_index"]
        opts = item.get("options")
        if isinstance(idx, int) and 0 <= idx < len(opts):
            return opts[idx]
    return None


def _normalize_text_for_match(s: str) -> str:
    # collapse whitespace and lowercase for more robust matching
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def build_mapping(benchmarks_dir: Path, recursive: bool = False) -> Dict[str, str]:
    mapping = {}
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"benchmarks dir not found: {benchmarks_dir}")
    # walk either direct children or recursively
    if recursive:
        files = sorted(benchmarks_dir.rglob("*.jsonl"))
    else:
        files = []
        for p in sorted(benchmarks_dir.iterdir()):
            if not p.is_dir():
                continue
            for filename in ("cloze_en.jsonl", "single_choice_en.jsonl"):
                fp = p / filename
                if fp.exists():
                    files.append(fp)

    for fp in files:
        name = fp.name
        if name not in ("cloze_en.jsonl", "single_choice_en.jsonl"):
            # skip unrelated jsonl files when recursive
            continue
        try:
            with fp.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qtext = build_question_text(obj)
                    ans = normalize_answer(obj)
                    if ans is not None:
                        key = _normalize_text_for_match(qtext)
                        mapping[key] = ans
        except Exception:
            # skip files we can't read
            continue
    return mapping


def _best_fuzzy_match(key: str, mapping_keys: list, threshold: float = 0.85):
    best = None
    best_score = 0.0
    for k in mapping_keys:
        score = difflib.SequenceMatcher(None, key, k).ratio()
        if score > best_score:
            best_score = score
            best = k
    if best_score >= threshold:
        return best, best_score
    return None, best_score


def process(combined_path: Path, benchmarks_dir: Path, out_path: Path, recursive: bool = False, fuzzy: bool = False, threshold: float = 0.85):
    mapping = build_mapping(benchmarks_dir, recursive=recursive)
    total = 0
    matched = 0
    with combined_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                # write unchanged
                fout.write(raw + "\n")
                continue
            # locate user message content (should be last user message)
            qtext = None
            try:
                msgs = rec.get("body", {}).get("messages", [])
                for m in reversed(msgs):
                    if m.get("role") == "user":
                        qtext = m.get("content", "")
                        break
            except Exception:
                qtext = None
            answer = None
            if qtext is not None:
                key = _normalize_text_for_match(qtext)
                if key in mapping:
                    answer = mapping[key]
                    matched += 1
                elif fuzzy and mapping:
                    best, score = _best_fuzzy_match(key, list(mapping.keys()), threshold=threshold)
                    if best:
                        answer = mapping[best]
                        matched += 1
            # attach answer at top-level
            if answer is not None:
                rec["answer"] = answer
            else:
                rec["answer"] = None
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Processed {total} records, matched answers for {matched} records ({matched}/{total}).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", required=True, help="Path to combined JSONL file")
    parser.add_argument("--benchmarks-dir", default="benchmarks_dataset", help="Path to benchmarks_dataset folder")
    parser.add_argument("--out", help="Output path (default: add .with_answers.jsonl suffix)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search benchmarks_dir for jsonl files")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching when exact match fails")
    parser.add_argument("--threshold", type=float, default=0.85, help="Fuzzy matching threshold (0-1)")
    args = parser.parse_args()

    combined = Path(args.combined)
    if not combined.exists():
        raise FileNotFoundError(f"combined file not found: {combined}")
    bdir = Path(args.benchmarks_dir)
    if not bdir.exists():
        # try relative to combined parent
        alt = combined.resolve().parent / args.benchmarks_dir
        if alt.exists():
            bdir = alt
        else:
            raise FileNotFoundError(f"benchmarks dir not found: {bdir}")

    if args.out:
        out = Path(args.out)
    else:
        out = combined.with_name(combined.stem + ".with_answers.jsonl")

    process(combined, bdir, out, recursive=args.recursive, fuzzy=args.fuzzy, threshold=args.threshold)


if __name__ == "__main__":
    main()
