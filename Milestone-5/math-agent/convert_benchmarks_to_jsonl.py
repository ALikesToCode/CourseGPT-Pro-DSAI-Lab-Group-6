import json
import os
import argparse
from typing import Iterable, Dict, Any


def process_single_choice(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert single-choice benchmark entry to target format."""
    question_text = str(entry.get("question", "")).strip()
    options = entry.get("options", []) or []

    content = (
        f"{question_text}\n\nOptions:\n"
        + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        + "\n\nSelect the correct answer."
    )

    out = {
        "body": {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000,
        }
    }
    # Optionally include label if present
    if "answer" in entry:
        out["label"] = entry["answer"]
    return out


def process_close_question(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert close (direct-answer) benchmark entry to target format."""
    question_text = str(entry.get("question", "")).strip()
    out = {
        "body": {
            "messages": [{"role": "user", "content": question_text}],
            "max_tokens": 1000,
        }
    }
    if "answer" in entry:
        out["label"] = entry["answer"]
    return out


def iter_entries_from_json_file(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a .json or .jsonl file.

    - For .json: expect a list of entries (or a dict with a top-level key containing a list).
    - For .jsonl: parse each line as a JSON object.
    """
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # skip bad lines but continue
                    continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # if dict with a single top-level list, try to find the list
            if isinstance(data, list):
                for entry in data:
                    yield entry
            elif isinstance(data, dict):
                # find first list-valued key
                for v in data.values():
                    if isinstance(v, list):
                        for entry in v:
                            yield entry
                        break


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark datasets to combined JSONL for evaluation")
    parser.add_argument("--input-dir", "-i", default="benchmarks_dataset", help="Path to folder containing benchmark JSON/JSONL files (default: benchmarks_dataset)")
    parser.add_argument("--output-file", "-o", default="combined_benchmark.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    if not os.path.exists(input_dir):
        raise SystemExit(f"ERROR: input directory not found: {input_dir}\nPlease check the path and try again.")

    combined_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if not (filename.endswith(".json") or filename.endswith(".jsonl")):
                    continue
                file_path = os.path.join(root, filename)
                print(f"Processing {file_path}...")
                for entry in iter_entries_from_json_file(file_path):
                    try:
                        if "options" in entry and isinstance(entry.get("options"), list):
                            record = process_single_choice(entry)
                        else:
                            record = process_close_question(entry)
                    except Exception:
                        # Skip malformed entries but continue
                        continue
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    combined_count += 1

    print(f"\n✅ Combined {combined_count} entries written to '{output_file}'")


if __name__ == "__main__":
    main()
