#!/usr/bin/env python3
"""OCR evaluation template for Milestone 5.

Fill in the TODO sections with actual OCR models/pipelines. This script is
intended to:
1. Detect whether inputs contain text (for images or pages extracted from PDFs/DOCX).
2. Extract recognised text when present.
3. Emit quality metrics (e.g., OCR confidence, character/word error rate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 5 OCR evaluation harness.")
    parser.add_argument("--input-dir", required=True, help="Folder containing images/PDF/DOCX assets.")
    parser.add_argument("--ground-truth", help="Optional path to JSON/CSV with reference text for metric computation.")
    parser.add_argument("--out", default="ocr_eval_results.json", help="Destination JSON report.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional limit for quick smoke tests.")
    return parser.parse_args()


# --- Placeholder implementations -------------------------------------------------

def detect_text_presence(file_path: Path) -> Tuple[bool, float]:
    """Return whether the document likely contains text plus a confidence score.

    Replace this stub with actual logic (e.g., vision model classifier,
    heuristics based on OCR engine confidence, etc.).
    """
    # TODO: Implement detection.
    return True, 0.0


def extract_text(file_path: Path) -> str:
    """Extract raw text from the asset (image/PDF/DOCX).

    Replace with calls to Tesseract, PaddleOCR, Azure Vision, etc.
    Handle multi-page documents by concatenating text in reading order.
    """
    # TODO: Implement extraction.
    return ""


def compute_quality_metrics(pred_text: str, ref_text: str | None) -> Dict[str, Any]:
    """Compute evaluation metrics for the OCR output.

    Replace with real metrics such as CER/WER once references are available.
    """
    metrics: Dict[str, Any] = {}
    if ref_text is not None:
        # TODO: compute CER/WER; placeholder returns perfect score.
        metrics["char_error_rate"] = 0.0
        metrics["word_error_rate"] = 0.0
    return metrics


# --- Pipeline orchestration ------------------------------------------------------

def load_references(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    ref_map: Dict[str, str] = {}
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        data = json.loads(path_obj.read_text())
        ref_map = {item["id"]: item["text"] for item in data}
    else:
        raise NotImplementedError("Only JSON reference files are supported in this template.")
    return ref_map


def iter_documents(input_dir: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.pdf", "*.docx"):
        yield from input_dir.rglob(ext)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    reference_map = load_references(args.ground_truth)

    records: list[Dict[str, Any]] = []
    for idx, path in enumerate(iter_documents(input_dir)):
        if args.max_samples and idx >= args.max_samples:
            break

        has_text, confidence = detect_text_presence(path)
        extracted_text = extract_text(path) if has_text else ""
        ref_text = reference_map.get(path.stem)
        metrics = compute_quality_metrics(extracted_text, ref_text)

        records.append(
            {
                "file": str(path),
                "has_text": has_text,
                "confidence": confidence,
                "extracted_text": extracted_text if len(extracted_text) <= 256 else extracted_text[:256] + "...",
                "metrics": metrics,
            }
        )

    report = {"input_dir": str(input_dir), "samples": len(records), "results": records}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote OCR evaluation report to {args.out}")


if __name__ == "__main__":
    main()
