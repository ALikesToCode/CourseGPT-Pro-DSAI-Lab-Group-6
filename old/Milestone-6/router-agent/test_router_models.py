"""Smoke-test the router models listed in the HF Space configuration.

This script reuses the inference utilities from `hf_space/app.py`
so that the behaviour matches the deployed Space:
  * Adapters automatically attach to their base checkpoints.
  * Groq-style conversational providers fall back to the chat API.
  * JSON extraction mirrors the Space, catching malformed responses.

Example:
    python Milestone-6/router-agent/test_router_models.py --models "Gemma 3 27B Router Adapter"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

# Optional dependency: python-dotenv makes reading .env seamless.
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fall back if dotenv absent.
    def load_dotenv(*_: object, **__: object) -> bool:
        return False

# Load .env file if present so HF_TOKEN/HUGGINGFACEHUB_API_TOKEN become available.
load_dotenv()


DEFAULT_PROMPT = (
    "Audit the router by producing a minimal plan that triages a debugging session "
    "for a failing machine learning pipeline. Focus on identifying next actions for "
    "code, math, and general-search specialists."
)


# Reuse the exact helper logic from the Space so we observe identical behaviour.
APP_MODULE_NAME = "router_space_app"
APP_PATH = Path(__file__).resolve().parent / "hf_space" / "app.py"
spec = importlib.util.spec_from_file_location(APP_MODULE_NAME, APP_PATH)
if spec is None or spec.loader is None:  # pragma: no cover - import errors surfaced to caller.
    print(f"Failed to load module spec from {APP_PATH}", file=sys.stderr)
    sys.exit(1)
router_app = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(router_app)  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover
    print(f"Failed to import router app module: {exc}", file=sys.stderr)
    sys.exit(1)


def detect_token() -> bool:
    return bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def iter_router_options(
    selected_labels: Iterable[str] | None,
) -> Iterable[Tuple[str, router_app.RouterOption]]:
    options: Dict[str, router_app.RouterOption] = router_app.AVAILABLE_ROUTER_OPTIONS
    if not selected_labels:
        for label, option in options.items():
            if not option.get("base"):
                # Skip the sample-plan placeholder.
                continue
            yield label, option
        return

    normalized = {label.lower(): label for label in options}
    for raw_label in selected_labels:
        key = raw_label.lower()
        if key in normalized:
            label = normalized[key]
            yield label, options[label]
            continue
        # Allow substring matches to reduce typing.
        matches = [label for label in options if key in label.lower()]
        if not matches:
            print(f"[WARN] No router option matched '{raw_label}'.", file=sys.stderr)
            continue
        for label in matches:
            yield label, options[label]


def run_probe(
    label: str,
    option: router_app.RouterOption,
    prompt: str,
    verbose: bool,
) -> Dict[str, Any]:
    start = time.time()
    result = router_app.call_router_model(prompt, option)
    duration = time.time() - start
    status = "ok"
    error = result.get("error")
    if error:
        status = "error"
    elif "route_plan" not in result:
        status = "invalid"
        error = "Response missing route_plan."
    elif not isinstance(result["route_plan"], list):
        status = "invalid"
        error = f"route_plan type mismatch: {type(result['route_plan']).__name__}"

    summary: Dict[str, Any] = {
        "label": label,
        "base": option.get("base"),
        "adapter": option.get("adapter"),
        "status": status,
        "latency_s": round(duration, 3),
    }
    if error:
        summary["error"] = error
    else:
        plan_preview = result["route_plan"][:2]  # type: ignore[index]
        summary["route_plan_preview"] = plan_preview
    if verbose:
        summary["full_response"] = result
    return summary


def pretty_print(report: Iterable[Dict[str, Any]]) -> None:
    print("=== Router Backend Smoke Test ===")
    for item in report:
        label = item["label"]
        base = item.get("base") or "N/A"
        adapter = item.get("adapter") or "None"
        status = item["status"]
        latency = item["latency_s"]
        print(f"\n[{label}]")
        print(f"  base    : {base}")
        print(f"  adapter : {adapter}")
        print(f"  status  : {status} (latency {latency:.3f}s)")
        if "error" in item:
            print(f"  error   : {item['error']}")
        else:
            preview = item.get("route_plan_preview") or []
            print(f"  preview : {json.dumps(preview, ensure_ascii=False)}")
        if "full_response" in item:
            print("  response:")
            print(json.dumps(item["full_response"], indent=2, ensure_ascii=False))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe router models via the HF Space logic.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Labels (or substrings) from the Space selector. Defaults to all real models.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to each router model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Dump the full JSON response for each model.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    if not detect_token():
        print(
            "[WARN] No HF token detected. Public or gated checkpoints may fail authentication.",
            file=sys.stderr,
        )
    report = [
        run_probe(label, option, args.prompt, args.verbose)
        for label, option in iter_router_options(args.models)
    ]
    pretty_print(report)
    failures = [item for item in report if item["status"] != "ok"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
