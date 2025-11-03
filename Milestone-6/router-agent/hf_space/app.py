from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import gradio as gr

# Ensure Milestone 5 evaluation utilities are importable when running inside the Space.
REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "Milestone-5" / "router-agent"
if EVAL_DIR.exists():
    sys.path.insert(0, str(EVAL_DIR))

try:
    from schema_score import (  # type: ignore
        run_schema_evaluation,
        tool_sequence,
        todo_covers_all_tools,
        todo_tool_alignment,
    )
except Exception as exc:  # pragma: no cover - handled gracefully in UI.
    run_schema_evaluation = None
    tool_sequence = None
    todo_covers_all_tools = None
    todo_tool_alignment = None
    SCHEMA_IMPORT_ERROR = str(exc)
else:
    SCHEMA_IMPORT_ERROR = ""

try:
    from router_benchmark_runner import (  # type: ignore
        load_thresholds,
        evaluate_thresholds,
    )
except Exception as exc:  # pragma: no cover
    load_thresholds = None
    evaluate_thresholds = None
    THRESHOLD_IMPORT_ERROR = str(exc)
else:
    THRESHOLD_IMPORT_ERROR = ""

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover
    InferenceClient = None  # type: ignore


HF_ROUTER_REPO = os.environ.get("HF_ROUTER_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN")

BENCH_GOLD_PATH = EVAL_DIR / "benchmarks" / "router_benchmark_hard.jsonl"
THRESHOLDS_PATH = EVAL_DIR / "router_benchmark_thresholds.json"

client = None
if HF_ROUTER_REPO and InferenceClient is not None:
    try:
        client = InferenceClient(model=HF_ROUTER_REPO, token=HF_TOKEN)
    except Exception as exc:  # pragma: no cover
        client = None
        ROUTER_LOAD_ERROR = str(exc)
    else:
        ROUTER_LOAD_ERROR = ""
else:
    ROUTER_LOAD_ERROR = "InferenceClient unavailable or HF_ROUTER_REPO unset."


SYSTEM_PROMPT = (
    "You are the Router Agent coordinating Math, Code, and General-Search specialists.\n"
    "Emit ONLY strict JSON with keys route_plan, route_rationale, expected_artifacts,\n"
    "thinking_outline, handoff_plan, todo_list, difficulty, tags, acceptance_criteria, metrics."
)


def load_sample_plan() -> Dict[str, Any]:
    try:
        if BENCH_GOLD_PATH.exists():
            first_line = BENCH_GOLD_PATH.read_text().splitlines()[0]
            record = json.loads(first_line)
            completion = json.loads(record["completion"])
            return completion
    except Exception:
        pass
    # Fallback minimal example.
    return {
        "route_plan": [
            "/general-search(query=\"site:arxiv.org meta-learning survey\", mode=web)",
            "/math(Outline a theoretical summary of Model-Agnostic Meta-Learning (MAML) and explain the inner/outer-loop updates.)",
            "/code(Implement a minimal MAML pseudo-code example to clarify the algorithm flow., using Python)",
        ],
        "route_rationale": (
            "Search surfaces authoritative meta-learning references; "
            "math distills the theory; code converts the derivation into an executable sketch."
        ),
        "expected_artifacts": [
            "Three bullet summary of seminal MAML papers.",
            "Equation block describing the meta-gradient.",
            "`maml_pseudocode.py` script with comments.",
        ],
        "thinking_outline": [
            "1. Gather citations describing MAML.",
            "2. Express the loss formulation and gradient steps.",
            "3. Provide annotated pseudo-code for the inner/outer loop.",
        ],
        "handoff_plan": "/general-search -> /math -> /code -> router QA",
        "todo_list": [
            "- [ ] /general-search: Collect recent survey or benchmark sources for MAML.",
            "- [ ] /math: Write the meta-objective and gradient derivation.",
            "- [ ] /code: Produce pseudo-code and comment on hyperparameters.",
            "- [ ] router QA: Ensure JSON schema compliance and cite sources.",
        ],
        "difficulty": "intermediate",
        "tags": ["meta-learning", "few-shot-learning"],
        "acceptance_criteria": [
            "- Includes at least two citations to reputable sources.",
            "- Meta-gradient expression matches the pseudo-code implementation.",
            "- JSON validates against the router schema.",
        ],
        "metrics": {
            "primary": ["Route accuracy >= 0.8 on benchmark."],
            "secondary": ["Report token count and inference latency."],
        },
    }


SAMPLE_PLAN = load_sample_plan()


def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    try:
        start = raw_text.index("{")
        end = raw_text.rfind("}")
        candidate = raw_text[start : end + 1]
        return json.loads(candidate)
    except Exception as exc:
        raise ValueError(f"Router output is not valid JSON: {exc}") from exc


def call_router_model(user_query: str) -> Dict[str, Any]:
    if client is None:
        return SAMPLE_PLAN

    prompt = f"{SYSTEM_PROMPT}\n\nUser query:\n{user_query.strip()}\n"
    try:
        raw = client.text_generation(
            prompt,
            max_new_tokens=900,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        return extract_json_from_text(raw)
    except Exception as exc:  # pragma: no cover
        return {
            "error": f"Router call failed ({exc}). Falling back to sample plan.",
            "sample_plan": SAMPLE_PLAN,
        }


def generate_plan(user_query: str) -> Dict[str, Any]:
    if not user_query.strip():
        raise gr.Error("Please provide a user query to route.")
    plan = call_router_model(user_query)
    return plan


def compute_structural_metrics(plan: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    route_plan = plan.get("route_plan", [])
    if tool_sequence is not None and isinstance(route_plan, list):
        tools = tool_sequence(route_plan)
        todo_list = plan.get("todo_list", []) if isinstance(plan.get("todo_list"), list) else []
        if todo_tool_alignment is not None:
            metrics["todo_tool_alignment"] = todo_tool_alignment(todo_list, tools)
        if todo_covers_all_tools is not None:
            metrics["todo_covers_all_tools"] = todo_covers_all_tools(todo_list, tools)
        handoff = plan.get("handoff_plan", "")
        metrics["handoff_mentions_all_tools"] = all(
            tool.lower() in (handoff or "").lower() for tool in tools
        )
    metrics["expected_artifacts_count"] = len(plan.get("expected_artifacts", []) or [])
    metrics["acceptance_criteria_count"] = len(plan.get("acceptance_criteria", []) or [])
    return metrics


def validate_plan(plan_input: Any) -> Dict[str, Any]:
    if isinstance(plan_input, str):
        try:
            plan = json.loads(plan_input)
        except json.JSONDecodeError as exc:
            return {"valid": False, "errors": [f"Invalid JSON: {exc}"]}
    else:
        plan = plan_input or {}
    errors = []
    required_keys = [
        "route_plan",
        "route_rationale",
        "expected_artifacts",
        "thinking_outline",
        "handoff_plan",
        "todo_list",
        "difficulty",
        "tags",
        "acceptance_criteria",
        "metrics",
    ]
    for key in required_keys:
        if key not in plan:
            errors.append(f"Missing required field: {key}")
    route_plan = plan.get("route_plan")
    if not isinstance(route_plan, list) or not route_plan:
        errors.append("route_plan must be a non-empty list of tool invocations.")
    else:
        for step in route_plan:
            if not isinstance(step, str):
                errors.append("Each route_plan entry must be a string.")
                break
    todo_list = plan.get("todo_list")
    if todo_list is not None and not isinstance(todo_list, list):
        errors.append("todo_list must be a list of strings.")
    metrics_block = plan.get("metrics")
    if metrics_block is not None and not isinstance(metrics_block, dict):
        errors.append("metrics must be a dictionary with primary/secondary lists.")

    structural = compute_structural_metrics(plan)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "structural_metrics": structural,
        "tool_count": len(route_plan) if isinstance(route_plan, list) else 0,
    }


def benchmark_predictions(pred_file: Any) -> Dict[str, Any]:
    if run_schema_evaluation is None or load_thresholds is None or evaluate_thresholds is None:
        return {
            "success": False,
            "error": "Benchmark utilities are unavailable.",
            "schema_import_error": SCHEMA_IMPORT_ERROR,
            "threshold_import_error": THRESHOLD_IMPORT_ERROR,
        }
    if not BENCH_GOLD_PATH.exists():
        return {
            "success": False,
            "error": f"Benchmark gold file missing: {BENCH_GOLD_PATH}",
        }
    if not THRESHOLDS_PATH.exists():
        return {
            "success": False,
            "error": f"Thresholds file missing: {THRESHOLDS_PATH}",
        }

    if pred_file is None:
        return {"success": False, "error": "Upload a .jsonl predictions file first."}

    if hasattr(pred_file, "name"):
        pred_path = Path(pred_file.name)
    elif isinstance(pred_file, str):
        pred_path = Path(pred_file)
    else:
        # Save uploaded bytes to a temp file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            tmp.write(pred_file.read())
            pred_path = Path(tmp.name)

    try:
        schema_report = run_schema_evaluation(
            str(BENCH_GOLD_PATH),
            str(pred_path),
            max_error_examples=10,
        )
    except Exception as exc:
        return {"success": False, "error": f"Schema evaluation failed: {exc}"}

    try:
        thresholds = load_thresholds(THRESHOLDS_PATH)
        threshold_results = evaluate_thresholds(schema_report["metrics"], thresholds)
    except Exception as exc:
        return {"success": False, "error": f"Threshold comparison failed: {exc}"}

    return {
        "success": True,
        "overall_pass": threshold_results.get("overall_pass"),
        "schema_metrics": schema_report["metrics"],
        "threshold_results": threshold_results,
        "error_samples": schema_report.get("error_samples", []),
    }


def describe_router_backend() -> str:
    if client is None:
        return f"Router backend not initialised. {ROUTER_LOAD_ERROR}"
    return f"Using Hugging Face Inference endpoint: `{HF_ROUTER_REPO}`"


with gr.Blocks(title="CourseGPT Router Control Room") as demo:
    gr.Markdown(
        "## CourseGPT Router Control Room\n"
        "Milestone 6 deployment scaffold for the router agent. Populate the router model "
        "environment variables to enable live inference, or rely on the bundled sample plan."
    )

    gr.Markdown(f"**Backend status:** {describe_router_backend()}")

    with gr.Tab("Router Planner"):
        user_query = gr.Textbox(
            label="User query",
            lines=8,
            placeholder="Describe the task that needs routing...",
        )
        generate_btn = gr.Button("Generate plan", variant="primary")
        plan_output = gr.JSON(label="Router plan")
        generate_btn.click(fn=generate_plan, inputs=user_query, outputs=plan_output)

        validate_btn = gr.Button("Run structural checks")
        validation_output = gr.JSON(label="Validation summary")
        validate_btn.click(fn=validate_plan, inputs=plan_output, outputs=validation_output)

    with gr.Tab("Benchmark"):
        gr.Markdown(
            "Upload a JSONL file of router predictions (one JSON object per line). "
            "The file must align with the `router_benchmark_hard.jsonl` gold split."
        )
        predictions_file = gr.File(label="Predictions (.jsonl)", file_types=[".jsonl"])
        benchmark_btn = gr.Button("Evaluate against thresholds", variant="primary")
        benchmark_output = gr.JSON(label="Benchmark report")
        benchmark_btn.click(fn=benchmark_predictions, inputs=predictions_file, outputs=benchmark_output)

    with gr.Tab("Docs & TODO"):
        gr.Markdown(
            "- Populate `/math`, `/code`, `/general-search` agent hooks for live orchestration.\n"
            "- Add citations and latency logging once the production router is connected.\n"
            "- Link to Milestone 5 benchmark reports and final project documentation."
        )

    demo.queue()


if __name__ == "__main__":  # pragma: no cover
    demo.launch()
