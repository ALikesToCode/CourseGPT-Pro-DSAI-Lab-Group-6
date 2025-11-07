from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable must be set for private router checkpoints.")

ROUTER_SYSTEM_PROMPT = """You are the Router Agent coordinating Math, Code, and General-Search specialists.\nEmit ONLY strict JSON with keys route_plan, route_rationale, expected_artifacts,\nthinking_outline, handoff_plan, todo_list, difficulty, tags, acceptance_criteria, metrics.\nEach route_plan entry must be a tool call (e.g., /math(...), /code(...), /general-search(...)).\nBe concise but precise. Do not include prose outside of the JSON object."""

MODELS = {
    "Router-Qwen3-32B-8bit": {
        "repo_id": "Alovestocode/router-qwen3-32b-merged",
        "description": "Router checkpoint on Qwen3 32B merged and quantized for 8-bit ZeroGPU inference.",
        "params_b": 32.0,
    },
    "Router-Gemma3-27B-8bit": {
        "repo_id": "Alovestocode/router-gemma3-merged",
        "description": "Router checkpoint on Gemma3 27B merged and quantized for 8-bit ZeroGPU inference.",
        "params_b": 27.0,
    },
}

REQUIRED_KEYS = [
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

PIPELINES: Dict[str, Any] = {}


def load_pipeline(model_name: str):
    if model_name in PIPELINES:
        return PIPELINES[model_name]

    repo = MODELS[model_name]["repo_id"]
    tokenizer = AutoTokenizer.from_pretrained(repo, token=HF_TOKEN)

    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        pipe = pipeline(
            task="text-generation",
            model=repo,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",
            model_kwargs={"quantization_config": quantization_config},
            use_cache=True,
            token=HF_TOKEN,
        )
        PIPELINES[model_name] = pipe
        return pipe
    except Exception as exc:
        print(f"8-bit load failed for {repo}: {exc}. Falling back to higher precision.")

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        try:
            pipe = pipeline(
                task="text-generation",
                model=repo,
                tokenizer=tokenizer,
                trust_remote_code=True,
                device_map="auto",
                dtype=dtype,
                use_cache=True,
                token=HF_TOKEN,
            )
            PIPELINES[model_name] = pipe
            return pipe
        except Exception:
            continue

    pipe = pipeline(
        task="text-generation",
        model=repo,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        use_cache=True,
        token=HF_TOKEN,
    )
    PIPELINES[model_name] = pipe
    return pipe


def build_router_prompt(
    user_task: str,
    context: str,
    acceptance: str,
    extra_guidance: str,
    difficulty: str,
    tags: str,
) -> str:
    prompt_parts = [ROUTER_SYSTEM_PROMPT.strip(), "\n### Router Inputs\n"]
    prompt_parts.append(f"Difficulty: {difficulty or 'intermediate'}")
    prompt_parts.append(f"Tags: {tags or 'general'}")
    if acceptance.strip():
        prompt_parts.append(f"Acceptance criteria: {acceptance.strip()}")
    if extra_guidance.strip():
        prompt_parts.append(f"Additional guidance: {extra_guidance.strip()}")
    if context.strip():
        prompt_parts.append("\n### Supporting context\n" + context.strip())
    prompt_parts.append("\n### User task\n" + user_task.strip())
    prompt_parts.append("\nReturn only JSON.")
    return "\n".join(prompt_parts)


def extract_json_from_text(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("Router output did not contain a JSON object.")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Router output JSON appears truncated.")


def validate_router_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in plan:
            issues.append(f"Missing key: {key}")
    route_plan = plan.get("route_plan")
    if not isinstance(route_plan, list) or not route_plan:
        issues.append("route_plan must be a non-empty list of tool calls")
    metrics = plan.get("metrics")
    if not isinstance(metrics, dict):
        issues.append("metrics must be an object containing primary/secondary entries")
    todo = plan.get("todo_list")
    if not isinstance(todo, list) or not todo:
        issues.append("todo_list must contain at least one checklist item")
    return len(issues) == 0, issues


def format_validation_message(ok: bool, issues: List[str]) -> str:
    if ok:
        return "✅ Router plan includes all required fields."
    bullets = "\n".join(f"- {issue}" for issue in issues)
    return f"❌ Issues detected:\n{bullets}"


@spaces.GPU(duration=600)
def generate_router_plan(
    user_task: str,
    context: str,
    acceptance: str,
    extra_guidance: str,
    difficulty: str,
    tags: str,
    model_choice: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, Dict[str, Any], str, str]:
    if not user_task.strip():
        raise gr.Error("User task is required.")
    
    if model_choice not in MODELS:
        raise gr.Error(f"Invalid model choice: {model_choice}. Available: {list(MODELS.keys())}")

    try:
        prompt = build_router_prompt(
            user_task=user_task,
            context=context,
            acceptance=acceptance,
            extra_guidance=extra_guidance,
            difficulty=difficulty,
            tags=tags,
        )

        generator = load_pipeline(model_choice)
        result = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )[0]["generated_text"]

        completion = result[len(prompt) :].strip() if result.startswith(prompt) else result.strip()

        try:
            json_block = extract_json_from_text(completion)
            plan = json.loads(json_block)
            ok, issues = validate_router_plan(plan)
            validation_msg = format_validation_message(ok, issues)
        except Exception as exc:
            plan = {}
            validation_msg = f"❌ JSON parsing failed: {exc}"

        return completion, plan, validation_msg, prompt
    except Exception as exc:
        error_msg = f"❌ Generation failed: {str(exc)}"
        return "", {}, error_msg, ""


def clear_outputs():
    return "", {}, "Awaiting generation.", ""


def build_ui():
    description = "Use the CourseGPT-Pro router checkpoints (Gemma3/Qwen3) hosted on ZeroGPU to generate structured routing plans."
    with gr.Blocks(theme=gr.themes.Soft(), css="""
        textarea { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
        .status-ok { color: #0d9488; font-weight: 600; }
        .status-bad { color: #dc2626; font-weight: 600; }
    """) as demo:
        gr.Markdown("# 🛰️ Router Control Room — ZeroGPU" )
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=3):
                user_task = gr.Textbox(
                    label="User Task / Problem Statement",
                    placeholder="Describe the homework-style query that needs routing...",
                    lines=8,
                    value="Explain how to solve a constrained optimization homework problem that mixes calculus and coding steps.",
                )
                context = gr.Textbox(
                    label="Supporting Context (optional)",
                    placeholder="Paste any retrieved evidence, PDFs, or rubric notes.",
                    lines=4,
                )
                acceptance = gr.Textbox(
                    label="Acceptance Criteria",
                    placeholder="Bullet list of 'definition of done' checks.",
                    lines=3,
                    value="- Provide citations for every claim.\n- Ensure /math verifies /code output.",
                )
                extra_guidance = gr.Textbox(
                    label="Additional Guidance",
                    placeholder="Special constraints, tools to avoid, etc.",
                    lines=3,
                )
            with gr.Column(scale=2):
                model_choice = gr.Dropdown(
                    label="Router Checkpoint",
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0] if MODELS else None,
                    allow_custom_value=False,
                )
                difficulty = gr.Radio(
                    label="Difficulty Tier",
                    choices=["introductory", "intermediate", "advanced"],
                    value="advanced",
                    interactive=True,
                )
                tags = gr.Textbox(
                    label="Tags",
                    placeholder="Comma-separated e.g. calculus, optimization, python",
                    value="calculus, optimization, python",
                )
                max_new_tokens = gr.Slider(256, 1024, value=640, step=32, label="Max New Tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

        generate_btn = gr.Button("Generate Router Plan", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Row():
            raw_output = gr.Textbox(label="Raw Model Output", lines=12)
            plan_json = gr.JSON(label="Parsed Router Plan")
        validation_msg = gr.Markdown("Awaiting generation.")
        prompt_view = gr.Textbox(label="Full Prompt", lines=10)

        generate_btn.click(
            generate_router_plan,
            inputs=[
                user_task,
                context,
                acceptance,
                extra_guidance,
                difficulty,
                tags,
                model_choice,
                max_new_tokens,
                temperature,
                top_p,
            ],
            outputs=[raw_output, plan_json, validation_msg, prompt_view],
        )

        clear_btn.click(fn=clear_outputs, outputs=[raw_output, plan_json, validation_msg, prompt_view])

    return demo


demo = build_ui()

if __name__ == "__main__":  # pragma: no cover
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
