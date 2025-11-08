from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer, pipeline
from threading import Thread

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable must be set for private router checkpoints.")

PLAN_END_TOKEN = "<|end_of_plan|>"
STOP_SEQUENCES = [PLAN_END_TOKEN, "</json>", "</JSON>"]

ROUTER_SYSTEM_PROMPT = """You are the Router Agent coordinating Math, Code, and General-Search specialists.\nEmit EXACTLY ONE strict JSON object with keys route_plan, route_rationale, expected_artifacts,\nthinking_outline, handoff_plan, todo_list, difficulty, tags, acceptance_criteria, metrics.\nRules:\n- No markdown/code fences, no natural-language prologues or epilogues.\n- route_plan must be an ordered list of tool invocations such as /math(...), /code(...), /general-search(...).\n- todo_list must map each checklist item to the responsible tool.\n- metrics must include primary and secondary arrays (add optional *_guidance fields when they exist).\n- After the closing brace of the JSON object, immediately append the sentinel <|end_of_plan|>.\nExample output:\n{\n  "route_plan": ["/general-search(...)"],\n  "route_rationale": "...",\n  ...\n}<|end_of_plan|>\nReturn nothing else."""

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
TOKENIZER_CACHE: Dict[str, Any] = {}
WARMED_REMAINING = False
TOOL_PATTERN = re.compile(r"^/[a-z0-9_-]+\(.*\)$", re.IGNORECASE)


def get_tokenizer(repo: str):
    tok = TOKENIZER_CACHE.get(repo)
    if tok is not None:
        return tok
    tok = AutoTokenizer.from_pretrained(repo, token=HF_TOKEN)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    TOKENIZER_CACHE[repo] = tok
    return tok


def load_pipeline(model_name: str):
    if model_name in PIPELINES:
        return PIPELINES[model_name]

    repo = MODELS[model_name]["repo_id"]
    tokenizer = get_tokenizer(repo)

    try:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        pipe = pipeline(
            task="text-generation",
            model=repo,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",
            model_kwargs={"quantization_config": quant_config},
            use_cache=True,
            token=HF_TOKEN,
        )
        pipe.model.eval()
        PIPELINES[model_name] = pipe
        _schedule_background_warm(model_name)
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
            pipe.model.eval()
            PIPELINES[model_name] = pipe
            _schedule_background_warm(model_name)
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
    pipe.model.eval()
    PIPELINES[model_name] = pipe
    _schedule_background_warm(model_name)
    return pipe


def _schedule_background_warm(loaded_model: str) -> None:
    global WARMED_REMAINING
    if WARMED_REMAINING:
        return
    warm_remaining = os.environ.get("ROUTER_WARM_REMAINING", "1")
    if warm_remaining not in {"1", "true", "True"}:
        return

    remaining = [name for name in MODELS if name not in PIPELINES]
    if not remaining:
        WARMED_REMAINING = True
        return

    def _warm_all():
        for name in remaining:
            try:
                print(f"Background warm start for {name}")
                load_pipeline(name)
            except Exception as exc:  # pragma: no cover
                print(f"Warm start failed for {name}: {exc}")
        WARMED_REMAINING = True

    Thread(target=_warm_all, daemon=True).start()


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


def is_function_call(text: str) -> bool:
    return bool(TOOL_PATTERN.match(text.strip()))


def validate_router_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in plan:
            issues.append(f"Missing key: {key}")

    route_plan = plan.get("route_plan")
    if isinstance(route_plan, str) and is_function_call(route_plan):
        plan["route_plan"] = [route_plan]
        route_plan = plan["route_plan"]
    if not isinstance(route_plan, list) or not route_plan:
        issues.append("route_plan must be a non-empty list of tool calls")
    else:
        cleaned: List[str] = []
        for entry in route_plan:
            if isinstance(entry, str) and is_function_call(entry.strip().strip("'\"")):
                cleaned.append(entry.strip().strip("'\""))
            else:
                issues.append(f"route_plan entry is not a tool call: {entry}")
        if cleaned:
            plan["route_plan"] = cleaned

    metrics = plan.get("metrics")
    if not isinstance(metrics, dict):
        issues.append("metrics must be an object containing primary/secondary entries")
    todo = plan.get("todo_list")
    if not isinstance(todo, list) or not todo:
        issues.append("todo_list must contain at least one checklist item")
    else:
        cleaned_todo: List[str] = []
        for entry in todo:
            if isinstance(entry, str):
                text = entry.strip()
                if not text.startswith("- ["):
                    text = text.lstrip("- ")
                    text = f"- [ ] {text}"
                cleaned_todo.append(text)
            else:
                issues.append("todo_list entry must be a string")
        if cleaned_todo:
            plan["todo_list"] = cleaned_todo

    return len(issues) == 0, issues


def format_validation_message(ok: bool, issues: List[str]) -> str:
    if ok:
        return "✅ Router plan includes all required fields."
    bullets = "\n".join(f"- {issue}" for issue in issues)
    return f"❌ Issues detected:\n{bullets}"


@spaces.GPU(duration=600)
def generate_router_plan_streaming(
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
):
    """Generator function for streaming token output."""
    if not user_task.strip():
        yield "", {}, "❌ User task is required.", ""
        return
    
    if model_choice not in MODELS:
        yield "", {}, f"❌ Invalid model choice: {model_choice}. Available: {list(MODELS.keys())}", ""
        return

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
        
        # Get the underlying model and tokenizer
        model = generator.model
        tokenizer = generator.tokenizer
        
        # Set up streaming
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Start generation in a separate thread
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "streamer": streamer,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }

        def _generate():
            with torch.inference_mode():
                model.generate(**generation_kwargs)

        thread = Thread(target=_generate)
        thread.start()
        
        # Stream tokens
        completion = ""
        parsed_plan: Dict[str, Any] | None = None
        validation_msg = "🔄 Generating..."

        for new_text in streamer:
            completion += new_text
            chunk = completion
            finished = False
            display_plan = parsed_plan or {}

            chunk, finished = trim_at_stop_sequences(chunk)

            try:
                json_block = extract_json_from_text(chunk)
                candidate_plan = json.loads(json_block)
                ok, issues = validate_router_plan(candidate_plan)
                validation_msg = format_validation_message(ok, issues)
                parsed_plan = candidate_plan if ok else parsed_plan
                display_plan = candidate_plan
            except Exception:
                # Ignore until JSON is complete
                pass

            yield chunk, display_plan, validation_msg, prompt

            if finished:
                completion = chunk
                break

        # Final processing after streaming completes
        thread.join()

        completion = trim_at_stop_sequences(completion.strip())[0]
        if parsed_plan is None:
            try:
                json_block = extract_json_from_text(completion)
                parsed_plan = json.loads(json_block)
                ok, issues = validate_router_plan(parsed_plan)
                validation_msg = format_validation_message(ok, issues)
            except Exception as exc:
                parsed_plan = {}
                validation_msg = f"❌ JSON parsing failed: {exc}"

        yield completion, parsed_plan, validation_msg, prompt
        
    except Exception as exc:
        error_msg = f"❌ Generation failed: {str(exc)}"
        yield "", {}, error_msg, ""


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
                max_new_tokens = gr.Slider(256, 20000, value=16000, step=32, label="Max New Tokens")
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
            generate_router_plan_streaming,
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
            show_progress="full",
            api_name="/generate_router_plan_streaming",
        )

        clear_btn.click(
            fn=clear_outputs,
            outputs=[raw_output, plan_json, validation_msg, prompt_view],
            api_name="/clear_outputs",
        )

    return demo



def _prefetch_from_env() -> None:
    entries = os.environ.get("ROUTER_PREFETCH_MODELS")
    if entries:
        names = [item.strip() for item in entries.split(",") if item.strip()]
    else:
        single = os.environ.get("ROUTER_PREFETCH_MODEL")
        names = [single] if single else []

    if names == ["ALL"] or names == ["all"]:
        names = list(MODELS.keys())

    for name in names:
        if name not in MODELS:
            print(f"Prefetch skipped, unknown model: {name}")
            continue
        try:
            load_pipeline(name)
            print(f"Prefetched router model: {name}")
        except Exception as exc:  # pragma: no cover
            print(f"Prefetch failed for {name}: {exc}")


_prefetch_from_env()

demo = build_ui()

if __name__ == "__main__":  # pragma: no cover
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_api=True
    )
def trim_at_stop_sequences(text: str) -> Tuple[str, bool]:
    earliest = None
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        return text[:earliest], True
    return text, False

