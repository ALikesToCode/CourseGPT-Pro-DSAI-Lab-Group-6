from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import gradio as gr
import spaces
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:  # Load optional .env so Spaces and local runs behave the same.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_: object, **__: object) -> bool:
        return False


load_dotenv()


MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "600"))
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.9"))
USE_4BIT = os.environ.get("LOAD_IN_4BIT", "1") not in {"0", "false", "False"}

MODEL_FALLBACKS = [
    "Alovestocode/router-qwen3-32b-merged",
    "Alovestocode/router-gemma3-merged",
]


def _initialise_tokenizer() -> tuple[str, AutoTokenizer]:
    errors: dict[str, str] = {}
    candidates = []
    explicit = os.environ.get("MODEL_REPO")
    if explicit:
        candidates.append(explicit)
    for name in MODEL_FALLBACKS:
        if name not in candidates:
            candidates.append(name)
    for candidate in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(candidate, use_fast=False)
            print(f"Loaded tokenizer from {candidate}")
            return candidate, tok
        except Exception as exc:  # pragma: no cover - download errors
            errors[candidate] = str(exc)
            print(f"Tokenizer load failed for {candidate}: {exc}")
    raise RuntimeError(
        "Unable to load any router model. Tried:\n" +
        "\n".join(f"- {k}: {v}" for k, v in errors.items())
    )


MODEL_ID, tokenizer = _initialise_tokenizer()


class GeneratePayload(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class GenerateResponse(BaseModel):
    text: str


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
_MODEL = None


@spaces.GPU(duration=120)
def get_model() -> AutoModelForCausalLM:
    global _MODEL
    if _MODEL is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if USE_4BIT:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            kwargs["torch_dtype"] = dtype
        _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).eval()
    return _MODEL


@lru_cache(maxsize=8)
def _build_system_prompt() -> str:
    return (
        "You are the Router Agent coordinating Math, Code, and General-Search specialists.\n"
        "Emit ONLY strict JSON with keys route_plan, route_rationale, expected_artifacts,\n"
        "thinking_outline, handoff_plan, todo_list, difficulty, tags, acceptance_criteria, metrics."
    )


def _generate(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    model = get_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eos = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos,
            pad_token_id=eos,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt) :].strip() or text.strip()


fastapi_app = FastAPI(title="Router Model API", version="1.0.0")


@fastapi_app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@fastapi_app.post("/v1/generate", response_model=GenerateResponse)
def generate_endpoint(payload: GeneratePayload) -> GenerateResponse:
    try:
        text = _generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens or MAX_NEW_TOKENS,
            temperature=payload.temperature or DEFAULT_TEMPERATURE,
            top_p=payload.top_p or DEFAULT_TOP_P,
        )
    except Exception as exc:  # pragma: no cover - errors bubbled to caller.
        raise HTTPException(status_code=500, detail=str(exc))
    return GenerateResponse(text=text)


def gradio_infer(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    return _generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


with gr.Blocks(title="Router Model ZeroGPU Backend") as demo:
    gr.Markdown(
        f"### {MODEL_ID}\n"
        "This Space serves a merged router checkpoint for the CourseGPT project. "
        "Use the `/v1/generate` REST endpoint for programmatic access."
    )
    with gr.Row():
        prompt_box = gr.Textbox(
            label="Prompt",
            placeholder="Router system prompt + user query…",
            lines=8,
        )
    with gr.Row():
        max_tokens = gr.Slider(64, 1024, MAX_NEW_TOKENS, step=16, label="max_new_tokens")
        temperature = gr.Slider(0.0, 1.5, DEFAULT_TEMPERATURE, step=0.05, label="temperature")
        top_p = gr.Slider(0.1, 1.0, DEFAULT_TOP_P, step=0.05, label="top_p")
    output_box = gr.Textbox(label="Router Response", lines=10)
    run_btn = gr.Button("Generate", variant="primary")
    run_btn.click(
        fn=gradio_infer,
        inputs=[prompt_box, max_tokens, temperature, top_p],
        outputs=output_box,
    )


demo.queue()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
