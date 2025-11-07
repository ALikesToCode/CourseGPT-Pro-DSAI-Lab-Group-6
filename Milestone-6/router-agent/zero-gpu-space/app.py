from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    import spaces  # type: ignore
except Exception:  # pragma: no cover
    class _SpacesShim:  # fallback for local runs
        @staticmethod
        def GPU(*_args, **_kwargs):
            def identity(fn):
                return fn

            return identity

    spaces = _SpacesShim()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "600"))
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.9"))
HF_TOKEN = os.environ.get("HF_TOKEN")

def _normalise_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


_strategy = os.environ.get("MODEL_LOAD_STRATEGY") or os.environ.get("LOAD_STRATEGY")
if _strategy:
    _strategy = _strategy.lower().strip()

# Backwards compatibility flags remain available for older deployments.
USE_8BIT = _normalise_bool(os.environ.get("LOAD_IN_8BIT"), default=True)
USE_4BIT = _normalise_bool(os.environ.get("LOAD_IN_4BIT"), default=False)

SKIP_WARM_START = _normalise_bool(os.environ.get("SKIP_WARM_START"), default=False)
ALLOW_WARM_START_FAILURE = _normalise_bool(
    os.environ.get("ALLOW_WARM_START_FAILURE"),
    default=False,
)


def _normalise_strategy(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    alias = name.lower().strip()
    mapping = {
        "8": "8bit",
        "8bit": "8bit",
        "int8": "8bit",
        "bnb8": "8bit",
        "llm.int8": "8bit",
        "4": "4bit",
        "4bit": "4bit",
        "int4": "4bit",
        "bnb4": "4bit",
        "nf4": "4bit",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "cpu": "cpu",
        "fp32": "cpu",
        "full": "cpu",
    }
    canonical = mapping.get(alias, alias)
    if canonical not in {"8bit", "4bit", "bf16", "fp16", "cpu"}:
        return None
    return canonical


def _strategy_sequence() -> List[str]:
    order: List[str] = []
    seen: set[str] = set()

    def push(entry: Optional[str]) -> None:
        canonical = _normalise_strategy(entry)
        if not canonical or canonical in seen:
            return
        seen.add(canonical)
        order.append(canonical)

    push(_strategy)
    for raw in os.environ.get("MODEL_LOAD_STRATEGIES", "").split(","):
        push(raw)

    # Compatibility: honour legacy boolean switches.
    if USE_8BIT:
        push("8bit")
    if USE_4BIT:
        push("4bit")
    if not (USE_8BIT or USE_4BIT):
        push("bf16" if torch.cuda.is_available() else "cpu")

    for fallback in ("8bit", "4bit", "bf16", "fp16", "cpu"):
        push(fallback)
    return order


DEFAULT_MODEL_FALLBACKS: List[str] = [
    "Alovestocode/router-gemma3-merged",
    "Alovestocode/router-llama31-merged",
    "Alovestocode/router-qwen3-32b-merged",
]


def _candidate_models() -> List[str]:
    explicit = os.environ.get("MODEL_REPO")
    overrides = [
        item.strip()
        for item in os.environ.get("MODEL_FALLBACKS", "").split(",")
        if item.strip()
    ]
    candidates: List[str] = []
    seen = set()
    for name in [explicit, *overrides, *DEFAULT_MODEL_FALLBACKS]:
        if not name or name in seen:
            continue
        seen.add(name)
        candidates.append(name)
    return candidates


def _initialise_tokenizer() -> tuple[str, AutoTokenizer]:
    errors: dict[str, str] = {}
    for candidate in _candidate_models():
        try:
            tok = AutoTokenizer.from_pretrained(
                candidate,
                use_fast=False,
                token=HF_TOKEN,
            )
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


_MODEL = None
ACTIVE_STRATEGY: Optional[str] = None


def _build_load_kwargs(strategy: str, gpu_compute_dtype: torch.dtype) -> Tuple[str, dict]:
    """Return kwargs for `from_pretrained` using the given strategy."""
    cuda_available = torch.cuda.is_available()
    strategy = strategy.lower()
    kwargs: dict = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "token": HF_TOKEN,
    }
    if strategy == "8bit":
        if not cuda_available:
            raise RuntimeError("8bit loading requires CUDA availability")
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        return "8bit", kwargs
    if strategy == "4bit":
        if not cuda_available:
            raise RuntimeError("4bit loading requires CUDA availability")
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=gpu_compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return "4bit", kwargs
    if strategy == "bf16":
        kwargs["device_map"] = "auto" if cuda_available else "cpu"
        kwargs["torch_dtype"] = torch.bfloat16 if cuda_available else torch.float32
        return "bf16", kwargs
    if strategy == "fp16":
        kwargs["device_map"] = "auto" if cuda_available else "cpu"
        kwargs["torch_dtype"] = torch.float16 if cuda_available else torch.float32
        return "fp16", kwargs
    if strategy == "cpu":
        kwargs["device_map"] = "cpu"
        kwargs["torch_dtype"] = torch.float32
        return "cpu", kwargs
    raise ValueError(f"Unknown load strategy: {strategy}")


def get_model() -> AutoModelForCausalLM:
    """Load the model. This function should be called within a @spaces.GPU decorated function."""
    global _MODEL, ACTIVE_STRATEGY
    if _MODEL is None:
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        attempts: List[Tuple[str, Exception]] = []
        strategies = _strategy_sequence()
        print(f"Attempting to load {MODEL_ID} with strategies: {strategies}")
        for candidate in strategies:
            try:
                label, kwargs = _build_load_kwargs(candidate, compute_dtype)
                print(f"Trying strategy '{label}' for {MODEL_ID} ...")
                model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
                _MODEL = model.eval()
                ACTIVE_STRATEGY = label
                print(f"Loaded {MODEL_ID} with strategy='{label}'")
                break
            except Exception as exc:  # pragma: no cover - depends on runtime
                attempts.append((candidate, exc))
                print(f"Strategy '{candidate}' failed: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if _MODEL is None:
            detail = "; ".join(f"{name}: {err}" for name, err in attempts) or "no details"
            last_exc = attempts[-1][1] if attempts else None
            raise RuntimeError(
                f"Unable to load {MODEL_ID}. Tried strategies {strategies}. Details: {detail}"
            ) from last_exc
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


@spaces.GPU(duration=300)
def _generate_with_gpu(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """Generate function wrapped with ZeroGPU decorator. Must be defined before FastAPI app for ZeroGPU detection."""
    return _generate(prompt, max_new_tokens, temperature, top_p)


fastapi_app = FastAPI(title="Router Model API", version="1.0.0")


@fastapi_app.get("/")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "strategy": ACTIVE_STRATEGY or "pending",
    }


@fastapi_app.on_event("startup")
def warm_start() -> None:
    """Warm start is disabled for ZeroGPU - model loads on first request."""
    # ZeroGPU functions decorated with @spaces.GPU cannot be called during startup.
    # They must be called within request handlers. Skip warm start for ZeroGPU.
    print("Warm start skipped for ZeroGPU. Model will load on first request.")
    return


@fastapi_app.post("/v1/generate", response_model=GenerateResponse)
def generate_endpoint(payload: GeneratePayload) -> GenerateResponse:
    try:
        text = _generate_with_gpu(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens or MAX_NEW_TOKENS,
            temperature=payload.temperature or DEFAULT_TEMPERATURE,
            top_p=payload.top_p or DEFAULT_TOP_P,
        )
    except Exception as exc:  # pragma: no cover - errors bubbled to caller.
        raise HTTPException(status_code=500, detail=str(exc))
    return GenerateResponse(text=text)


@fastapi_app.get("/gradio", response_class=HTMLResponse)
def interactive_ui() -> str:
    return """
    <!doctype html>
    <html>
    <head>
      <title>Router Control Room</title>
      <style>
        body { font-family: sans-serif; margin: 40px; max-width: 900px; }
        textarea, input { width: 100%; }
        textarea { height: 180px; }
        pre { background: #111; color: #0f0; padding: 16px; border-radius: 8px; }
      </style>
    </head>
    <body>
      <h1>Router Control Room</h1>
      <p>This lightweight UI calls <code>/v1/generate</code>. Provide a full router prompt below.</p>
      <label>Prompt</label>
      <textarea id="prompt" placeholder="Include system text + user query..."></textarea>
      <label>Max new tokens</label>
      <input id="max_tokens" type="number" value="600" min="64" max="1024" step="16" />
      <label>Temperature</label>
      <input id="temperature" type="number" value="0.2" min="0" max="2" step="0.05" />
      <label>Top-p</label>
      <input id="top_p" type="number" value="0.9" min="0" max="1" step="0.05" />
      <button onclick="callRouter()">Generate plan</button>
      <h2>Response</h2>
      <pre id="response">(waiting)</pre>
      <script>
        async function callRouter() {
          const resp = await fetch("/v1/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prompt: document.getElementById("prompt").value,
              max_new_tokens: Number(document.getElementById("max_tokens").value),
              temperature: Number(document.getElementById("temperature").value),
              top_p: Number(document.getElementById("top_p").value)
            })
          });
          const json = await resp.json();
          document.getElementById("response").textContent = JSON.stringify(json, null, 2);
        }
      </script>
    </body>
    </html>
    """


# Gradio interface for ZeroGPU detection - ZeroGPU requires Gradio SDK
import gradio as gr

@spaces.GPU(duration=300)
def gradio_generate(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """Gradio interface function with GPU decorator for ZeroGPU detection."""
    return _generate(prompt, max_new_tokens, temperature, top_p)

# Create Gradio interface - this ensures ZeroGPU detects the GPU function
gradio_interface = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=5, placeholder="Enter your router prompt here..."),
        gr.Slider(minimum=64, maximum=2048, value=MAX_NEW_TOKENS, step=16, label="Max New Tokens"),
        gr.Slider(minimum=0.0, maximum=2.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=DEFAULT_TOP_P, step=0.05, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Generated Response", lines=10),
    title="Router Model API - ZeroGPU",
    description=f"Model: {MODEL_ID} | Strategy: {ACTIVE_STRATEGY or 'pending'}",
)

# Set app to Gradio interface for Spaces - ZeroGPU requires Gradio SDK
# For Spaces, Gradio will handle launching and FastAPI routes can be accessed via Gradio's app
app = gradio_interface

# Mount FastAPI routes to Gradio's underlying FastAPI app
# This happens after Gradio app is created during launch
original_launch = gradio_interface.launch

def launch_with_fastapi(*args, **kwargs):
    """Launch Gradio and mount FastAPI routes."""
    result = original_launch(*args, **kwargs)
    try:
        # Mount FastAPI routes to Gradio's FastAPI app
        gradio_app = gradio_interface.app
        gradio_app.mount("/v1", fastapi_app)
        gradio_app.mount("/gradio", fastapi_app)
    except (AttributeError, Exception):
        # Routes mounting will be handled by Spaces or on first request
        pass
    return result

gradio_interface.launch = launch_with_fastapi

if __name__ == "__main__":  # pragma: no cover
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
