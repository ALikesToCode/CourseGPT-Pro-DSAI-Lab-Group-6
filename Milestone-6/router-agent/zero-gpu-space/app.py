from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from fastapi import APIRouter, HTTPException
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
    "Alovestocode/router-qwen3-32b-merged",
    "Alovestocode/router-llama31-merged",
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
    """Generate function wrapped with ZeroGPU decorator."""
    return _generate(prompt, max_new_tokens, temperature, top_p)


def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "strategy": ACTIVE_STRATEGY or "pending",
    }


def warm_start() -> None:
    """Warm start is disabled for ZeroGPU - model loads on first request."""
    print("Warm start skipped for ZeroGPU. Model will load on first request.")


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




# Gradio interface for ZeroGPU detection - ZeroGPU requires Gradio SDK
import gradio as gr

STATUS_IDLE = "Status: awaiting prompt."


def _format_status(message: str, *, success: bool) -> str:
    prefix = "✅" if success else "❌"
    return f"{prefix} {message}"


def gradio_generate_handler(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> tuple[str, str]:
    """Wrapper used by the Gradio UI with friendly status messages."""
    if not prompt.strip():
        return (
            "ERROR: Prompt must not be empty.",
            _format_status("Prompt required before generating.", success=False),
        )
    try:
        # Reuse the same GPU-decorated generator as the API so behaviour matches.
        text = _generate_with_gpu(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as exc:  # pragma: no cover - runtime/hardware dependent
        print(f"UI generation failed: {exc}")
        return (
            f"ERROR: {exc}",
            _format_status("Generation failed. Check logs for details.", success=False),
        )
    return text, _format_status("Plan generated successfully.", success=True)

# Create Gradio Blocks app to mount FastAPI routes properly
with gr.Blocks(
    title="Router Model API - ZeroGPU",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .info-box {
        background: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    """
) as gradio_app:
    gr.HTML("""
    <div class="main-header">
        <h1>🚀 Router Model API - ZeroGPU</h1>
        <p>Intelligent routing agent for coordinating specialized AI agents</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            with gr.Accordion("Model Information", open=False):
                gr.Markdown(f"""
                **Model:** `{MODEL_ID}`  
                **Strategy:** `{ACTIVE_STRATEGY or 'pending'}`  
                **Max Tokens:** `{MAX_NEW_TOKENS}`  
                **Default Temperature:** `{DEFAULT_TEMPERATURE}`  
                **Default Top-p:** `{DEFAULT_TOP_P}`
                """)
            
            gr.Markdown("### 📝 Input")
            prompt_input = gr.Textbox(
                label="Router Prompt",
                lines=8,
                placeholder="Enter your router prompt here...\n\nExample:\nYou are the Router Agent coordinating Math, Code, and General-Search specialists.\nUser query: Solve the integral of x^2 from 0 to 1",
                value="",
            )
            
            with gr.Accordion("⚙️ Generation Parameters", open=True):
                max_tokens_input = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=MAX_NEW_TOKENS,
                    step=16,
                    label="Max New Tokens",
                    info="Maximum number of tokens to generate"
                )
                temp_input = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=DEFAULT_TEMPERATURE,
                    step=0.05,
                    label="Temperature",
                    info="Controls randomness: lower = more deterministic"
                )
                top_p_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_TOP_P,
                    step=0.05,
                    label="Top-p (Nucleus Sampling)",
                    info="Probability mass to consider for sampling"
                )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Output")
            output = gr.Textbox(
                label="Generated Response",
                lines=22,
                placeholder="Generated response will appear here...",
                show_copy_button=True,
            )
            status_display = gr.Markdown(STATUS_IDLE)
            
            with gr.Accordion("📚 API Information", open=False):
                gr.Markdown("""
                ### API Endpoints
                
                **POST** `/v1/generate`
                ```json
                {
                  "prompt": "Your prompt here",
                  "max_new_tokens": 600,
                  "temperature": 0.2,
                  "top_p": 0.9
                }
                ```
                
                **GET** `/health` - JSON health check  
                **GET** `/` - Full Gradio UI
                """)
    
    # Event handlers
    generate_btn.click(
        fn=gradio_generate_handler,
        inputs=[prompt_input, max_tokens_input, temp_input, top_p_input],
        outputs=[output, status_display],
    )

    clear_btn.click(
        fn=lambda: ("", "", STATUS_IDLE),
        outputs=[prompt_input, output, status_display],
    )
    
    # Note: API routes will be added after Blocks context to avoid interfering with Gradio's static assets

# Attach API routes directly onto Gradio's FastAPI instance
api_router = APIRouter()


@api_router.get("/health")
def api_health() -> dict[str, str]:
    return healthcheck()


@api_router.post("/v1/generate", response_model=GenerateResponse)
def api_generate(payload: GeneratePayload) -> GenerateResponse:
    return generate_endpoint(payload)


gradio_app.app.include_router(api_router)
warm_start()

# Enable queued execution so ZeroGPU can schedule GPU work reliably
gradio_app.queue(max_size=8)

app = gradio_app

if __name__ == "__main__":  # pragma: no cover
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
