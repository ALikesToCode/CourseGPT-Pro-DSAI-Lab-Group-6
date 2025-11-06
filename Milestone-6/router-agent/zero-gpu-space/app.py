from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    import spaces  # type: ignore
except Exception:  # pragma: no cover
    spaces = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

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


_MODEL = None


def _spaces_gpu(*args, **kwargs):
    if spaces is None:
        def identity(fn):
            return fn
        return identity
    return spaces.GPU(*args, **kwargs)


@_spaces_gpu(duration=120)
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


app = fastapi_app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
