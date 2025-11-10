from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, TextIteratorStreamer, pipeline
from threading import Thread

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True

# Ensure CUDA is visible to vLLM on ZeroGPU
# vLLM needs explicit CUDA device configuration
# ZeroGPU uses MIG UUIDs, but vLLM needs numeric device index
if torch.cuda.is_available():
    # Set CUDA_VISIBLE_DEVICES if not already set or if it's a MIG UUID
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible or not cuda_visible.isdigit():
        # If CUDA_VISIBLE_DEVICES is a MIG UUID or empty, use "0" for single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
else:
    print("WARNING: CUDA not available - vLLM will not work")

# Try to import vLLM (primary inference engine)
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    print("Warning: vLLM not available, falling back to Transformers")

# Try to import LLM Compressor (for quantization - optional, vLLM has native AWQ support)
# Note: llm-compressor is only needed for quantizing models, not for loading pre-quantized AWQ models
# vLLM can load AWQ models natively without llm-compressor
try:
    # Try both package names (llm-compressor and llmcompressor)
    try:
        from llmcompressor import oneshot
        # Correct import path: AWQModifier is in modifiers.awq, not modifiers.quantization
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        # Try alternative package name
        import sys
        import subprocess
        # Package might be named llm-compressor (with hyphen)
        try:
            import importlib.util
            spec = importlib.util.find_spec("llm_compressor")
            if spec is None:
                raise ImportError("llm-compressor not found")
            from llm_compressor import oneshot
            from llm_compressor.modifiers.awq import AWQModifier
        except ImportError:
            raise ImportError("Neither llmcompressor nor llm-compressor found")
    LLM_COMPRESSOR_AVAILABLE = True
    print("Info: LLM Compressor available (for quantizing models)")
except ImportError:
    LLM_COMPRESSOR_AVAILABLE = False
    # This is fine - vLLM has native AWQ support, so we don't need llm-compressor for loading
    print("Info: LLM Compressor not available (not needed - vLLM has native AWQ support for pre-quantized models)")

# Try to import AWQ (deprecated, but kept for fallback compatibility)
# Note: AutoAWQ is deprecated; vLLM handles AWQ natively via llm-compressor
try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")
except ImportError:
    AWQ_AVAILABLE = False
    print("Info: AutoAWQ not available (using vLLM native AWQ support instead)")

# Always import BitsAndBytesConfig for fallback
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None
    print("Warning: BitsAndBytes not available")

# Try to import FlashAttention-2
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: FlashAttention-2 not available")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable must be set for private router checkpoints.")

PLAN_END_TOKEN = "<|end_of_plan|>"
STOP_SEQUENCES = [PLAN_END_TOKEN, "</json>", "</JSON>"]

ROUTER_SYSTEM_PROMPT = """You are the Router Agent coordinating Math, Code, and General-Search specialists.\nEmit EXACTLY ONE strict JSON object with keys route_plan, route_rationale, expected_artifacts,\nthinking_outline, handoff_plan, todo_list, difficulty, tags, acceptance_criteria, metrics.\nRules:\n- No markdown/code fences, no natural-language prologues or epilogues.\n- route_plan must be an ordered list of tool invocations such as /math(...), /code(...), /general-search(...).\n- todo_list must map each checklist item to the responsible tool.\n- metrics must include primary and secondary arrays (add optional *_guidance fields when they exist).\n- After the closing brace of the JSON object, immediately append the sentinel <|end_of_plan|>.\nExample output:\n{\n  "route_plan": ["/general-search(...)"],\n  "route_rationale": "...",\n  ...\n}<|end_of_plan|>\nReturn nothing else."""

MODELS = {
    "Router-Qwen3-32B-AWQ": {
        "repo_id": "Alovestocode/router-qwen3-32b-merged-awq",  # AWQ quantized model
        "tokenizer_repo": "Alovestocode/router-qwen3-32b-merged",  # Tokenizer from original repo
        "description": "Router checkpoint on Qwen3 32B merged, optimized with AWQ quantization via vLLM.",
        "params_b": 32.0,
        "quantization": "awq",  # vLLM will auto-detect AWQ
    },
    "Router-Gemma3-27B-AWQ": {
        "repo_id": "Alovestocode/router-gemma3-merged-awq",  # AWQ quantized model
        "tokenizer_repo": "Alovestocode/router-gemma3-merged",  # Tokenizer from original repo
        "description": "Router checkpoint on Gemma3 27B merged, optimized with AWQ quantization via vLLM.",
        "params_b": 27.0,
        "quantization": "awq",  # vLLM will auto-detect AWQ
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

PIPELINES: Dict[str, Any] = {}  # For Transformers fallback
VLLM_MODELS: Dict[str, Any] = {}  # For vLLM models
TOKENIZER_CACHE: Dict[str, Any] = {}
WARMED_REMAINING = False
TOOL_PATTERN = re.compile(r"^/[a-z0-9_-]+\(.*\)$", re.IGNORECASE)


def get_tokenizer(repo: str, tokenizer_repo: str = None):
    """Get tokenizer, preferring tokenizer_repo if provided (for AWQ models)."""
    # Use tokenizer_repo if provided (for AWQ models where tokenizer is in original repo)
    actual_repo = tokenizer_repo if tokenizer_repo else repo
    tok = TOKENIZER_CACHE.get(actual_repo)
    if tok is not None:
        return tok
    tok = AutoTokenizer.from_pretrained(
        actual_repo, 
        token=HF_TOKEN,
        use_fast=True,
        trust_remote_code=True
    )
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    TOKENIZER_CACHE[actual_repo] = tok
    return tok


def load_vllm_model(model_name: str):
    """Load model with vLLM (supports AWQ natively, continuous batching, PagedAttention)."""
    if model_name in VLLM_MODELS:
        return VLLM_MODELS[model_name]
    
    model_config = MODELS[model_name]
    repo = model_config["repo_id"]
    quantization = model_config.get("quantization", None)
    
    # For AWQ models, vLLM should point to repo root (not default/ subfolder)
    # vLLM will find quantization_config.json at root, which points to default/ subfolder
    # The quantization_config.json tells vLLM where the actual model files are
    if quantization == "awq":
        # Point to repo root - vLLM will auto-detect AWQ via quantization_config.json
        # The config file at root tells vLLM the model files are in default/ subfolder
        model_path = repo  # Use repo root, not repo/default
        print(f"Loading {model_path} with vLLM (AWQ quantization, vLLM will find files in default/ via quantization_config.json)...")
    else:
        model_path = repo
        print(f"Loading {model_path} with vLLM (quantization: {quantization})...")
    
    try:
        # Detect device explicitly for vLLM
        # vLLM needs explicit device configuration on ZeroGPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - vLLM requires GPU. Falling back to Transformers pipeline.")
        
        print(f"  → CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  → CUDA device count: {torch.cuda.device_count()}")
        print(f"  → CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        
        # vLLM configuration optimized for ZeroGPU H200 slice
        # vLLM natively supports AWQ via llm-compressor (replaces deprecated AutoAWQ)
        # Note: HF_TOKEN is passed via environment variable, not as a parameter
        # vLLM auto-detects CUDA from torch.cuda.is_available() and CUDA_VISIBLE_DEVICES
        # For AWQ models with files in default/ subfolder, vLLM should auto-detect via quantization_config.json
        llm_kwargs = {
            "model": model_path,  # Use model_path which may point to default/ subfolder
            "trust_remote_code": True,
            "dtype": "bfloat16",  # Prefer bf16 over int8 for speed
            "gpu_memory_utilization": 0.90,  # Leave headroom for KV cache
            "max_model_len": 16384,  # Adjust based on GPU memory
            "enable_chunked_prefill": True,  # Better for long prompts
            "tensor_parallel_size": 1,  # Single GPU for ZeroGPU
            "max_num_seqs": 256,  # Continuous batching capacity
            "enable_prefix_caching": True,  # Cache prompts for faster TTFT
        }
        
        # Ensure CUDA_VISIBLE_DEVICES is set correctly for vLLM device detection
        # ZeroGPU uses MIG UUIDs, but vLLM needs numeric device index
        # IMPORTANT: Set this BEFORE creating LLM() instance, as vLLM checks device during init
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible or not cuda_visible.isdigit():
            # If CUDA_VISIBLE_DEVICES is a MIG UUID or empty, use "0" for single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print(f"  → Set CUDA_VISIBLE_DEVICES=0 (was: {cuda_visible})")
        
        # Force torch to see the correct device after setting CUDA_VISIBLE_DEVICES
        # This ensures vLLM's device detection works correctly
        import torch
        if torch.cuda.is_available():
            # Verify device is accessible
            device_name = torch.cuda.get_device_name(0)
            print(f"  → Verified CUDA device accessible: {device_name}")
        
        # Add quantization if specified (vLLM auto-detects AWQ via llm-compressor)
        if quantization == "awq":
            llm_kwargs["quantization"] = "awq"
            # AWQ model files are in the 'default' subfolder
            # vLLM should auto-detect this via quantization_config.json at repo root
            # If auto-detection fails, we can explicitly point to default/ subfolder
            # Enable FP8 KV cache for 50% memory reduction (allows longer contexts)
            # FP8 KV cache is compatible with AWQ quantization
            try:
                llm_kwargs["kv_cache_dtype"] = "fp8"
                print(f"  → AWQ quantization + FP8 KV cache enabled (vLLM native support)")
                print(f"  → FP8 KV cache reduces memory by ~50%, enabling longer contexts")
                print(f"  → Loading AWQ model from: {model_path} (files in default/ subfolder)")
            except Exception:
                # Fallback if FP8 KV cache not supported
                print(f"  → AWQ quantization enabled (FP8 KV cache not available)")
                print(f"  → Loading AWQ model from: {model_path} (files in default/ subfolder)")
        elif quantization == "fp8":
            # Try FP8 quantization if available (faster than AWQ)
            try:
                llm_kwargs["quantization"] = "fp8"
                llm_kwargs["dtype"] = "float8_e5m2"
                print(f"  → FP8 quantization enabled (~2x faster than AWQ)")
            except Exception:
                print(f"  → FP8 quantization not available, falling back to bf16")
        
        print(f"  → Loading with vLLM (continuous batching, PagedAttention)...")
        llm = LLM(**llm_kwargs)
        VLLM_MODELS[model_name] = llm
        print(f"✅ vLLM model loaded: {model_name}")
        print(f"   - Continuous batching: enabled (max {llm_kwargs['max_num_seqs']} concurrent)")
        print(f"   - Prefix caching: enabled")
        print(f"   - Quantization: {quantization or 'none (bf16)'}")
        return llm
    except Exception as exc:
        print(f"❌ vLLM load failed for {repo}: {exc}")
        import traceback
        traceback.print_exc()
        raise


def load_awq_pipeline(repo: str, tokenizer):
    """Load AWQ-quantized model with FlashAttention-2 and torch.compile (Transformers fallback)."""
    model = AutoAWQForCausalLM.from_quantized(
        repo,
        fuse_layers=True,
        trust_remote_code=True,
        device_map="auto",
        token=HF_TOKEN,
    )
    
    # Prepare model kwargs with FlashAttention-2 if available
    model_kwargs = {}
    if FLASH_ATTN_AVAILABLE:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        model_kwargs=model_kwargs,
        use_cache=True,
        torch_dtype=torch.bfloat16,  # Prefer bf16 over int8 for speed
    )
    pipe.model.eval()
    
    # Apply torch.compile for kernel fusion (~10-20% speedup after first call)
    try:
        if hasattr(torch, 'compile'):
            print("Applying torch.compile for kernel fusion...")
            pipe.model = torch.compile(pipe.model, mode="reduce-overhead")
            print("✅ torch.compile applied (first call will be slower, subsequent calls faster)")
    except Exception as exc:
        print(f"⚠️ torch.compile failed: {exc} (continuing without compilation)")
    
    return pipe


def load_pipeline(model_name: str):
    """Load model with vLLM (preferred) or Transformers (fallback).
    
    Fallback chain:
    1. vLLM with AWQ (best performance, continuous batching)
    2. vLLM with FP16 (if AWQ not available)
    3. Transformers with AWQ (via AutoAWQ - deprecated but functional)
    4. Transformers with BitsAndBytes 8-bit
    5. Transformers with FP16/FP32
    """
    # Try vLLM first (best performance with native AWQ support via llm-compressor)
    # vLLM handles AWQ natively, so AutoAWQ deprecation doesn't affect us
    if VLLM_AVAILABLE:
        try:
            print(f"🔄 Attempting to load {model_name} with vLLM (native AWQ support)...")
            return load_vllm_model(model_name)
        except Exception as exc:
            print(f"⚠️ vLLM load failed: {exc}")
            print(f"   → Falling back to Transformers pipeline...")
            import traceback
            traceback.print_exc()
    
    # Fallback to Transformers pipeline
    if model_name in PIPELINES:
        print(f"✅ Using cached Transformers pipeline for {model_name}")
        return PIPELINES[model_name]

    model_config = MODELS[model_name]
    repo = model_config["repo_id"]
    tokenizer_repo = model_config.get("tokenizer_repo", None)
    quantization = model_config.get("quantization", None)
    
    # For AWQ models, the AWQ repo doesn't have standard model files (they're in default/)
    # Use the original repo for Transformers fallback, not the AWQ repo
    if quantization == "awq" and tokenizer_repo:
        # AWQ repos have files in default/ subfolder which Transformers can't load directly
        # Use the original repo for Transformers fallback
        transformers_repo = tokenizer_repo  # Use original repo for Transformers
        print(f"⚠️ AWQ model detected - Transformers fallback will use original repo: {transformers_repo}")
    else:
        transformers_repo = repo
    
    tokenizer = get_tokenizer(repo, tokenizer_repo=tokenizer_repo)

    # Try AWQ first if available (Transformers fallback path)
    if AWQ_AVAILABLE:
        try:
            print(f"🔄 Loading {transformers_repo} with Transformers + AutoAWQ (fallback path)...")
            pipe = load_awq_pipeline(transformers_repo, tokenizer)
            PIPELINES[model_name] = pipe
            _schedule_background_warm(model_name)
            # Warm kernels immediately after loading
            Thread(target=lambda: _warm_kernels(model_name), daemon=True).start()
            print(f"✅ Transformers + AutoAWQ pipeline loaded: {model_name}")
            return pipe
        except Exception as exc:
            print(f"⚠️ AutoAWQ load failed for {transformers_repo}: {exc}")
            print(f"   → Falling back to BitsAndBytes 8-bit...")

    # Fallback to BitsAndBytes 8-bit
    if BITSANDBYTES_AVAILABLE:
        try:
            print(f"🔄 Loading {transformers_repo} with BitsAndBytes 8-bit quantization...")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs = {"quantization_config": quant_config}
            if FLASH_ATTN_AVAILABLE:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            pipe = pipeline(
                task="text-generation",
                model=transformers_repo,
                tokenizer=tokenizer,
                trust_remote_code=True,
                device_map="auto",
                model_kwargs=model_kwargs,
                use_cache=True,
                token=HF_TOKEN,
                torch_dtype=torch.bfloat16,
            )
            
            pipe.model.eval()
            
            # Apply torch.compile for kernel fusion (~10-20% speedup after first call)
            try:
                if hasattr(torch, 'compile'):
                    pipe.model = torch.compile(pipe.model, mode="reduce-overhead")
            except Exception:
                pass
            
            PIPELINES[model_name] = pipe
            _schedule_background_warm(model_name)
            print(f"✅ BitsAndBytes 8-bit pipeline loaded: {model_name}")
            return pipe
        except Exception as exc:
            print(f"⚠️ BitsAndBytes 8-bit load failed for {repo}: {exc}")
            print(f"   → Falling back to FP16/FP32...")

    # Fallback to bfloat16/fp16/fp32 (unquantized)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        dtype_name = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]
        try:
            print(f"🔄 Loading {repo} with {dtype_name} precision...")
            model_kwargs = {}
            if FLASH_ATTN_AVAILABLE:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            pipe = pipeline(
                task="text-generation",
                model=repo,
                tokenizer=tokenizer,
                trust_remote_code=True,
                device_map="auto",
                dtype=dtype,
                model_kwargs=model_kwargs,
                use_cache=True,
                token=HF_TOKEN,
            )
            pipe.model.eval()
            
            # Apply torch.compile for kernel fusion
            try:
                if hasattr(torch, 'compile'):
                    pipe.model = torch.compile(pipe.model, mode="reduce-overhead")
            except Exception:
                pass
            
            PIPELINES[model_name] = pipe
            _schedule_background_warm(model_name)
            print(f"✅ {dtype_name} pipeline loaded: {model_name}")
            return pipe
        except Exception as exc:
            print(f"⚠️ {dtype_name} load failed: {exc}")
            continue

    # Final fallback (no quantization, no FlashAttention)
    print(f"⚠️ All quantization methods failed, using basic pipeline...")
    model_kwargs = {}
    if FLASH_ATTN_AVAILABLE:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    pipe = pipeline(
        task="text-generation",
        model=repo,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        model_kwargs=model_kwargs,
        use_cache=True,
        token=HF_TOKEN,
    )
    pipe.model.eval()
    
    # Apply torch.compile for kernel fusion
    try:
        if hasattr(torch, 'compile'):
            pipe.model = torch.compile(pipe.model, mode="reduce-overhead")
    except Exception:
        pass
    
    PIPELINES[model_name] = pipe
    _schedule_background_warm(model_name)
    print(f"✅ Basic pipeline loaded: {model_name}")
    return pipe


def _warm_kernels(model_name: str) -> None:
    """Warm up CUDA kernels with a small dummy generation."""
    try:
        # Check if using vLLM
        if VLLM_AVAILABLE and model_name in VLLM_MODELS:
            llm = VLLM_MODELS[model_name]
            # vLLM handles warmup internally, but we can trigger a small generation
            sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
            _ = llm.generate("test", sampling_params)
            print(f"vLLM kernels warmed for {model_name}")
            return
        
        # Transformers pipeline warmup
        pipe = PIPELINES.get(model_name)
        if pipe is None:
            return
        
        tokenizer = pipe.tokenizer
        # Create a minimal prompt for warmup
        warmup_text = "test"
        inputs = tokenizer(warmup_text, return_tensors="pt")
        if hasattr(pipe.model, 'device'):
            inputs = {k: v.to(pipe.model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Run a tiny generation to JIT-fuse kernels
        with torch.inference_mode():
            _ = pipe.model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                use_cache=True,
            )
        print(f"Transformers kernels warmed for {model_name}")
    except Exception as exc:
        print(f"Kernel warmup failed for {model_name}: {exc}")


def _schedule_background_warm(loaded_model: str) -> None:
    global WARMED_REMAINING
    if WARMED_REMAINING:
        return
    warm_remaining = os.environ.get("ROUTER_WARM_REMAINING", "1")
    if warm_remaining not in {"1", "true", "True"}:
        return

    # Check both PIPELINES and VLLM_MODELS for remaining models
    loaded_models = set(PIPELINES.keys()) | set(VLLM_MODELS.keys())
    remaining = [name for name in MODELS if name not in loaded_models]
    if not remaining:
        WARMED_REMAINING = True
        return

    def _warm_all():
        for name in remaining:
            try:
                print(f"Background warm start for {name}")
                load_pipeline(name)
                # Warm kernels after loading
                _warm_kernels(name)
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


def trim_at_stop_sequences(text: str) -> Tuple[str, bool]:
    """Trim text at stop sequences and return trimmed text and whether a stop was found."""
    earliest = None
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        return text[:earliest], True
    return text, False


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


def _generate_router_plan_streaming_internal(
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
    gpu_duration: int,
):
    """Internal generator function for streaming token output."""
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

        print(f"[DEBUG] Loading model: {model_choice}")
        generator = load_pipeline(model_choice)
        print(f"[DEBUG] Model loaded successfully: {type(generator)}")
        
        # Check if using vLLM or Transformers
        is_vllm = VLLM_AVAILABLE and isinstance(generator, LLM)
        
        if is_vllm:
            # Use vLLM streaming API with continuous batching
            # Optimized sampling parameters for router plan generation
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                stop=STOP_SEQUENCES,
                skip_special_tokens=False,  # Keep special tokens for parsing
                spaces_between_special_tokens=False,  # Don't add spaces around special tokens
                include_stop_str_in_output=False,  # Don't include stop sequences in output
            )
            
            # vLLM streaming generation (non-blocking, continuous batching)
            completion = ""
            parsed_plan: Dict[str, Any] | None = None
            validation_msg = "🔄 Generating..."
            
            # vLLM's generate with stream=True returns RequestOutput iterator
            # Each RequestOutput contains incremental text updates
            stream = generator.generate(prompt, sampling_params, stream=True)
            
            prev_text_len = 0
            for request_output in stream:
                if not request_output.outputs:
                    continue
                
                # Get the latest output (vLLM provides incremental updates)
                output = request_output.outputs[0]
                current_text = output.text
                
                # Extract only new tokens since last update
                if len(current_text) > prev_text_len:
                    new_text = current_text[prev_text_len:]
                    completion += new_text
                    prev_text_len = len(current_text)
                    
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
                
                # Check if generation is finished
                if request_output.finished:
                    break
        else:
            # Use Transformers pipeline (fallback)
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

            generation_error = None
            
            def _generate():
                nonlocal generation_error
                try:
                    with torch.inference_mode():
                        model.generate(**generation_kwargs)
                except Exception as e:
                    generation_error = e
                    print(f"[DEBUG] Generation thread error: {e}")
                    import traceback
                    traceback.print_exc()

            thread = Thread(target=_generate)
            thread.start()

            # Stream tokens
            completion = ""
            parsed_plan: Dict[str, Any] | None = None
            validation_msg = "🔄 Generating..."
            
            print(f"[DEBUG] Starting to consume streamer...")
            token_count = 0
            
            try:
                for new_text in streamer:
                    if generation_error:
                        raise generation_error
                    
                    if new_text:
                        token_count += 1
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
                
                print(f"[DEBUG] Streamer finished. Received {token_count} tokens.")
            except Exception as stream_error:
                print(f"[DEBUG] Streamer error: {stream_error}")
                import traceback
                traceback.print_exc()
                # Wait for thread to finish
                thread.join(timeout=5.0)
                if generation_error:
                    raise generation_error
                raise stream_error
            
            # Final processing after streaming completes
            thread.join(timeout=30.0)
            if thread.is_alive():
                print("[DEBUG] WARNING: Generation thread still running after timeout")
            
            if generation_error:
                raise generation_error

        completion = trim_at_stop_sequences(completion.strip())[0]
        print(f"[DEBUG] Final completion length: {len(completion)}")
        
        if not completion:
            print("[DEBUG] WARNING: Completion is empty - model may not have generated output")
            validation_msg = "⚠️ Model generated empty output. Check GPU allocation and model loading."
        elif parsed_plan is None:
            try:
                json_block = extract_json_from_text(completion)
                parsed_plan = json.loads(json_block)
                ok, issues = validate_router_plan(parsed_plan)
                validation_msg = format_validation_message(ok, issues)
            except Exception as exc:
                parsed_plan = {}
                validation_msg = f"❌ JSON parsing failed: {exc}"
                print(f"[DEBUG] JSON parsing error: {exc}")

        yield completion, parsed_plan, validation_msg, prompt
        
    except Exception as exc:
        import traceback
        print(f"[DEBUG] Exception in generation: {exc}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        error_msg = f"❌ Generation failed: {str(exc)}"
        yield "", {}, error_msg, ""


# Pre-create GPU wrappers for common durations at module load time
# This ensures spaces.GPU decorators are detected during startup
_GPU_WRAPPERS: Dict[int, Any] = {}

# Create wrappers for durations: 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 
# 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800 (every 60s from 60 to 1800)
def _make_gpu_wrapper(duration: int):
    """Factory function to create GPU-decorated wrapper with closure over duration."""
    @spaces.GPU(duration=duration)
    def wrapper(
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
        gpu_duration: int,
    ):
        yield from _generate_router_plan_streaming_internal(
            user_task, context, acceptance, extra_guidance,
            difficulty, tags, model_choice, max_new_tokens,
            temperature, top_p, duration
        )
    return wrapper

# Pre-create all wrappers at module load time
for duration in range(60, 1801, 60):
    _GPU_WRAPPERS[duration] = _make_gpu_wrapper(duration)


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
    gpu_duration: int = 600,
):
    """
    Generate router plan with streaming output.
    
    Uses user-specified gpu_duration to select the appropriate GPU wrapper.
    """
    # Round to nearest 60 seconds and clamp between 60 and 1800
    rounded_duration = ((gpu_duration + 30) // 60) * 60
    rounded_duration = max(60, min(1800, rounded_duration))
    
    # Get the pre-created wrapper with this duration
    wrapper = _GPU_WRAPPERS[rounded_duration]
    yield from wrapper(
        user_task, context, acceptance, extra_guidance,
        difficulty, tags, model_choice, max_new_tokens,
        temperature, top_p, rounded_duration
    )


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
                gpu_duration = gr.Slider(60, 1800, value=600, step=60, label="GPU Duration (seconds)", info="Maximum GPU time allocation for this request")

        with gr.Row():
            generate_btn = gr.Button("Generate Router Plan", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

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
                gpu_duration,
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
    # Support both Hugging Face Spaces and Google Cloud Run
    # Cloud Run uses PORT, Hugging Face Spaces uses GRADIO_SERVER_PORT
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    
    demo.launch(
        server_name=server_name,
        server_port=port,
        show_api=True
    )

