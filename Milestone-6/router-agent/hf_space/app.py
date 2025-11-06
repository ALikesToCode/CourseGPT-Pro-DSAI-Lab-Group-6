from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib.util
import re
import requests

import gradio as gr

try:
    import spaces  # type: ignore
except Exception:  # pragma: no cover - spaces is only available inside HF Spaces.
    spaces = None


def _gpu_decorator(*args, **kwargs):
    if spaces is None:
        def identity(fn):
            return fn
        return identity
    return spaces.GPU(*args, **kwargs)

# Ensure Milestone 5 evaluation utilities are importable when running inside the Space.
APP_PATH = Path(__file__).resolve()
PARENTS = APP_PATH.parents
try:
    REPO_ROOT = PARENTS[3]
except IndexError:
    # Hugging Face Spaces flatten uploads under /app; fall back to parent.
    REPO_ROOT = APP_PATH.parent

EVAL_DIR = REPO_ROOT / "Milestone-5" / "router-agent"
LOCAL_LIB_DIR = APP_PATH.parent
if EVAL_DIR.exists():
    sys.path.insert(0, str(EVAL_DIR))
elif (LOCAL_LIB_DIR / "schema_score.py").exists():
    sys.path.insert(0, str(LOCAL_LIB_DIR))

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


HF_ROUTER_REPO_ENV = os.environ.get("HF_ROUTER_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_ROUTER_API = os.environ.get("HF_ROUTER_API", "").strip()

RouterOption = Dict[str, Optional[str]]

def _make_option(
    label: str,
    base: Optional[str] = None,
    adapter: Optional[str] = None,
    api: Optional[str] = None,
) -> RouterOption:
    return {"label": label, "base": base, "adapter": adapter, "api": api}


ROUTER_OPTIONS_LIST: List[RouterOption] = [
    _make_option("Use sample plan (no remote model)", ""),
    _make_option(
        "Llama 3.1 8B Router Adapter",
        "meta-llama/Llama-3.1-8B",
        "CourseGPT-Pro-DSAI-Lab-Group-6/router-llama31-peft",
    ),
    _make_option("Llama 3.1 8B Base Model", "meta-llama/Llama-3.1-8B"),
    _make_option(
        "Gemma 3 27B Router Adapter",
        "google/gemma-3-27b-pt",
        "CourseGPT-Pro-DSAI-Lab-Group-6/router-gemma3-peft",
    ),
    _make_option("Gemma 3 27B Base Model", "google/gemma-3-27b-pt"),
    _make_option(
        "Qwen3 32B Router Adapter",
        "Qwen/Qwen3-32B",
        "CourseGPT-Pro-DSAI-Lab-Group-6/router-qwen3-32b-peft",
    ),
    _make_option("Qwen3 32B Base Model", "Qwen/Qwen3-32B"),
]

if HF_ROUTER_API:
    ROUTER_OPTIONS_LIST.append(
        _make_option("Custom Router API (HF_ROUTER_API)", api=HF_ROUTER_API)
    )

AVAILABLE_ROUTER_OPTIONS: OrderedDict[str, RouterOption] = OrderedDict(
    (opt["label"], opt) for opt in ROUTER_OPTIONS_LIST
)


def _match_env_to_option(env_value: str) -> Optional[str]:
    for label, option in AVAILABLE_ROUTER_OPTIONS.items():
        if (
            option.get("base") == env_value
            or option.get("adapter") == env_value
            or option.get("api") == env_value
        ):
            return label
    return None


DEFAULT_ROUTER_LABEL = _match_env_to_option(HF_ROUTER_REPO_ENV) or "Use sample plan (no remote model)"

if HF_ROUTER_REPO_ENV and DEFAULT_ROUTER_LABEL == "Use sample plan (no remote model)" and HF_ROUTER_REPO_ENV:
    custom_label = f"Env configured model ({HF_ROUTER_REPO_ENV})"
    AVAILABLE_ROUTER_OPTIONS[custom_label] = _make_option(custom_label, HF_ROUTER_REPO_ENV)
    DEFAULT_ROUTER_LABEL = custom_label

if HF_ROUTER_API and DEFAULT_ROUTER_LABEL == "Use sample plan (no remote model)":
    DEFAULT_ROUTER_LABEL = "Custom Router API (HF_ROUTER_API)"

CLIENT_CACHE: Dict[str, Optional[Any]] = {}
CLIENT_ERRORS: Dict[str, str] = {}
DEFAULT_ROUTER_OPTION = AVAILABLE_ROUTER_OPTIONS[DEFAULT_ROUTER_LABEL]


def _option_key(option: RouterOption) -> str:
    base = option.get("base") or ""
    adapter = option.get("adapter") or ""
    api = option.get("api") or ""
    return f"{base}||{adapter}||{api}"

if EVAL_DIR.exists():
    BENCH_GOLD_PATH = EVAL_DIR / "benchmarks" / "router_benchmark_hard.jsonl"
    THRESHOLDS_PATH = EVAL_DIR / "router_benchmark_thresholds.json"
else:
    BENCH_GOLD_PATH = LOCAL_LIB_DIR / "router_benchmark_hard.jsonl"
    THRESHOLDS_PATH = LOCAL_LIB_DIR / "router_benchmark_thresholds.json"


SYSTEM_PROMPT = (
    "You are the Router Agent coordinating Math, Code, and General-Search specialists.\n"
    "Emit ONLY strict JSON with keys route_plan, route_rationale, expected_artifacts,\n"
    "thinking_outline, handoff_plan, todo_list, difficulty, tags, acceptance_criteria, metrics."
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)

AGENT_LOAD_LOG: List[str] = []


def _load_module(module_name: str, file_path: Path):
    if not file_path.exists():
        AGENT_LOAD_LOG.append(f"Missing module: {file_path}")
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        AGENT_LOAD_LOG.append(f"Unable to load spec for {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:
        AGENT_LOAD_LOG.append(f"Failed to import {file_path.name}: {exc}")
        return None
    return module


M6_ROOT = REPO_ROOT / "Milestone-6"
AGENT_BASE_PATH = M6_ROOT / "agents" / "base.py"
BASE_MODULE = _load_module("router_agents_base", AGENT_BASE_PATH)

if BASE_MODULE:
    AgentRequest = getattr(BASE_MODULE, "AgentRequest", None)
    AgentResult = getattr(BASE_MODULE, "AgentResult", None)
else:
    AgentRequest = None
    AgentResult = None
    AGENT_LOAD_LOG.append("Agent base definitions unavailable; agent execution disabled.")


class GeminiFallbackManager:
    """Fallback generator powered by Gemini 2.5 Pro (if configured)."""

    def __init__(self) -> None:
        self.available = False
        self.error: Optional[str] = None
        self.model = None
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-exp-0801")
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.error = f"google-generativeai import failed: {exc}"
            AGENT_LOAD_LOG.append(f"Gemini fallback disabled: {self.error}")
            return
        if not api_key:
            self.error = "GOOGLE_API_KEY (or GEMINI_API_KEY) not set."
            AGENT_LOAD_LOG.append(f"Gemini fallback disabled: {self.error}")
            return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as exc:  # pragma: no cover
            self.error = f"Failed to initialise Gemini model: {exc}"
            AGENT_LOAD_LOG.append(f"Gemini fallback disabled: {self.error}")
            return
        self.available = True
        AGENT_LOAD_LOG.append(f"Gemini fallback ready (model={self.model_name}).")

    def generate(self, tool_name: str, request: Any, error: Optional[str] = None) -> Any:
        if not self.available or self.model is None or AgentResult is None:
            raise RuntimeError("Gemini fallback not available.")
        if isinstance(request, dict):
            context = request.get("context") or {}
            step_instruction = request.get("user_query", "")
        else:
            context = getattr(request, "context", {}) or {}
            step_instruction = getattr(request, "user_query", "")
        original_query = context.get("original_query", "")

        prompt = (
            f"You are the fallback specialist for router tool `{tool_name}`.\n"
            "Provide a thoughtful, self-contained response even when primary agents fail.\n"
            "Instructions:\n"
            "- Derive or explain any mathematics rigorously with step-by-step reasoning.\n"
            "- When code is required, output Python snippets and describe expected outputs; "
            "assume execution in a safe environment but do not fabricate results without caveats.\n"
            "- When internet search is needed, hypothesise likely high-quality sources and cite them "
            "as inline references (e.g., [search:keyword] or known publications).\n"
            "- Make assumptions explicit, and flag any gaps that require real execution or live search.\n"
            "- Return the final answer in Markdown.\n"
        )
        prompt += f"\nOriginal user query:\n{original_query or 'N/A'}\n"
        prompt += f"\nCurrent routed instruction:\n{step_instruction}\n"
        if error:
            prompt += f"\nPrevious agent error: {error}\n"
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "top_p": 0.8},
            )
            text = getattr(response, "text", None)
            if text is None and hasattr(response, "candidates"):
                text = response.candidates[0].content.parts[0].text  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Gemini fallback generation failed: {exc}") from exc
        if not text:
            text = "Fallback model did not return content."
        metrics = {"status": "fallback", "model": self.model_name}
        if error:
            metrics["upstream_error"] = error
        return AgentResult(content=text, metrics=metrics)


fallback_manager = GeminiFallbackManager()


def _load_agent_class(
    agent_name: str,
    primary_path: Path,
    primary_class: str,
    fallback_path: Optional[Path] = None,
    fallback_class: Optional[str] = None,
):
    module = _load_module(f"{agent_name}_primary", primary_path)
    if module and hasattr(module, primary_class):
        AGENT_LOAD_LOG.append(f"Loaded {primary_class} from {primary_path}")
        return getattr(module, primary_class)
    if fallback_path and fallback_class:
        fallback_module = _load_module(f"{agent_name}_fallback", fallback_path)
        if fallback_module and hasattr(fallback_module, fallback_class):
            AGENT_LOAD_LOG.append(f"Using fallback {fallback_class} for {agent_name}")
            return getattr(fallback_module, fallback_class)
    AGENT_LOAD_LOG.append(f"No implementation available for {agent_name}")
    return None


AGENT_REGISTRY: Dict[str, Any] = {}


def _register_agent(name: str, agent_obj: Any) -> None:
    AGENT_REGISTRY[name] = agent_obj
    if name.startswith("/"):
        AGENT_REGISTRY[name.lstrip("/")] = agent_obj
    else:
        AGENT_REGISTRY[f"/{name}"] = agent_obj


if AgentRequest is not None and AgentResult is not None:
    # Math agent
    math_class = _load_agent_class(
        "math_agent",
        M6_ROOT / "math-agent" / "handler.py",
        "MathAgent",
        fallback_path=M6_ROOT / "math-agent" / "math_agent_template.py",
        fallback_class="TemplateMathAgent",
    )
    # Code agent
    code_class = _load_agent_class(
        "code_agent",
        M6_ROOT / "code-agent" / "handler.py",
        "CodeAgent",
    )
    # General-search agent
    general_class = _load_agent_class(
        "general_agent",
        M6_ROOT / "general-agent" / "handler.py",
        "GeneralSearchAgent",
    )

    class _StubAgent:
        def __init__(self, tool_name: str, message: str):
            self.name = tool_name
            self._message = message

        def invoke(self, request: Any) -> Any:
            if fallback_manager.available:
                try:
                    return fallback_manager.generate(self.name, request)
                except Exception as exc:  # pragma: no cover
                    AGENT_LOAD_LOG.append(f"Gemini fallback failed for {self.name}: {exc}")
            return AgentResult(
                content=self._message,
                metrics={"status": "stub", "tool": self.name},
            )

    if math_class is None:
        math_agent = _StubAgent("/math", "Math agent not yet implemented.")
    else:
        try:
            math_agent = math_class()
        except Exception as exc:
            AGENT_LOAD_LOG.append(f"MathAgent instantiation failed: {exc}")
            math_agent = _StubAgent("/math", f"Math agent load error: {exc}")
    _register_agent("/math", math_agent)

    if code_class is None:
        code_agent = _StubAgent("/code", "Code agent not yet implemented.")
    else:
        try:
            code_agent = code_class()
        except Exception as exc:
            AGENT_LOAD_LOG.append(f"CodeAgent instantiation failed: {exc}")
            code_agent = _StubAgent("/code", f"Code agent load error: {exc}")
    _register_agent("/code", code_agent)

    if general_class is None:
        general_agent = _StubAgent("/general-search", "General-search agent not yet implemented.")
    else:
        try:
            general_agent = general_class()
        except Exception as exc:
            AGENT_LOAD_LOG.append(f"GeneralSearchAgent instantiation failed: {exc}")
            general_agent = _StubAgent("/general-search", f"General agent load error: {exc}")
    _register_agent("/general-search", general_agent)
else:
    AGENT_LOAD_LOG.append("AgentRequest/AgentResult undefined; skipping agent registry.")


AGENT_STATUS_MARKDOWN = (
    "\n".join(f"- {line}" for line in AGENT_LOAD_LOG) if AGENT_LOAD_LOG else "- Agent stubs loaded successfully."
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

TOOL_REGEX = re.compile(r"^\s*(/[a-zA-Z0-9_-]+)")


def get_inference_client(option: RouterOption) -> Optional[Any]:
    if option.get("api"):
        return None
    base_repo = option.get("base")
    if not base_repo:
        return None
    if InferenceClient is None:
        CLIENT_ERRORS[_option_key(option)] = "huggingface_hub InferenceClient unavailable in this runtime."
        CLIENT_CACHE[_option_key(option)] = None
        return None
    key = _option_key(option)
    if key in CLIENT_CACHE:
        return CLIENT_CACHE[key]
    try:
        client = InferenceClient(model=base_repo, token=HF_TOKEN)
    except Exception as exc:  # pragma: no cover
        CLIENT_ERRORS[key] = str(exc)
        CLIENT_CACHE[key] = None
        return None
    CLIENT_ERRORS.pop(key, None)
    CLIENT_CACHE[key] = client
    return client


def describe_router_backend(option: RouterOption) -> str:
    api_endpoint = option.get("api") or ""
    if api_endpoint:
        key = _option_key(option)
        error = CLIENT_ERRORS.get(key)
        if error:
            return f"Custom router API `{api_endpoint}` error: {error}"
        return f"Using custom router API endpoint: `{api_endpoint}`"
    base_repo = option.get("base") or ""
    adapter_repo = option.get("adapter") or ""
    if not base_repo:
        return "Router backend not configured; using bundled sample plan."
    if InferenceClient is None:
        return "Router backend unavailable: huggingface_hub.InferenceClient is not installed."
    key = _option_key(option)
    error = CLIENT_ERRORS.get(key)
    if error:
        return f"Router backend failed to initialise `{base_repo}`: {error}"
    if key not in CLIENT_CACHE:
        # Ensure lazy init so that status reflects actual connection when possible.
        get_inference_client(option)
        error = CLIENT_ERRORS.get(key)
        if error:
            return f"Router backend failed to initialise `{base_repo}`: {error}"
    if adapter_repo:
        return f"Using base model `{base_repo}` with adapter `{adapter_repo}`"
    return f"Using Hugging Face Inference endpoint: `{base_repo}`"


def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    if start == -1:
        raise ValueError("Router output did not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw_text)):
        char = raw_text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_text[start : idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Router output is not valid JSON: {exc}") from exc
                finally:
                    break
    raise ValueError("Router output contained an unterminated JSON object.")


def _clean_router_text(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", text)
    fence_match = FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1)
    return cleaned.strip()


def call_router_model(user_query: str, option: RouterOption) -> Dict[str, Any]:
    prompt = f"{SYSTEM_PROMPT}\n\nUser query:\n{user_query.strip()}\n"
    api_endpoint = option.get("api") or ""
    base_repo = option.get("base") or ""
    adapter_repo = option.get("adapter") or ""

    if api_endpoint:
        key = _option_key(option)
        payload = {
            "prompt": prompt,
            "max_new_tokens": 900,
            "temperature": 0.2,
            "top_p": 0.9,
        }
        try:
            response = requests.post(api_endpoint, json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            CLIENT_ERRORS[key] = str(exc)
            return {
                "error": f"Router API request failed ({exc}). Falling back to sample plan.",
                "sample_plan": SAMPLE_PLAN,
            }
        try:
            data = response.json()
        except ValueError as exc:
            CLIENT_ERRORS[key] = f"Invalid JSON from router API: {exc}"
            return {
                "error": f"Router API returned non-JSON payload ({exc}). Falling back to sample plan.",
                "sample_plan": SAMPLE_PLAN,
            }
        text_output = data.get("text") or data.get("output")
        if not isinstance(text_output, str):
            CLIENT_ERRORS[key] = "API response missing 'text' field."
            return {
                "error": "Router API response missing 'text' field. Falling back to sample plan.",
                "sample_plan": SAMPLE_PLAN,
            }
        cleaned_text = _clean_router_text(text_output)
        try:
            result = extract_json_from_text(cleaned_text)
        except ValueError as exc:
            CLIENT_ERRORS[key] = str(exc)
            return {
                "error": f"Router API output parse error ({exc}). Falling back to sample plan.",
                "sample_plan": SAMPLE_PLAN,
            }
        CLIENT_ERRORS.pop(key, None)
        return result

    if not base_repo:
        return SAMPLE_PLAN
    inference_client = get_inference_client(option)
    if inference_client is None:
        error = CLIENT_ERRORS.get(_option_key(option)) or "Unknown client initialisation error."
        return {
            "error": f"Router call skipped: {error}",
            "sample_plan": SAMPLE_PLAN,
        }
    try:
        generation_kwargs = {
            "max_new_tokens": 900,
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        if adapter_repo:
            generation_kwargs["adapter_id"] = adapter_repo
        raw = inference_client.text_generation(
            prompt,
            **generation_kwargs,
        )
        if raw.lstrip().startswith("<"):
            raise ValueError(
                "Provider returned HTML (likely authentication or access denied). "
                "Verify that your HF token is authorised for this checkpoint."
            )
        cleaned_raw = _clean_router_text(raw)
        return extract_json_from_text(cleaned_raw)
    except Exception as exc:  # pragma: no cover
        error_message = f"{exc.__class__.__name__}: {exc}"
        # Some providers (e.g. Groq) expose conversational-only endpoints.
        if "not supported for provider" in str(exc).lower() and "conversational" in str(exc).lower():
            chat_kwargs: Dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query.strip()},
                ],
                "max_tokens": 900,
                "temperature": 0.2,
                "top_p": 0.9,
                "response_format": {"type": "json_object"},
            }
            if adapter_repo:
                chat_kwargs["extra_body"] = {"adapter_id": adapter_repo}
            try:
                try:
                    chat_response = inference_client.chat_completion(**chat_kwargs)
                except Exception as first_chat_exc:
                    # Some providers reject adapter_id/response_format; retry without.
                    lowered = str(first_chat_exc).lower()
                    retried = False
                    if "adapter_id" in lowered and "unsupported" in lowered and "extra_body" in chat_kwargs:
                        chat_kwargs.pop("extra_body", None)
                        retried = True
                    if "response_format" in lowered and "unsupported" in lowered:
                        chat_kwargs.pop("response_format", None)
                        retried = True
                    if retried:
                        chat_response = inference_client.chat_completion(**chat_kwargs)
                    else:
                        raise first_chat_exc
                choice = chat_response.choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                else:
                    message = getattr(choice, "message", None)
                content: Any = ""
                if isinstance(message, dict):
                    content = message.get("content", "")
                elif message is not None and hasattr(message, "content"):
                    content = getattr(message, "content")
                elif isinstance(choice, dict):
                    content = choice.get("content", "")
                else:
                    content = getattr(choice, "content", "")
                if isinstance(content, list):
                    # Some providers return a list of content parts; join text items.
                    content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
                if not isinstance(content, str):
                    content = str(content)
                cleaned_content = _clean_router_text(content)
                try:
                    return extract_json_from_text(cleaned_content)
                except ValueError as parse_exc:
                    snippet = cleaned_content.strip().splitlines()
                    excerpt = " ".join(snippet[:3])[:200]
                    raise ValueError(f"{parse_exc}; content excerpt: {excerpt}") from parse_exc
            except Exception as chat_exc:  # pragma: no cover
                error_message = f"{error_message}; chat fallback failed ({chat_exc.__class__.__name__}: {chat_exc})"

        CLIENT_ERRORS[_option_key(option)] = error_message
        return {
            "error": f"Router call failed ({error_message}). Falling back to sample plan.",
            "sample_plan": SAMPLE_PLAN,
        }


@_gpu_decorator(duration=120)
def gpu_router_plan(user_query: str, option: RouterOption) -> Dict[str, Any]:
    """GPU-backed wrapper so ZeroGPU hardware provisions the accelerator on demand."""
    return call_router_model(user_query, option)


def generate_plan(user_query: str, router_option_label: str) -> Dict[str, Any]:
    if not user_query.strip():
        raise gr.Error("Please provide a user query to route.")
    option = AVAILABLE_ROUTER_OPTIONS.get(router_option_label, AVAILABLE_ROUTER_OPTIONS["Use sample plan (no remote model)"])
    use_gpu = spaces is not None and os.environ.get("ROUTER_FORCE_CPU", "").lower() not in {"1", "true", "yes"}
    plan = (
        gpu_router_plan(user_query, option)
        if use_gpu
        else call_router_model(user_query, option)
    )
    return plan


def generate_plan_and_store(user_query: str, router_option_label: str) -> tuple[Dict[str, Any], str]:
    plan = generate_plan(user_query, router_option_label)
    return plan, user_query


def _resolve_plan_object(plan_input: Any) -> Optional[Dict[str, Any]]:
    plan_obj: Optional[Dict[str, Any]]
    if isinstance(plan_input, str):
        try:
            plan_obj = json.loads(plan_input)
        except json.JSONDecodeError:
            return None
    elif isinstance(plan_input, dict):
        plan_obj = plan_input
    else:
        return None
    if "route_plan" not in plan_obj and isinstance(plan_obj.get("sample_plan"), dict):
        plan_obj = plan_obj["sample_plan"]
    return plan_obj if isinstance(plan_obj, dict) else None


def execute_plan(plan_input: Any, original_query: str) -> Dict[str, Any]:
    if AgentRequest is None or AgentResult is None:
        return {"success": False, "error": "Agent interfaces unavailable; cannot execute plan."}
    plan_obj = _resolve_plan_object(plan_input)
    if not plan_obj:
        return {"success": False, "error": "Plan must be valid JSON with a route_plan field."}
    route_plan = plan_obj.get("route_plan")
    if not isinstance(route_plan, list):
        return {"success": False, "error": "Plan is missing a route_plan list."}

    results: List[Dict[str, Any]] = []
    for step_index, step in enumerate(route_plan):
        if not isinstance(step, str):
            results.append(
                {
                    "step_index": step_index,
                    "status": "invalid_step",
                    "message": "Route step must be a string.",
                }
            )
            continue
        match = TOOL_REGEX.match(step)
        tool_name = match.group(1) if match else "unknown"
        agent = AGENT_REGISTRY.get(tool_name) or AGENT_REGISTRY.get(tool_name.lstrip("/"))
        if agent is None:
            results.append(
                {
                    "step_index": step_index,
                    "tool": tool_name,
                    "status": "skipped",
                    "message": "No agent registered for this tool.",
                }
            )
            continue

        request = AgentRequest(
            user_query=step,
            context={"original_query": original_query},
            plan_metadata={"step_index": step_index, "raw_step": step},
        )
        try:
            agent_result = agent.invoke(request)
        except Exception as exc:
            if fallback_manager.available:
                try:
                    agent_result = fallback_manager.generate(tool_name, request, error=str(exc))
                except Exception as fallback_exc:  # pragma: no cover
                    results.append(
                        {
                            "step_index": step_index,
                            "tool": tool_name,
                            "status": "error",
                            "message": f"{exc}; fallback failed: {fallback_exc}",
                        }
                    )
                    continue
            else:
                results.append(
                    {
                        "step_index": step_index,
                        "tool": tool_name,
                        "status": "error",
                        "message": str(exc),
                    }
                )
                continue
        results.append(
            {
                "step_index": step_index,
                "tool": tool_name,
                "content": getattr(agent_result, "content", ""),
                "citations": getattr(agent_result, "citations", []),
                "artifacts": getattr(agent_result, "artifacts", []),
                "metrics": getattr(agent_result, "metrics", {}),
            }
        )
    return {"success": True, "results": results}


def run_startup_benchmark() -> Dict[str, Any]:
    if run_schema_evaluation is None or load_thresholds is None or evaluate_thresholds is None:
        return {"status": "unavailable", "message": "Benchmark utilities not available in this environment."}
    prediction_path = os.environ.get("ROUTER_BENCHMARK_PREDICTIONS")
    if not prediction_path:
        return {"status": "skipped", "message": "Set ROUTER_BENCHMARK_PREDICTIONS to auto-run benchmarks."}
    pred_path = Path(prediction_path)
    if not pred_path.exists():
        return {"status": "error", "message": f"Predictions file not found: {pred_path}"}
    if not BENCH_GOLD_PATH.exists() or not THRESHOLDS_PATH.exists():
        return {"status": "error", "message": "Benchmark gold or thresholds file missing."}
    try:
        schema_report = run_schema_evaluation(
            str(BENCH_GOLD_PATH),
            str(pred_path),
            max_error_examples=5,
        )
        thresholds = load_thresholds(THRESHOLDS_PATH)
        threshold_results = evaluate_thresholds(schema_report["metrics"], thresholds)
    except Exception as exc:
        return {"status": "error", "message": f"Benchmark run failed: {exc}"}
    status = "pass" if threshold_results.get("overall_pass") else "fail"
    return {
        "status": status,
        "message": f"Benchmark {status.upper()} on startup.",
        "report": {
            "schema_report": schema_report,
            "threshold_results": threshold_results,
        },
        "predictions_path": str(pred_path),
    }


def _compute_startup_benchmark() -> Dict[str, Any]:
    """Run the optional startup benchmark without crashing the Space on failure."""
    try:
        return run_startup_benchmark()
    except NameError:
        return {
            "status": "unavailable",
            "message": (
                "run_startup_benchmark() is not defined in the deployed bundle. "
                "Redeploy the latest app.py."
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive guard for remote exec.
        return {
            "status": "error",
            "message": f"Startup benchmark execution failed: {exc}",
        }


STARTUP_BENCHMARK_RESULT = _compute_startup_benchmark()


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


with gr.Blocks(title="CourseGPT Router Control Room") as demo:
    gr.Markdown(
        "## CourseGPT Router Control Room\n"
        "Milestone 6 deployment scaffold for the router agent. Populate the router model "
        "environment variables or select a hosted checkpoint to enable live inference."
    )

    initial_backend_status = f"**Backend status:** {describe_router_backend(DEFAULT_ROUTER_OPTION)}"
    backend_status_md = gr.Markdown(initial_backend_status)

    with gr.Tab("Router Planner"):
        router_option_state = gr.State(DEFAULT_ROUTER_LABEL)
        model_selector = gr.Radio(
            label="Router checkpoint",
            choices=list(AVAILABLE_ROUTER_OPTIONS.keys()),
            value=DEFAULT_ROUTER_LABEL,
        )

        def _set_router_option(choice: str) -> tuple[str, str]:
            option = AVAILABLE_ROUTER_OPTIONS.get(choice, AVAILABLE_ROUTER_OPTIONS["Use sample plan (no remote model)"])
            status = f"**Backend status:** {describe_router_backend(option)}"
            return choice, status

        model_selector.change(
            fn=_set_router_option,
            inputs=model_selector,
            outputs=[router_option_state, backend_status_md],
        )

        user_query_state = gr.State("")
        user_query = gr.Textbox(
            label="User query",
            lines=8,
            placeholder="Describe the task that needs routing...",
        )
        generate_btn = gr.Button("Generate plan", variant="primary")
        plan_output = gr.JSON(label="Router plan")
        generate_btn.click(
            fn=generate_plan_and_store,
            inputs=[user_query, router_option_state],
            outputs=[plan_output, user_query_state],
        )

        validate_btn = gr.Button("Run structural checks")
        validation_output = gr.JSON(label="Validation summary")
        validate_btn.click(fn=validate_plan, inputs=plan_output, outputs=validation_output)

        execute_btn = gr.Button("Simulate agent execution")
        execution_output = gr.JSON(label="Agent execution log")
        execution_markdown = gr.Markdown("")

        def execute_plan_and_format(plan_input: Any, original_query: str) -> tuple[Dict[str, Any], str]:
            result = execute_plan(plan_input, original_query)
            if not result.get("success"):
                return result, ""
            combined_sections: List[str] = []
            for entry in result.get("results", []):
                content = entry.get("content")
                if content:
                    step_idx = entry.get("step_index")
                    tool = entry.get("tool", "tool")
                    header = f"### Step {step_idx + 1} ({tool})" if isinstance(step_idx, int) else f"### {tool}"
                    combined_sections.append(f"{header}\n{content}")
            combined_markdown = "\n\n".join(combined_sections)
            return result, combined_markdown

        execute_btn.click(
            fn=execute_plan_and_format,
            inputs=[plan_output, user_query_state],
            outputs=[execution_output, execution_markdown],
        )

    with gr.Tab("Benchmark"):
        gr.Markdown(
            "Upload a JSONL file of router predictions (one JSON object per line). "
            "The file must align with the `router_benchmark_hard.jsonl` gold split."
        )
        startup_status = STARTUP_BENCHMARK_RESULT.get("message", "Benchmark not run.")
        gr.Markdown(f"**Startup benchmark status:** {startup_status}")
        if STARTUP_BENCHMARK_RESULT.get("report"):
            gr.JSON(
                value=STARTUP_BENCHMARK_RESULT["report"],
                label="Startup benchmark report",
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
        gr.Markdown("**Agent load summary:**\n" + AGENT_STATUS_MARKDOWN)

    demo.queue()


if __name__ == "__main__":  # pragma: no cover
    demo.launch()
