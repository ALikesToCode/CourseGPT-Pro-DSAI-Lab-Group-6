#!/usr/bin/env python3
"""
Gemini-powered synthetic dataset generator for the router agent.

The resulting dataset helps a router LLM learn when to dispatch sub-requests to
specialized agents:
    - /math(context) for math-leaning instructions.
    - /code(context) for Python-centric coding tasks.
    - /general-search(context, mode=web|rag|both) for a general search agent.

Usage:
    python gemini_router_dataset.py --count 12 --output data/router.jsonl

Environment:
    - Place GOOGLE_API_KEY (or GEMINI_API_KEY) in a .env or shell env.
    - Optionally set GEMINI_MODEL_NAME for the Gemini 2.5 Pro model identifier.
    - Requires the Google Gen AI `google-genai` SDK.

Features:
    - Parallel generation via --concurrency with thread-local Gemini clients.
    - Incremental JSONL appends after every successful example.
    - Automatic resume support that continues numbering after existing records.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sys
import textwrap
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency.
    def load_dotenv(*_args: Any, **_kwargs: Any) -> None:
        """Fallback no-op when python-dotenv is unavailable."""


try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - handled at runtime.
    genai = None  # type: ignore
    genai_errors = None  # type: ignore
    genai_types = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
except ImportError as exc:  # pragma: no cover - handled at runtime.
    Console = None  # type: ignore
    Panel = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    BarColumn = None  # type: ignore
    _RICH_IMPORT_ERROR = exc
else:
    _RICH_IMPORT_ERROR = None


ALLOWED_TOOLS = ["/math", "/code", "/general-search"]
ROUTE_VARIANTS = [
    {
        "name": "math_only",
        "required": ["/math"],
        "description": "Elite-level theoretical math reasoning, algebraic topology, tensor calculus, competition-grade proofs.",
    },
    {
        "name": "code_only",
        "required": ["/code"],
        "description": "Complex Python systems programming or ML experimentation requiring optimized, well-tested code.",
    },
    {
        "name": "general_only",
        "required": ["/general-search"],
        "description": "Curating authoritative research references, benchmarks, or competition archives to ground the solution.",
    },
    {
        "name": "math_plus_code",
        "required": ["/math", "/code"],
        "description": "Deep mathematical derivations whose validity is checked via rigorous Python experimentation or simulation.",
    },
    {
        "name": "math_plus_general",
        "required": ["/math", "/general-search"],
        "description": "Advanced math problems that demand literature review for the latest theorems, benchmarks, or constants.",
    },
    {
        "name": "code_plus_general",
        "required": ["/code", "/general-search"],
        "description": "Cutting-edge coding tasks guided by evidence from state-of-the-art papers, blogs, or competition problems.",
    },
    {
        "name": "tri_route",
        "required": ["/math", "/code", "/general-search"],
        "description": "Full-stack reasoning: derive theory, validate via Python, and gather references to justify every claim.",
    },
]

GENERAL_SEARCH_MODES = ["web", "rag", "both"]
ROUTER_DATASET_SCHEMA = {
    "id": "Unique identifier, e.g., router_0001.",
    "user_query": "Natural language user input that triggers the router.",
    "task_summary": "One sentence summary of the underlying intent.",
    "route_plan": "Ordered list of tool command strings.",
    "route_rationale": "Short explanation of why these tools were chosen.",
    "expected_artifacts": "List describing deliverables expected from sub-agents.",
    "thinking_outline": "Numbered list of 4-6 expert reasoning steps the router expects downstream agents to follow.",
    "handoff_plan": "Succinct narrative describing how sub-agents hand results to each other, including verification or fallback.",
    "difficulty": "introductory | intermediate | advanced",
    "tags": "List of topical tags or skills.",
    "quality_score": "Heuristic score (0-100) reflecting prompt richness and verification coverage.",
    "todo_list": "Ordered list of granular TODO items the router uses to brief sub-agents and track progress.",
    "acceptance_criteria": "Bullet list of pass/fail checks for graders.",
    "metrics": "Dictionary naming primary/secondary metrics with computation guidance.",
    "compute_budget": "Resource envelope (GPU minutes, CPU minutes, VRAM).",
    "repro": "Reproducibility contract (seed, determinism flags, frameworks).",
    "requires_browse": "Boolean indicating whether the task expects literature search.",
    "citation_policy": "Explicit citation expectations (e.g., minimum # of arXiv IDs).",
    "io_schema": "Declared input/output artifacts and file formats."
}


ADVANCED_THEMES = [
    "Transformers and large language model optimization",
    "Multivariable calculus and constrained optimization for deep learning",
    "Matrix calculus and spectral analysis in neural networks",
    "Tensor decomposition for representation learning",
    "Probabilistic graphical models and variational inference",
    "Non-convex optimization and saddle point analysis",
    "Reinforcement learning with policy gradients",
    "Competitive programming: IOI-level dynamic programming and graph theory",
    "Combinatorial game theory with algorithmic complexity",
    "Numerical linear algebra for GPU-accelerated simulation",
    "Stochastic processes for Bayesian machine learning",
    "Information theory and coding for generative AI safety",
    "Partial differential equations in physics-informed ML",
    "Topological data analysis for manifold learning",
    "Advanced statistics: causal inference under interventions",
    "Quantum machine learning and variational quantum eigensolvers",
    "Diffusion models and score-based generative modeling theory",
    "Meta-learning and few-shot adaptation strategies",
    "Neural architecture search and differentiable AutoML pipelines",
    "Federated learning with privacy-preserving aggregation",
    "Graph neural networks for molecular dynamics simulations",
    "Attention mechanisms and self-supervised representation learning theory",
    "Primal-dual convex optimization and ADMM variants",
    "Causal discovery and structural causal modeling under interventions",
    "Neural-symbolic theorem proving and program synthesis",
    "Quantum error correction and fault-tolerant computation",
    "Bayesian deep learning with stochastic differential equations",
    "Reinforcement learning for robotics with safety constraints",
    "Sparse coding and dictionary learning for compressed sensing",
]

VERIFY_KEYWORDS = (
    "verify",
    "validate",
    "check",
    "double-check",
    "cross-check",
    "audit",
    "prove",
    "sanity-check",
    "test",
    "ensure",
    "confirm",
    "cross-validate",
    "stress-test",
)

DOMAIN_KEYWORDS = {
    "tensor",
    "gradient",
    "lagrange",
    "bayesian",
    "stochastic",
    "transformer",
    "manifold",
    "spectral",
    "eigenvalue",
    "convergence",
    "variational",
    "federated",
    "diffusion",
    "graph neural",
    "meta-learning",
    "automl",
    "reinforcement",
    "policy gradient",
    "kkt",
    "admm",
    "primal-dual",
    "causal",
    "structural",
    "quantum",
    "vqe",
    "score-based",
    "sde",
    "topological",
    "gan",
    "variational inference",
    "markov",
    "graphcut",
    "sat solver",
    "symbolic",
    "fptas",
    "complexity",
    "ioi",
    "python",
    "numpy",
    "pytorch",
    "jax",
    "cuda",
    "arxiv",
    "mit",
    "lagrangian",
    "jacobian",
    "optimality",
    "stochastic gradient",
    "spectral norm",
    "fid",
    "nash",
    "monte carlo",
    "statistics",
    "regression",
    "curriculum",
    "prototype",
    "usability",
    "robotics",
    "encryption",
    "database",
    "cloud",
    "climate",
    "biology",
    "physics",
    "chemistry",
    "education",
    "finance",
    "econometrics",
    "thermodynamics",
    "biomedical",
    "cybersecurity",
    "supply chain",
    "control theory",
    "optimization",
    "modeling",
    "energy",
    "materials",
    "euler-lagrange",
    "hamiltonian",
    "poisson",
    "navier-stokes",
    "bayes",
    "likelihood",
    "prior",
    "posterior",
    "gradient boosting",
    "bootstrap",
    "currying",
    "memoization",
    "game theory",
    "pareto",
    "variance",
    "confidence interval",
    "precision",
    "recall",
    "roc",
    "auprc",
    "inception score",
    "score matching",
    "langevin",
    "sobolev",
    "laplacian",
    "hermite",
    "chebyshev",
    "fourier",
    "bernoulli",
    "mcmc",
    "kalman",
    "smith-waterman",
    "bioinformatics",
    "metropolis",
    "adversarial",
    "bandit",
    "curricular",
    "erlang",
    "lambda calculus"
}

MIN_USER_QUERY_LEN = 100
MIN_TASK_SUMMARY_LEN = 50
MIN_ROUTE_CONTEXT_LEN = 40

ANTI_PATTERNS = (
    "quadratic formula",
    "binary search",
    "fibonacci",
    "hello world",
    "two-sum",
    "prime factor",
    "bubble sort",
    "gaussian elimination homework",
)

DIFFICULTY_DISTRIBUTION = [
    "advanced",
    "advanced",
    "advanced",
    "intermediate",
    "advanced",
    "introductory",
    "advanced",
    "intermediate",
    "advanced",
]

TOOL_TODO_TEMPLATES = {
    "/math": "- [ ] /math: formalize proofs, check constraints, and document verification steps.",
    "/code": "- [ ] /code: implement the experiment in Python, run tests, and log metrics for validation.",
    "/general-search": "- [ ] /general-search: gather high-authority references and cross-check facts.",
    "router qa": "- [ ] router QA: consolidate tool outputs, verify citations, and approve the response.",
}

DEFAULT_ACCEPTANCE_CRITERIA = [
    "All expected artifacts enumerated in `expected_artifacts` are delivered and referenced in the final report.",
    "Primary metrics defined in `metrics` are computed, plotted, and meet the stated thresholds.",
    "Citations satisfy `citation_policy`, with persistent identifiers included in the report.",
]

DEFAULT_METRICS = {
    "primary": [
        "Define at least one domain-specific performance metric with formulas and acceptance thresholds.",
    ],
    "secondary": [
        "Report runtime/memory usage and summarize failure cases discovered during evaluation.",
    ],
}

DEFAULT_COMPUTE_BUDGET = {
    "gpu_minutes": 30,
    "cpu_minutes": 10,
    "vram_gb": 16,
}

DEFAULT_REPRO = {
    "seed": 1337,
    "deterministic": True,
    "framework": "specify (pytorch|jax|numpy)",
}

DEFAULT_IO_SCHEMA = {
    "artifacts": [
        "report.md",
        "figures/*.png",
        "metrics.json",
        "code/*.py",
    ],
    "logs": "logs/run.log",
}


def _contains_domain_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in DOMAIN_KEYWORDS)


def _count_domain_keywords(text: str) -> int:
    lowered = text.lower()
    return sum(1 for keyword in DOMAIN_KEYWORDS if keyword in lowered)


def _strip_code_fence(payload: str) -> str:
    """Remove markdown code fences when present."""
    fence_re = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)
    return fence_re.sub("", payload).strip()


def _as_json(text_block: str) -> Dict[str, Any]:
    """Attempt to coerce model text into JSON."""
    cleaned = _strip_code_fence(text_block)
    return json.loads(cleaned)


def _infer_api_key() -> Optional[str]:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"):
        candidate = os.getenv(key)
        if candidate:
            return candidate
    return None


def _ensure_packages() -> None:
    if _IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing google-genai dependency. Install it via:\n"
            "  pip install google-genai python-dotenv\n"
            f"Original import error: {_IMPORT_ERROR}"
        )


def _get_console() -> "Console":
    if _RICH_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing rich dependency. Install it via:\n"
            "  pip install rich\n"
            f"Original import error: {_RICH_IMPORT_ERROR}"
        )
    assert Console is not None  # For type checkers.
    return Console()


@dataclass
class RouterExample:
    """Container for validated dataset entries."""

    id: str
    user_query: str
    task_summary: str
    route_plan: List[str]
    route_rationale: str
    expected_artifacts: List[str]
    thinking_outline: List[str]
    handoff_plan: str
    todo_list: List[str]
    difficulty: str
    tags: List[str]
    quality_score: float
    acceptance_criteria: List[str]
    metrics: Dict[str, Any]
    compute_budget: Dict[str, Any]
    repro: Dict[str, Any]
    requires_browse: bool
    citation_policy: str
    io_schema: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "user_query": self.user_query,
                "task_summary": self.task_summary,
                "route_plan": self.route_plan,
                "route_rationale": self.route_rationale,
                "expected_artifacts": self.expected_artifacts,
                "thinking_outline": self.thinking_outline,
                "handoff_plan": self.handoff_plan,
                "todo_list": self.todo_list,
                "difficulty": self.difficulty,
                "tags": self.tags,
                "quality_score": round(self.quality_score, 2),
                "acceptance_criteria": self.acceptance_criteria,
                "metrics": self.metrics,
                "compute_budget": self.compute_budget,
                "repro": self.repro,
                "requires_browse": self.requires_browse,
                "citation_policy": self.citation_policy,
                "io_schema": self.io_schema,
            },
            ensure_ascii=True,
        )


class GeminiRouterDatasetBuilder:
    """Generate router dataset rows with Gemini."""

    def __init__(
        self,
        model_name: str,
        seed: Optional[int],
        temperature: float,
        max_retries: int,
        sleep_base: float,
        console: Optional["Console"] = None,
        freshness_window: int = 6,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.sleep_base = sleep_base
        self.console = console or _get_console()
        self._seed = seed if seed is not None else 7
        self._thread_local: threading.local = threading.local()
        self._freshness_window = max(1, freshness_window)
        self._recent_themes: deque[str] = deque(maxlen=self._freshness_window)
        self._recent_tags: deque[str] = deque(maxlen=self._freshness_window * 2)

        _ensure_packages()
        api_key = _infer_api_key()
        if not api_key:
            raise SystemExit(
                "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        self._api_key = api_key

    def variant_for_index(self, idx: int) -> Dict[str, Any]:
        """Return the routing variant for a given example index."""
        return ROUTE_VARIANTS[idx % len(ROUTE_VARIANTS)]

    def theme_for_index(self, idx: int) -> str:
        """Select an advanced STEM/AI theme for the user query."""
        rng_seed = (self._seed + 13) * (idx + 3)
        rng = random.Random(rng_seed)
        candidates = [theme for theme in ADVANCED_THEMES if theme not in self._recent_themes]
        if not candidates:
            candidates = ADVANCED_THEMES
        return rng.choice(candidates)

    def difficulty_for_index(self, idx: int) -> str:
        """Sample a difficulty level for diversified training."""
        rng_seed = (self._seed + 11) * (idx + 29)
        rng = random.Random(rng_seed)
        return rng.choice(DIFFICULTY_DISTRIBUTION)

    def generate_example(self, idx: int, variant: Dict[str, Any]) -> RouterExample:
        """Call Gemini to synthesize a single router-training example."""
        theme = self.theme_for_index(idx)
        difficulty = self.difficulty_for_index(idx)
        requested_mode = self._search_mode_for_index(idx)
        prompt = self._build_prompt(idx, variant, theme, difficulty, requested_mode)
        for attempt in range(1, self.max_retries + 1):
            try:
                client = self._get_client()
                assert genai_types is not None  # mypy safeguard.
                config = genai_types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                )
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                payload_obj = getattr(response, "parsed", None)
                if isinstance(payload_obj, dict):
                    data = payload_obj
                else:
                    payload_text = getattr(response, "text", "") or ""
                    data = _as_json(payload_text)
                self._sanitize_payload(
                    data,
                    variant,
                    difficulty,
                    idx,
                    requested_mode,
                    theme,
                )
                self._validate_payload(data, variant, difficulty)
                difficulty_value = data["difficulty"]
                quality_score = self._compute_quality_score(data, theme, difficulty_value)
                self.console.log(
                    f"[cyan]example {idx} ({difficulty_value}): quality_score={quality_score:.1f}[/cyan]"
                )
                example = RouterExample(
                    id=data["id"],
                    user_query=data["user_query"],
                    task_summary=data["task_summary"],
                    route_plan=data["route_plan"],
                    route_rationale=data["route_rationale"],
                    expected_artifacts=data["expected_artifacts"],
                    thinking_outline=data["thinking_outline"],
                    handoff_plan=data["handoff_plan"],
                    todo_list=data["todo_list"],
                    difficulty=difficulty_value,
                    tags=data["tags"],
                    quality_score=quality_score,
                    acceptance_criteria=data["acceptance_criteria"],
                    metrics=data["metrics"],
                    compute_budget=data["compute_budget"],
                    repro=data["repro"],
                    requires_browse=data["requires_browse"],
                    citation_policy=data["citation_policy"],
                    io_schema=data["io_schema"],
                )
                self._recent_themes.append(theme)
                for tag in example.tags:
                    if isinstance(tag, str) and tag.strip():
                        self._recent_tags.append(tag.strip())
                return example
            except (json.JSONDecodeError, AssertionError, KeyError, ValueError) as exc:
                self._handle_retry(exc, attempt, idx)
            except genai_errors.APIError as exc:  # pragma: no cover
                self._handle_retry(exc, attempt, idx)
        raise RuntimeError(
            f"Failed to synthesize example {idx} after {self.max_retries} attempts."
        )

    def _get_client(self) -> "genai.Client":
        """Provide a thread-local Gemini client for safe parallel usage."""
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = genai.Client(api_key=self._api_key)
            self._thread_local.client = client
        return client

    def _handle_retry(self, exc: Exception, attempt: int, idx: int) -> None:
        wait = self.sleep_base * (2 ** (attempt - 1))
        reason = f"[example {idx}] attempt {attempt} failed: {exc}"
        self.console.log(f"[yellow]{reason}[/yellow]")
        time.sleep(wait)

    @staticmethod
    def _split_command(command: str) -> Tuple[str, str]:
        if "(" in command and command.endswith(")"):
            prefix, rest = command.split("(", 1)
            return prefix, rest[:-1]
        return command, ""

    def _sanitize_payload(
        self,
        payload: Dict[str, Any],
        variant: Dict[str, Any],
        difficulty: str,
        idx: int,
        requested_mode: str,
        theme: str,
    ) -> None:
        payload["id"] = f"router_{idx:04d}"

        route_plan = payload.get("route_plan", [])
        sanitized_plan: List[str] = []
        commands_present: List[str] = []
        prev_prefix: Optional[str] = None
        domain_requirement = 2 if difficulty == "advanced" else 1
        for entry in route_plan:
            if not isinstance(entry, str):
                continue
            prefix, body = self._split_command(entry)
            if prev_prefix == prefix:
                continue
            if prefix == "/general-search" and "mode=" not in body:
                body = body.rstrip()
                if body:
                    body = body + f", mode={requested_mode}"
                else:
                    body = f"mode={requested_mode}"
            if prefix == "/code" and "python" not in body.lower():
                body = body + (", using Python" if body else "using Python")

            needed = max(0, domain_requirement - _count_domain_keywords(body))
            if needed > 0:
                additions = []
                lowered = body.lower()
                for keyword in DOMAIN_KEYWORDS:
                    if keyword in lowered or keyword in additions:
                        continue
                    additions.append(keyword)
                    if len(additions) >= needed:
                        break
                if additions:
                    qualifier = ", includes " + ", ".join(additions)
                    body = body + qualifier

            sanitized_plan.append(f"{prefix}({body})")
            commands_present.append(prefix)
            prev_prefix = prefix

        payload["route_plan"] = sanitized_plan

        todo_list = payload.get("todo_list")
        if not isinstance(todo_list, list):
            todo_list = []

        # Ensure router QA final task
        if not todo_list or "router qa" not in todo_list[-1].lower():
            todo_list.append(TOOL_TODO_TEMPLATES["router qa"])

        # Ensure each todo entry formatted and long enough
        formatted_todos: List[str] = []
        for item in todo_list:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if not stripped.startswith("- [ ]"):
                stripped = "- [ ] " + stripped.lstrip("- ")
            if len(stripped) < 15:
                stripped += " (expand details)"
            formatted_todos.append(stripped)
        todo_list = formatted_todos or [TOOL_TODO_TEMPLATES["router qa"]]

        # Ensure tool coverage
        for tool in commands_present:
            if tool not in ALLOWED_TOOLS:
                continue
            if not any(tool in item for item in todo_list):
                template = TOOL_TODO_TEMPLATES.get(
                    tool,
                    f"- [ ] {tool}: execute task, log outputs, and prepare summary.",
                )
                todo_list.insert(-1, template)

        # Verification todos
        min_verification = 2 if difficulty == "advanced" else 1
        todo_verify = sum(
            1 for item in todo_list if any(keyword in item.lower() for keyword in VERIFY_KEYWORDS)
        )
        if todo_verify < min_verification:
            for idx_todo, item in enumerate(todo_list[:-1]):
                if todo_verify >= min_verification:
                    break
                lowered = item.lower()
                if any(keyword in lowered for keyword in VERIFY_KEYWORDS):
                    continue
                todo_list[idx_todo] = item.rstrip(".") + " (verify results)"
                todo_verify += 1
        while todo_verify < min_verification:
            todo_list.insert(-1, "- [ ] router QA: verify intermediate outputs and cross-check metrics.")
            todo_verify += 1

        todo_min, todo_max = (
            (5, 8)
            if difficulty == "advanced"
            else (4, 7)
            if difficulty == "intermediate"
            else (3, 6)
        )

        while len(todo_list) < todo_min:
            todo_list.insert(
                -1,
                "- [ ] /general-search: expand references and validate context with additional sources.",
            )
        while len(todo_list) > todo_max and len(todo_list) > todo_min:
            for idx_todo, item in enumerate(todo_list[:-1]):
                if "router qa" in item.lower():
                    continue
                # Prefer removing entries that do not explicitly reference any required tool
                if not any(tool in item for tool in commands_present):
                    del todo_list[idx_todo]
                    break
                # If all remaining items are tool-specific, skip removal to preserve coverage
            else:
                todo_list = todo_list[: todo_max - 1] + [todo_list[-1]]

        # Final coverage pass: ensure each tool in route_plan has a corresponding TODO entry
        for tool in commands_present:
            if tool in ALLOWED_TOOLS and not any(tool in entry for entry in todo_list):
                template = TOOL_TODO_TEMPLATES.get(
                    tool,
                    f"- [ ] {tool}: execute task, log outputs, and prepare summary.",
                )
                todo_list.insert(-1, template)

        # Re-check verification steps after potential mutations
        todo_verify = sum(
            1 for item in todo_list if any(keyword in item.lower() for keyword in VERIFY_KEYWORDS)
        )
        if todo_verify < min_verification:
            for idx_todo, item in enumerate(todo_list[:-1]):
                if todo_verify >= min_verification:
                    break
                lowered = item.lower()
                if any(keyword in lowered for keyword in VERIFY_KEYWORDS):
                    continue
                todo_list[idx_todo] = item.rstrip(".") + " (verify results)"
                todo_verify += 1
        while todo_verify < min_verification:
            todo_list.insert(-1, "- [ ] router QA: verify intermediate outputs and cross-check metrics.")
            todo_verify += 1

        # Ensure TODO list stays within tier-specific bounds after adjustments
        while len(todo_list) < todo_min:
            todo_list.insert(
                -1,
                "- [ ] /general-search: expand references and validate context with additional sources.",
            )
        # When trimming to the max, avoid removing mission-critical tool entries
        while len(todo_list) > todo_max:
            removed = False
            for idx_todo, item in enumerate(todo_list[:-1]):
                if "router qa" in item.lower():
                    continue
                contains_tools = [tool for tool in commands_present if tool in item]
                # skip removal if this item is the only entry covering a required tool
                if any(
                    sum(1 for entry in todo_list if tool in entry) <= 1
                    for tool in contains_tools
                ):
                    continue
                del todo_list[idx_todo]
                removed = True
                break
            if not removed:
                break

        payload["todo_list"] = todo_list

        acceptance = payload.get("acceptance_criteria")
        if not isinstance(acceptance, list) or not acceptance:
            payload["acceptance_criteria"] = DEFAULT_ACCEPTANCE_CRITERIA.copy()
        else:
            payload["acceptance_criteria"] = [str(item).strip() for item in acceptance if str(item).strip()]

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            payload["metrics"] = json.loads(json.dumps(DEFAULT_METRICS))
        else:
            primary = metrics.get("primary")
            secondary = metrics.get("secondary")
            if not isinstance(primary, list) or not primary:
                metrics["primary"] = DEFAULT_METRICS["primary"].copy()
            if not isinstance(secondary, list) or not secondary:
                metrics["secondary"] = DEFAULT_METRICS["secondary"].copy()
            payload["metrics"] = metrics

        compute_budget = payload.get("compute_budget")
        if not isinstance(compute_budget, dict) or not compute_budget:
            payload["compute_budget"] = DEFAULT_COMPUTE_BUDGET.copy()
        else:
            payload["compute_budget"] = {
                "gpu_minutes": compute_budget.get("gpu_minutes", DEFAULT_COMPUTE_BUDGET["gpu_minutes"]),
                "cpu_minutes": compute_budget.get("cpu_minutes", DEFAULT_COMPUTE_BUDGET["cpu_minutes"]),
                "vram_gb": compute_budget.get("vram_gb", DEFAULT_COMPUTE_BUDGET["vram_gb"]),
            }

        repro = payload.get("repro")
        if not isinstance(repro, dict) or not repro:
            payload["repro"] = DEFAULT_REPRO.copy()
        else:
            framework_value = repro.get("framework", DEFAULT_REPRO["framework"])
            if isinstance(framework_value, list):
                framework_value = framework_value[0] if framework_value else DEFAULT_REPRO["framework"]
            elif isinstance(framework_value, dict):
                framework_value = next(
                    (
                        str(value).strip()
                        for value in framework_value.values()
                        if isinstance(value, str) and value.strip()
                    ),
                    DEFAULT_REPRO["framework"],
                )
            elif not isinstance(framework_value, str) or not framework_value.strip():
                framework_value = DEFAULT_REPRO["framework"]

            payload["repro"] = {
                "seed": repro.get("seed", DEFAULT_REPRO["seed"]),
                "deterministic": repro.get("deterministic", DEFAULT_REPRO["deterministic"]),
                "framework": framework_value.strip(),
            }

        requires_browse = payload.get("requires_browse")
        payload["requires_browse"] = bool(requires_browse) if isinstance(requires_browse, bool) else True

        citation_policy = payload.get("citation_policy")
        if not isinstance(citation_policy, str) or not citation_policy.strip():
            payload["citation_policy"] = "Cite ≥2 authoritative sources (arXiv DOI/URL) and include them in the final report."
        else:
            payload["citation_policy"] = citation_policy.strip()

        io_schema = payload.get("io_schema")
        if not isinstance(io_schema, dict) or not io_schema:
            payload["io_schema"] = json.loads(json.dumps(DEFAULT_IO_SCHEMA))
        else:
            artifacts = io_schema.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                io_schema["artifacts"] = DEFAULT_IO_SCHEMA["artifacts"].copy()
            io_schema.setdefault("logs", DEFAULT_IO_SCHEMA["logs"])
            logs_field = io_schema["logs"]
            if isinstance(logs_field, list):
                io_schema["logs"] = logs_field[0] if logs_field else DEFAULT_IO_SCHEMA["logs"]
            if not isinstance(io_schema["logs"], str) or not io_schema["logs"].strip():
                io_schema["logs"] = DEFAULT_IO_SCHEMA["logs"]
            outputs = io_schema.get("outputs")
            if outputs is None or not isinstance(outputs, dict):
                io_schema["outputs"] = {"artifacts": DEFAULT_IO_SCHEMA["artifacts"].copy()}
            else:
                artifacts_field = outputs.get("artifacts")
                if isinstance(artifacts_field, dict):
                    cleaned = [
                        str(value).strip()
                        for value in artifacts_field.values()
                        if isinstance(value, str) and value.strip()
                    ]
                    outputs["artifacts"] = cleaned or DEFAULT_IO_SCHEMA["artifacts"].copy()
                elif isinstance(artifacts_field, list):
                    cleaned = [
                        str(item).strip()
                        for item in artifacts_field
                        if isinstance(item, str) and item.strip()
                    ]
                    outputs["artifacts"] = cleaned or DEFAULT_IO_SCHEMA["artifacts"].copy()
                elif isinstance(artifacts_field, str) and artifacts_field.strip():
                    outputs["artifacts"] = [artifacts_field.strip()]
                else:
                    outputs["artifacts"] = DEFAULT_IO_SCHEMA["artifacts"].copy()
                logs_field = outputs.get("logs")
                if isinstance(logs_field, list):
                    outputs["logs"] = logs_field[0] if logs_field else DEFAULT_IO_SCHEMA["logs"]
                elif isinstance(logs_field, dict):
                    first = next(
                        (
                            str(value).strip()
                            for value in logs_field.values()
                            if isinstance(value, str) and value.strip()
                        ),
                        DEFAULT_IO_SCHEMA["logs"],
                    )
                    outputs["logs"] = first
                elif not isinstance(logs_field, str) or not logs_field.strip():
                    outputs["logs"] = DEFAULT_IO_SCHEMA["logs"]
            payload["io_schema"] = io_schema


    def _build_prompt(
        self,
        idx: int,
        variant: Dict[str, Any],
        theme: str,
        difficulty: str,
        requested_mode: str,
    ) -> str:
        required_commands = variant["required"]
        command_specs = []
        for tool in required_commands:
            if tool == "/math":
                command_specs.append(
                    "/math(context) where context gives math objective and known data."
                )
            elif tool == "/code":
                command_specs.append(
                    "/code(context) where context explains the Python program needed."
                )
            else:
                command_specs.append(
                    f"/general-search(context, mode={requested_mode}) tailored to gather sources; set tool_use_color to blue."
                )

        schema_lines = "\n".join(f"- {k}: {v}" for k, v in ROUTER_DATASET_SCHEMA.items())
        reference_hints = (
            "Focus on elite university / IOI calibre scenarios that require multistep reasoning, "
            "formal derivations, and careful verification. Blend advanced STEM mathematics with "
            "applications to deep learning, generative AI, or algorithmic competitions as appropriate."
        )
        search_hint = (
            "If /general-search is used, craft the query to target authoritative resources (e.g.,"
            " 'site:arxiv.org transformer convergence proof' or 'site:math.mit.edu multivariable optimization')."
        )
        glossary_preview = ", ".join(sorted(DOMAIN_KEYWORDS)[:40])
        anti_pattern_preview = ", ".join(ANTI_PATTERNS[:5])
        recent_theme_list = list(self._recent_themes)
        recent_tag_list = list(dict.fromkeys(self._recent_tags))
        recent_themes_str = ", ".join(recent_theme_list) if recent_theme_list else "none (full flexibility)"
        recent_tags_str = ", ".join(recent_tag_list) if recent_tag_list else "none (introduce fresh tags)"

        if difficulty == "advanced":
            user_line = (
                f"Provide a realistic, student-facing user_query centered on advanced STEM maths and {theme}, "
                f"at least 100 characters long, packed with domain-specific terminology, and NEVER mention introductory homework phrases ({anti_pattern_preview})."
            )
            summary_line = (
                "task_summary must be at least 50 characters and capture the graduate-level objective and constraints."
            )
            rigor_line = "Every scenario must feel graduate-level or IOI difficulty and require deep reasoning."
            thinking_range = "4-6"
            verification_text = "AT LEAST TWO"
            route_context_line = (
                "Each route_plan context must be ≥40 characters, include at least two glossary tokens, and cite concrete constraints or metrics—never reference introductory tasks."
            )
            todo_verify_text = (
                "Include at least two verification todo items (e.g., '- [ ] /math: verify derivation aligns with implementation')."
            )
            todo_line = (
                "todo_list must include 5-8 entries formatted as '- [ ] /tool: action ...', covering every tool, incorporating verification steps, and ending with a router QA review task."
            )
        elif difficulty == "intermediate":
            user_line = (
                f"Provide a competition-oriented user_query for {theme}, at least 80 characters, including at least one glossary token and avoiding intro-level phrasing."
            )
            summary_line = (
                "task_summary must be at least 40 characters summarizing the applied objective and success criteria."
            )
            rigor_line = "Scenario should challenge advanced undergrads or early graduate students while remaining actionable for the router."
            thinking_range = "4-5"
            verification_text = "AT LEAST ONE"
            route_context_line = (
                "Each route_plan context must be ≥35 characters, include at least one glossary token, and specify evaluation criteria or constraints (e.g., time complexity, accuracy threshold, safety margin)."
            )
            todo_verify_text = (
                "Include at least one verification todo item (e.g., '- [ ] /code: validate outputs against analytical baseline')."
            )
            todo_line = (
                "todo_list must include 4-7 entries using '- [ ] /tool: ...' phrasing, referencing each tool at least once and closing with a router QA validation task."
            )
        else:
            user_line = (
                f"Provide an ambitious but approachable user_query (≥60 characters) that introduces {theme} to a motivated learner, referencing at least one glossary token or precise domain phrase while avoiding banned homework tasks ({anti_pattern_preview})."
            )
            summary_line = (
                "task_summary must be at least 35 characters, clarifying the learning or applied outcome sought."
            )
            rigor_line = "Scenario should feel like a real user exploring the topic with practical stakes (project planning, applied research, mentoring)."
            thinking_range = "3-5"
            verification_text = "AT LEAST ONE"
            route_context_line = (
                "Each route_plan context must be ≥30 characters, include at least one glossary token or precise descriptor, and articulate why the tool is necessary."
            )
            todo_verify_text = (
                "Include at least one verification todo item to sanity-check outputs before router QA."
            )
            todo_line = (
                "todo_list must include 3-6 entries formatted '- [ ] ...', guiding the user through the tool sequence and concluding with router QA approval."
            )

        domain_requirement_count = 2 if difficulty == "advanced" else 1
        glossary_line = (
            f"When crafting route_plan contexts, explicitly include at least {domain_requirement_count} tokens chosen from this glossary (case-insensitive): {glossary_preview}, ... You may use other advanced terms too, but avoid repeating the same token twice in one context."
        )
        thinking_line = (
            f"thinking_outline must list {thinking_range} numbered strings (\"1.\", \"2.\", ...) that expose the reasoning chain; include {verification_text} verification/validation step(s) with words like \"verify\", \"check\", \"validate\", or \"audit\"."
        )
        difficulty_line = f"difficulty must be \"{difficulty}\"."
        tags_line = "Provide 1-3 topical tags (include at least one token tied to the theme)."
        route_examples = (
            "/math(Apply Karush-Kuhn-Tucker optimality conditions to the transformer sparsification Lagrangian with spectral norm and Lipschitz constraints while referencing the Hessian eigenvalue budget.)",
            "/code(Write JAX to implement score-based diffusion sampling with Hutchinson trace estimator, tracking FID metrics and enforcing spectral norm regularization.)",
            f"/general-search(query=\"site:arxiv.org diffusion score matching Hessian Lipschitz analysis\", mode={requested_mode})",
        )

        prompt = textwrap.dedent(
            f"""
            You are Gemini 2.5 Pro. Produce one JSON object describing a router-training example. No extra prose.

            ### Scenario Inputs
            - Variant: {variant['name']} → {variant['description']}
            - Theme: {theme}
            - Target difficulty: {difficulty}
            - Allowed tools: {', '.join(ALLOWED_TOOLS)}
            - Recent themes within the freshness window ({self._freshness_window}): {recent_themes_str}
            - Recently used tags (aim for novelty): {recent_tags_str}

            ### JSON Schema (all keys required exactly once)
            {schema_lines}

            ### Field requirements
            - {user_line}
            - {summary_line}
            - {rigor_line}
            - {thinking_line}
            - `route_rationale` must justify each tool and, whenever /general-search appears, explicitly reference the "blue" general-search agent.
            - {tags_line}
            - {difficulty_line}
            - Populate `expected_artifacts` with 3–5 bullet strings covering proofs, code, citations, and verification outputs.
            - {todo_verify_text}
            - {todo_line}
            - Ensure `todo_list` contains at least one entry for every tool present in `route_plan`, plus a final router QA review item.
            - `handoff_plan` must use ASCII arrows (`->`) and explicitly mention verification/fallback behavior (e.g., `/general-search -> /math -> /code -> router QA (verification: …; fallback: …)`).
            - `acceptance_criteria` should list ≥3 pass/fail bullets aligned with metrics and artifacts.
            - `metrics.primary` and `metrics.secondary` must spell out domain metrics and diagnostics with short computation guidance.
            - Provide plausible `compute_budget` and `repro` dictionaries (seed, determinism, framework).
            - State whether browsing is required (`requires_browse`) and provide a concrete `citation_policy` (e.g., “Cite ≥2 arXiv papers with IDs”).
            - Define `io_schema.artifacts` (report, plots, metrics JSON, code) using non-empty string paths, and set `io_schema.logs` so automation knows what to collect.

            ### Route-planning rules
            1. route_plan MUST be an ordered list of tool call strings.
            2. Use tool literals exactly as `/math`, `/code`, `/general-search`.
            3. When using /general-search, include `mode={requested_mode}` and describe high-authority queries (e.g., `site:arxiv.org`).
            4. When using /code, explicitly reference Python tooling and runtime validation checks.
            5. Never emit identical tools consecutively—merge reasoning or expand context instead.
            6. {route_context_line}
            7. {glossary_line}
            8. Treat /general-search as the “blue” tool inside `route_rationale` and justify every hand-off.
            9. Avoid reusing the exact same theme/tag mix listed above unless no alternatives remain; if you must reuse, explain why in `acceptance_criteria`.
            10. Example route entries (shuffle order as appropriate): "{route_examples[0]}", "{route_examples[1]}", "{route_examples[2]}".

            ### Acceptance, metrics, and citations
            - {todo_verify_text}
            - Align acceptance criteria, metrics, and expected artifacts so graders can evaluate pass/fail conclusively.
            - Cite ≥2 authoritative sources that satisfy the declared `citation_policy`, listing arXiv IDs/DOIs in the final artifact.
            - Budget (`compute_budget`) should reflect realistic GPU/CPU minutes and VRAM for the difficulty tier.

            ### Research expectations
            - {reference_hints}
            - {search_hint}
            - Encourage downstream agents to capture reasoning traces, verified computations, and feed results back to router QA.

            ### Anti-patterns to avoid
            - Do NOT create trivial homework prompts (e.g., {', '.join(ANTI_PATTERNS[:4])}).
            - Do not omit verification tasks, citations, or metric reporting.
            - Avoid vague requests (“discuss”, “explain”) without concrete acceptance hooks.
            - Avoid reusing recent themes/tags unless justified as described above.

            Return ONLY valid JSON (double-quoted keys/values). No markdown fences, commentary, or trailing prose.
            """
        ).strip()
        return prompt

    def _search_mode_for_index(self, idx: int) -> str:
        """Select a deterministic /general-search mode for the given index."""
        rng_seed = (self._seed + 1) * (idx + 1)
        rng = random.Random(rng_seed)
        return rng.choice(GENERAL_SEARCH_MODES)

    def _validate_payload(
        self,
        payload: Dict[str, Any],
        variant: Dict[str, Any],
        target_difficulty: str,
    ) -> None:
        required_keys = [
            "id",
            "user_query",
            "task_summary",
            "route_plan",
            "route_rationale",
            "expected_artifacts",
            "thinking_outline",
            "handoff_plan",
            "todo_list",
            "difficulty",
            "tags",
        ]
        for key in required_keys:
            if key not in payload:
                raise KeyError(f"Missing key `{key}` in payload.")

        difficulty = payload["difficulty"]
        if difficulty != target_difficulty:
            raise ValueError(
                f"Difficulty mismatch: expected '{target_difficulty}' but received '{difficulty}'."
            )

        if difficulty == "advanced":
            min_user_len = MIN_USER_QUERY_LEN
            min_summary_len = MIN_TASK_SUMMARY_LEN
            min_context_len = MIN_ROUTE_CONTEXT_LEN
            min_domain_query = 2
            min_domain_context = 2
            min_verification_steps = 2
            min_thinking_steps = 4
        elif difficulty == "intermediate":
            min_user_len = 80
            min_summary_len = 40
            min_context_len = 35
            min_domain_query = 1
            min_domain_context = 1
            min_verification_steps = 1
            min_thinking_steps = 4
        else:
            min_user_len = 60
            min_summary_len = 35
            min_context_len = 30
            min_domain_query = 1
            min_domain_context = 1
            min_verification_steps = 1
            min_thinking_steps = 3

        user_query = payload["user_query"].strip()
        if len(user_query) < min_user_len:
            raise ValueError(
                f"user_query too short for {difficulty} difficulty (min {min_user_len} chars)."
            )
        lowered_query = user_query.lower()
        if _count_domain_keywords(user_query) < min_domain_query:
            raise ValueError("user_query must reference the theme or domain terminology.")
        if any(pattern in lowered_query for pattern in ANTI_PATTERNS):
            raise ValueError("user_query appears to reference an introductory homework pattern.")

        task_summary = payload["task_summary"].strip()
        if len(task_summary) < min_summary_len:
            raise ValueError(
                f"task_summary too short for {difficulty} difficulty (min {min_summary_len} chars)."
            )

        if not isinstance(payload["route_plan"], list) or not payload["route_plan"]:
            raise ValueError("route_plan must be a non-empty list.")

        commands_present = set()
        previous_command = None
        for cmd in payload["route_plan"]:
            if not isinstance(cmd, str):
                raise ValueError("Each route_plan entry must be a string.")
            if not any(cmd.startswith(tool) for tool in ALLOWED_TOOLS):
                raise ValueError(f"route_plan entry uses unsupported tool: {cmd}")
            for tool in ALLOWED_TOOLS:
                if cmd.startswith(tool):
                    commands_present.add(tool)
            command_name = cmd.split("(", 1)[0]
            if previous_command is not None and command_name == previous_command:
                raise ValueError("route_plan should not repeat identical consecutive tool calls.")
            previous_command = command_name
            try:
                _, raw_context = cmd.split("(", 1)
                context = raw_context.rstrip(")")
            except ValueError:
                context = ""
            stripped_context = context.strip()
            if len(stripped_context) < min_context_len:
                raise ValueError(
                    f"Each route_plan context must provide ≥{min_context_len} characters of detailed guidance."
                )
            if _count_domain_keywords(stripped_context) < min_domain_context:
                raise ValueError("route_plan contexts must cite domain-specific terminology or constraints.")
            lower_context = stripped_context.lower()
            if any(pattern in lower_context for pattern in ANTI_PATTERNS):
                raise ValueError("route_plan context appears to reference prohibited introductory tasks.")
            if "/general-search" in cmd and "mode=" not in cmd:
                raise ValueError("General-search command missing mode parameter.")
            if "/code" in cmd and "Python" not in cmd:
                raise ValueError("Coding command must mention Python context explicitly.")

        for required_tool in variant["required"]:
            if required_tool not in commands_present:
                raise ValueError(
                    f"Required tool `{required_tool}` missing for variant {variant['name']}."
                )

        expected_artifacts = payload["expected_artifacts"]
        if not isinstance(expected_artifacts, list) or not (3 <= len(expected_artifacts) <= 5):
            raise ValueError("expected_artifacts must contain 3-5 descriptive deliverables.")
        for artifact in expected_artifacts:
            if not isinstance(artifact, str) or len(artifact.strip()) < 10:
                raise ValueError("Each expected_artifacts entry must be a descriptive string.")

        todo_list = payload["todo_list"]
        if not isinstance(todo_list, list):
            raise ValueError("todo_list must be an array of strings.")
        if difficulty == "advanced":
            todo_min, todo_max = 5, 8
        elif difficulty == "intermediate":
            todo_min, todo_max = 4, 7
        else:
            todo_min, todo_max = 3, 6
        if not (todo_min <= len(todo_list) <= todo_max):
            raise ValueError(
                f"todo_list must contain between {todo_min} and {todo_max} entries for {difficulty} prompts."
            )
        todo_verify = 0
        covered_tools = set()
        for todo in todo_list:
            if not isinstance(todo, str) or len(todo.strip()) < 15:
                raise ValueError("Each todo_list entry must be a descriptive string (>=15 chars).")
            if "[ ]" not in todo:
                raise ValueError("Each todo_list item must start with '[ ]' to signal an unenforced task.")
            lower_todo = todo.lower()
            if any(keyword in lower_todo for keyword in VERIFY_KEYWORDS):
                todo_verify += 1
            for tool in ALLOWED_TOOLS:
                if tool in todo:
                    covered_tools.add(tool)
            if "router qa" in lower_todo:
                covered_tools.add("router qa")
        if todo_verify < min_verification_steps:
            raise ValueError(
                f"todo_list must include at least {min_verification_steps} verification/validation tasks."
            )
        if "router qa" not in todo_list[-1].lower():
            raise ValueError("The final todo item must route results back to router QA for approval.")
        missing_tools = {tool for tool in commands_present if tool in ALLOWED_TOOLS and tool not in covered_tools}
        if missing_tools:
            raise ValueError(
                f"todo_list must include tasks referencing each tool in route_plan. Missing: {sorted(missing_tools)}"
            )
        if "router qa" not in covered_tools:
            raise ValueError("todo_list must include an explicit router QA review task.")

        acceptance = payload["acceptance_criteria"]
        if not isinstance(acceptance, list) or not acceptance:
            raise ValueError("acceptance_criteria must be a non-empty list of success checks.")
        for item in acceptance:
            if not isinstance(item, str) or len(item.strip()) < 10:
                raise ValueError("Each acceptance criterion must be a descriptive string (>=10 chars).")

        metrics = payload["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dictionary with primary/secondary entries.")
        primary = metrics.get("primary")
        secondary = metrics.get("secondary")
        if not isinstance(primary, list) or not primary:
            raise ValueError("metrics.primary must enumerate domain metrics to compute.")
        if not isinstance(secondary, list) or not secondary:
            raise ValueError("metrics.secondary must enumerate auxiliary diagnostics.")
        if any(not isinstance(m, str) or not m.strip() for m in primary + secondary):
            raise ValueError("All metric descriptions must be non-empty strings.")

        compute_budget = payload["compute_budget"]
        if not isinstance(compute_budget, dict):
            raise ValueError("compute_budget must be a dictionary with resource estimates.")
        for key in ("gpu_minutes", "cpu_minutes", "vram_gb"):
            if key not in compute_budget:
                raise ValueError(f"compute_budget missing key '{key}'.")
            if not isinstance(compute_budget[key], (int, float)):
                raise ValueError(f"compute_budget.{key} must be numeric.")

        repro = payload["repro"]
        if not isinstance(repro, dict) or "seed" not in repro:
            raise ValueError("repro must define reproducibility parameters (seed, determinism, framework).")
        if not isinstance(repro.get("framework"), str) or not repro["framework"].strip():
            raise ValueError("repro.framework must be a non-empty string naming the primary framework.")

        if not isinstance(payload["requires_browse"], bool):
            raise ValueError("requires_browse must be boolean.")

        citation_policy = payload["citation_policy"]
        if not isinstance(citation_policy, str) or not citation_policy.strip():
            raise ValueError("citation_policy must be a non-empty string.")

        io_schema = payload["io_schema"]
        if not isinstance(io_schema, dict):
            raise ValueError("io_schema must be a dictionary describing deliverables.")
        if "artifacts" not in io_schema or not isinstance(io_schema["artifacts"], list):
            raise ValueError("io_schema.artifacts must enumerate expected files.")
        if any(not isinstance(path, str) or not path.strip() for path in io_schema["artifacts"]):
            raise ValueError("Each io_schema.artifacts entry must be a non-empty string path.")
        logs_field = io_schema.get("logs")
        if not isinstance(logs_field, str) or not logs_field.strip():
            raise ValueError("io_schema.logs must be a non-empty string path.")
        outputs = io_schema.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("io_schema.outputs must be an object containing an artifacts list.")
        artifacts_list = outputs.get("artifacts")
        if not isinstance(artifacts_list, list) or not artifacts_list:
            raise ValueError("io_schema.outputs.artifacts must be a non-empty list of paths.")
        if any(not isinstance(path, str) or not path.strip() for path in artifacts_list):
            raise ValueError("Each entry in io_schema.outputs.artifacts must be a non-empty string path.")
        logs_output = outputs.get("logs")
        if logs_output is not None and (not isinstance(logs_output, str) or not logs_output.strip()):
            raise ValueError("io_schema.outputs.logs must be a non-empty string path if provided.")

        thinking_outline = payload["thinking_outline"]
        if not isinstance(thinking_outline, list) or len(thinking_outline) < min_thinking_steps:
            raise ValueError(
                f"thinking_outline must be a list with at least {min_thinking_steps} reasoning steps."
            )
        verification_steps = 0
        for step in thinking_outline:
            if not isinstance(step, str) or len(step.strip()) < 10:
                raise ValueError("Each thinking_outline entry must be a descriptive string.")
            stripped = step.strip()
            if not stripped[0].isdigit():
                raise ValueError("thinking_outline steps should be numbered (e.g., '1. ...').")
            lowered = stripped.lower()
            if any(keyword in lowered for keyword in VERIFY_KEYWORDS):
                verification_steps += 1
        if verification_steps < min_verification_steps:
            raise ValueError(
                f"thinking_outline must include at least {min_verification_steps} explicit verification or validation steps."
            )

        handoff_plan = payload["handoff_plan"]
        if not isinstance(handoff_plan, str) or len(handoff_plan.strip()) < 40:
            raise ValueError("handoff_plan must be a descriptive string (>= 40 chars).")
        normalized_handoff = handoff_plan.lower()
        if "->" not in handoff_plan and "→" not in handoff_plan:
            raise ValueError("handoff_plan must illustrate flow using '->' (e.g., /general-search -> /math -> /code).")
        # Encourage mention of verification/fallback.
        if "verify" not in normalized_handoff and "fallback" not in normalized_handoff and "loop" not in normalized_handoff:
            raise ValueError("handoff_plan must mention verification, fallback, or loop behaviour.")

        tags = payload["tags"]
        if not isinstance(tags, list) or not tags:
            raise ValueError("tags must be a non-empty list.")

        route_rationale = payload["route_rationale"]
        if "/general-search" in commands_present and "blue" not in route_rationale.lower():
            raise ValueError("Route rationale must mention blue when general-search is used.")


    def _compute_quality_score(
        self,
        payload: Dict[str, Any],
        theme: str,
        difficulty: str,
    ) -> float:
        user_query = payload["user_query"].strip()
        task_summary = payload["task_summary"].strip()
        route_plan = payload["route_plan"]
        thinking_outline = payload["thinking_outline"]
        tags = payload.get("tags", [])
        todo_list = payload.get("todo_list", [])

        contexts: List[str] = []
        for cmd in route_plan:
            try:
                _, raw = cmd.split("(", 1)
                context = raw.rstrip(")")
            except ValueError:
                context = ""
            contexts.append(context.strip())
        avg_context_len = sum(len(ctx) for ctx in contexts) / max(len(contexts), 1)

        verification_steps = sum(
            1
            for step in thinking_outline
            if any(keyword in step.lower() for keyword in VERIFY_KEYWORDS)
        )

        lowered_query = user_query.lower()
        domain_hits = sum(1 for keyword in DOMAIN_KEYWORDS if keyword in lowered_query)
        theme_lower = theme.lower()
        theme_hit = 1 if theme_lower in lowered_query or any(theme_lower in tag.lower() for tag in tags) else 0
        todo_len = len(todo_list)
        todo_verify = sum(
            1
            for todo in todo_list
            if isinstance(todo, str) and any(keyword in todo.lower() for keyword in VERIFY_KEYWORDS)
        )
        todo_router = 1 if todo_list and "router qa" in todo_list[-1].lower() else 0

        difficulty_scaler = {
            "advanced": 1.0,
            "intermediate": 0.8,
            "introductory": 0.65,
        }
        scale = difficulty_scaler.get(difficulty, 1.0)
        user_norm = 150.0 * scale
        summary_norm = 80.0 * scale
        context_norm = 80.0 * scale
        expected_verification = 2 if difficulty == "advanced" else 1
        expected_todos = 5 if difficulty == "advanced" else (4 if difficulty == "intermediate" else 3)

        score = 0.0
        score += min(len(user_query) / user_norm, 1.5) * 20
        score += min(len(task_summary) / summary_norm, 1.0) * 10
        score += min(avg_context_len / context_norm, 1.5) * 25
        score += min(verification_steps / max(expected_verification, 1), 1.2) * 20
        score += min(domain_hits, 6) * 3
        score += theme_hit * 7
        score += min(todo_len / max(expected_todos, 1), 1.3) * 8
        score += min(todo_verify / max(expected_verification, 1), 1.1) * 4
        score += todo_router * 3

        return max(0.0, min(score, 100.0))


@dataclass
class GenerationFailure:
    """Capture metadata for a failed example synthesis."""

    index: int
    variant: str
    error: str


@dataclass
class GenerationSummary:
    """Report aggregate statistics for a generation run."""

    successes: int
    failures: List[GenerationFailure]
    duration_seconds: float


class JsonlWriter:
    """Thread-safe JSONL writer that flushes on every record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_output_dir(path)
        self._ensure_trailing_newline(path)
        self._stream = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def append(self, example: RouterExample) -> None:
        record = example.to_json()
        with self._lock:
            self._stream.write(record + "\n")
            self._stream.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc: Optional[BaseException], _: Optional[BaseException]) -> None:
        self.close()

    def _ensure_trailing_newline(self, path: Path) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            last = handle.read(1)
            if last not in (b"\n", b"\r"):
                handle.write(b"\n")


class DatasetOrchestrator:
    """Coordinate concurrent Gemini requests and incremental file writes."""

    def __init__(
        self,
        builder: GeminiRouterDatasetBuilder,
        writer: JsonlWriter,
        console: "Console",
    ) -> None:
        self.builder = builder
        self.writer = writer
        self.console = console
        self._variant_counts: Dict[str, int] = {variant["name"]: 0 for variant in ROUTE_VARIANTS}
        self._theme_counts: Dict[str, int] = {}
        self._theme_alerts: set[str] = set()
        self._difficulty_counts: Dict[str, int] = {}
        self._difficulty_alerted = False
        self._imbalance_warned = False
        self._last_signature: Optional[Tuple[str, ...]] = None

    def generate(
        self,
        *,
        count: int,
        start_index: int,
        concurrency: int,
    ) -> GenerationSummary:
        start_time = time.time()
        failures: List[GenerationFailure] = []
        successes = 0

        assert (
            Progress is not None
            and SpinnerColumn is not None
            and TextColumn is not None
            and BarColumn is not None
        )
        columns = [
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
        ]

        with Progress(*columns, console=self.console) as progress:
            task_id = progress.add_task(
                "Synthesizing router examples",
                total=count,
            )

            if concurrency <= 1:
                for offset in range(count):
                    example_idx = start_index + offset
                    variant = self.builder.variant_for_index(example_idx)
                    try:
                        example = self.builder.generate_example(example_idx, variant)
                        self.writer.append(example)
                        successes += 1
                        self._check_diversity(example, variant["name"])
                    except Exception as exc:  # pragma: no cover - rely on runtime signals.
                        failures.append(
                            GenerationFailure(
                                index=example_idx,
                                variant=variant["name"],
                                error=str(exc),
                            )
                        )
                        self.console.log(
                            f"[red]Failed example {example_idx}: {exc}[/red]"
                        )
                    finally:
                        progress.advance(task_id, 1)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_map = {}
                    for offset in range(count):
                        example_idx = start_index + offset
                        variant = self.builder.variant_for_index(example_idx)
                        future = executor.submit(
                            self.builder.generate_example,
                            example_idx,
                            variant,
                        )
                        future_map[future] = (example_idx, variant)
                    try:
                        for future in as_completed(future_map):
                            example_idx, variant = future_map[future]
                            try:
                                example = future.result()
                                self.writer.append(example)
                                successes += 1
                                self._check_diversity(example, variant["name"])
                            except Exception as exc:  # pragma: no cover - runtime path.
                                failures.append(
                                    GenerationFailure(
                                        index=example_idx,
                                        variant=variant["name"],
                                        error=str(exc),
                                    )
                                )
                                self.console.log(
                                    f"[red]Failed example {example_idx}: {exc}[/red]"
                                )
                            finally:
                                progress.advance(task_id, 1)
                    except KeyboardInterrupt:
                        for future in future_map:
                            future.cancel()
                        raise

        duration = time.time() - start_time
        return GenerationSummary(
            successes=successes,
            failures=failures,
            duration_seconds=duration,
        )

    def _check_diversity(self, example: RouterExample, variant_name: str) -> None:
        signature = tuple(cmd.split("(", 1)[0] for cmd in example.route_plan)
        if self._last_signature == signature:
            self.console.log(
                "[yellow]Consecutive examples share identical route signature; consider diversifying tool order.[/yellow]"
            )
        self._last_signature = signature

        self._variant_counts[variant_name] = self._variant_counts.get(variant_name, 0) + 1
        total = sum(self._variant_counts.values())
        if total >= len(self._variant_counts) * 3 and not self._imbalance_warned:
            avg = total / len(self._variant_counts)
            imbalanced = [
                name for name, count in self._variant_counts.items() if count > avg + 2
            ]
            if imbalanced:
                self.console.log(
                    f"[yellow]Variant utilization skew detected: {imbalanced}. Rotate prompts for balance.[/yellow]"
                )
                self._imbalance_warned = True

        theme_token = example.tags[0] if example.tags else ""
        if theme_token:
            self._theme_counts[theme_token] = self._theme_counts.get(theme_token, 0) + 1
            if (
                self._theme_counts[theme_token] >= 3
                and theme_token not in self._theme_alerts
            ):
                self.console.log(
                    f"[yellow]Theme '{theme_token}' appearing frequently ({self._theme_counts[theme_token]} times). Inject variety.[/yellow]"
                )
                self._theme_alerts.add(theme_token)

        diff = example.difficulty
        self._difficulty_counts[diff] = self._difficulty_counts.get(diff, 0) + 1
        total_diff = sum(self._difficulty_counts.values())
        if total_diff >= 8 and not self._difficulty_alerted:
            adv_ratio = self._difficulty_counts.get("advanced", 0) / total_diff
            if adv_ratio > 0.75 or adv_ratio < 0.25:
                self.console.log(
                    "[yellow]Difficulty mix skew detected (advanced ratio {:.0%}). Consider prompting for more varied tiers.[/yellow]".format(adv_ratio)
                )
                self._difficulty_alerted = True
# End Patch
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini 2.5 Pro synthetic dataset generator for the router agent."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of dataset rows to generate (default: 12).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("router_dataset.jsonl"),
        help="Path for the dataset file (JSONL).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro"),
        help="Gemini model name (default from GEMINI_MODEL_NAME or gemini-2.5-pro).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Sampling temperature passed to Gemini (default: 0.85).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum Gemini retry attempts per example (default: 4).",
    )
    parser.add_argument(
        "--sleep-base",
        type=float,
        default=1.5,
        help="Base seconds for exponential backoff between retries (default: 1.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic variant ordering.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel Gemini requests to run (default: 1).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "Zero-based index for dataset IDs. Defaults to appending after the existing"
            " number of records in the output file."
        ),
    )
    parser.add_argument(
        "--repair-ids",
        type=Path,
        help=(
            "If set, renumber the JSONL file so ids follow router_XXXX order in file sequence and exit."
        ),
    )
    parser.add_argument(
        "--repair-schema",
        type=Path,
        help=(
            "If set, patch JSONL records to include all schema fields with sensible defaults and exit."
        ),
    )
    parser.add_argument(
        "--repair-start",
        type=int,
        default=0,
        help="Starting integer to use when repairing ids (default: 0 → router_0000).",
    )
    return parser.parse_args(argv)


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def repair_jsonl_ids(file_path: Path, start: int) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot repair ids: {file_path} does not exist")

    with file_path.open("r", encoding="utf-8") as reader:
        lines = [line.rstrip("\n") for line in reader if line.strip()]

    repaired_lines: List[str] = []
    for offset, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Line {offset + 1} in {file_path} is not valid JSON and cannot be repaired"
            ) from exc

        payload["id"] = f"router_{start + offset:04d}"
        repaired_lines.append(json.dumps(payload, ensure_ascii=True))

    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    file_path.rename(backup_path)
    with file_path.open("w", encoding="utf-8") as writer:
        writer.write("\n".join(repaired_lines) + "\n")

    print(f"Repaired ids written to {file_path} (backup saved as {backup_path})")


def _schema_backup_path(file_path: Path, label: str) -> Path:
    candidate = file_path.with_name(f"{file_path.name}.{label}.bak")
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        alt = file_path.with_name(f"{file_path.name}.{label}.bak{idx}")
        if not alt.exists():
            return alt
        idx += 1


def repair_jsonl_schema(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot repair schema: {file_path} does not exist")

    with file_path.open("r", encoding="utf-8") as reader:
        raw_lines = [line.rstrip("\n") for line in reader if line.strip()]

    if not raw_lines:
        print(f"No records detected in {file_path}; nothing to repair.")
        return

    def _copy_default(obj: Any) -> Any:
        return json.loads(json.dumps(obj))

    updated_lines: List[str] = []
    field_repairs: Dict[str, int] = {}
    records_repaired = 0

    for idx, raw_line in enumerate(raw_lines):
        line_no = idx + 1
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_no} in {file_path} is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_no} in {file_path} is not a JSON object.")

        changed = False

        def mark(field: str) -> None:
            nonlocal changed
            field_repairs[field] = field_repairs.get(field, 0) + 1
            changed = True

        def clean_string(value: Any, *, field: str, allow_empty: bool = False) -> str:
            if isinstance(value, str):
                stripped = value.strip()
            elif value is None:
                stripped = ""
            else:
                stripped = str(value).strip()
            if not stripped and not allow_empty:
                raise ValueError(f"Line {line_no}: field '{field}' must be a non-empty string.")
            return stripped

        def optional_string(value: Any) -> Optional[str]:
            if isinstance(value, str):
                stripped = value.strip()
            elif value is None:
                return None
            else:
                stripped = str(value).strip()
            return stripped or None

        def clean_string_list(
            value: Any,
            *,
            field: str,
            fallback: Optional[List[str]] = None,
            allow_empty: bool = False,
        ) -> List[str]:
            if isinstance(value, list):
                source_iter = value
            elif isinstance(value, (tuple, set)):
                source_iter = list(value)
            elif isinstance(value, str):
                source_iter = [value]
            elif value is None:
                source_iter = []
            else:
                source_iter = [value]

            items: List[str] = []
            for entry in source_iter:
                maybe = optional_string(entry)
                if maybe:
                    items.append(maybe)
            if not items:
                if fallback is not None:
                    return list(fallback)
                if allow_empty:
                    return []
                raise ValueError(f"Line {line_no}: field '{field}' must provide at least one entry.")
            return items

        for token in ("id", "user_query", "task_summary", "route_rationale", "handoff_plan"):
            original = payload.get(token)
            cleaned = clean_string(original, field=token)
            if cleaned != original:
                payload[token] = cleaned
                mark(token)

        difficulty_original = payload.get("difficulty")
        difficulty_clean = clean_string(difficulty_original, field="difficulty")
        normalized_difficulty = difficulty_clean.lower()
        if normalized_difficulty not in {"introductory", "intermediate", "advanced"}:
            normalized_difficulty = "advanced"
        if normalized_difficulty != difficulty_original:
            payload["difficulty"] = normalized_difficulty
            mark("difficulty")

        route_plan_original = payload.get("route_plan")
        route_plan = clean_string_list(route_plan_original, field="route_plan")
        if route_plan != route_plan_original:
            payload["route_plan"] = route_plan
            mark("route_plan")

        expected_artifacts_original = payload.get("expected_artifacts")
        expected_artifacts = clean_string_list(
            expected_artifacts_original,
            field="expected_artifacts",
        )
        if expected_artifacts != expected_artifacts_original:
            payload["expected_artifacts"] = expected_artifacts
            mark("expected_artifacts")

        thinking_outline_original = payload.get("thinking_outline")
        thinking_outline = clean_string_list(
            thinking_outline_original,
            field="thinking_outline",
        )
        if thinking_outline != thinking_outline_original:
            payload["thinking_outline"] = thinking_outline
            mark("thinking_outline")

        tags_original = payload.get("tags")
        tags = clean_string_list(
            tags_original,
            field="tags",
            fallback=["router"],
        )
        if tags != tags_original:
            payload["tags"] = tags
            mark("tags")

        todo_original = payload.get("todo_list")
        todo_entries = clean_string_list(
            todo_original,
            field="todo_list",
            allow_empty=True,
        )
        todo_changed = False
        if not todo_entries:
            todo_entries = [TOOL_TODO_TEMPLATES["router qa"]]
            todo_changed = True
        if todo_entries and "router qa" not in todo_entries[-1].lower():
            todo_entries.append(TOOL_TODO_TEMPLATES["router qa"])
            todo_changed = True
        if todo_changed or todo_entries != todo_original:
            payload["todo_list"] = todo_entries
            mark("todo_list")

        acceptance_original = payload.get("acceptance_criteria")
        acceptance = clean_string_list(
            acceptance_original,
            field="acceptance_criteria",
            fallback=DEFAULT_ACCEPTANCE_CRITERIA,
        )
        if acceptance != acceptance_original:
            payload["acceptance_criteria"] = acceptance
            mark("acceptance_criteria")

        metrics_original = payload.get("metrics")
        if isinstance(metrics_original, dict):
            sanitized_metrics: Dict[str, Any] = {
                key: value
                for key, value in metrics_original.items()
                if key not in {"primary", "secondary"}
            }
            primary_list = clean_string_list(
                metrics_original.get("primary"),
                field="metrics.primary",
                fallback=DEFAULT_METRICS["primary"],
            )
            secondary_list = clean_string_list(
                metrics_original.get("secondary"),
                field="metrics.secondary",
                fallback=DEFAULT_METRICS["secondary"],
            )
            sanitized_metrics["primary"] = primary_list
            sanitized_metrics["secondary"] = secondary_list
        else:
            sanitized_metrics = _copy_default(DEFAULT_METRICS)
        if sanitized_metrics != metrics_original:
            payload["metrics"] = sanitized_metrics
            mark("metrics")

        def coerce_numeric(value: Any, base: Any) -> Any:
            target_type = type(base)
            if isinstance(value, (int, float)):
                coerced = target_type(value)
                if target_type is int:
                    coerced = int(round(float(value)))
                return coerced
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return base
                try:
                    parsed = float(stripped)
                except ValueError:
                    return base
                coerced = target_type(parsed)
                if target_type is int:
                    coerced = int(round(parsed))
                return coerced
            return base

        compute_original = payload.get("compute_budget")
        cleaned_budget: Dict[str, Any] = {}
        for key, base in DEFAULT_COMPUTE_BUDGET.items():
            source = compute_original.get(key) if isinstance(compute_original, dict) else None
            cleaned_budget[key] = coerce_numeric(source, base)
        if cleaned_budget != compute_original:
            payload["compute_budget"] = cleaned_budget
            mark("compute_budget")

        repro_original = payload.get("repro") if isinstance(payload.get("repro"), dict) else {}
        seed_value = coerce_numeric(repro_original.get("seed"), DEFAULT_REPRO["seed"])
        deterministic_raw = repro_original.get("deterministic")
        if isinstance(deterministic_raw, bool):
            deterministic_value = deterministic_raw
        elif isinstance(deterministic_raw, str):
            lowered = deterministic_raw.strip().lower()
            if lowered in {"false", "0", "no", "n"}:
                deterministic_value = False
            elif lowered in {"true", "1", "yes", "y"}:
                deterministic_value = True
            else:
                deterministic_value = DEFAULT_REPRO["deterministic"]
        elif deterministic_raw is None:
            deterministic_value = DEFAULT_REPRO["deterministic"]
        else:
            deterministic_value = bool(deterministic_raw)
        framework_value = optional_string(repro_original.get("framework")) or DEFAULT_REPRO["framework"]
        cleaned_repro = {
            "seed": int(seed_value),
            "deterministic": deterministic_value,
            "framework": framework_value,
        }
        if cleaned_repro != payload.get("repro"):
            payload["repro"] = cleaned_repro
            mark("repro")

        requires_browse_original = payload.get("requires_browse")
        if isinstance(requires_browse_original, bool):
            requires_browse_clean = requires_browse_original
        elif isinstance(requires_browse_original, str):
            requires_browse_clean = requires_browse_original.strip().lower() not in {"false", "0", "no", "n"}
        elif requires_browse_original is None:
            requires_browse_clean = True
        else:
            requires_browse_clean = bool(requires_browse_original)
        if requires_browse_clean != requires_browse_original:
            payload["requires_browse"] = requires_browse_clean
            mark("requires_browse")

        citation_original = payload.get("citation_policy")
        citation_clean = optional_string(citation_original) or "Cite ≥2 authoritative sources (arXiv DOI/URL) and include them in the final report."
        if citation_clean != citation_original:
            payload["citation_policy"] = citation_clean
            mark("citation_policy")

        io_schema_original = payload.get("io_schema")
        if isinstance(io_schema_original, dict):
            cleaned_io_schema: Dict[str, Any] = {
                key: value
                for key, value in io_schema_original.items()
                if key not in {"artifacts", "logs", "outputs"}
            }
            artifacts_clean = clean_string_list(
                io_schema_original.get("artifacts"),
                field="io_schema.artifacts",
                fallback=DEFAULT_IO_SCHEMA["artifacts"],
            )
            cleaned_io_schema["artifacts"] = artifacts_clean

            logs_candidate = io_schema_original.get("logs")
            logs_clean = optional_string(logs_candidate)
            if logs_clean is None:
                if isinstance(logs_candidate, (list, tuple, set)):
                    for entry in logs_candidate:
                        logs_clean = optional_string(entry)
                        if logs_clean:
                            break
                elif isinstance(logs_candidate, dict):
                    for entry in logs_candidate.values():
                        logs_clean = optional_string(entry)
                        if logs_clean:
                            break
            if logs_clean is None:
                logs_clean = DEFAULT_IO_SCHEMA["logs"]
            cleaned_io_schema["logs"] = logs_clean

            outputs_original = io_schema_original.get("outputs")
            if isinstance(outputs_original, dict):
                outputs_clean: Dict[str, Any] = {
                    key: value
                    for key, value in outputs_original.items()
                    if key not in {"artifacts", "logs"}
                }
                outputs_artifacts = clean_string_list(
                    outputs_original.get("artifacts"),
                    field="io_schema.outputs.artifacts",
                    fallback=DEFAULT_IO_SCHEMA["artifacts"],
                )
                outputs_clean["artifacts"] = outputs_artifacts
                outputs_log = optional_string(outputs_original.get("logs")) or logs_clean
                outputs_clean["logs"] = outputs_log
            else:
                outputs_clean = {
                    "artifacts": artifacts_clean.copy(),
                    "logs": logs_clean,
                }
            cleaned_io_schema["outputs"] = outputs_clean
        else:
            cleaned_io_schema = _copy_default(DEFAULT_IO_SCHEMA)
        if cleaned_io_schema != io_schema_original:
            payload["io_schema"] = cleaned_io_schema
            mark("io_schema")

        quality_score_original = payload.get("quality_score")
        if isinstance(quality_score_original, (int, float)):
            quality_score_clean = float(quality_score_original)
        elif isinstance(quality_score_original, str):
            try:
                quality_score_clean = float(quality_score_original.strip())
            except ValueError:
                quality_score_clean = 0.0
        else:
            quality_score_clean = 0.0
        quality_score_clean = max(0.0, min(100.0, round(quality_score_clean, 2)))
        if not isinstance(quality_score_original, (int, float)) or round(float(quality_score_original), 2) != quality_score_clean:
            payload["quality_score"] = quality_score_clean
            mark("quality_score")

        updated_lines.append(json.dumps(payload, ensure_ascii=True))
        if changed:
            records_repaired += 1

    if records_repaired == 0:
        print(f"{file_path} already conforms to the router schema; no changes made.")
        return

    backup_path = _schema_backup_path(file_path, "schema")
    file_path.rename(backup_path)
    with file_path.open("w", encoding="utf-8") as writer:
        writer.write("\n".join(updated_lines) + "\n")

    summary = ", ".join(
        f"{field}×{count}"
        for field, count in sorted(field_repairs.items())
    ) or "no field-level adjustments recorded"
    print(
        f"Schema repair complete for {records_repaired} record(s). "
        f"Backup saved as {backup_path}. Updated fields: {summary}"
    )


def _count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main(argv: Optional[List[str]] = None) -> None:
    load_dotenv()
    args = parse_args(argv)

    if args.repair_ids and args.repair_schema:
        raise SystemExit("Use either --repair-ids or --repair-schema, not both.")

    if args.repair_ids:
        repair_jsonl_ids(args.repair_ids, args.repair_start)
        return
    if args.repair_schema:
        repair_jsonl_schema(args.repair_schema)
        return

    console = _get_console()
    console.rule("[bold cyan]Gemini Router Dataset Builder[/bold cyan]")
    console.print(f"[bold]Model:[/] {args.model}")
    console.print(f"[bold]Examples:[/] {args.count}")
    console.print(f"[bold]Output file:[/] {args.output}")

    if args.count <= 0:
        console.print(
            Panel("`--count` must be a positive integer.", border_style="red", title="Invalid count")
        )
        sys.exit(2)
    if args.concurrency <= 0:
        console.print(
            Panel(
                "`--concurrency` must be a positive integer.",
                border_style="red",
                title="Invalid concurrency",
            )
        )
        sys.exit(2)

    existing_records = _count_existing_rows(args.output)
    start_index = args.start_index if args.start_index is not None else existing_records

    console.print(f"[bold]Concurrency:[/] {args.concurrency}")
    console.print(f"[bold]Start index:[/] {start_index}")
    console.print(
        "[bold]Difficulty mix:[/] adaptive rotation across introductory/intermediate/advanced tiers"
    )

    if existing_records > 0 and args.start_index is None:
        console.print(
            Panel(
                f"Detected {existing_records} existing records. Appending new examples starting at index {start_index}.",
                border_style="cyan",
                title="Resume mode",
            )
        )
    if args.start_index is not None and args.start_index < existing_records:
        console.print(
            Panel(
                (
                    "The requested start index is lower than the number of existing records."
                    " This may create duplicate IDs."
                ),
                border_style="yellow",
                title="Start index overlap",
            )
        )

    try:
        builder = GeminiRouterDatasetBuilder(
            model_name=args.model,
            seed=args.seed,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sleep_base=args.sleep_base,
            console=console,
        )
        with JsonlWriter(args.output) as writer:
            orchestrator = DatasetOrchestrator(builder, writer, console)
            summary = orchestrator.generate(
                count=args.count,
                start_index=start_index,
                concurrency=args.concurrency,
            )
    except KeyboardInterrupt:
        console.print(
            Panel(
                "Generation interrupted by user. Partial results (if any) are already saved.",
                border_style="yellow",
                title="Interrupted",
            )
        )
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        console.print(Panel(str(exc), border_style="red", title="Generation failed"))
        sys.exit(1)

    success_msg = (
        f"Generated {summary.successes} example(s) in {summary.duration_seconds:.1f} seconds.\n"
        f"Location: {args.output.resolve()}"
    )
    if summary.failures:
        failure_lines = "\n".join(
            f"- idx {failure.index} ({failure.variant}): {failure.error}"
            for failure in summary.failures[:10]
        )
        if len(summary.failures) > 10:
            failure_lines += f"\n... and {len(summary.failures) - 10} more."
        console.print(
            Panel(
                success_msg + "\n\nFailures:\n" + failure_lines,
                border_style="yellow",
                title="Completed with issues",
            )
        )
        sys.exit(1)
    else:
        console.print(
            Panel(
                success_msg,
                border_style="green",
                title="Generation complete",
            )
        )


if __name__ == "__main__":
    main()
