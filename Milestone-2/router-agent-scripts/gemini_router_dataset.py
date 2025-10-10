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


def _contains_domain_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in DOMAIN_KEYWORDS)


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
    difficulty: str
    tags: List[str]
    quality_score: float

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
                "difficulty": self.difficulty,
                "tags": self.tags,
                "quality_score": round(self.quality_score, 2),
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
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.sleep_base = sleep_base
        self.console = console or _get_console()
        self._seed = seed if seed is not None else 7
        self._thread_local: threading.local = threading.local()

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
        return rng.choice(ADVANCED_THEMES)

    def generate_example(self, idx: int, variant: Dict[str, Any]) -> RouterExample:
        """Call Gemini to synthesize a single router-training example."""
        theme = self.theme_for_index(idx)
        prompt = self._build_prompt(idx, variant, theme)
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
                self._validate_payload(data, variant)
                quality_score = self._compute_quality_score(data, theme)
                self.console.log(
                    f"[cyan]example {idx}: quality_score={quality_score:.1f}[/cyan]"
                )
                return RouterExample(
                    id=data["id"],
                    user_query=data["user_query"],
                    task_summary=data["task_summary"],
                    route_plan=data["route_plan"],
                    route_rationale=data["route_rationale"],
                    expected_artifacts=data["expected_artifacts"],
                    thinking_outline=data["thinking_outline"],
                    handoff_plan=data["handoff_plan"],
                    difficulty=data["difficulty"],
                    tags=data["tags"],
                    quality_score=quality_score,
                )
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

    def _build_prompt(self, idx: int, variant: Dict[str, Any], theme: str) -> str:
        requested_mode = self._search_mode_for_index(idx)
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
        prompt = textwrap.dedent(
            f"""
            You are Gemini 2.5 Pro creating synthetic router-training data.
            Generate a single JSON object that follows this schema:
            {schema_lines}

            Hard rules:
              * route_plan MUST be an ordered list of tool call strings.
              * Only use these commands exactly as literals: {', '.join(ALLOWED_TOOLS)}.
              * When using /general-search, include a mode parameter set to {requested_mode}.
              * Treat the general search tool as color-coded "blue" and mention this in route_rationale.
              * When /code is used, explicitly mention Python in the command context.
              * The request variant is `{variant['name']}` which means: {variant['description']}
              * Provide a realistic, student-facing user_query centered on advanced STEM maths and {theme}.
              * Every scenario must feel graduate-level or IOI difficulty and require deep reasoning.
              * thinking_outline must list 4-6 numbered strings ("1.", "2.", ...) that expose the reasoning chain the router expects; include at least one verification/quality-check step.
              * Provide 1-3 topical tags (include at least one token tied to {theme}).
              * difficulty must be "advanced".
              * Populate expected_artifacts with 3-5 bullet-point style strings covering proofs, code, and citations.
              * Provide handoff_plan describing agent-to-agent flow (use arrows like "/general-search -> /math -> /code"), including verification loops or fallback contingencies.
              * Route_plan contexts must specify precise notation, algorithm names, library/tooling constraints, and evaluation metrics so sub-agents can execute without ambiguity.
              * The user_query should read like a graduate researcher or competitive programmer seeking help on a novel experiment, not a basic homework request.
            Anti-patterns to AVOID:
              * Do NOT create elementary prompts like "derive the quadratic formula" or "implement binary search".
              * Avoid introductory calculus, routine linear algebra drills, or toy coding warmups.
              * Do not omit verification requirements, citations, or constraint descriptions.
              * Use ids shaped like router_{idx:04d}.

            Required command coverage guidance (include each):
              {chr(10).join(f'- {spec}' for spec in command_specs)}

            Context expectations:
              * Theme focus: {theme}.
              * {reference_hints}
              * {search_hint}
              * Ensure the user query invites the router to explicitly request rigorous proofs, derivations, or algorithmic analysis.
              * Make the task description explicit about constraints (e.g., asymptotics, error bounds, convergence guarantees).
              * Encourage downstream agents to return detailed reasoning traces, verified computations, citations, AND report back to the router for sign-off.
              * Highlight markers of graduate-level complexity: research-grade citations, multi-constraint optimization, stochastic analysis, or competitive-programming difficulty tiers.
            
            Return ONLY valid JSON with double quotes. Do not wrap in markdown.
            """
        ).strip()
        return prompt

    def _search_mode_for_index(self, idx: int) -> str:
        """Select a deterministic /general-search mode for the given index."""
        rng_seed = (self._seed + 1) * (idx + 1)
        rng = random.Random(rng_seed)
        return rng.choice(GENERAL_SEARCH_MODES)

    def _validate_payload(self, payload: Dict[str, Any], variant: Dict[str, Any]) -> None:
        required_keys = [
            "id",
            "user_query",
            "task_summary",
            "route_plan",
            "route_rationale",
            "expected_artifacts",
            "thinking_outline",
            "handoff_plan",
            "difficulty",
            "tags",
        ]
        for key in required_keys:
            if key not in payload:
                raise KeyError(f"Missing key `{key}` in payload.")

        user_query = payload["user_query"].strip()
        if len(user_query) < MIN_USER_QUERY_LEN:
            raise ValueError("user_query must be at least 100 characters for advanced prompts.")
        lowered_query = user_query.lower()
        if not _contains_domain_keyword(user_query):
            raise ValueError("user_query must reference graduate-level domain terminology.")
        if any(pattern in lowered_query for pattern in ANTI_PATTERNS):
            raise ValueError("user_query appears to reference an introductory homework pattern.")

        task_summary = payload["task_summary"].strip()
        if len(task_summary) < MIN_TASK_SUMMARY_LEN:
            raise ValueError("task_summary must be at least 50 characters long.")

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
            if previous_command is not None and cmd.split("(", 1)[0] == previous_command:
                raise ValueError("route_plan should not repeat identical consecutive tool calls.")
            previous_command = cmd.split("(", 1)[0]
            try:
                _, raw_context = cmd.split("(", 1)
                context = raw_context.rstrip(")")
            except ValueError:
                context = ""
            if len(context.strip()) < MIN_ROUTE_CONTEXT_LEN:
                raise ValueError("Each route_plan context must provide ≥40 characters of detailed guidance.")
            if not _contains_domain_keyword(context):
                raise ValueError("route_plan contexts must cite domain-specific terminology or constraints.")
            lower_context = context.lower()
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

        difficulty = payload["difficulty"]
        if difficulty != "advanced":
            raise ValueError(
                f"Invalid difficulty level: {difficulty}. Expected 'advanced' for high-rigor training."
            )

        thinking_outline = payload["thinking_outline"]
        if not isinstance(thinking_outline, list) or len(thinking_outline) < 4:
            raise ValueError("thinking_outline must be a list with at least four reasoning steps.")
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
        if verification_steps < 2:
            raise ValueError("thinking_outline must include at least two explicit verification or validation steps.")

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


    def _compute_quality_score(self, payload: Dict[str, Any], theme: str) -> float:
        user_query = payload["user_query"].strip()
        task_summary = payload["task_summary"].strip()
        route_plan = payload["route_plan"]
        thinking_outline = payload["thinking_outline"]
        tags = payload.get("tags", [])

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

        score = 0.0
        score += min(len(user_query) / 150.0, 1.5) * 20
        score += min(len(task_summary) / 80.0, 1.0) * 10
        score += min(avg_context_len / 80.0, 1.5) * 25
        score += min(verification_steps / 3.0, 1.0) * 20
        score += min(domain_hits, 6) * 3
        score += theme_hit * 7

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
    return parser.parse_args(argv)


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main(argv: Optional[List[str]] = None) -> None:
    load_dotenv()
    args = parse_args(argv)
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
