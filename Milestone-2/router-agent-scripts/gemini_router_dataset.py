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
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError as exc:  # pragma: no cover - handled at runtime.
    Console = None  # type: ignore
    Panel = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    _RICH_IMPORT_ERROR = exc
else:
    _RICH_IMPORT_ERROR = None


ALLOWED_TOOLS = ["/math", "/code", "/general-search"]
ROUTE_VARIANTS = [
    {
        "name": "math_only",
        "required": ["/math"],
        "description": "Pure math reasoning, symbolic manipulation, proofs, derivations.",
    },
    {
        "name": "code_only",
        "required": ["/code"],
        "description": "Implementation-heavy Python coding without math derivations.",
    },
    {
        "name": "general_only",
        "required": ["/general-search"],
        "description": "Answer hinges on external references, factual lookup, or RAG.",
    },
    {
        "name": "math_plus_code",
        "required": ["/math", "/code"],
        "description": "Math derivations that also need a validating Python snippet.",
    },
    {
        "name": "math_plus_general",
        "required": ["/math", "/general-search"],
        "description": "Math-heavy request that also needs cited references or context.",
    },
    {
        "name": "code_plus_general",
        "required": ["/code", "/general-search"],
        "description": "Coding-focused prompt plus web/RAG context gathering.",
    },
    {
        "name": "tri_route",
        "required": ["/math", "/code", "/general-search"],
        "description": "Blended ask requiring math, Python, and retrieval steps.",
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
    "difficulty": "introductory | intermediate | advanced",
    "tags": "List of topical tags or skills.",
}


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
    difficulty: str
    tags: List[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "user_query": self.user_query,
                "task_summary": self.task_summary,
                "route_plan": self.route_plan,
                "route_rationale": self.route_rationale,
                "expected_artifacts": self.expected_artifacts,
                "difficulty": self.difficulty,
                "tags": self.tags,
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
        self.seed = seed
        self.temperature = temperature
        self.max_retries = max_retries
        self.sleep_base = sleep_base
        self.console = console or _get_console()
        self._random = random.Random(seed)
        self._client = None
        _ensure_packages()
        api_key = _infer_api_key()
        if not api_key:
            raise SystemExit(
                "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        self._client = genai.Client(api_key=api_key)

    def build_dataset(
        self,
        count: int,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[RouterExample]:
        examples: List[RouterExample] = []
        variants = ROUTE_VARIANTS.copy()
        for idx in range(count):
            variant = variants[idx % len(variants)]
            example = self._generate_example(idx, variant)
            examples.append(example)
            if progress_cb:
                progress_cb(idx + 1, count)
        return examples

    def _generate_example(self, idx: int, variant: Dict[str, Any]) -> RouterExample:
        prompt = self._build_prompt(idx, variant)
        for attempt in range(1, self.max_retries + 1):
            try:
                assert self._client is not None and genai_types is not None
                config = genai_types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                )
                response = self._client.models.generate_content(
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
                return RouterExample(
                    id=data["id"],
                    user_query=data["user_query"],
                    task_summary=data["task_summary"],
                    route_plan=data["route_plan"],
                    route_rationale=data["route_rationale"],
                    expected_artifacts=data["expected_artifacts"],
                    difficulty=data["difficulty"],
                    tags=data["tags"],
                )
            except (json.JSONDecodeError, AssertionError, KeyError, ValueError) as exc:
                self._handle_retry(exc, attempt, idx)
            except genai_errors.APIError as exc:  # pragma: no cover
                self._handle_retry(exc, attempt, idx)
        raise RuntimeError(f"Failed to synthesize example after {self.max_retries} tries.")

    def _handle_retry(self, exc: Exception, attempt: int, idx: int) -> None:
        wait = self.sleep_base * (2 ** (attempt - 1))
        reason = f"[example {idx}] attempt {attempt} failed: {exc}"
        self.console.log(f"[yellow]{reason}[/yellow]")
        time.sleep(wait)

    def _build_prompt(self, idx: int, variant: Dict[str, Any]) -> str:
        requested_mode = self._random.choice(GENERAL_SEARCH_MODES)
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
              * Provide a realistic, student-facing user_query about higher-ed STEM topics.
              * Provide 1-3 topical tags.
              * difficulty must be one of: introductory, intermediate, advanced.
              * Populate expected_artifacts with 2-4 bullet-point style strings.
              * Use ids shaped like router_{idx:04d}.

            Required command coverage guidance (include each):
              {chr(10).join(f'- {spec}' for spec in command_specs)}

            Return ONLY valid JSON with double quotes. Do not wrap in markdown.
            """
        ).strip()
        return prompt

    def _validate_payload(self, payload: Dict[str, Any], variant: Dict[str, Any]) -> None:
        required_keys = [
            "id",
            "user_query",
            "task_summary",
            "route_plan",
            "route_rationale",
            "expected_artifacts",
            "difficulty",
            "tags",
        ]
        for key in required_keys:
            if key not in payload:
                raise KeyError(f"Missing key `{key}` in payload.")

        if not isinstance(payload["route_plan"], list) or not payload["route_plan"]:
            raise ValueError("route_plan must be a non-empty list.")

        commands_present = set()
        for cmd in payload["route_plan"]:
            if not isinstance(cmd, str):
                raise ValueError("Each route_plan entry must be a string.")
            if not any(cmd.startswith(tool) for tool in ALLOWED_TOOLS):
                raise ValueError(f"route_plan entry uses unsupported tool: {cmd}")
            for tool in ALLOWED_TOOLS:
                if cmd.startswith(tool):
                    commands_present.add(tool)
            if "/general-search" in cmd and "mode=" not in cmd:
                raise ValueError("General-search command missing mode parameter.")
            if "/code" in cmd and "Python" not in cmd:
                raise ValueError("Coding command must mention Python context explicitly.")

        for required_tool in variant["required"]:
            if required_tool not in commands_present:
                raise ValueError(
                    f"Required tool `{required_tool}` missing for variant {variant['name']}."
                )

        difficulty = payload["difficulty"]
        if difficulty not in {"introductory", "intermediate", "advanced"}:
            raise ValueError(f"Invalid difficulty level: {difficulty}.")

        tags = payload["tags"]
        if not isinstance(tags, list) or not tags:
            raise ValueError("tags must be a non-empty list.")

        route_rationale = payload["route_rationale"]
        if "/general-search" in commands_present and "blue" not in route_rationale.lower():
            raise ValueError("Route rationale must mention blue when general-search is used.")


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
    return parser.parse_args(argv)


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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

    builder: Optional[GeminiRouterDatasetBuilder] = None
    try:
        builder = GeminiRouterDatasetBuilder(
            model_name=args.model,
            seed=args.seed,
            temperature=args.temperature,
            max_retries=args.max_retries,
            sleep_base=args.sleep_base,
            console=console,
        )
        assert Progress is not None and SpinnerColumn is not None and TextColumn is not None
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Synthesizing router examples", total=args.count)

            def progress_cb(completed: int, _total: int) -> None:
                progress.update(task_id, completed=completed)

            dataset = builder.build_dataset(args.count, progress_cb=progress_cb)

        ensure_output_dir(args.output)
        with console.status("Writing dataset to disk...", spinner="dots"):
            with args.output.open("w", encoding="utf-8") as stream:
                for row in dataset:
                    stream.write(row.to_json() + "\n")

        console.print(
            Panel(
                f"Saved {len(dataset)} router examples to {args.output.resolve()}",
                border_style="green",
                title="Generation complete",
            )
        )
    except SystemExit:
        raise
    except Exception as exc:
        error_panel = Panel(str(exc), border_style="red", title="Generation failed")
        console.print(error_panel)
        sys.exit(1)
    finally:
        if builder is not None:
            client = getattr(builder, "_client", None)
            if client is not None:
                try:
                    client.close()
                except Exception:  # pragma: no cover - best effort cleanup.
                    pass


if __name__ == "__main__":
    main()
