from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.route_after_tools import route_after_tools  # noqa: E402
from graph.should_goto_tools import router_should_goto_tools  # noqa: E402


def test_router_fallback_math():
    state = {
        "messages": [
            HumanMessage(content="Solve the integral of x^2 dx."),
            AIMessage(content="(router did not emit tool calls)"),
        ]
    }
    assert router_should_goto_tools(state) == "math_agent"


def test_router_fallback_code():
    state = {
        "messages": [
            HumanMessage(content="Python bug: traceback shows KeyError in dict."),
            AIMessage(content="(router did not emit tool calls)"),
        ]
    }
    assert router_should_goto_tools(state) == "code_agent"


def test_router_fallback_general():
    state = {
        "messages": [
            HumanMessage(content="Plan a weekend study schedule for algorithms."),
            AIMessage(content="(router did not emit tool calls)"),
        ]
    }
    assert router_should_goto_tools(state) == "general_agent"


def test_route_after_tools_handles_handoff():
    state = {"messages": [ToolMessage(content="ok", name="math_agent_handoff", tool_call_id="t0")]}
    assert route_after_tools(state) == "math_agent"


def test_router_dataset_entries_have_handoff_plan():
    dataset_path = Path(__file__).resolve().parents[2] / "old/Milestone-2/router-agent-scripts/output.jsonl"
    assert dataset_path.exists(), "router dataset is missing"
    with dataset_path.open() as f:
        lines = [next(f) for _ in range(3)]
    for line in lines:
        entry = json.loads(line)
        for key in ("user_query", "route_plan", "handoff_plan", "todo_list"):
            assert key in entry and entry[key], f"missing {key} in router dataset entry"
