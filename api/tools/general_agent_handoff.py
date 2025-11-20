from typing import Any, Dict, List, Union

from langchain.tools import tool

from .handoff_utils import normalize_dict, normalize_list


@tool
def general_agent_handoff(
    task_summary: str,
    thinking_outline: Union[str, List[str]],
    route_plan: Union[str, List[str]],
    route_rationale: str,
    handoff_plan: str,
    todo_list: Union[str, List[str]],
    acceptance_criteria: Union[str, List[str]],
    expected_artifacts: Union[str, List[str]],
    difficulty: str,
    tags: Union[str, List[str]],
    metrics: Union[str, Dict[str, Any]],
    compute_budget: Union[str, Dict[str, Any]],
    repro: Union[str, Dict[str, Any]],
    citation_policy: str,
    io_schema: Union[str, Dict[str, Any]],
    requires_browse: bool,
):
    """
    Signals that the router agent should hand off to the general agent.

    This tool normalizes incoming values to avoid type errors from the router.
    """
    return {
        "handoff": "general_agent",
        "task_summary": task_summary,
        "thinking_outline": normalize_list(thinking_outline),
        "route_plan": normalize_list(route_plan),
        "route_rationale": route_rationale,
        "handoff_plan": handoff_plan,
        "todo_list": normalize_list(todo_list),
        "acceptance_criteria": normalize_list(acceptance_criteria),
        "expected_artifacts": normalize_list(expected_artifacts),
        "difficulty": difficulty,
        "tags": normalize_list(tags),
        "metrics": normalize_dict(metrics),
        "compute_budget": normalize_dict(compute_budget),
        "repro": normalize_dict(repro),
        "citation_policy": citation_policy,
        "io_schema": normalize_dict(io_schema),
        "requires_browse": requires_browse,
    }
