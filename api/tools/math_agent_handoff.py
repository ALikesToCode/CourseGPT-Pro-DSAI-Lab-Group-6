from langchain.tools import tool


@tool
def math_agent_handoff(task_summary: str, thinking_outline: list[str], route_plan: list[str], route_rationale: str, handoff_plan: str, todo_list: list[str], acceptance_criteria: list[str], expected_artifacts: list[str], difficulty: str, tags: list[str], metrics: dict, compute_budget: dict, repro: dict, citation_policy: str, io_schema: dict, requires_browse: bool):
    """
    Signals that the router agent should hand off to the math agent.
    
    Args:
        task_summary: One-sentence synopsis of the ask.
        thinking_outline: Numbered steps exposing the reasoning chain.
        route_plan: The high-level plan for routing.
        route_rationale: The reasoning behind the routing decision.
        handoff_plan: Specific instructions for the math agent.
        todo_list: Checkbox-style plan for sub-agents.
        acceptance_criteria: A list of criteria to verify the task completion.
        expected_artifacts: List of deliverables.
        difficulty: Task difficulty (introductory/intermediate/advanced).
        tags: Domain-specific tags.
        metrics: Success metrics.
        compute_budget: Resource constraints.
        repro: Reproducibility settings.
        citation_policy: Citation requirements.
        io_schema: Input/output schema.
        requires_browse: Whether web browsing is required.
    """
    return f"Handoff to Math Agent initiated.\nPlan: {handoff_plan}\nContext: {route_rationale}"
