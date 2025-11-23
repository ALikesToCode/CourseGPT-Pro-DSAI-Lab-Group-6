from .states.main_state import CourseGPTState
from langgraph.graph import StateGraph, START, END
import re


def should_goto_tools(state: CourseGPTState):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(
            f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def router_should_goto_tools(state: CourseGPTState):
    """
    Router-specific helper: if the router didn't emit a tool call, fall back to
    the general agent instead of ending the graph with no answer. Includes
    lightweight heuristics to push obviously-math queries to math_agent and
    code-like queries to code_agent when the router LLM fails to emit a tool call.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(
            f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    # Heuristic fallback: inspect the last user message to avoid misrouting.
    user_text = ""
    if isinstance(state, list):
        prior = state[:-1]
    else:
        prior = state.get("messages", [])[:-1]
    for msg in reversed(prior):
        if getattr(msg, "type", getattr(msg, "role", None)) in ("human", "user"):
            user_text = getattr(msg, "content", "") or ""
            break

    if user_text:
        lower = user_text.lower()
        # Math-ish patterns: numbers + operators or explicit math keywords, or stats vocabulary.
        math_keywords = [
            "solve", "derivative", "integral", "equation", "quadratic", "roots",
            "normal distribution", "mean", "standard deviation", "variance", "sigma", "z-value", "z score",
            "probability", "demand", "inventory", "newsvendor", "optimal order", "critical ratio", "service level",
            "forecast", "gaussian", "poisson", "binomial",
        ]
        if re.search(r"[0-9]", user_text) and any(kw in lower for kw in math_keywords):
            return "math_agent"
        if re.search(r"[0-9][^a-zA-Z]*(?:\+|-|\*|/|\^)", user_text):
            return "math_agent"
        # Code-ish patterns: language names or obvious coding verbs.
        if any(kw in lower for kw in ["python", "javascript", "java", "c++", "c#", "code", "bug", "traceback"]):
            return "code_agent"

    return "general_agent"
