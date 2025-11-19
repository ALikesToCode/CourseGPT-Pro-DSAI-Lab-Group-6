from langchain.tools import tool


@tool
def math_agent_handoff():
    """
    Signals that the router agent should hand off to the math agent.
    """
    pass
