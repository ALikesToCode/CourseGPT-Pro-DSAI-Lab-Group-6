from langchain.tools import tool


@tool
def general_agent_handoff():
    """
    Signals that the router agent should hand off to the general agent.
    """
    pass
