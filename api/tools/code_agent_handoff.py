from langchain.tools import tool


@tool
def code_agent_handoff():
    """
    Signals that the router agent should hand off to the code agent.
    """
    pass
