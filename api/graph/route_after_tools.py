from .states.main_state import CourseGPTState
from langgraph.graph import END

def route_after_tools(state: CourseGPTState):
    """
    Determine the next node based on the tool call that was just executed.
    """
    if isinstance(state, list):
        # In case state is a list of messages
        messages = state
    else:
        messages = state.get("messages", [])
    
    if not messages:
        return END

    last_message = messages[-1]
    
    # Check if the last message is a ToolMessage (result of tool execution)
    # We need to look at the tool name. 
    # However, ToolMessage usually has 'name' or 'artifact'.
    # But we can also look at the tool call in the PREVIOUS message (AIMessage).
    
    # Actually, in LangGraph, the state after ToolNode contains the ToolMessage.
    # We can check the name of the tool that produced this message.
    
    if hasattr(last_message, "name"):
        tool_name = last_message.name
        if tool_name == "general_agent_handoff":
            return "general_agent"
        elif tool_name == "code_agent_handoff":
            return "code_agent"
        elif tool_name == "math_agent_handoff":
            return "math_agent"
            
    # If we can't determine from the last message (maybe it's not a ToolMessage yet?),
    # we might need to look at the tool calls in the second to last message.
    # But this function is called AFTER ToolNode.
    
    return "router_agent" # Default loop back if unknown
