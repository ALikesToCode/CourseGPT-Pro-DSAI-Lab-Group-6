# from IPython.display import Image, display
from .should_goto_tools import should_goto_tools, router_should_goto_tools
from .states.main_state import CourseGPTState
from .agents.math_agent import math_agent, math_agent_tools
from .agents.general_agent import general_agent, general_agent_tools
from .agents.router_agent import router_agent, router_agent_tools
from .agents.code_agent import code_agent, code_agent_tools
from dotenv import load_dotenv
from langchain.tools import tool
from requests_html import HTMLSession
import requests
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
import sys
from pathlib import Path

# Add the repository root to sys.path so `api.*` imports work even when running
# from inside the `api` directory (uvicorn main:app).
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


load_dotenv()


graph = StateGraph(MessagesState)
checkpointer = InMemorySaver()


graph.add_node(router_agent)
graph.add_node(code_agent)
graph.add_node(general_agent)
graph.add_node(math_agent)


graph.add_node("router_agent_tools", ToolNode(router_agent_tools))
graph.add_node("code_agent_tools", ToolNode(code_agent_tools))
graph.add_node("general_agent_tools", ToolNode(general_agent_tools))
graph.add_node("math_agent_tools", ToolNode(math_agent_tools))

from .route_after_tools import route_after_tools

graph.add_edge(START, "router_agent")
# Removed static edges to allow conditional routing via tools

graph.add_conditional_edges(
    'router_agent',
    router_should_goto_tools,
    {
        "tools": "router_agent_tools",
        "math_agent": "math_agent",
        "code_agent": "code_agent",
        "general_agent": "general_agent",
        END: END
    }
)

graph.add_conditional_edges(
    'router_agent_tools',
    route_after_tools,
    {
        "general_agent": "general_agent",
        "code_agent": "code_agent",
        "math_agent": "math_agent",
        "router_agent": "router_agent",
        END: END
    }
)


graph.add_conditional_edges(
    'code_agent',
    should_goto_tools,
    {
        "tools": "code_agent_tools",
        END: END
    }
)

graph.add_edge('code_agent_tools', 'code_agent')

graph.add_conditional_edges(
    'general_agent',
    should_goto_tools,
    {
        "tools": "general_agent_tools",
        END: END
    }
)

graph.add_edge('general_agent_tools', 'general_agent')


graph.add_conditional_edges(
    'math_agent',
    should_goto_tools,
    {
        "tools": "math_agent_tools",
        END: END
    }
)

graph.add_edge('math_agent_tools', 'math_agent')


graph = graph.compile()


# Removed IPython import and graph visualization for production
