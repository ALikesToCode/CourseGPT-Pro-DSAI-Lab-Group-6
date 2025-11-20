
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
import requests
from ..states.main_state import CourseGPTState
from dotenv import load_dotenv
import os
from api.tools.code_agent_handoff import code_agent_handoff
from langchain_openai import ChatOpenAI
from api.config import get_settings
load_dotenv()

general_agent_tools = [code_agent_handoff]

general_agent_prompt = """You are a general-purpose assistant that helps users with project planning, coordination, and integration tasks.
When given a request, determine if any of the available tools can help you accomplish the task.
If a tool is needed, call the appropriate tool with the necessary parameters.
If no tool is needed, provide the best possible answer based on your knowledge.
Available tools:

{tools_list}
When responding, follow this format:
If using a tool:
Tool: <tool_name>
Input: <input_parameters>
If not using a tool:
Answer: <your_answer>
"""


from api.services.gemini_client import Gemini3Client

def general_agent(state: CourseGPTState):
    settings = get_settings()

    if settings.general_agent_url:
        llm = ChatOpenAI(
            base_url=settings.general_agent_url,
            api_key=settings.general_agent_api_key or "dummy",
            model=settings.general_agent_model or "default",
            temperature=0
        )
    else:
        # Default to Gemini 3 Pro
        llm = Gemini3Client(
            api_key=settings.google_api_key,
            model=settings.gemini_model
        )
        llm.enable_google_search()

    llm.bind_tools(general_agent_tools, parallel_tool_calls=False)

    system_message = SystemMessage(content=general_agent_prompt)

    response = llm.invoke([system_message] + state["messages"])

    return {"messages": response}
