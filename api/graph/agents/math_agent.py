
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
import requests
from ..states.main_state import CourseGPTState
from dotenv import load_dotenv
import os
from api.tools.general_agent_handoff import general_agent_handoff
from langchain_openai import ChatOpenAI
from api.config import get_settings

load_dotenv()

math_agent_tools = [general_agent_handoff]


math_agent_prompt = """You are a mathematics assistant. Provide clear, step-by-step solutions and where appropriate show work using LaTeX for equations.
When given a math problem, prefer to show reasoning and intermediate steps. If a numerical answer is requested, also provide the derivation.
If any of the available tools can help (e.g., computation, plotting, symbolic manipulation), choose the tool and call it.
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

def math_agent(state: CourseGPTState):

    settings = get_settings()

    if settings.math_agent_url:
        llm = ChatOpenAI(
            base_url=settings.math_agent_url,
            api_key=settings.math_agent_api_key or "dummy",
            model=settings.math_agent_model or "default",
            temperature=0
        )
    else:
        # Default to Gemini 3 Pro
        llm = Gemini3Client(
            api_key=settings.google_api_key,
            model=settings.gemini_model
        )

    llm.bind_tools(math_agent_tools, parallel_tool_calls=False)

    system_message = SystemMessage(content=math_agent_prompt)

    response = llm.invoke([system_message] + state["messages"])

    return {"messages": response}
