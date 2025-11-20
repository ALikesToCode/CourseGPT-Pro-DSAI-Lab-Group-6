from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..states.main_state import CourseGPTState
from api.tools.general_agent_handoff import general_agent_handoff
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from api.config import get_settings

load_dotenv()

code_agent_tools = [general_agent_handoff]


code_agent_prompt = """You are a code assistant that helps users with programming tasks.
When given a request, you should determine if any of the available tools can help you accomplish the task.
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

def code_agent(state: CourseGPTState):

    settings = get_settings()

    if settings.code_agent_url:
        llm = ChatOpenAI(
            base_url=settings.code_agent_url,
            api_key=settings.code_agent_api_key or "dummy",
            model=settings.code_agent_model or "default",
            temperature=0
        )
    else:
        # Default to Gemini 3 Pro
        llm = Gemini3Client(
            api_key=settings.google_api_key,
            model=settings.gemini_model
        )
        llm.enable_code_execution()
        
    llm.bind_tools(code_agent_tools, parallel_tool_calls=False)

    system_message = SystemMessage(content=code_agent_prompt)

    response = llm.invoke([system_message] + state["messages"])

    return {"messages": response}
