
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
import requests
from ..states.main_state import CourseGPTState
from dotenv import load_dotenv
import os
from tools.general_agent_handoff import general_agent_handoff
load_dotenv()

general_agent_tools = [general_agent_handoff]

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


def general_agent(state: CourseGPTState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm.bind_tools(general_agent_tools, parallel_tool_calls=False)

    system_message = SystemMessage(content=general_agent_prompt)

    response = llm.invoke([system_message] + state["messages"])

    return {"messages": response}
