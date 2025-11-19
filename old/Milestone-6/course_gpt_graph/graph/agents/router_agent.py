
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
import requests
from states.main_state import CourseGPTState
from dotenv import load_dotenv
import os
from tools.general_agent_handoff import general_agent_handoff
from tools.code_agent_handoff import code_agent_handoff
from tools.math_agent_handoff import math_agent_handoff
load_dotenv()

router_agent_tools = [general_agent_handoff, code_agent_handoff, math_agent_handoff]


router_agent_prompt = """You are a routing assistant whose job is to decide which specialized agent or tool should handle incoming requests.
When given a user request, choose the most appropriate agent (e.g., `code_agent`, `math_agent`, `general_agent`) or one of the available tools.
If you select another agent, respond with the exact routing format below so the caller can dispatch the request.
Available tools:

{tools_list}
When responding, follow this format exactly:
If routing to another agent:
Route: <agent_name>
Message: <message_to_forward>
If calling a tool directly:
Tool: <tool_name>
Input: <input_parameters>
If answering directly yourself:
Answer: <your_answer>
"""


def router_agent(state: CourseGPTState):

    # TODO: replace with your custom fine tuned model or different LLM as needed

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm.bind_tools(router_agent_tools, parallel_tool_calls=False)

    system_message = SystemMessage(content=router_agent_prompt)

    response = llm.invoke([system_message] + state["messages"])

    return {"messages": response}
