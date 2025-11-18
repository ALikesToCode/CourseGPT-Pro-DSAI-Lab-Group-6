
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
import requests
from states.main_state import CourseGPTState
from dotenv import load_dotenv
import os
from tools.general_agent_handoff import general_agent_handoff

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


def math_agent(state: CourseGPTState):

    # TODO: replace with your custom fine tuned model or different LLM as needed

    llm = ChatGoogleGenerativeAI(
        model="gemii-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm.bind_tools(math_agent_tools, parallel_tool_calls=False)

    system_message = SystemMessage(content=math_agent_prompt)

    response = llm.invoke([system_message] + state["messages"])

    return {"messages": response}
