from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from states.main_state import CourseGPTState
from tools.general_agent_handoff import general_agent_handoff
from dotenv import load_dotenv
import os

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


def code_agent(state: CourseGPTState):

    # TODO: replace with your custom fine tuned model or different LLM as needed

    llm = ChatGoogleGenerativeAI(
        model="gemii-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm.bind_tools(code_agent_tools, parallel_tool_calls=False)

    system_message = SystemMessage(content=code_agent_prompt)

    response = llm.invoke([system_message] + state["messages"])

    return {"messages": response}
