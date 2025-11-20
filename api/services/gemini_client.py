import os
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

class Gemini3Client:
    """
    A wrapper for the Google Gen AI SDK (v1alpha/beta) to support Gemini 3 models
    in a way that is compatible with LangChain's invoke/bind_tools pattern.
    """
    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tools = []
        self.tool_config = None
        self.native_tools = []

    def enable_google_search(self):
        """Enable Google Search grounding."""
        self.native_tools.append({"google_search": {}})

    def enable_code_execution(self):
        """Enable Code Execution."""
        self.native_tools.append({"code_execution": {}})

    def bind_tools(self, tools: List[BaseTool], parallel_tool_calls: bool = False):
        """
        Bind tools to the client. This converts LangChain tools to Gemini tool definitions.
        """
        self.tools = tools
        # Convert LangChain tools to Gemini function declarations
        # For simplicity, we pass the callables directly as the SDK supports it.
        # We will combine these with any native tools in invoke.
        pass

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Invoke the model with a list of LangChain messages.
        """
        # Convert LangChain messages to Gemini content
        contents = []
        system_instruction = None
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.content
            elif isinstance(msg, HumanMessage):
                contents.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
            elif isinstance(msg, AIMessage):
                # Handle tool calls in AI message if present
                parts = []
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=tool_call["name"],
                                args=tool_call["args"]
                            )
                        ))
                
                if msg.content:
                    parts.append(types.Part(text=msg.content))
                
                contents.append(types.Content(role="model", parts=parts))
            elif isinstance(msg, ToolMessage):
                # Tool response
                contents.append(types.Content(role="user", parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=msg.name, 
                            response={"result": msg.content} 
                        )
                    )
                ]))

        # Call the API
        config = {}
        all_tools = []
        
        # Add custom tools (LangChain tools)
        if self.tools:
            all_tools.extend(self.tools)
            
        # Add native tools
        if self.native_tools:
            all_tools.extend(self.native_tools)
            
        if all_tools:
            config["tools"] = all_tools

        # Set thinking level if using Gemini 3 Pro
        if "gemini-3" in self.model:
            config["thinking_config"] = {"include_thoughts": True} 

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                **config
            )
        )

        # Convert response back to LangChain AIMessage
        content = response.text or ""
        tool_calls = []
        
        # Check for function calls in candidates
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append({
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                        "id": "call_" + part.function_call.name # Dummy ID
                    })
                # Handle executable code result if present (for code execution tool)
                if part.executable_code:
                    # The model generated code to run. 
                    # The SDK might handle execution if configured, or we see the code here.
                    # For now, we just append it to content if it's not already there.
                    # content += f"\n```python\n{part.executable_code.code}\n```"
                    pass
                if part.code_execution_result:
                    # Result of code execution
                    # content += f"\nOutput:\n```\n{part.code_execution_result.output}\n```"
                    pass
        
        return AIMessage(content=content, tool_calls=tool_calls)
