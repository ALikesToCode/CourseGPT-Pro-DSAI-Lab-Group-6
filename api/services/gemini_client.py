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
        self.function_declarations: List[types.FunctionDeclaration] = []
        self.native_tools: List[str] = []

    def enable_google_search(self):
        """Enable Google Search grounding."""
        if "google_search" not in self.native_tools:
            self.native_tools.append("google_search")

    def enable_code_execution(self):
        """Enable Code Execution."""
        if "code_execution" not in self.native_tools:
            self.native_tools.append("code_execution")

    def bind_tools(self, tools: List[BaseTool], parallel_tool_calls: bool = False):
        """
        Bind tools to the client. This converts LangChain tools to Gemini tool definitions.
        """
        self.tools = tools
        self.function_declarations = []
        for tool in tools:
            # Map Python types to JSON schema types
            def map_type(py_type):
                type_str = str(py_type).lower()
                if "str" in type_str: return "STRING"
                if "int" in type_str: return "INTEGER"
                if "float" in type_str: return "NUMBER"
                if "bool" in type_str: return "BOOLEAN"
                if "list" in type_str: return "ARRAY"
                if "dict" in type_str: return "OBJECT"
                return "STRING" # Default

            properties = {}
            required = []
            
            # Inspect the args_schema if available, otherwise try to infer from args
            if tool.args_schema:
                schema = tool.args_schema.schema()
                for prop_name, prop_info in schema.get("properties", {}).items():
                    properties[prop_name] = {
                        "type": map_type(prop_info.get("type", "string")),
                        "description": prop_info.get("description", "")
                    }
                    # Handle array items if present
                    if properties[prop_name]["type"] == "ARRAY":
                         # Simplified assumption: array of strings if not specified
                         properties[prop_name]["items"] = {"type": "STRING"}

                required = schema.get("required", [])

            self.function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=required
                    )
                )
            )

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

        tool_kwargs: Dict[str, Any] = {}
        if self.function_declarations:
            tool_kwargs["function_declarations"] = self.function_declarations
        if "google_search" in self.native_tools:
            tool_kwargs["google_search"] = types.GoogleSearch()
        if "code_execution" in self.native_tools:
            tool_kwargs["code_execution"] = types.ToolCodeExecution()

        if tool_kwargs:
            all_tools.append(types.Tool(**tool_kwargs))

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
