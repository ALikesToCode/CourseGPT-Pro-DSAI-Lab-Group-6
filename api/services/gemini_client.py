import os
import json
import base64
import binascii
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator
from google import genai
from google.genai import types
from google.oauth2 import service_account
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun

class Gemini3Client(BaseChatModel):
    """
    A wrapper for the Google Gen AI SDK (v1alpha/beta) to support Gemini 3 models
    in a way that is compatible with LangChain's invoke/bind_tools pattern.
    """
    model_name: str = "gemini-3-pro-preview"
    fallback_models: List[str] = []
    use_vertex: bool = False
    client: Any = None
    api_key: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_credentials_b64: Optional[str] = None
    native_tools: List[str] = []
    # We don't store tools/functions here directly for bind_tools mutation anymore,
    # but we keep them for internal state if needed or rely on kwargs in _generate.
    
    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gemini-3-pro-preview",
        fallback_models: List[str] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials_b64: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model
        self.fallback_models = fallback_models or []
        self.use_vertex = bool(vertex_project and vertex_credentials_b64)
        self.api_key = api_key
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        self.vertex_credentials_b64 = vertex_credentials_b64
        self.client = None # Lazy init
        self.native_tools = []

    def _get_client(self):
        if self.client:
            return self.client
            
        if self.use_vertex:
            try:
                decoded = base64.b64decode(self.vertex_credentials_b64)
                creds_dict = json.loads(decoded)
            except (binascii.Error, json.JSONDecodeError) as exc:
                raise ValueError("VERTEX_CREDENTIALS_JSON_B64 is not valid base64-encoded JSON") from exc
            scopes = [
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/generative-language",
            ]
            creds = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=scopes,
            )
            self.client = genai.Client(
                vertexai=True,
                project=self.vertex_project,
                location=self.vertex_location or "us-central1",
                credentials=creds,
            )
        else:
            if not self.api_key:
                raise ValueError("Missing GEMINI_API_KEY or Vertex credentials")
            self.client = genai.Client(api_key=self.api_key)
        return self.client

    @property
    def _llm_type(self) -> str:
        return "gemini-3-client"

    def enable_google_search(self):
        """Enable Google Search grounding."""
        if "google_search" not in self.native_tools:
            self.native_tools.append("google_search")

    def enable_code_execution(self):
        """Enable Code Execution."""
        if "code_execution" not in self.native_tools:
            self.native_tools.append("code_execution")

    def bind_tools(self, tools: List[BaseTool], **kwargs):
        """
        Bind tools to the client. Returns a new runnable with tools bound.
        """
        # Convert LangChain tools to Gemini FunctionDeclarations
        function_declarations = []
        for tool in tools:
            def map_type(py_type):
                type_str = str(py_type).lower()
                if "str" in type_str: return "STRING"
                if "int" in type_str: return "INTEGER"
                if "float" in type_str: return "NUMBER"
                if "bool" in type_str: return "BOOLEAN"
                if "list" in type_str: return "ARRAY"
                if "dict" in type_str: return "OBJECT"
                return "STRING"

            properties = {}
            required = []
            
            if tool.args_schema:
                schema = tool.args_schema.schema()
                for prop_name, prop_info in schema.get("properties", {}).items():
                    properties[prop_name] = {
                        "type": map_type(prop_info.get("type", "string")),
                        "description": prop_info.get("description", "")
                    }
                    if properties[prop_name]["type"] == "ARRAY":
                         properties[prop_name]["items"] = {"type": "STRING"}

                required = schema.get("required", [])

            function_declarations.append(
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
        
        return self.bind(function_declarations=function_declarations, **kwargs)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[types.Content]:
        contents = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System messages are handled in config, but if mixed in, we might need to handle differently.
                # For now, we assume system message is extracted before calling this or passed as config.
                pass 
            elif isinstance(msg, HumanMessage):
                contents.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
            elif isinstance(msg, AIMessage):
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
                contents.append(types.Content(role="user", parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=msg.name, 
                            response={"result": msg.content} 
                        )
                    )
                ]))
        return contents

    def _prepare_config(self, stop: Optional[List[str]] = None, **kwargs) -> types.GenerateContentConfig:
        config_args = {}
        all_tools = []
        
        # Handle bound tools (passed via kwargs from bind)
        fds = kwargs.get("function_declarations", [])
        if fds:
            config_args["function_declarations"] = fds
        
        if "google_search" in self.native_tools:
            config_args["google_search"] = types.GoogleSearch()
        if "code_execution" in self.native_tools:
            config_args["code_execution"] = types.ToolCodeExecution()

        if config_args:
            all_tools.append(types.Tool(**config_args))

        config_params = {}
        if all_tools:
            config_params["tools"] = all_tools
        
        if stop:
            config_params["stop_sequences"] = stop

        # Thinking config
        if "gemini-3" in self.model_name:
             config_params["thinking_config"] = {"include_thoughts": True}

        return types.GenerateContentConfig(**config_params)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        contents = self._convert_messages(messages)
        system_instruction = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.content
                break

        config = self._prepare_config(stop, **kwargs)
        
        models_to_try = []
        seen_models = set()
        for m in [self.model_name] + self.fallback_models:
            if m not in seen_models:
                models_to_try.append(m)
                seen_models.add(m)

        response = None
        last_exception = None

        for model_name in models_to_try:
            try:
                # Update thinking config based on model name
                current_config = config
                # We already set thinking in _prepare_config based on self.model_name, 
                # but if fallback is different, we might need to adjust. 
                # For simplicity, we assume _prepare_config logic holds or we'd need to rebuild config per model.
                
                response = self._get_client().models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=current_config.copy(update=dict(system_instruction=system_instruction))
                )
                break
            except Exception as e:
                last_exception = e
                print(f"Warning: Model {model_name} failed: {e}")
                continue

        if response is None:
            if last_exception:
                raise last_exception
            raise RuntimeError("No response generated.")

        # Convert response
        content = response.text or ""
        tool_calls = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append({
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                        "id": "call_" + part.function_call.name
                    })

        msg = AIMessage(content=content, tool_calls=tool_calls)
        generation = ChatGeneration(message=msg)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        contents = self._convert_messages(messages)
        system_instruction = None
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.content
                break

        config = self._prepare_config(stop, **kwargs)
        
        # Streaming only supports primary model for now to keep it simple
        # or we could loop try/catch but streaming fallback is trickier.
        # We'll stick to primary model.
        
        stream = self._get_client().models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config.copy(update=dict(system_instruction=system_instruction))
                )

        for chunk in stream:
            content = chunk.text or ""
            # Handle tool calls in chunks if they arrive (Gemini usually sends them at end)
            tool_call_chunks = []
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if part.function_call:
                        tool_call_chunks.append({
                            "name": part.function_call.name,
                            "args": part.function_call.args, # args might be partial? SDK usually aggregates.
                            "id": "call_" + part.function_call.name,
                            "index": 0 # Required for chunk
                        })
            
            msg_chunk = AIMessage(content=content, tool_calls=tool_call_chunks) # AIMessageChunk?
            # LangChain expects AIMessageChunk for streaming
            from langchain_core.messages import AIMessageChunk
            
            # Construct proper chunk
            # Note: Gemini SDK aggregates args in function_call, so it might not be partial.
            # But for streaming, we just emit what we get.
            
            # If tool calls are present, we need to format them for AIMessageChunk
            tc_chunks = []
            for tc in tool_call_chunks:
                tc_chunks.append({
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"], # args must be str for chunk?
                    "id": tc["id"],
                    "index": tc["index"]
                })

            chunk_msg = AIMessageChunk(content=content, tool_call_chunks=tc_chunks)
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=chunk_msg)
            yield ChatGenerationChunk(message=chunk_msg)
