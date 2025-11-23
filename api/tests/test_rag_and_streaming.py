import sys
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import asyncio
from langchain_core.messages import AIMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.tools.rag_search import rag_search
from api.routes.graph_call import _extract_router_debug

# --- Test RAG Tool ---

def test_rag_search_not_configured():
    with patch("api.tools.rag_search.get_settings") as mock_settings:
        mock_settings.return_value.has_ai_search = False
        
        result = rag_search.invoke("test query")
        assert "RAG search is not configured" in result

def test_rag_search_success():
    with patch("api.tools.rag_search.get_settings") as mock_settings, \
         patch("api.tools.rag_search.AISearchService") as MockService:
        
        mock_settings.return_value.has_ai_search = True
        
        mock_service_instance = MockService.return_value
        mock_service_instance.is_configured = True
        
        # Mock the search method to return a valid result
        # Since the tool runs the async method in a loop, we need to make sure the mock returns a coroutine result if called async,
        # or just a return value if the tool logic handles it. 
        # The tool uses: loop.run_until_complete(service.search(...))
        # So service.search must be an async method (or return a future).
        
        async def mock_search(payload):
            return {
                "result": [
                    {"score": 0.95, "payload": {"text": "Document content 1"}},
                    {"score": 0.80, "text": "Document content 2"} # Test fallback key
                ]
            }
        
        mock_service_instance.search = mock_search
        
        result = rag_search.invoke("test query")
        
        assert "Document content 1" in result
        assert "Document content 2" in result
        assert "Score: 0.95" in result

def test_rag_search_no_results():
    with patch("api.tools.rag_search.get_settings") as mock_settings, \
         patch("api.tools.rag_search.AISearchService") as MockService:
        
        mock_settings.return_value.has_ai_search = True
        mock_service_instance = MockService.return_value
        mock_service_instance.is_configured = True
        
        async def mock_search(payload):
            return {"result": []}
        
        mock_service_instance.search = mock_search
        
        result = rag_search.invoke("test query")
        assert "No relevant documents found" in result

# --- Test Router Debug Extraction ---

def test_extract_router_debug_tool_calls():
    # Test extraction from AIMessage with tool_calls
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "general_agent_handoff", "args": {"task_summary": "test"}, "id": "123"}
        ]
    )
    debug = _extract_router_debug([msg])
    assert debug is not None
    assert debug["tool"] == "general_agent_handoff"
    assert debug["content"] == {"task_summary": "test"}

def test_extract_router_debug_legacy():
    # Test extraction from ToolMessage (legacy/fallback)
    msg = ToolMessage(
        content='{"task_summary": "test"}',
        name="general_agent_handoff",
        tool_call_id="123"
    )
    debug = _extract_router_debug([msg])
    assert debug is not None
    assert debug["tool"] == "general_agent_handoff"
    assert debug["content"] == {"task_summary": "test"}

# --- Test SSE Event Format (Mocking Graph Output) ---

def test_streaming_format():
    # We'll simulate the generator logic from graph_ask
    
    async def mock_graph_stream():
        # 1. Router Handoff
        yield {
            "router_agent": {
                "messages": [
                    AIMessage(content="", tool_calls=[{"name": "general_agent_handoff", "args": {"plan": "foo"}, "id": "1"}])
                ]
            }
        }
        # 2. Tool Use
        yield {
            "general_agent": {
                "messages": [
                    AIMessage(content="", tool_calls=[{"name": "rag_search", "args": {"query": "bar"}, "id": "2"}])
                ]
            }
        }
        # 3. Token
        yield {
            "general_agent": {
                "messages": [AIMessage(content="Hello")]
            }
        }
    
    async def run_test():
        events = []
        async for event in mock_graph_stream():
            for node_name, update in event.items():
                # logic copied/adapted from graph_call.py for verification
                if node_name == "router_agent":
                    messages = update.get("messages", [])
                    debug_info = _extract_router_debug(messages)
                    if debug_info:
                        events.append({"type": "handoff", "content": debug_info})
                
                if node_name in ["general_agent"]:
                    messages = update.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if not tc.get("name", "").endswith("_handoff"):
                                    events.append({"type": "tool_use", "tool": tc["name"], "input": tc["args"]})
                        
                        if hasattr(msg, "content") and msg.content:
                            events.append({"type": "token", "content": msg.content})

        assert len(events) == 3
        assert events[0]["type"] == "handoff"
        assert events[0]["content"]["tool"] == "general_agent_handoff"
        
        assert events[1]["type"] == "tool_use"
        assert events[1]["tool"] == "rag_search"
        assert events[1]["input"] == {"query": "bar"}
        
        assert events[2]["type"] == "token"
        assert events[2]["content"] == "Hello"

    asyncio.run(run_test())
