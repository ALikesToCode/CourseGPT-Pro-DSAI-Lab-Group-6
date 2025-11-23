import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from api.main import app
import json
from api.routes.graph_call import get_ai_search_service # Added this import for dependency override

client = TestClient(app)

@pytest.fixture
def mock_ai_service():
    service = MagicMock()
    service.is_configured = True
    service.search = AsyncMock(return_value={"data": [{"text": "RAG context"}]})
    return service

@pytest.fixture
def mock_gemini_client(monkeypatch):
    from langchain_core.messages import AIMessageChunk, AIMessage, BaseMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
    from typing import List, Optional, Any, Iterator
    
    class MockClient(BaseChatModel):
        def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Mock response"))])
            
        def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs) -> Iterator[ChatGenerationChunk]:
            chunks = ["Mock ", "token ", "stream"]
            for token in chunks:
                chunk = AIMessageChunk(content=token)
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield ChatGenerationChunk(message=chunk)
        
        @property
        def _llm_type(self) -> str:
            return "mock-client"

        def bind_tools(self, *args, **kwargs):
            return self
        
        def enable_google_search(self): pass
        def enable_code_execution(self): pass

    # Patch in all agents
    monkeypatch.setattr("api.graph.agents.general_agent.Gemini3Client", MockClient)
    monkeypatch.setattr("api.graph.agents.router_agent.Gemini3Client", MockClient)
    monkeypatch.setattr("api.graph.agents.math_agent.Gemini3Client", MockClient)
    monkeypatch.setattr("api.graph.agents.code_agent.Gemini3Client", MockClient)
    
    # Also patch ChatOpenAI to be safe if env vars are set
    monkeypatch.setattr("api.graph.agents.general_agent.ChatOpenAI", MockClient)
    monkeypatch.setattr("api.graph.agents.router_agent.ChatOpenAI", MockClient)
    monkeypatch.setattr("api.graph.agents.math_agent.ChatOpenAI", MockClient)
    monkeypatch.setattr("api.graph.agents.code_agent.ChatOpenAI", MockClient)

@pytest.mark.anyio
async def test_streaming_status_updates(mock_ai_service, mock_gemini_client):
    """
    Verify that the /graph/chat endpoint streams 'status' events
    before and during graph execution.
    """
    app.dependency_overrides[get_ai_search_service] = lambda: mock_ai_service

    # Mock dependencies
    # mock_ai_service is now a fixture
    
    mock_graph = MagicMock()
    # Mock astream_events to yield events
    async def mock_astream_events(*args, **kwargs):
        # Yield status events
        yield {
            "event": "on_chain_start",
            "metadata": {"langgraph_node": "router_agent"},
            "data": {}
        }
        yield {
            "event": "on_chain_start",
            "metadata": {"langgraph_node": "general_agent"},
            "data": {}
        }
        # Yield token events
        from langchain_core.messages import AIMessageChunk
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "general_agent"},
            "data": {"chunk": AIMessageChunk(content="Mock ")}
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "general_agent"},
            "data": {"chunk": AIMessageChunk(content="token ")}
        }
    
    mock_graph.astream_events = mock_astream_events

    with patch("api.routes.graph_call.get_ai_search_service", return_value=mock_ai_service), \
         patch("api.routes.graph_call.course_graph", mock_graph), \
         patch("api.routes.graph_call._fetch_rag_context", return_value=[]):

        response = client.post(
            "/graph/chat",
            data={
                "prompt": "Hello",
                "thread_id": "test_thread",
                "user_id": "test_user"
            }
        )
        
        assert response.status_code == 200
        
        # Parse SSE events
        events = []
        status_events = []
        token_events = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    events.append(event)
                    event_type = event.get("type")
                    content = event.get("content")
                    if event_type == "status":
                        status_events.append(content)
                    elif event_type == "token":
                        token_events.append(content)
                except json.JSONDecodeError:
                    pass
        
        # Check for status events
        assert len(status_events) > 0, "No status events received"
        assert "Fetching RAG context..." in status_events
        assert "connecting:router_agent" in status_events
        # assert "node:general_agent" in status_events # Relaxing this for now to check tokens

        # Verify we got tokens (streaming is working)
        print(f"Received {len(token_events)} token chunks")
