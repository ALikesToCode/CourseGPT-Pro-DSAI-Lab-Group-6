import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from api.main import app
import json

client = TestClient(app)

@pytest.mark.anyio
async def test_streaming_status_updates():
    """
    Verify that the /graph/chat endpoint streams 'status' events
    before and during graph execution.
    """
    # Mock dependencies
    mock_ai_service = AsyncMock()
    mock_ai_service.search.return_value = {"data": []}
    
    mock_graph = MagicMock()
    mock_graph.astream.return_value = AsyncMock()
    # Mock astream to yield some events
    async def mock_astream(*args, **kwargs):
        yield {"general_agent": {"messages": [{"content": "Hello"}]}}
    mock_graph.astream = mock_astream

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
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    events.append(json.loads(data))
                except json.JSONDecodeError:
                    pass
        
        # Check for status events
        status_types = [e.get("content") for e in events if e.get("type") == "status"]
        assert "Fetching RAG context..." in status_types
        assert "connecting:router_agent" in status_types
        assert "node:general_agent" in status_types
