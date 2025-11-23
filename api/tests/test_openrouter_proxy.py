from __future__ import annotations

import json

import httpx
import pytest
import respx
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app),
        base_url="http://test",
        follow_redirects=True,
    ) as client:
        yield client


@respx.mock
@pytest.mark.anyio("asyncio")
async def test_openrouter_proxy_non_stream(api_client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"id": "cmpl-1", "choices": [{"message": {"content": "ok"}}]},
        )
    )

    payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    resp = await api_client.post("/openrouter/chat/completions", json=payload)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert mock_route.called
    sent = mock_route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer test-key"
    assert json.loads(sent.content)["model"] == "test-model"


@respx.mock
@pytest.mark.anyio("asyncio")
async def test_openrouter_proxy_stream(api_client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    stream_body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: [DONE]\n\n"
    mock_route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=stream_body,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "stream"}],
        "stream": True,
    }
    resp = await api_client.post("/openrouter/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.text
    assert "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}" in body
    assert "data: [DONE]" in body
    assert mock_route.called
