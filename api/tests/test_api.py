from __future__ import annotations

import sys
from pathlib import Path

import anyio
import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from api.config import get_settings  # noqa: E402

pytestmark = pytest.mark.anyio("asyncio")

app = main.app


@pytest.fixture
def anyio_backend():
    # Force asyncio backend to avoid optional trio dependency.
    return "asyncio"


@pytest.fixture
async def api_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=True,
    ) as client:
        yield client


async def _request(client: httpx.AsyncClient, method: str, url: str, **kwargs):
    # Guard requests to avoid hanging the ASGI transport on multipart/file routes.
    with anyio.fail_after(15):
        return await client.request(method, url, **kwargs)


async def test_health_endpoint(api_client):
    response = await _request(api_client, "GET", "/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(not get_settings().cloudflare_r2_bucket, reason="R2 not configured")
async def test_upload_and_manage_files(api_client):
    response = await _request(
        api_client,
        "POST",
        "/files/",
        files={"file": ("note.txt", b"hello world", "text/plain")},
        data={"prefix": "notes"},
    )
    assert response.status_code == 201
    uploaded_key = response.json()["file"]["key"]

    list_response = await _request(api_client, "GET", "/files/")
    assert list_response.status_code == 200

    view_response = await _request(
        api_client,
        "GET",
        f"/files/view/{uploaded_key}",
        params={"expires_in": 300},
    )
    assert view_response.status_code == 200

    delete_response = await _request(api_client, "DELETE", f"/files/{uploaded_key}")
    assert delete_response.status_code == 200


@pytest.mark.skipif(not get_settings().has_ai_search, reason="AI Search not configured")
async def test_ai_search_query_success(api_client):
    response = await _request(api_client, "POST", "/ai-search/query", json={"query": "hello"})
    assert response.status_code == 200
    assert response.json().get("result") is not None


@pytest.mark.skipif(not get_settings().has_tavily, reason="Tavily not configured")
async def test_tavily_map(api_client):
    response = await _request(
        api_client,
        "POST",
        "/ai-search/tavily/map",
        json={"url": "https://docs.tavily.com", "max_depth": 1, "limit": 2},
    )
    assert response.status_code == 200
    assert response.json().get("results") is not None


@pytest.mark.skipif(not get_settings().has_tavily, reason="Tavily not configured")
async def test_tavily_crawl(api_client):
    response = await _request(
        api_client,
        "POST",
        "/ai-search/tavily/crawl",
        json={"url": "https://docs.tavily.com", "include_images": False, "limit": 2},
    )
    assert response.status_code == 200
    assert response.json().get("results") is not None


@pytest.mark.skipif(not get_settings().has_tavily, reason="Tavily not configured")
async def test_tavily_extract(api_client):
    response = await _request(
        api_client,
        "POST",
        "/ai-search/tavily/extract",
        json={"urls": "https://example.com"},
    )
    assert response.status_code == 200
    assert response.json().get("results") is not None
