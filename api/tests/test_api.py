from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import httpx
import anyio

import main
from api.dependencies import get_ai_search_service, get_r2_service
from api.services.ai_search import CloudflareConfigurationError
from api.dependencies import get_tavily_service

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend():
    return "asyncio"


app = main.app


class FakeR2Service:
    def __init__(self) -> None:
        self.storage: Dict[str, bytes] = {}

    def upload_fileobj(self, file_obj, key: str, content_type=None, metadata=None) -> Dict[str, str]:
        self.storage[key] = file_obj.read()
        return {"key": key, "url": f"https://example.com/{key}"}

    def list_objects(self, prefix=None, max_items=500) -> List[Dict[str, Any]]:
        return [
            {"key": key, "size": len(content), "url": f"https://example.com/{key}"}
            for key, content in self.storage.items()
        ]

    def delete_object(self, key: str) -> None:
        self.storage.pop(key, None)

    def generate_presigned_get_url(self, key: str, expires_in: int = 900) -> str:
        if key not in self.storage:
            raise RuntimeError("missing key")
        return f"https://example.com/{key}?exp={expires_in}"


class FakeAISearchService:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    async def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.should_fail:
            raise CloudflareConfigurationError("missing config")
        return {"result": [{"id": "doc1", "score": 0.99, "payload": payload}]}

    async def list_files(self, **params) -> Dict[str, Any]:
        return {"result": [], "result_info": {"count": 0}}

    async def trigger_sync(self) -> Dict[str, Any]:
        return {"success": True}


class FakeTavilyService:
    def search(self, **kwargs):
        return {"results": [{"query": kwargs.get("query"), "source": "search"}]}

    def map(self, **kwargs):
        return {"results": ["https://example.com"], "config": kwargs}

    def crawl(self, **kwargs):
        return {"base_url": kwargs.get("url"), "results": [{"url": "https://example.com/page"}]}

    def extract(self, **kwargs):
        urls = kwargs.get("urls")
        return {"results": [{"url": urls, "raw_content": "content"}]}


@pytest.fixture(autouse=True)
def override_dependencies():
    fake_r2 = FakeR2Service()
    fake_ai = FakeAISearchService()
    fake_tavily = FakeTavilyService()

    app.dependency_overrides[get_r2_service] = lambda: fake_r2
    app.dependency_overrides[get_ai_search_service] = lambda: fake_ai
    app.dependency_overrides[get_tavily_service] = lambda: fake_tavily
    yield fake_r2, fake_ai
    app.dependency_overrides.clear()


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
    with anyio.fail_after(5):
        return await client.request(method, url, **kwargs)


async def test_health_endpoint(api_client):
    response = await _request(api_client, "GET", "/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_upload_and_manage_files(override_dependencies, api_client):
    fake_r2, _ = override_dependencies

    response = await _request(
        api_client,
        "POST",
        "/files/",
        files={"file": ("note.txt", b"hello world", "text/plain")},
        data={"prefix": "notes"},
    )
    assert response.status_code == 201
    uploaded_key = response.json()["file"]["key"]
    assert uploaded_key in fake_r2.storage

    list_response = await _request(api_client, "GET", "/files/")
    assert list_response.status_code == 200
    assert list_response.json()["files"]

    view_response = await _request(
        api_client,
        "GET",
        f"/files/view/{uploaded_key}",
        params={"expires_in": 300},
    )
    assert view_response.status_code == 200
    assert view_response.json()["url"].endswith("exp=300")

    delete_response = await _request(api_client, "DELETE", f"/files/{uploaded_key}")
    assert delete_response.status_code == 200
    assert uploaded_key not in fake_r2.storage


async def test_ai_search_query_success(api_client):
    response = await _request(api_client, "POST", "/ai-search/query", json={"query": "hello"})
    assert response.status_code == 200
    assert response.json()["result"]


async def test_ai_search_query_missing_config(api_client):
    failing_service = FakeAISearchService(should_fail=True)

    app.dependency_overrides[get_ai_search_service] = lambda: failing_service

    response = await _request(api_client, "POST", "/ai-search/query", json={"query": "hello"})
    assert response.status_code == 503
    assert "missing config" in response.json()["detail"]


async def test_tavily_map(api_client):
    response = await _request(
        api_client,
        "POST",
        "/ai-search/tavily/map",
        json={"url": "https://docs.tavily.com", "max_depth": 2, "limit": 3},
    )
    assert response.status_code == 200
    assert response.json()["results"]


async def test_tavily_crawl(api_client):
    response = await _request(
        api_client,
        "POST",
        "/ai-search/tavily/crawl",
        json={"url": "https://docs.tavily.com", "include_images": True},
    )
    assert response.status_code == 200
    assert response.json()["results"]


async def test_tavily_extract(api_client):
    response = await _request(
        api_client,
        "POST",
        "/ai-search/tavily/extract",
        json={"urls": "https://example.com"},
    )
    assert response.status_code == 200
    assert response.json()["results"]
