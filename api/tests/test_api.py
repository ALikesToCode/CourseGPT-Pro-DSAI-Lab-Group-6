from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

import main
from dependencies import get_ai_search_service, get_r2_service
from services.ai_search import CloudflareConfigurationError


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


@pytest.fixture(autouse=True)
def override_dependencies():
    fake_r2 = FakeR2Service()
    fake_ai = FakeAISearchService()

    app.dependency_overrides[get_r2_service] = lambda: fake_r2
    app.dependency_overrides[get_ai_search_service] = lambda: fake_ai
    yield fake_r2, fake_ai
    app.dependency_overrides.clear()


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_manage_files(override_dependencies):
    fake_r2, _ = override_dependencies
    client = TestClient(app)

    response = client.post(
        "/files",
        files={"file": ("note.txt", b"hello world", "text/plain")},
        data={"prefix": "notes"},
    )
    assert response.status_code == 201
    uploaded_key = response.json()["file"]["key"]
    assert uploaded_key in fake_r2.storage

    list_response = client.get("/files")
    assert list_response.status_code == 200
    assert list_response.json()["files"]

    view_response = client.get(f"/files/view/{uploaded_key}", params={"expires_in": 300})
    assert view_response.status_code == 200
    assert view_response.json()["url"].endswith("exp=300")

    delete_response = client.delete(f"/files/{uploaded_key}")
    assert delete_response.status_code == 200
    assert uploaded_key not in fake_r2.storage


def test_ai_search_query_success():
    client = TestClient(app)
    response = client.post("/ai-search/query", json={"query": "hello"})
    assert response.status_code == 200
    assert response.json()["result"]


def test_ai_search_query_missing_config():
    failing_service = FakeAISearchService(should_fail=True)

    app.dependency_overrides[get_ai_search_service] = lambda: failing_service
    client = TestClient(app)

    response = client.post("/ai-search/query", json={"query": "hello"})
    assert response.status_code == 503
    assert "missing config" in response.json()["detail"]

