from __future__ import annotations

import sys
import uuid
from pathlib import Path

import httpx
import pytest
import anyio

pytestmark = pytest.mark.anyio("asyncio")

@pytest.fixture
def anyio_backend():
    return "asyncio"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from config import get_settings  # noqa: E402


def _r2_credentials_available() -> bool:
    try:
        settings = get_settings()
    except RuntimeError:
        return False

    return all(
        [
            settings.cloudflare_access_key,
            settings.cloudflare_secret_access_key,
            settings.cloudflare_r2_bucket,
            settings.cloudflare_r2_endpoint,
        ]
    )


skip_missing_r2 = pytest.mark.skipif(
    not _r2_credentials_available(),
    reason="Cloudflare R2 credentials not configured (.env missing).",
)

async def _request(client: httpx.AsyncClient, method: str, url: str, **kwargs):
    # Guard requests to avoid hanging the ASGI transport on multipart/file routes.
    # Give extra headroom for real network + R2 I/O.
    with anyio.fail_after(30):
        return await client.request(method, url, **kwargs)


@skip_missing_r2
async def test_r2_file_lifecycle_integration():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        follow_redirects=True,
    ) as client:
        payload = f"integration test payload {uuid.uuid4()}".encode("utf-8")
        response = await _request(
            client,
            "POST",
            "/files/",
            files={"file": ("integration.txt", payload, "text/plain")},
            data={"prefix": "integration-tests"},
        )
        assert response.status_code == 201, response.text
        uploaded_key = response.json()["file"]["key"]

        try:
            view_resp = await _request(
                client,
                "GET",
                f"/files/view/{uploaded_key}",
                params={"expires_in": 120},
            )
            assert view_resp.status_code == 200, view_resp.text
            presigned_url = view_resp.json()["url"]

            fetched = httpx.get(presigned_url, timeout=30.0)
            assert fetched.status_code == 200
            assert payload in fetched.content
        finally:
            delete_resp = await _request(client, "DELETE", f"/files/{uploaded_key}")
            if delete_resp.status_code != 200:
                pytest.fail(f"Cleanup failed for {uploaded_key}: {delete_resp.text}")
