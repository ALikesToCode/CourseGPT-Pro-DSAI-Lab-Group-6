from __future__ import annotations

import json
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from api.config import get_settings

router = APIRouter()
# Reuse a single HTTP client to avoid re-establishing TLS connections per request.
_shared_client: httpx.AsyncClient | None = None
try:
    import importlib.util
    _HTTP2_ENABLED = importlib.util.find_spec("h2") is not None
except Exception:
    _HTTP2_ENABLED = False


def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(timeout=60.0, http2=_HTTP2_ENABLED)
    return _shared_client


@router.on_event("shutdown")
async def _close_client() -> None:
    global _shared_client
    if _shared_client:
        await _shared_client.aclose()
        _shared_client = None


async def _error_response(resp: httpx.Response) -> HTTPException:
    try:
        payload = resp.json()
    except Exception:
        payload = resp.text
    return HTTPException(
        status_code=resp.status_code,
        detail={"error": payload},
    )


@router.post("/openrouter/chat/completions")
async def openrouter_chat_completion(body: Dict[str, Any], settings=Depends(get_settings)):
    """
    Thin proxy to OpenRouter's /chat/completions endpoint.
    Mirrors the OpenRouter API for both streaming and non-streaming calls.
    Requires OPENROUTER_API_KEY (or falls back to *AGENT_API_KEY values).
    """
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    base_url = settings.openrouter_base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    stream = bool(body.get("stream"))
    client = _get_client()
    if stream:
        async def event_stream():
            try:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code >= 400:
                        raise await _error_response(resp)
                    async for line in resp.aiter_lines():
                        if line is None:
                            continue
                        yield line + "\n"
            except HTTPException as http_exc:
                error_payload = json.dumps({"error": http_exc.detail})
                yield f"data: {error_payload}\n\n"
            except Exception as exc:  # noqa: BLE001
                error_payload = json.dumps({"error": str(exc)})
                yield f"data: {error_payload}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    resp = await client.post(url, headers=headers, json=body)
    if resp.status_code >= 400:
        raise await _error_response(resp)
    try:
        payload = resp.json()
    except ValueError:
        raise HTTPException(status_code=500, detail="OpenRouter returned non-JSON response")
    return JSONResponse(content=payload, status_code=resp.status_code)
