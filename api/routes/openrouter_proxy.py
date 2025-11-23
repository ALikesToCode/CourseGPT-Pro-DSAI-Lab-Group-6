from __future__ import annotations

import json
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from api.config import get_settings

router = APIRouter()


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

    client = httpx.AsyncClient(timeout=60.0)
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
            finally:
                await client.aclose()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            raise await _error_response(resp)
        try:
            payload = resp.json()
        except ValueError:
            raise HTTPException(status_code=500, detail="OpenRouter returned non-JSON response")
        return JSONResponse(content=payload, status_code=resp.status_code)
    finally:
        await client.aclose()
