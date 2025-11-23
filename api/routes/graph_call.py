from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from langchain.messages import HumanMessage
from pypdf import PdfReader

from api.dependencies import get_ai_search_service
from api.services.ai_search import (
    AISearchService,
    CloudflareConfigurationError,
    CloudflareRequestError,
)

from api.graph.graph import graph as course_graph

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_RAG_RESULTS = 5
MAX_CONTEXT_CHARS = 6000
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL")
OCR_SERVICE_TOKEN = os.getenv("OCR_SERVICE_TOKEN")


def _get_state_field(result_state, field):
    if isinstance(result_state, dict):
        return result_state.get(field)
    return getattr(result_state, field, None)


def _extract_latest_message(messages):
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if hasattr(last, "content"):
            return last.content
        if isinstance(last, dict):
            return last.get("content") or last.get("text") or str(last)
        return str(last)
    if isinstance(messages, str):
        return messages
    return None


def _extract_router_debug(messages):
    """
    Return the first handoff tool call content for debugging the router output.
    Handles both AIMessage (with tool_calls) and ToolMessage (with name).
    """
    import json

    if not isinstance(messages, list):
        return None

    for msg in messages:
        # Case 1: AIMessage with tool_calls (from router_agent node)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("name", "").endswith("_handoff"):
                    return {"tool": tool_call["name"], "content": tool_call["args"]}

        # Case 2: ToolMessage (legacy/fallback)
        if hasattr(msg, "name") and isinstance(msg.name, str) and msg.name.endswith("_handoff"):
            payload = msg.content
            # Try to parse JSON payloads if content is a string representation
            if isinstance(payload, str):
                try:
                    payload_json = json.loads(payload)
                    payload = payload_json
                except Exception:
                    pass
            return {"tool": msg.name, "content": payload}
    return None


def _truncate_text(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_text_from_pdf_bytes(payload: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to parse uploaded PDF for OCR fallback: %s", exc)
        return ""

    chunks: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to extract text from page %s: %s", idx, exc)
            extracted = ""
        if extracted:
            chunks.append(extracted.strip())
    return "\n\n".join(chunks)


def _coalesce_ocr_response(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"]
    pages = payload.get("pages") or []
    texts = [page.get("text", "") for page in pages if isinstance(page, dict) and page.get("text")]
    return "\n\n".join(filter(None, texts))


async def _call_remote_ocr(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str],
) -> str:
    if not OCR_SERVICE_URL:
        return ""

    headers = {}
    if OCR_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {OCR_SERVICE_TOKEN}"

    files = {
        "file": (
            filename or "upload.pdf",
            file_bytes,
            content_type or "application/pdf",
        )
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(OCR_SERVICE_URL, files=files, headers=headers)
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Remote OCR request failed: %s", exc)
        return ""

    try:
        data = response.json()
    except ValueError:
        logger.warning("OCR service returned non-JSON payload")
        return ""
    return _coalesce_ocr_response(data)


async def _process_uploaded_file(file: UploadFile) -> Optional[Dict[str, Any]]:
    file_bytes = await file.read()
    file.file.seek(0)
    if not file_bytes:
        return None

    text = await _call_remote_ocr(file_bytes, file.filename or "", file.content_type)
    if not text:
        text = await asyncio.to_thread(_extract_text_from_pdf_bytes, file_bytes)

    if not text:
        return None

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "text": _truncate_text(text),
    }


async def _fetch_rag_context(
    ai_service: AISearchService,
    prompt: str,
    user_id: str,
    additional_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not getattr(ai_service, "is_configured", False):
        return []

    query = prompt.strip()
    if additional_context:
        query = f"{prompt.strip()}\n\nDocument context:\n{_truncate_text(additional_context, 1500)}"

    payload: Dict[str, Any] = {"query": query, "max_num_results": DEFAULT_RAG_RESULTS}
    # Safety: ensure no filters key slips through to Cloudflare to avoid 7001 errors
    payload.pop("filters", None)

    try:
        response = await ai_service.search(payload)
    except (CloudflareConfigurationError, CloudflareRequestError) as exc:
        msg = str(exc)
        if "filters" in msg.lower():
            logger.info("Skipping RAG fetch due to filters validation: %s", msg)
        else:
            logger.warning("Unable to fetch RAG context: %s", msg)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during RAG fetch")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw_hits = (
        response.get("result")
        or response.get("results")
        or response.get("data")
        or []
    )
    
    if isinstance(raw_hits, dict) and "data" in raw_hits:
        hits = raw_hits["data"]
    elif isinstance(raw_hits, list):
        hits = raw_hits
    else:
        hits = []

    contexts: List[Dict[str, Any]] = []
    for idx, hit in enumerate(hits[:DEFAULT_RAG_RESULTS]):
        payload_blob = hit.get("payload")
        payload_text = ""
        if isinstance(payload_blob, dict):
            payload_text = (
                payload_blob.get("text")
                or payload_blob.get("content")
                or payload_blob.get("snippet")
            )

        # Extract text from various possible fields
        raw_text = (
            hit.get("text")
            or hit.get("content")
            or hit.get("snippet")
            or payload_text
        )

        text = ""
        if isinstance(raw_text, str):
            text = raw_text
        elif isinstance(raw_text, list):
            # Handle list of content chunks (e.g. Cloudflare structure)
            parts = []
            for item in raw_text:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            text = "\n\n".join(filter(None, parts))
        
        text = text.strip()

        contexts.append(
            {
                "id": hit.get("id") or hit.get("document_id") or f"hit-{idx}",
                "score": hit.get("score"),
                "metadata": hit.get("metadata") or hit.get("payload") or hit.get("attributes"),
                "text": _truncate_text(text) if text else "",
            }
        )
    return contexts


from fastapi.responses import StreamingResponse
import json

@router.post("/chat")
async def graph_ask(
    prompt: str = Form(...),
    thread_id: str = Form(...),
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    ai_service: AISearchService = Depends(get_ai_search_service),
):
    """Accepts form-data (multipart) with an optional file upload.
    Returns a StreamingResponse with SSE events.
    """

    try:
        # if a file was uploaded, enforce PDF-only and attach bytes+metadata
        uploaded_context: Optional[Dict[str, Any]] = None
        if file is not None:
            filename = (file.filename or "").lower()
            is_pdf = (file.content_type == "application/pdf") or filename.endswith(".pdf")
            if not is_pdf:
                raise HTTPException(status_code=400, detail="Only PDF file uploads are accepted")
            uploaded_context = await _process_uploaded_file(file)

        rag_context = await _fetch_rag_context(
            ai_service,
            prompt,
            user_id=user_id,
            additional_context=uploaded_context["text"] if uploaded_context else None,
        )

        config_payload: Dict[str, Any] = {
            "thread_id": thread_id,
            "user_id": user_id,
        }
        if rag_context:
            config_payload["rag_documents"] = rag_context
        if uploaded_context:
            config_payload["uploaded_file"] = uploaded_context

        config = {
            "configurable": config_payload
        }

        async def event_generator():
            # Track how much content has already been streamed per node so we can
            # emit incremental deltas instead of re-sending full messages.
            streamed_lengths: Dict[str, int] = {}
            try:
                # Stream updates from the graph
                async for event in course_graph.astream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="updates"
                ):
                    # event is a dict {node_name: update}
                    for node_name, update in event.items():
                        # Handle Router Agent Handoffs
                        if node_name == "router_agent":
                            messages = update.get("messages", [])
                            if messages:
                                # Check for handoff tool calls
                                debug_info = _extract_router_debug(messages)
                                if debug_info:
                                    yield f"data: {json.dumps({'type': 'handoff', 'content': debug_info})}\n\n"
                        
                        # Handle Agent Responses (Final Answer)
                        # We assume the last message from a leaf agent is the answer
                        if node_name in ["general_agent", "math_agent", "code_agent"]:
                            messages = update.get("messages", [])
                            if messages:
                                # Debug logging
                                print(f"DEBUG: Node {node_name} returned messages type: {type(messages)}")
                                
                                if isinstance(messages, list):
                                    last_msg = messages[-1]
                                else:
                                    # It's a single message object (e.g. AIMessage)
                                    last_msg = messages

                                content = ""
                                if hasattr(last_msg, "content"):
                                    content = last_msg.content
                                elif isinstance(last_msg, dict):
                                    content = last_msg.get("content") or ""
                                
                                if content:
                                    if delta:
                                        yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
                                        streamed_lengths[node_name] = prev_len + len(delta)
                        
                        # Handle Tool Calls (e.g. RAG Search)
                        messages = update.get("messages", [])
                        if messages:
                            if isinstance(messages, list):
                                for msg in messages:
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        for tc in msg.tool_calls:
                                            # Skip handoff tools as they are handled by router logic
                                            if not tc.get("name", "").endswith("_handoff"):
                                                yield f"data: {json.dumps({'type': 'tool_use', 'tool': tc['name'], 'input': tc['args']})}\n\n"
                
                # Signal end of stream
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Error during stream")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph chat invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
