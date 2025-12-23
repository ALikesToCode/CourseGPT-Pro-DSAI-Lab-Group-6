from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
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
SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif", ".bmp")
DEFAULT_PDF_MIME = "application/pdf"


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


def _normalize_messages(messages):
    """
    Ensure we always iterate over a list of message-like objects.
    """
    if not messages:
        return []
    if isinstance(messages, list):
        return messages
    return [messages]


def _message_content_text(message: Any) -> str:
    """
    Extract textual content from different message shapes (AIMessage, ToolMessage, dict).
    """
    content = ""
    if hasattr(message, "content"):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content") or message.get("text")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                text_val = chunk.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "".join(parts)
    return str(content) if content else ""


def _truncate_text(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _truncate_for_log(text: str, limit: int = 200) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


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

    is_pdf = _is_pdf(filename, content_type)
    mime_type = content_type or (DEFAULT_PDF_MIME if is_pdf else "application/octet-stream")
    upload_name = filename or ("upload.pdf" if is_pdf else "upload.bin")

    files = {
        "file": (
            upload_name,
            file_bytes,
            mime_type,
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


def _is_pdf(filename: str, content_type: Optional[str]) -> bool:
    """
    Validate PDFs by mime or extension to gate parsing logic.
    """
    normalized = (filename or "").lower()
    mime = (content_type or "").lower()
    return mime == DEFAULT_PDF_MIME or normalized.endswith(".pdf")


def _is_image(filename: str, content_type: Optional[str]) -> bool:
    """
    Broad image detection to allow camera/photos uploads without hard-failing.
    """
    normalized = (filename or "").lower()
    mime = (content_type or "").lower()
    return mime.startswith("image/") or normalized.endswith(SUPPORTED_IMAGE_EXTS)


async def _extract_uploaded_context(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Extract readable text from uploaded PDFs or images. Images rely on the OCR
    microservice; PDFs fall back to pypdf if OCR is unavailable.
    """
    text = await _call_remote_ocr(file_bytes, filename, content_type)
    if not text and _is_pdf(filename, content_type):
        text = await asyncio.to_thread(_extract_text_from_pdf_bytes, file_bytes)

    if not text:
        return None

    return {
        "filename": filename,
        "content_type": content_type or DEFAULT_PDF_MIME,
        "text": _truncate_text(text),
    }


async def _process_uploaded_file(file: UploadFile) -> Optional[Dict[str, Any]]:
    file_bytes = await file.read()
    file.file.seek(0)
    if not file_bytes:
        return None

    return await _extract_uploaded_context(
        file_bytes=file_bytes,
        filename=file.filename or "",
        content_type=file.content_type,
    )


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
        prompt_text = prompt.strip()
        if not prompt_text:
            prompt_text = "Please review the attached file."

        start_time = time.monotonic()
        logger.info(
            "Graph chat start thread=%s user=%s prompt_preview=%s",
            thread_id,
            user_id,
            _truncate_for_log(prompt),
        )

        # We need to read the file here because UploadFile is not async-safe to pass into the generator directly
        # if we close the request context. However, for StreamingResponse, it's safer to read bytes now.
        file_bytes = None
        filename = ""
        content_type = ""
        file_kind = None
        image_data_uri: Optional[str] = None
        if file is not None:
            filename = file.filename or ""
            content_type = file.content_type or ""
            is_pdf = _is_pdf(filename, content_type)
            is_image = _is_image(filename, content_type)
            if not is_pdf and not is_image:
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF or image uploads are accepted",
                )
            file_kind = "pdf" if is_pdf else "image"
            file_bytes = await file.read()
            content_type = content_type or (DEFAULT_PDF_MIME if is_pdf else "application/octet-stream")
            file.file.seek(0)
            if is_image and file_bytes:
                encoded = base64.b64encode(file_bytes).decode("ascii")
                image_data_uri = f"data:{content_type or 'image/png'};base64,{encoded}"

        async def event_generator():
            # Guard the graph execution so slow upstream models don't hang the SSE forever.
            timeout_seconds = float(os.getenv("GRAPH_STREAM_TIMEOUT", "600"))
            streamed_lengths: Dict[str, int] = {}
            
            try:
                # 1. Process File Upload (if any)
                uploaded_context: Optional[Dict[str, Any]] = None
                if file_bytes:
                    uploaded_context = await _extract_uploaded_context(
                        file_bytes=file_bytes,
                        filename=filename or "upload",
                        content_type=content_type,
                    )

                    if uploaded_context:
                        uploaded_context["kind"] = file_kind or "file"

                # 2. Fetch RAG Context
                yield f"data: {json.dumps({'type': 'status', 'content': 'Preparing context...'})}\n\n"
                rag_context = await _fetch_rag_context(
                    ai_service,
                    prompt_text,
                    user_id=user_id,
                    additional_context=uploaded_context["text"] if uploaded_context else None,
                )

                # 3. Prepare Config
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
                
                logger.info(
                    "Graph config thread=%s user=%s rag_docs=%s uploaded=%s",
                    thread_id,
                    user_id,
                    len(rag_context),
                    bool(uploaded_context),
                )

                # 4. Run Graph
                yield f"data: {json.dumps({'type': 'status', 'content': 'connecting:router_agent'})}\n\n"
                
                async with asyncio.timeout(timeout_seconds):
                    user_message_content: Any = prompt_text
                    if image_data_uri:
                        user_message_content = [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ]

                    async for event in course_graph.astream_events(
                        {"messages": [HumanMessage(content=user_message_content)]},
                        config=config,
                        version="v2"
                    ):
                        kind = event["event"]
                        node_name = event.get("metadata", {}).get("langgraph_node", "")
                        
                        if kind == "on_chat_model_stream":
                            # Stream tokens from agents
                            content = event["data"]["chunk"].content
                            if content:
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        
                        elif kind == "on_tool_start":
                            # Tool usage status
                            tool_name = event["name"]
                            if not tool_name.endswith("_handoff"):
                                yield f"data: {json.dumps({'type': 'status', 'content': f'Running tool {tool_name}'})}\n\n"
                                yield f"data: {json.dumps({'type': 'tool_use', 'tool': tool_name, 'input': event['data'].get('input')})}\n\n"

                        elif kind == "on_chat_model_end":
                            # Check for Router Handoffs
                            if node_name == "router_agent":
                                output = event["data"]["output"]
                                # output is ChatResult or AIMessage depending on runner? 
                                # In astream_events, output is usually the result of the runnable.
                                # For ChatModel, it's AIMessage.
                                if hasattr(output, "tool_calls") and output.tool_calls:
                                    for tool_call in output.tool_calls:
                                        if tool_call.get("name", "").endswith("_handoff"):
                                            args = tool_call.get("args") or {}
                                            # Populate defaults so UI always shows target + rationale.
                                            derived_handoff = args.get("handoff") or tool_call.get("name", "").replace("_handoff", "")
                                            args.setdefault("handoff", derived_handoff)
                                            args.setdefault(
                                                "route_rationale",
                                                f"Router selected {derived_handoff.replace('_', ' ')} based on the task."
                                            )
                                            args.setdefault(
                                                "task_summary",
                                                args.get("handoff_plan")
                                                or args.get("route_plan")
                                                or "Routing to specialized agent."
                                            )
                                            debug_info = {"tool": tool_call["name"], **args}
                                            logger.info(
                                                "Router handoff thread=%s user=%s tool=%s",
                                                thread_id,
                                                user_id,
                                                debug_info.get("tool"),
                                            )
                                            yield f"data: {json.dumps({'type': 'handoff', 'content': debug_info})}\n\n"
                        
                        elif kind == "on_chain_start":
                            # Node transitions
                            if node_name and node_name != "__start__":
                                yield f"data: {json.dumps({'type': 'status', 'content': f'node:{node_name}'})}\n\n"

                # Signal end of stream
                yield "data: [DONE]\n\n"

            except asyncio.TimeoutError:
                logger.warning(
                    "Graph stream timeout thread=%s user=%s after %.1fs",
                    thread_id,
                    user_id,
                    timeout_seconds,
                )
                yield f"data: {json.dumps({'type': 'error', 'content': f'Graph streaming timed out after {timeout_seconds}s'})}\n\n"
            except Exception as e:
                logger.exception("Error during stream")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        response = StreamingResponse(event_generator(), media_type="text/event-stream")
        elapsed = time.monotonic() - start_time
        logger.info("Graph chat setup complete thread=%s user=%s elapsed=%.2fs", thread_id, user_id, elapsed)
        return response

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph chat invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
