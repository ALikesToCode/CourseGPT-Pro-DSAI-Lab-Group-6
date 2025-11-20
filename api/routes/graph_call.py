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

    filters: Dict[str, Any] = {}
    if user_id:
        filters["user_id"] = user_id

    query = prompt.strip()
    if additional_context:
        query = f"{prompt.strip()}\n\nDocument context:\n{_truncate_text(additional_context, 1500)}"

    payload: Dict[str, Any] = {"query": query, "max_num_results": DEFAULT_RAG_RESULTS}
    if filters:
        payload["filters"] = filters

    try:
        response = await ai_service.search(payload)
    except (CloudflareConfigurationError, CloudflareRequestError) as exc:
        logger.warning("Unable to fetch RAG context: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during RAG fetch")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    hits = (
        response.get("result")
        or response.get("results")
        or response.get("data")
        or []
    )

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

        text = (
            hit.get("text")
            or hit.get("content")
            or hit.get("snippet")
            or payload_text
            or ""
        ).strip()

        contexts.append(
            {
                "id": hit.get("id") or hit.get("document_id") or f"hit-{idx}",
                "score": hit.get("score"),
                "metadata": hit.get("metadata") or hit.get("payload"),
                "text": _truncate_text(text) if text else "",
            }
        )
    return contexts


@router.post("/chat")
async def graph_ask(
    prompt: str = Form(...),
    thread_id: str = Form(...),
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    ai_service: AISearchService = Depends(get_ai_search_service),
):
    """Accepts form-data (multipart) with an optional file upload.

    Fields:
    - `prompt`: the user prompt
    - `thread_id`: thread identifier
    - `user_id`: user identifier
    - `file`: optional file uploaded with the request
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

        # invoke the compiled state graph with a HumanMessage
        result_state = course_graph.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

        messages = _get_state_field(result_state, "messages")
        latest_message = _extract_latest_message(messages)

        return {"latest_message": latest_message}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph chat invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
