from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from dependencies import get_ai_search_service
from services.ai_search import (
    AISearchService,
    CloudflareConfigurationError,
    CloudflareRequestError,
)

router = APIRouter(prefix="/ai-search", tags=["ai-search"])


class AISearchQuery(BaseModel):
    query: str = Field(..., description="Natural language query for the RAG pipeline")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")
    max_num_results: Optional[int] = Field(default=None, ge=1, le=50)
    ranking_options: Optional[Dict[str, Any]] = None
    reranking: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


def _handle_cloudflare_errors(exc: Exception) -> None:
    if isinstance(exc, CloudflareConfigurationError):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    if isinstance(exc, CloudflareRequestError):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    raise exc


@router.post("/query", summary="Query Cloudflare AI Search (AutoRAG)")
async def run_ai_search(
    payload: AISearchQuery,
    ai_service: AISearchService = Depends(get_ai_search_service),
):
    try:
        result = await ai_service.search(payload.model_dump(exclude_none=True))
    except Exception as exc:  # noqa: BLE001
        _handle_cloudflare_errors(exc)
    else:
        return result


@router.get("/files", summary="List documents registered in AI Search")
async def list_ai_search_files(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=50),
    status_filter: Optional[str] = Query(None, description="Optional ingestion status filter"),
    ai_service: AISearchService = Depends(get_ai_search_service),
):
    try:
        return await ai_service.list_files(page=page, per_page=per_page, status=status_filter)
    except Exception as exc:  # noqa: BLE001
        _handle_cloudflare_errors(exc)


@router.patch("/sync", summary="Trigger AI Search to sync the configured R2 data source")
async def trigger_ai_search_sync(
    ai_service: AISearchService = Depends(get_ai_search_service),
):
    try:
        return await ai_service.trigger_sync()
    except Exception as exc:  # noqa: BLE001
        _handle_cloudflare_errors(exc)

