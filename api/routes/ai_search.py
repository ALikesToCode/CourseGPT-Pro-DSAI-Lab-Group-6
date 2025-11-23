from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import get_ai_search_service, get_tavily_service
from api.services.ai_search import (
    AISearchService,
    CloudflareConfigurationError,
    CloudflareRequestError,
)
from api.services.tavily_search import TavilyService

router = APIRouter(prefix="/ai-search", tags=["ai-search"])


class AISearchQuery(BaseModel):
    query: str = Field(..., description="Natural language query for the RAG pipeline")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")
    max_num_results: Optional[int] = Field(default=None, ge=1, le=50)
    ranking_options: Optional[Dict[str, Any]] = None
    reranking: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class TavilySearchQuery(BaseModel):
    query: str = Field(..., description="Search query")
    search_depth: str = "basic"
    include_answer: bool = False
    include_raw_content: bool = False
    max_results: int = 5
    include_images: bool = False

    model_config = {"extra": "allow"}


class TavilyMapQuery(BaseModel):
    url: str = Field(..., description="Root URL to map")
    instructions: Optional[str] = None
    max_depth: int = Field(1, ge=1, le=5)
    max_breadth: int = Field(20, ge=1)
    limit: int = Field(50, ge=1)
    select_paths: Optional[List[str]] = None
    select_domains: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    allow_external: bool = True
    timeout: Optional[float] = Field(
        default=None,
        ge=10,
        le=150,
        description="Optional timeout in seconds (10-150) for Tavily Map",
    )


class TavilyCrawlQuery(BaseModel):
    url: str = Field(..., description="Root URL to crawl")
    instructions: Optional[str] = None
    max_depth: int = Field(1, ge=1, le=5)
    max_breadth: int = Field(20, ge=1)
    limit: int = Field(50, ge=1)
    select_paths: Optional[List[str]] = None
    select_domains: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    include_images: bool = False
    include_favicon: bool = False
    extract_depth: str = Field("basic", description="basic or advanced")
    format: str = Field("markdown", description="markdown or text")
    allow_external: bool = True
    timeout: Optional[float] = Field(
        default=150,
        ge=10,
        le=150,
        description="Timeout in seconds (10-150) for crawl operations",
    )


class TavilyExtractQuery(BaseModel):
    urls: Union[str, List[str]] = Field(..., description="URL or list of URLs to extract")
    include_images: bool = False
    include_favicon: bool = False
    extract_depth: str = Field("basic", description="basic or advanced")
    format: str = Field("markdown", description="markdown or text")
    timeout: Optional[float] = Field(
        default=None, description="Optional timeout (seconds) between 1 and 60"
    )


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


@router.post("/tavily/search", summary="Query Tavily Search API")
async def run_tavily_search(
    payload: TavilySearchQuery,
    tavily_service: TavilyService = Depends(get_tavily_service),
):
    try:
        result = tavily_service.search(**payload.model_dump(exclude_none=True))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    else:
        return result


@router.post("/tavily/map", summary="Call Tavily Map")
async def run_tavily_map(
    payload: TavilyMapQuery,
    tavily_service: TavilyService = Depends(get_tavily_service),
):
    try:
        return tavily_service.map(**payload.model_dump(exclude_none=True))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/tavily/crawl", summary="Call Tavily Crawl")
async def run_tavily_crawl(
    payload: TavilyCrawlQuery,
    tavily_service: TavilyService = Depends(get_tavily_service),
):
    try:
        return tavily_service.crawl(**payload.model_dump(exclude_none=True))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/tavily/extract", summary="Call Tavily Extract")
async def run_tavily_extract(
    payload: TavilyExtractQuery,
    tavily_service: TavilyService = Depends(get_tavily_service),
):
    try:
        return tavily_service.extract(**payload.model_dump(exclude_none=True))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


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
