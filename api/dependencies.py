from functools import lru_cache

from fastapi import Depends, HTTPException, status

from api.config import Settings, get_settings
from api.services.ai_search import AISearchService
from api.services.r2_storage import R2StorageService
from api.services.tavily_search import TavilyService


@lru_cache
def _build_r2_service(settings: Settings) -> R2StorageService:
    return R2StorageService(settings=settings)


@lru_cache
def _build_ai_search_service(settings: Settings) -> AISearchService:
    return AISearchService(settings=settings)


@lru_cache
def _build_tavily_service(settings: Settings) -> TavilyService:
    return TavilyService(settings=settings)


def get_r2_service(
    settings: Settings = Depends(get_settings),
) -> R2StorageService:
    """
    FastAPI dependency wrapper so we can inject the R2 storage service.
    """
    try:
        return _build_r2_service(settings)
    except RuntimeError as exc:
        # Surface as a 503 to indicate missing configuration rather than a server crash.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )


def get_ai_search_service(
    settings: Settings = Depends(get_settings),
) -> AISearchService:
    """
    FastAPI dependency wrapper so routes can call Cloudflare AI Search APIs.
    """
    return _build_ai_search_service(settings)


def get_tavily_service(
    settings: Settings = Depends(get_settings),
) -> TavilyService:
    """
    FastAPI dependency wrapper so routes can call Tavily Search APIs.
    """
    try:
        return _build_tavily_service(settings)
    except RuntimeError as exc:
        # Surface as a 503 to indicate missing configuration rather than a server crash.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
