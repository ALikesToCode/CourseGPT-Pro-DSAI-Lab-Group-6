from functools import lru_cache

from fastapi import Depends

from config import Settings, get_settings
from services.ai_search import AISearchService
from services.r2_storage import R2StorageService


@lru_cache
def _build_r2_service(settings: Settings) -> R2StorageService:
    return R2StorageService(settings=settings)


@lru_cache
def _build_ai_search_service(settings: Settings) -> AISearchService:
    return AISearchService(settings=settings)


def get_r2_service(
    settings: Settings = Depends(get_settings),
) -> R2StorageService:
    """
    FastAPI dependency wrapper so we can inject the R2 storage service.
    """
    return _build_r2_service(settings)


def get_ai_search_service(
    settings: Settings = Depends(get_settings),
) -> AISearchService:
    """
    FastAPI dependency wrapper so routes can call Cloudflare AI Search APIs.
    """
    return _build_ai_search_service(settings)

