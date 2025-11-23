from .health import router as health_router
from .files import router as files_router
from .ai_search import router as ai_search_router
from .openrouter_proxy import router as openrouter_router

__all__ = ["health_router", "files_router", "ai_search_router", "openrouter_router"]
