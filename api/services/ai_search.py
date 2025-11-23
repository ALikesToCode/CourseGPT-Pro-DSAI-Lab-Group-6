from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from api.config import Settings


logger = logging.getLogger(__name__)
try:
    import importlib.util
    _HTTP2_ENABLED = importlib.util.find_spec("h2") is not None
except Exception:
    _HTTP2_ENABLED = False


class CloudflareConfigurationError(RuntimeError):
    """Raised when Cloudflare AI Search credentials are missing."""


class CloudflareRequestError(RuntimeError):
    """Raised when the Cloudflare API returns an error."""


class AISearchService:
    """
    Wrapper around the Cloudflare AI Search (AutoRAG) REST API.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._base_url: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        if settings.has_ai_search:
            self._base_url = (
                f"https://api.cloudflare.com/client/v4/accounts/"
                f"{settings.cloudflare_account_id}/autorag/rags/{settings.cloudflare_rag_id}"
            )

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url and self._settings.cloudflare_ai_search_token)

    def _require_configuration(self) -> None:
        if not self.is_configured:
            raise CloudflareConfigurationError(
                "Cloudflare AI Search is not fully configured. "
                "Set CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_RAG_ID, and CLOUDFLARE_AI_SEARCH_TOKEN."
            )

    async def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._require_configuration()
        return await self._request("POST", "search", json=payload)

    async def list_files(self, page: int = 1, per_page: int = 20, status: Optional[str] = None) -> Dict[str, Any]:
        self._require_configuration()
        params = {"page": page, "per_page": per_page}
        if status:
            params["status"] = status
        return await self._request("GET", "files", params=params)

    async def trigger_sync(self) -> Dict[str, Any]:
        self._require_configuration()
        return await self._request("PATCH", "sync")

    def _client_instance(self) -> httpx.AsyncClient:
        """
        Lazily create a shared HTTP client so we reuse the connection pool across requests.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0, http2=_HTTP2_ENABLED)
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self._base_url  # guarded by _require_configuration

        url = f"{self._base_url}/{path}"
        headers = {
            "Authorization": f"Bearer {self._settings.cloudflare_ai_search_token}",
            "Content-Type": "application/json",
        }

        client = self._client_instance()
        response = await client.request(method, url, json=json, params=params, headers=headers)

        if response.status_code >= 400:
            logger.error("Cloudflare API error (%s %s): %s", method, path, response.text)
            raise CloudflareRequestError(
                f"Cloudflare API error ({response.status_code}): {response.text}"
            )

        return response.json()
