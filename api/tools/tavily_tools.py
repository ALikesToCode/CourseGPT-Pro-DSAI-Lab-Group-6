"""LangChain tools that wrap the Tavily web APIs.

These helpers expose Tavily Search / Map / Crawl / Extract to the agents so
they can fetch fresh web content when configured with ``TAVILY_API_KEY``.
Each tool is defensive: if the API key is missing or a request fails, the
error message is returned to the model instead of raising.
"""

from __future__ import annotations

from typing import List, Optional, Union

from langchain.tools import tool
from tavily import TavilyClient

from api.config import get_settings


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    if not settings.has_tavily:
        raise RuntimeError("TAVILY_API_KEY is not configured.")
    return TavilyClient(api_key=settings.tavily_api_key)


@tool
def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_answer: bool = False,
    include_raw_content: bool = False,
    include_images: bool = False,
) -> str:
    """Run a Tavily web search. Use for general web questions or fresh data.

    Args:
        query: Natural language query.
        search_depth: "basic" (cheap) or "advanced" (deeper, more expensive).
        max_results: Number of results to return (1-10 typical).
        include_answer: Whether Tavily should synthesize an answer.
        include_raw_content: Include raw page content snippets.
        include_images: Include image URLs when available.
    """

    try:
        client = _get_tavily_client()
        result = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Tavily search failed: {exc}"


@tool
def tavily_map(
    url: str,
    instructions: Optional[str] = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    select_paths: Optional[List[str]] = None,
    select_domains: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    allow_external: bool = True,
    timeout: Optional[float] = None,
) -> str:
    """Build a link map starting from a URL (Tavily Map).

    Args mirror the Tavily Map API. Depth/breadth are capped by Tavily (depth<=5).
    """

    try:
        client = _get_tavily_client()
        result = client.map(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            select_paths=select_paths,
            select_domains=select_domains,
            exclude_paths=exclude_paths,
            exclude_domains=exclude_domains,
            allow_external=allow_external,
            timeout=timeout,
        )
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Tavily map failed: {exc}"


@tool
def tavily_crawl(
    url: str,
    instructions: Optional[str] = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    select_paths: Optional[List[str]] = None,
    select_domains: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_images: bool = False,
    include_favicon: bool = False,
    extract_depth: str = "basic",
    format: str = "markdown",
    allow_external: bool = True,
    timeout: Optional[float] = 150,
) -> str:
    """Crawl a site/section and return extracted content (Tavily Crawl)."""

    try:
        client = _get_tavily_client()
        result = client.crawl(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            select_paths=select_paths,
            select_domains=select_domains,
            exclude_paths=exclude_paths,
            exclude_domains=exclude_domains,
            include_images=include_images,
            include_favicon=include_favicon,
            extract_depth=extract_depth,
            format=format,
            allow_external=allow_external,
            timeout=timeout,
        )
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Tavily crawl failed: {exc}"


@tool
def tavily_extract(
    urls: Union[str, List[str]],
    include_images: bool = False,
    include_favicon: bool = False,
    extract_depth: str = "basic",
    format: str = "markdown",
    timeout: Optional[float] = None,
) -> str:
    """Extract page content for one or more URLs (Tavily Extract)."""

    try:
        client = _get_tavily_client()
        result = client.extract(
            urls=urls,
            include_images=include_images,
            include_favicon=include_favicon,
            extract_depth=extract_depth,
            format=format,
            timeout=timeout,
        )
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Tavily extract failed: {exc}"
