from tavily import TavilyClient

from api.config import Settings


class TavilyService:
    def __init__(self, settings: Settings):
        if not settings.has_tavily:
            raise RuntimeError(
                "Tavily API key not configured. Please set TAVILY_API_KEY."
            )
        self.client = TavilyClient(api_key=settings.tavily_api_key)

    def search(self, query: str, **kwargs):
        return self.client.search(query, **kwargs)

    def map(self, **kwargs):
        return self.client.map(**kwargs)

    def crawl(self, **kwargs):
        return self.client.crawl(**kwargs)

    def extract(self, **kwargs):
        return self.client.extract(**kwargs)
