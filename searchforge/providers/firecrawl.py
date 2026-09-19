"""Firecrawl adapter. Called with Firecrawl's ordinary default settings."""

from __future__ import annotations

import httpx

from searchforge.providers.base import PageContent, ProviderError, SearchResult
from searchforge.providers.transport import TIMEOUT, request_json

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"


class FirecrawlProvider:
    name = "firecrawl"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def search(self, query: str, num_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                FIRECRAWL_SEARCH_URL,
                provider=self.name,
                headers=self._headers,
                payload={"query": query, "limit": num_results, "sources": ["web"]},
            )
        data = response.json().get("data") or {}
        # v2 groups hits by source (`data.web`); a bare list is the v1 shape.
        items = data.get("web") or [] if isinstance(data, dict) else data

        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                excerpt=item.get("description", ""),
                date=item.get("date"),
                rank=rank,
            )
            for rank, item in enumerate(items[:num_results], start=1)
        ]

    async def fetch(self, url: str) -> PageContent:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                FIRECRAWL_SCRAPE_URL,
                provider=self.name,
                headers=self._headers,
                payload={
                    "url": url,
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                },
            )
        data = response.json().get("data") or {}
        if not data:
            raise ProviderError(f"firecrawl returned no content for {url}")
        metadata = data.get("metadata") or {}
        return PageContent(
            url=metadata.get("sourceURL") or url,
            title=metadata.get("title", ""),
            text=data.get("markdown") or "",
            date=metadata.get("publishedTime"),
        )
