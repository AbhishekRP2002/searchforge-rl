"""Tavily adapter. `search_depth`/`extract_depth` stay on their `basic` defaults."""

from __future__ import annotations

from datetime import date

import httpx

from searchforge.providers.base import PageContent, ProviderError, SearchResult
from searchforge.providers.transport import TIMEOUT, request_json

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class TavilyProvider:
    name = "tavily"
    supports_date_filter = True

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        num_results: int,
        *,
        after: date | None = None,
        before: date | None = None,
    ) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                TAVILY_SEARCH_URL,
                provider=self.name,
                headers=self._headers,
                payload={
                    "query": query,
                    "max_results": num_results,
                    "search_depth": "basic",
                    "topic": "general",
                    **({"start_date": after.isoformat()} if after else {}),
                    **({"end_date": before.isoformat()} if before else {}),
                },
            )
        # `include_answer` is left off: Tavily's synthesised answer is the same
        # capability asymmetry as Serper's answerBox, not better retrieval.
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                excerpt=item.get("content", ""),
                date=item.get("published_date"),
                rank=rank,
            )
            for rank, item in enumerate(
                (response.json().get("results") or [])[:num_results], start=1
            )
        ]

    async def fetch(self, url: str) -> PageContent:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                TAVILY_EXTRACT_URL,
                provider=self.name,
                headers=self._headers,
                payload={
                    "urls": [url],
                    "extract_depth": "basic",
                    "format": "markdown",
                },
            )
        results = response.json().get("results") or []
        if not results:
            raise ProviderError(f"tavily returned no content for {url}")
        item = results[0]
        return PageContent(
            url=item.get("url") or url,
            title=item.get("title", ""),
            text=item.get("raw_content") or "",
            date=item.get("published_date"),
        )
