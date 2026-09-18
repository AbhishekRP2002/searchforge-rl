"""Exa adapter using provider defaults and extractive content only."""

from __future__ import annotations

import httpx

from searchforge.providers.base import PageContent, ProviderError, SearchResult
from searchforge.providers.transport import TIMEOUT, post_json

EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_CONTENTS_URL = "https://api.exa.ai/contents"


class ExaProvider:
    name = "exa"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "Content-Type": "application/json"}

    async def search(self, query: str, num_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await post_json(
                client,
                EXA_SEARCH_URL,
                provider=self.name,
                headers=self._headers,
                payload={
                    "query": query,
                    "type": "auto",
                    "numResults": num_results,
                    "contents": {"highlights": True},
                },
            )

        results = []
        for rank, item in enumerate(response.json().get("results") or [], start=1):
            highlights = item.get("highlights") or []
            excerpt = "\n\n".join(highlights) if highlights else item.get("text", "")
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    excerpt=excerpt,
                    date=item.get("publishedDate"),
                    rank=rank,
                )
            )
        return results[:num_results]

    async def fetch(self, url: str) -> PageContent:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await post_json(
                client,
                EXA_CONTENTS_URL,
                provider=self.name,
                headers=self._headers,
                payload={"urls": [url], "text": True},
            )

        results = response.json().get("results") or []
        if not results:
            raise ProviderError(f"exa returned no content for {url}")
        item = results[0]
        return PageContent(
            url=item.get("url") or url,
            title=item.get("title", ""),
            text=item.get("text", ""),
            date=item.get("publishedDate"),
        )
