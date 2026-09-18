"""Serper adapter. Called with Serper's ordinary default settings — no `tbs`
recency filter, no cache controls. See spec section 5, "Provider defaults"."""

from __future__ import annotations

import httpx

from searchforge.providers.base import PageContent, SearchResult
from searchforge.providers.transport import TIMEOUT, post_json

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_SCRAPE_URL = "https://scrape.serper.dev"


class SerperProvider:
    name = "serper"

    def __init__(self, api_key: str, *, include_answer_box: bool = False) -> None:
        self._api_key = api_key
        self._include_answer_box = include_answer_box

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-KEY": self._api_key, "Content-Type": "application/json"}

    async def search(self, query: str, num_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await post_json(
                client,
                SERPER_SEARCH_URL,
                provider=self.name,
                headers=self._headers,
                payload={"q": query, "num": num_results},
            )
        payload = response.json()

        results: list[SearchResult] = []
        if self._include_answer_box and (box := payload.get("answerBox")):
            results.append(
                SearchResult(
                    title=box.get("title", ""),
                    url=box.get("link", ""),
                    excerpt=box.get("snippet") or box.get("answer") or "",
                    date=None,
                    rank=0,
                )
            )
        for item in (payload.get("organic") or [])[:num_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    excerpt=item.get("snippet", ""),
                    date=item.get("date"),
                    rank=item.get("position", len(results) + 1),
                )
            )
        return results

    async def fetch(self, url: str) -> PageContent:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await post_json(
                client,
                SERPER_SCRAPE_URL,
                provider=self.name,
                headers=self._headers,
                payload={"url": url, "includeMarkdown": True},
            )
        payload = response.json()
        metadata = payload.get("metadata") or {}
        text = payload.get("text") or payload.get("markdown") or ""

        return PageContent(
            url=url,
            title=metadata.get("title", ""),
            text=text,
            date=metadata.get("published_time"),
        )
