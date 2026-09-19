"""Keenable adapter. Search is POST; fetch is a GET carrying query parameters."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from searchforge.providers.base import PageContent, ProviderError, SearchResult
from searchforge.providers.transport import TIMEOUT, request_json

KEENABLE_SEARCH_URL = "https://api.keenable.ai/v1/search"
KEENABLE_FETCH_URL = "https://api.keenable.ai/v1/fetch"

KEENABLE_MAX_CHARS = 1_000_000
"""Keenable is the one provider that truncates server-side by default, at 50,000
characters. Leaving that default in place would cap this arm and no other, which is
exactly the "same cap, unequal arms" asymmetry the spec refuses. Set high enough to
be inert; lower it only as part of a cap policy that applies to every provider.
"""


"""Without this the endpoint serves only URLs already in Keenable's index and errors
on anything else, so a fetch comparison would silently run on a different URL
population than the other arms.
"""
KEENABLE_LIVE = True


def _iso(published_at: object) -> str | None:
    """Keenable dates a page with a Unix timestamp; every other adapter emits ISO."""
    if not isinstance(published_at, (int, float)) or isinstance(published_at, bool):
        return None
    return datetime.fromtimestamp(published_at, tz=UTC).isoformat()


class KeenableProvider:
    name = "keenable"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key, "Content-Type": "application/json"}

    async def search(self, query: str, num_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                KEENABLE_SEARCH_URL,
                provider=self.name,
                headers=self._headers,
                payload={"query": query, "max_results": num_results},
            )
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                # `snippet` is the page excerpt; `description` is a summary line.
                # Prefer the excerpt so this arm carries the same kind of text.
                excerpt=item.get("snippet") or item.get("description", ""),
                date=item.get("published_at"),
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
                KEENABLE_FETCH_URL,
                provider=self.name,
                headers=self._headers,
                method="GET",
                params={
                    "url": url,
                    "max_chars": KEENABLE_MAX_CHARS,
                    "live": KEENABLE_LIVE,
                },
            )
        item = response.json() or {}
        # Raise only when the provider returned no result object at all, matching
        # every other adapter. A result whose text is empty is a thin page, which
        # the spec treats as ordinary tool output the agent recovers from.
        if not item:
            raise ProviderError(f"keenable returned no result for {url}")
        return PageContent(
            url=item.get("url") or url,
            title=item.get("title", ""),
            text=item.get("content", ""),
            date=_iso(item.get("published_at")),
        )
