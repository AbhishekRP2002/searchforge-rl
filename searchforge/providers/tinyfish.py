"""TinyFish adapter. Search is a GET endpoint; fetch is a POST that takes a batch.

Both are on TinyFish's free tier, which changes nothing about the contract: the
adapter still normalises to the same five search fields and four page fields.
"""

from __future__ import annotations

import httpx

from searchforge.providers.base import PageContent, ProviderError, SearchResult
from searchforge.providers.transport import TIMEOUT, request_json

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"
TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"


class TinyFishProvider:
    name = "tinyfish"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key, "Content-Type": "application/json"}

    async def search(self, query: str, num_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                TINYFISH_SEARCH_URL,
                provider=self.name,
                headers=self._headers,
                method="GET",
                params={"query": query},
            )
        # No result-count parameter is documented, so the cap is applied here. The
        # slice keeps every arm at the same `num_results` the comparison asserts.
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                excerpt=item.get("snippet", ""),
                date=item.get("date"),
                rank=item.get("position", rank),
            )
            for rank, item in enumerate(
                (response.json().get("results") or [])[:num_results], start=1
            )
        ]

    async def fetch(self, url: str) -> PageContent:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                TINYFISH_FETCH_URL,
                provider=self.name,
                headers=self._headers,
                payload={"urls": [url], "format": "markdown"},
            )
        body = response.json()
        results = body.get("results") or []
        if not results:
            # Per-URL failures land in `errors`, not as an HTTP status, so an empty
            # `results` is the only signal that this fetch produced nothing.
            errors = body.get("errors") or []
            raise ProviderError(
                f"tinyfish returned no content for {url}"
                + (f": {errors[0]}" if errors else "")
            )
        item = results[0]
        return PageContent(
            url=item.get("final_url") or item.get("url") or url,
            title=item.get("title", ""),
            text=item.get("text") or "",
            date=item.get("published_date"),
        )
