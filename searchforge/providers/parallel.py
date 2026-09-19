"""Parallel adapter.

Parallel's Search API is shaped differently from every other provider here: it
accepts a natural-language `objective` alongside keyword `search_queries`, and
returns `excerpts[]` per result rather than one snippet.

`objective` is deliberately never sent. It exists so the provider can focus
excerpts against the caller's underlying goal, which is the provider doing part of
the agent's reasoning -- the same category the spec rejects Exa's `deep` and
`deep-reasoning` modes for (section 5). This arm therefore receives exactly what
every other arm receives: the agent's literal query, and nothing about why it was
asked.
"""

from __future__ import annotations

import httpx

from searchforge.providers.base import PageContent, ProviderError, SearchResult
from searchforge.providers.transport import TIMEOUT, request_json

PARALLEL_SEARCH_URL = "https://api.parallel.ai/v1/search"
PARALLEL_EXTRACT_URL = "https://api.parallel.ai/v1/extract"

PARALLEL_MODE = "fast"
"""`advanced` is the API default and spends ~3s per call doing more retrieval work.
Pinned rather than inherited so a vendor default change cannot silently re-tier one
arm of a labelled comparison.
"""

PARALLEL_MAX_RESULTS_DEFAULT = 10
"""What the API returns when `max_results` is omitted ("Defaults to 10 if not
provided"). Recorded because it differs from this environment's `num_results`
default of 5: left unset, this arm would carry twice the results of every other one.

The knob lives at `advanced_settings.max_results`, not at the top level. The
request model sets `additionalProperties: false`, so a top-level `max_results` is
rejected with a 422 rather than ignored — which is how the live gate caught it.
"""


class ParallelProvider:
    name = "parallel"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "Content-Type": "application/json"}

    async def search(self, query: str, num_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await request_json(
                client,
                PARALLEL_SEARCH_URL,
                provider=self.name,
                headers=self._headers,
                payload={
                    "search_queries": [query],
                    "mode": PARALLEL_MODE,
                    # Only max_results is set here. source_policy, fetch_policy
                    # and excerpt_settings stay unset, so everything except the
                    # result count keeps the provider's ordinary defaults.
                    "advanced_settings": {"max_results": num_results},
                },
            )
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                # One result carries several excerpts; joined the same way the Exa
                # adapter joins highlights, so both multi-excerpt providers render
                # identically and neither gains from the formatting.
                excerpt="\n\n".join(item.get("excerpts") or []),
                date=item.get("publish_date"),
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
                PARALLEL_EXTRACT_URL,
                provider=self.name,
                headers=self._headers,
                # No `objective` here either: with one omitted the extract returns
                # whole-page content rather than excerpts selected against a goal,
                # which is what `web_fetch` is supposed to deliver.
                payload={"urls": [url]},
            )
        body = response.json()
        results = body.get("results") or []
        if not results:
            errors = body.get("errors") or []
            raise ProviderError(
                f"parallel returned no content for {url}"
                + (f": {errors[0]}" if errors else "")
            )
        item = results[0]
        return PageContent(
            url=item.get("url") or url,
            title=item.get("title", ""),
            text=item.get("full_content") or "\n\n".join(item.get("excerpts") or []),
            date=item.get("publish_date"),
        )
