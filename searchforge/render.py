"""Turn normalised results into the text the model sees.

Markdown is the default because JSON keys, braces and quoting repeat on every
result and cost sequence budget — and policy updates cost 3-5x rollout generation.
"""

from __future__ import annotations

import json

from searchforge.providers.base import PageContent, SearchResult

NO_RESULTS = "No results found for that query."


def render_results(results: list[SearchResult], fmt: str = "markdown") -> str:
    if fmt == "markdown":
        return _markdown(results)
    if fmt == "json":
        return json.dumps(
            [
                {
                    "rank": r.rank,
                    "title": r.title,
                    "url": r.url,
                    "excerpt": r.excerpt,
                    **({"date": r.date} if r.date else {}),
                }
                for r in results
            ],
            ensure_ascii=False,
        )
    raise ValueError(f"unknown result_format {fmt!r}; expected 'markdown' or 'json'")


def _markdown(results: list[SearchResult]) -> str:
    if not results:
        return NO_RESULTS
    blocks = []
    for r in results:
        head = f"[{r.rank}] {r.title}"
        if r.date:
            head += f" ({r.date})"
        blocks.append(f"{head}\n    {r.url}\n    {r.excerpt}")
    return "\n\n".join(blocks)


def render_page(page: PageContent) -> str:
    head = page.title or page.url
    if page.date:
        head += f" ({page.date})"
    return f"{head}\n{page.url}\n\n{page.text}"
