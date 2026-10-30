"""Length caps on what the tools return. Pure text maths: no verifiers, no network.

Every provider's output is cut by this module, after normalisation, at the same
number. A cap applied inside an adapter would let one arm's provider-specific
text shape decide how much the model sees, which is a hidden provider advantage.

The cut keeps the head. Web page tails are mostly navigation and footers, and a
head cut is what nearly every published search environment does. The marker is
always visible, so the model knows more exists, and it is the one place the raw
length is written down: `fetch_chars_raw` and `fetch_truncated` are parsed back
out of it from the trace (`parse_fetch_marker`), because the tools are pure
functions with no state channel to report through.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

from searchforge.providers.base import SearchResult

EXCERPT_ELLIPSIS = " […]"

_TRUNCATED = re.compile(r"\[truncated: showing ([\d,]+) of ([\d,]+) chars\]")
_PAGED = re.compile(r"\[page (\d+) of (\d+): chars [\d,]+-[\d,]+ of ([\d,]+)\b")


def _n(value: int) -> str:
    return f"{value:,}"


def truncation_marker(shown: int, total: int) -> str:
    return f"[truncated: showing {_n(shown)} of {_n(total)} chars]"


def page_marker(page: int, pages: int, start: int, end: int, total: int) -> str:
    more = (
        f" Call web_fetch with page={page + 1} for more."
        if page < pages
        else " This is the last page."
    )
    return f"[page {page} of {pages}: chars {_n(start + 1)}-{_n(end)} of {_n(total)}.{more}]"


def cap_page(text: str, limit: int | None) -> str:
    """Head-keep `text` to `limit` characters, with a marker when anything was cut.

    `None` is uncapped, and a page that fits comes back byte-identical, so a cap
    that never binds changes nothing the model sees.
    """
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n{truncation_marker(limit, len(text))}"


def page_count(total: int, size: int) -> int:
    return max(1, math.ceil(total / size))


def slice_page(text: str, size: int, page: int) -> str | None:
    """Page `page` (1-based) of `text` in `size`-character pages, or None when
    that page does not exist. A one-page document carries no marker, matching
    `cap_page` on a page that fits."""
    pages = page_count(len(text), size)
    if page < 1 or page > pages:
        return None
    if pages == 1:
        return text
    start = (page - 1) * size
    end = min(start + size, len(text))
    return f"{text[start:end]}\n\n{page_marker(page, pages, start, end, len(text))}"


def cap_excerpts(results: list[SearchResult], limit: int | None) -> list[SearchResult]:
    """Head-keep every excerpt to `limit` characters. Cheap insurance: today's
    providers return ~1.3k characters per whole result list, but one that
    returned page-length excerpts would otherwise get far more text per call."""
    if limit is None:
        return results
    return [
        replace(r, excerpt=r.excerpt[:limit] + EXCERPT_ELLIPSIS)
        if len(r.excerpt) > limit
        else r
        for r in results
    ]


def parse_fetch_marker(content: str) -> tuple[int, bool] | None:
    """(raw page length, whether the model saw less than all of it), read back
    from a cap or page marker. None when the content carries no marker."""
    if match := _TRUNCATED.search(content):
        return int(match.group(2).replace(",", "")), True
    if match := _PAGED.search(content):
        return int(match.group(3).replace(",", "")), int(match.group(2)) > 1
    return None
