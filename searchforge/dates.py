"""Date-filter maths. No verifiers imports, no network.

Two jobs: validate what the agent asks for, and decide whether a result's own
date falls inside the requested window when the filter is emulated.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_DISPLAY_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y")
"""Absolute display dates as SERP providers print them ("Mar 5, 2024").
Relative ones ("2 days ago") are deliberately absent: they resolve against the
provider's clock at crawl time, not ours, so any reading of them is a guess."""


def parse_bound(value: str | None, field: str) -> tuple[date | None, str | None]:
    """Parse an agent-supplied bound. Returns (date, None) or (None, message).

    Strictly YYYY-MM-DD. `date.fromisoformat` alone would also accept "20240301"
    and week dates on 3.11+, and the tool schema promises one format.
    """
    if value is None or value == "":
        return None, None
    if not isinstance(value, str) or not _ISO_DAY.match(value.strip()):
        return None, (
            f"Invalid date {value!r} for `{field}`: use YYYY-MM-DD, e.g. 2024-03-01."
        )
    try:
        return date.fromisoformat(value.strip()), None
    except ValueError:
        return None, f"Invalid date {value!r} for `{field}`: not a real calendar day."


def check_window(after: date | None, before: date | None) -> str | None:
    if after and before and after > before:
        return (
            f"Invalid date range: `after` ({after.isoformat()}) is later than "
            f"`before` ({before.isoformat()})."
        )
    return None


def parse_result_date(value: object) -> date | None:
    """A provider's result date, as a calendar day, or None when it cannot be read.

    Accepts ISO dates and timestamps (with `Z` or an offset), Unix seconds, and
    absolute display dates. Anything else is None rather than a best guess: a
    wrong date silently moves a result across the window edge.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=UTC).date()
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not (text := value.strip()):
        return None
    try:
        parsed = datetime.fromisoformat(text)  # 3.11+ reads a trailing Z
    except ValueError:
        pass
    else:
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC)
        return parsed.date()
    for fmt in _DISPLAY_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def within(day: date | None, after: date | None, before: date | None) -> bool:
    """Inclusive at both ends. An undated result is never inside a window."""
    if day is None:
        return False
    if after and day < after:
        return False
    return not (before and day > before)


def google_tbs(after: date | None, before: date | None) -> str:
    """Google's custom-date-range `tbs`, as Serper and Firecrawl both forward it.

    Month and day are unpadded, as Google writes them. An open end is omitted
    rather than sent empty.
    """
    parts = ["cdr:1"]
    if after:
        parts.append(f"cd_min:{after.month}/{after.day}/{after.year}")
    if before:
        parts.append(f"cd_max:{before.month}/{before.day}/{before.year}")
    return ",".join(parts)
