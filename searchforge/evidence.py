"""Pure evidence maths. No verifiers imports, no network — see trace_reader.py
for the Trace side.

Citations are always checked against what this episode's own tools returned,
never against a curated gold-URL list. Gold URLs encode the retrieval path of
whichever provider built the dataset, which would silently rig a provider
comparison.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_eid", "ref_src", "igshid")

_URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")
_TRAILING = ".,;:!?'\""


def normalize_url(url: str) -> str:
    """Canonical form for comparison: lowercase scheme and host, no fragment,
    no tracking parameters, no trailing slash. Path case is preserved."""
    parts = urlsplit(url.strip())
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith(TRACKING_PREFIXES)
        ]
    )
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def urls_in_text(text: str) -> set[str]:
    return {match.rstrip(_TRAILING) for match in _URL_RE.findall(text or "")}


def grounded_fraction(cited: set[str], retrieved: set[str]) -> float:
    """Share of cited URLs that this episode actually retrieved.

    No citations scores 1.0: not citing is a different behaviour from citing
    something fabricated, and conflating them would penalise honest short answers.
    """
    if not cited:
        return 1.0
    seen = {normalize_url(u) for u in retrieved}
    hits = sum(1 for u in cited if normalize_url(u) in seen)
    return hits / len(cited)


def _canonical(text: str) -> str:
    # Commas are removed, not replaced with a space: "56,000" must canonicalise
    # to "56000" so it matches a gold answer written without the separator.
    return re.sub(r"\s+", " ", (text or "").lower().replace(",", "")).strip()


def answer_present(answer: str | list[str], haystack: str) -> bool:
    """Did the gold answer appear in this text? Used for `answer_in_results`,
    which separates retrieval failure from reasoning failure."""
    hay = _canonical(haystack)
    items = answer if isinstance(answer, list) else [answer]
    return all(_canonical(str(item)) in hay for item in items)
