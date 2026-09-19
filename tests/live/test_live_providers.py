"""Provider round trips. Run explicitly with `pytest -m live`.

This is the Phase 5 live gate: every adapter in the registry must complete a real
search and a real fetch. Offline fixtures prove normalisation against a response
we wrote down; only this proves it against the response the provider actually
sends. A provider without a configured key skips rather than fails, so partial
credential coverage still exercises everything it can.
"""

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from searchforge.servers.tool import PROVIDERS

pytestmark = pytest.mark.live
ENV_FILE = Path(__file__).parents[2] / ".env"


def _key(name: str) -> str:
    value = os.environ.get(name) or dotenv_values(ENV_FILE).get(name)
    if not value:
        pytest.skip(f"{name} is not configured in the process or .env")
    return value


@pytest.mark.parametrize("provider", sorted(PROVIDERS), ids=str)
async def test_live_search_and_fetch(provider):
    key_name, construct = PROVIDERS[provider]
    client = construct(_key(key_name))

    results = await client.search("official Python programming language website", 3)
    assert results, f"{provider} returned no search results"
    assert len(results) <= 3, f"{provider} ignored num_results"
    top = results[0]
    assert top.url.startswith("http"), f"{provider} search: url field mismapped"
    assert top.title, f"{provider} search: title field mismapped"
    assert top.excerpt, f"{provider} search: excerpt field mismapped"
    assert top.rank == 1, f"{provider} search: rank is not 1-based"

    page = await client.fetch(top.url)
    assert page.url.startswith("http"), f"{provider} fetch: url field mismapped"
    assert page.text.strip(), f"{provider} fetch: text field mismapped"
