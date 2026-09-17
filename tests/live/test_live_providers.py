"""Provider round trips. Run explicitly with `pytest -m live`."""

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from searchforge.providers.exa import ExaProvider
from searchforge.providers.serper import SerperProvider

pytestmark = pytest.mark.live
ENV_FILE = Path(__file__).parents[2] / ".env"


def _key(name: str) -> str:
    value = os.environ.get(name) or dotenv_values(ENV_FILE).get(name)
    if not value:
        pytest.skip(f"{name} is not configured in the process or .env")
    return value


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [(SerperProvider, "SERPER_API_KEY"), (ExaProvider, "EXA_API_KEY")],
)
async def test_live_search_and_fetch(provider, key_name):
    client = provider(_key(key_name))

    results = await client.search("official Python programming language website", 3)
    assert results and all(result.url.startswith("http") for result in results)

    page = await client.fetch(results[0].url)
    assert page.url.startswith("http")
    assert page.text.strip()
