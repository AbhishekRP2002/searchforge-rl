"""One contract, every provider.

Spec section 12 gates Phase 5 on "parameterised contract tests per adapter". The
adapters differ only in request shape and field names — normalisation, the shared
retry policy and the empty-result rule are identical by design, so they are
asserted once here rather than copied per provider. Request-shape assertions that
are genuinely provider-specific stay in that provider's own test module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import httpx
import pytest
import respx
from tenacity import wait_none

from searchforge.providers import transport
from searchforge.providers.base import ProviderError, SearchProvider
from searchforge.providers.exa import EXA_CONTENTS_URL, EXA_SEARCH_URL, ExaProvider
from searchforge.providers.firecrawl import (
    FIRECRAWL_SCRAPE_URL,
    FIRECRAWL_SEARCH_URL,
    FirecrawlProvider,
)
from searchforge.providers.keenable import (
    KEENABLE_FETCH_URL,
    KEENABLE_SEARCH_URL,
    KeenableProvider,
)
from searchforge.providers.parallel import (
    PARALLEL_EXTRACT_URL,
    PARALLEL_SEARCH_URL,
    ParallelProvider,
)
from searchforge.providers.serper import (
    SERPER_SCRAPE_URL,
    SERPER_SEARCH_URL,
    SerperProvider,
)
from searchforge.providers.tavily import (
    TAVILY_EXTRACT_URL,
    TAVILY_SEARCH_URL,
    TavilyProvider,
)
from searchforge.providers.tinyfish import (
    TINYFISH_FETCH_URL,
    TINYFISH_SEARCH_URL,
    TinyFishProvider,
)


@dataclass(frozen=True)
class Contract:
    """Everything that differs between adapters, so the assertions need not."""

    name: str
    build: Callable[[str], SearchProvider]
    search_url: str
    fetch_url: str
    search_fixture: str
    fetch_fixture: str
    no_results: dict
    """A 200 response that carries no search hits."""
    no_page: dict
    """A 200 response that carries no fetched page object."""
    search_method: str = "POST"
    fetch_method: str = "POST"
    fetch_raises_when_empty: bool = True
    """Serper's scrape always returns a page object, so it has nothing to raise on."""


CONTRACTS = [
    Contract(
        "serper", SerperProvider, SERPER_SEARCH_URL, SERPER_SCRAPE_URL,
        "serper_search.json", "serper_scrape.json",
        no_results={"organic": []}, no_page={}, fetch_raises_when_empty=False,
    ),
    Contract(
        "exa", ExaProvider, EXA_SEARCH_URL, EXA_CONTENTS_URL,
        "exa_search.json", "exa_contents.json",
        no_results={"results": []}, no_page={"results": []},
    ),
    Contract(
        "firecrawl", FirecrawlProvider, FIRECRAWL_SEARCH_URL, FIRECRAWL_SCRAPE_URL,
        "firecrawl_search.json", "firecrawl_scrape.json",
        no_results={"success": True, "data": {"web": []}},
        no_page={"success": True, "data": {}},
    ),
    Contract(
        "tavily", TavilyProvider, TAVILY_SEARCH_URL, TAVILY_EXTRACT_URL,
        "tavily_search.json", "tavily_extract.json",
        no_results={"results": []}, no_page={"results": [], "failed_results": []},
    ),
    Contract(
        "parallel", ParallelProvider, PARALLEL_SEARCH_URL, PARALLEL_EXTRACT_URL,
        "parallel_search.json", "parallel_extract.json",
        no_results={"results": []}, no_page={"results": [], "errors": []},
    ),
    Contract(
        "tinyfish", TinyFishProvider, TINYFISH_SEARCH_URL, TINYFISH_FETCH_URL,
        "tinyfish_search.json", "tinyfish_fetch.json",
        no_results={"results": []}, no_page={"results": [], "errors": []},
        search_method="GET",
    ),
    Contract(
        "keenable", KeenableProvider, KEENABLE_SEARCH_URL, KEENABLE_FETCH_URL,
        "keenable_search.json", "keenable_fetch.json",
        no_results={"results": []}, no_page={},
        fetch_method="GET",
    ),
]

CASES = pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)

PAGE_URL = "https://example.com/nintendo-history"


def _mock(method: str, url: str, **kwargs) -> respx.Route:
    # url__startswith, not an exact match: the GET adapters carry their arguments
    # in the query string, which an equality match would reject.
    return respx.route(method=method, url__startswith=url).mock(**kwargs)


@CASES
@respx.mock
async def test_search_normalises_to_the_shared_result_fields(contract, load_fixture):
    _mock(
        contract.search_method,
        contract.search_url,
        return_value=httpx.Response(200, json=load_fixture(contract.search_fixture)),
    )

    results = await contract.build("k").search("who founded nintendo", 5)

    assert results, f"{contract.name} produced no results from its own fixture"
    first = results[0]
    assert first.rank == 1
    assert first.url.startswith("http")
    assert first.title
    assert "Fusajiro Yamauchi" in first.excerpt
    assert all(isinstance(r.date, (str, type(None))) for r in results)


@CASES
@respx.mock
async def test_search_never_returns_more_than_num_results(contract, load_fixture):
    _mock(
        contract.search_method,
        contract.search_url,
        return_value=httpx.Response(200, json=load_fixture(contract.search_fixture)),
    )

    results = await contract.build("k").search("who founded nintendo", 1)

    assert len(results) == 1


@CASES
@respx.mock
async def test_empty_search_is_an_empty_list_not_an_error(contract):
    """Spec section 7: empty results are ordinary tool output, not a failure."""
    _mock(
        contract.search_method,
        contract.search_url,
        return_value=httpx.Response(200, json=contract.no_results),
    )

    assert await contract.build("k").search("nothing matches this", 5) == []


@CASES
@respx.mock
async def test_fetch_normalises_to_the_shared_page_fields(contract, load_fixture):
    _mock(
        contract.fetch_method,
        contract.fetch_url,
        return_value=httpx.Response(200, json=load_fixture(contract.fetch_fixture)),
    )

    page = await contract.build("k").fetch(PAGE_URL)

    assert page.url.startswith("http")
    assert "Fusajiro Yamauchi" in page.text
    assert isinstance(page.title, str)
    assert isinstance(page.date, (str, type(None)))


@CASES
@respx.mock
async def test_fetch_without_a_result_object_is_a_permanent_error(contract):
    if not contract.fetch_raises_when_empty:
        pytest.skip(f"{contract.name} always returns a page object")
    _mock(
        contract.fetch_method,
        contract.fetch_url,
        return_value=httpx.Response(200, json=contract.no_page),
    )

    with pytest.raises(ProviderError) as exc:
        await contract.build("k").fetch(PAGE_URL)

    assert exc.value.retryable is False


@CASES
@respx.mock
async def test_transient_failure_is_retried_then_succeeds(
    contract, load_fixture, monkeypatch
):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = _mock(
        contract.search_method,
        contract.search_url,
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=load_fixture(contract.search_fixture)),
        ],
    )

    results = await contract.build("k").search("who founded nintendo", 5)

    assert route.call_count == 2
    assert results


@CASES
@respx.mock
async def test_rate_limit_is_retryable_and_auth_failure_is_not(
    contract, monkeypatch
):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = _mock(
        contract.search_method, contract.search_url,
        return_value=httpx.Response(429),
    )
    with pytest.raises(ProviderError) as rate_limited:
        await contract.build("k").search("q", 5)
    assert rate_limited.value.retryable is True
    assert route.call_count == transport.RETRY_ATTEMPTS

    respx.reset()
    route = _mock(
        contract.search_method, contract.search_url,
        return_value=httpx.Response(401),
    )
    with pytest.raises(ProviderError) as unauthorised:
        await contract.build("bad").search("q", 5)
    assert unauthorised.value.retryable is False
    assert route.call_count == 1, "an auth failure must never be retried"


@CASES
def test_every_contract_is_registered_with_a_credential_variable(contract):
    from searchforge.servers.tool import PROVIDERS

    variable, construct = PROVIDERS[contract.name]
    assert variable.endswith("_API_KEY")
    assert construct is contract.build


def test_the_contract_table_covers_every_registered_provider():
    """A new adapter in the registry must arrive with contract coverage."""
    from searchforge.servers.tool import PROVIDERS

    assert {c.name for c in CONTRACTS} == set(PROVIDERS)
