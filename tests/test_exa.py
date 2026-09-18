import json

import httpx
import pytest
import respx
from tenacity import wait_none

from searchforge.providers import exa, transport
from searchforge.providers.base import ProviderError
from searchforge.providers.exa import EXA_CONTENTS_URL, EXA_SEARCH_URL, ExaProvider


@respx.mock
async def test_search_uses_auto_with_uncapped_highlights(load_fixture):
    route = respx.post(EXA_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("exa_search.json"))
    )

    results = await ExaProvider("k").search("who founded nintendo", 5)

    request = route.calls.last.request
    assert request.headers["x-api-key"] == "k"
    assert json.loads(request.content) == {
        "query": "who founded nintendo",
        "type": "auto",
        "numResults": 5,
        "contents": {"highlights": True},
    }
    assert results[0].rank == 1
    assert results[0].title == "Nintendo - History"
    assert "Fusajiro Yamauchi" in results[0].excerpt
    assert results[0].date == "2025-01-02T00:00:00.000Z"
    assert results[1].excerpt == "A short company profile."


@respx.mock
async def test_fetch_requests_full_text_without_a_character_cap(load_fixture):
    route = respx.post(EXA_CONTENTS_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("exa_contents.json"))
    )

    page = await ExaProvider("k").fetch("https://example.com/nintendo-history")

    assert json.loads(route.calls.last.request.content) == {
        "urls": ["https://example.com/nintendo-history"],
        "text": True,
    }
    assert page.title == "Nintendo - History"
    assert "Fusajiro Yamauchi" in page.text


@respx.mock
async def test_fetch_empty_results_is_a_non_retryable_provider_error():
    respx.post(EXA_CONTENTS_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    with pytest.raises(ProviderError) as exc:
        await ExaProvider("k").fetch("https://example.com/missing")

    assert exc.value.retryable is False


@respx.mock
async def test_transient_exa_failure_retries_then_succeeds(
    monkeypatch, load_fixture
):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = respx.post(EXA_SEARCH_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=load_fixture("exa_search.json")),
        ]
    )

    results = await ExaProvider("k").search("q", 5)

    assert route.call_count == 2
    assert results


@respx.mock
async def test_exa_auth_failure_is_not_retried():
    route = respx.post(EXA_SEARCH_URL).mock(return_value=httpx.Response(401))

    with pytest.raises(ProviderError) as exc:
        await ExaProvider("bad").search("q", 5)

    assert route.call_count == 1
    assert exc.value.retryable is False
