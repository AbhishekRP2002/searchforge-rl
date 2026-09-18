import httpx
import pytest
import respx
from tenacity import wait_none

from searchforge.providers.base import ProviderError
from searchforge.providers import serper, transport
from searchforge.providers.serper import SERPER_SEARCH_URL, SerperProvider


@respx.mock
async def test_search_maps_serper_fields_to_the_normalised_contract(load_fixture):
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_search.json"))
    )
    results = await SerperProvider("k").search("who founded nintendo", 5)

    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].title == "Nintendo - Wikipedia"
    assert results[0].url == "https://en.wikipedia.org/wiki/Nintendo"
    assert "Fusajiro Yamauchi" in results[0].excerpt
    assert results[0].date == "Aug 14, 2026"
    # missing date must be None, never a fabricated value
    assert results[1].date is None


@respx.mock
async def test_search_respects_num_results(load_fixture):
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_search.json"))
    )
    results = await SerperProvider("k").search("q", 2)
    assert len(results) == 2


@respx.mock
async def test_search_sends_the_api_key_as_a_header_not_a_query_param(load_fixture):
    route = respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_search.json"))
    )
    await SerperProvider("secret-key").search("q", 5)

    request = route.calls.last.request
    assert request.headers["X-API-KEY"] == "secret-key"
    assert "secret-key" not in str(request.url)


@respx.mock
async def test_answer_box_is_dropped_by_default(load_fixture):
    """Google's direct answer is a capability Exa has no equivalent for. Passing
    it through measures a different thing, not better retrieval. Spec section 5."""
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=load_fixture("serper_search_answerbox.json")
        )
    )
    results = await SerperProvider("k").search("who founded nintendo", 5)

    assert len(results) == 1
    joined = " ".join(r.excerpt + r.title for r in results)
    assert "Nintendo / Founder" not in joined


@respx.mock
async def test_answer_box_is_included_when_explicitly_enabled(load_fixture):
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=load_fixture("serper_search_answerbox.json")
        )
    )
    results = await SerperProvider("k", include_answer_box=True).search("q", 5)

    assert results[0].rank == 0
    assert "Fusajiro Yamauchi" in results[0].excerpt


@respx.mock
async def test_empty_organic_returns_an_empty_list_not_an_error():
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"organic": []})
    )
    assert await SerperProvider("k").search("zzzz", 5) == []


@respx.mock
async def test_auth_failure_is_not_retryable():
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(403, json={"message": "Unauthorized."})
    )
    with pytest.raises(ProviderError) as exc:
        await SerperProvider("bad").search("q", 5)

    assert exc.value.status == 403
    assert exc.value.retryable is False


@respx.mock
async def test_server_error_is_retried_then_succeeds(monkeypatch, load_fixture):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = respx.post(SERPER_SEARCH_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=load_fixture("serper_search.json")),
        ]
    )

    results = await SerperProvider("k").search("q", 5)

    assert route.call_count == 2
    assert results[0].title == "Nintendo - Wikipedia"


@respx.mock
async def test_transport_error_is_retried_then_succeeds(monkeypatch, load_fixture):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = respx.post(SERPER_SEARCH_URL).mock(
        side_effect=[
            httpx.ConnectError("connection reset"),
            httpx.Response(200, json=load_fixture("serper_search.json")),
        ]
    )

    results = await SerperProvider("k").search("q", 5)

    assert route.call_count == 2
    assert results


@respx.mock
async def test_persistent_server_error_stops_after_three_attempts(monkeypatch):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = respx.post(SERPER_SEARCH_URL).mock(return_value=httpx.Response(503))
    with pytest.raises(ProviderError) as exc:
        await SerperProvider("k").search("q", 5)

    assert route.call_count == 3
    assert exc.value.retryable is True


@respx.mock
async def test_rate_limit_is_retryable_and_distinguishable(monkeypatch):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = respx.post(SERPER_SEARCH_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(ProviderError) as exc:
        await SerperProvider("k").search("q", 5)

    assert route.call_count == 3
    assert exc.value.status == 429
    assert exc.value.retryable is True


@respx.mock
async def test_auth_failure_is_not_retried():
    route = respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(403, json={"message": "Unauthorized."})
    )

    with pytest.raises(ProviderError):
        await SerperProvider("bad").search("q", 5)

    assert route.call_count == 1
