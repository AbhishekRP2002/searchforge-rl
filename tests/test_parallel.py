"""Parallel request-shape tests.

The shared contract lives in test_provider_contract.py. What is asserted here is
the part unique to this provider: it is the only adapter whose API offers to do
some of the agent's reasoning, and the request body is where that offer is
declined.
"""

import json

import httpx
import respx

from searchforge.providers.parallel import (
    PARALLEL_EXTRACT_URL,
    PARALLEL_SEARCH_URL,
    ParallelProvider,
)


@respx.mock
async def test_search_sends_queries_without_an_objective(load_fixture):
    route = respx.post(PARALLEL_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("parallel_search.json"))
    )

    await ParallelProvider("k").search("who founded nintendo", 5)

    request = route.calls.last.request
    assert request.headers["x-api-key"] == "k"
    assert json.loads(request.content) == {
        "search_queries": ["who founded nintendo"],
        "mode": "fast",
        "advanced_settings": {"max_results": 5},
    }


@respx.mock
async def test_objective_is_never_sent_on_either_endpoint(load_fixture):
    """`objective` focuses results against the caller's goal, which is reasoning
    the policy is supposed to do. Sending it would make this arm incomparable."""
    search = respx.post(PARALLEL_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("parallel_search.json"))
    )
    extract = respx.post(PARALLEL_EXTRACT_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("parallel_extract.json"))
    )

    provider = ParallelProvider("k")
    await provider.search("who founded nintendo", 5)
    await provider.fetch("https://example.com/nintendo-history")

    for route in (search, extract):
        body = json.loads(route.calls.last.request.content)
        assert "objective" not in body
        assert "search_queries" not in body or body["search_queries"] == [
            "who founded nintendo"
        ]


@respx.mock
async def test_max_results_is_explicit_and_correctly_nested(load_fixture):
    """Omitting it returns 10, against this environment's default of 5. It belongs
    under advanced_settings: the request model forbids extra top-level keys, so a
    misplaced one is a 422, not a silently ignored field."""
    route = respx.post(PARALLEL_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("parallel_search.json"))
    )

    await ParallelProvider("k").search("q", 3)

    body = json.loads(route.calls.last.request.content)
    assert body["advanced_settings"]["max_results"] == 3
    assert "max_results" not in body, "a top-level max_results is rejected with 422"


@respx.mock
async def test_several_excerpts_join_into_one_excerpt(load_fixture):
    respx.post(PARALLEL_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("parallel_search.json"))
    )

    results = await ParallelProvider("k").search("who founded nintendo", 5)

    assert "Fusajiro Yamauchi" in results[0].excerpt
    assert "handmade playing cards" in results[0].excerpt
    assert results[0].excerpt.count("\n\n") == 1, "excerpts join like Exa highlights"
    assert results[0].date == "2025-01-02"


@respx.mock
async def test_fetch_prefers_full_content_then_falls_back_to_excerpts():
    respx.post(PARALLEL_EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "excerpts": ["only an excerpt"],
                    }
                ]
            },
        )
    )

    page = await ParallelProvider("k").fetch("https://example.com/a")

    assert page.text == "only an excerpt"
