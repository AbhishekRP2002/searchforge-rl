import httpx
import pytest
import respx
from tenacity import wait_none

from searchforge.providers.base import ProviderError
from searchforge.providers import serper, transport
from searchforge.providers.serper import SERPER_SCRAPE_URL, SerperProvider


@respx.mock
async def test_fetch_maps_scrape_fields(load_fixture):
    respx.post(SERPER_SCRAPE_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_scrape.json"))
    )
    page = await SerperProvider("k").fetch("https://en.wikipedia.org/wiki/Nintendo")

    assert page.url == "https://en.wikipedia.org/wiki/Nintendo"
    assert page.title == "Nintendo - Wikipedia"
    assert "Fusajiro Yamauchi" in page.text
    assert page.date == "2026-08-14T00:00:00Z"


@respx.mock
async def test_fetch_sends_the_url_in_the_body(load_fixture):
    import json

    route = respx.post(SERPER_SCRAPE_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_scrape.json"))
    )
    await SerperProvider("k").fetch("https://example.com/a")

    assert json.loads(route.calls.last.request.content)["url"] == "https://example.com/a"


@respx.mock
async def test_fetch_falls_back_to_markdown_when_text_is_absent(load_fixture):
    payload = load_fixture("serper_scrape.json")
    del payload["text"]
    respx.post(SERPER_SCRAPE_URL).mock(return_value=httpx.Response(200, json=payload))

    page = await SerperProvider("k").fetch("https://example.com")
    assert "Nintendo Co., Ltd." in page.text


@respx.mock
async def test_fetch_of_an_empty_page_returns_empty_text_not_an_error():
    respx.post(SERPER_SCRAPE_URL).mock(
        return_value=httpx.Response(200, json={"text": "", "metadata": {}})
    )
    page = await SerperProvider("k").fetch("https://example.com")

    assert page.text == ""
    assert page.title == ""


@respx.mock
async def test_fetch_auth_failure_is_not_retryable():
    route = respx.post(SERPER_SCRAPE_URL).mock(
        return_value=httpx.Response(403, json={"message": "Unauthorized."})
    )
    with pytest.raises(ProviderError) as exc:
        await SerperProvider("bad").fetch("https://example.com")

    assert route.call_count == 1
    assert exc.value.retryable is False


@respx.mock
async def test_fetch_retries_rate_limit_then_succeeds(monkeypatch, load_fixture):
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    route = respx.post(SERPER_SCRAPE_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=load_fixture("serper_scrape.json")),
        ]
    )

    page = await SerperProvider("k").fetch("https://example.com")

    assert route.call_count == 2
    assert "Fusajiro Yamauchi" in page.text
