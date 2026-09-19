import httpx
import pytest
import respx
from tenacity import wait_none

from searchforge.providers import serper, transport
from searchforge.providers.base import ProviderError
from searchforge.providers.serper import SERPER_SCRAPE_URL, SERPER_SEARCH_URL
from searchforge.servers.tool import WebToolset, WebToolsetConfig


def _toolset(**kwargs) -> WebToolset:
    return WebToolset(WebToolsetConfig(**kwargs))


@respx.mock
async def test_search_returns_rendered_markdown(monkeypatch, load_fixture):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_search.json"))
    )
    out = await _toolset().search("who founded nintendo")

    assert isinstance(out, str)
    assert "[1] Nintendo - Wikipedia" in out
    assert "https://en.wikipedia.org/wiki/Nintendo" in out


@respx.mock
async def test_fetch_returns_rendered_page(monkeypatch, load_fixture):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    respx.post(SERPER_SCRAPE_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_scrape.json"))
    )
    out = await _toolset().fetch("https://en.wikipedia.org/wiki/Nintendo")

    assert "Nintendo - Wikipedia" in out
    assert "Fusajiro Yamauchi" in out


async def test_missing_api_key_raises_naming_the_variable_and_the_cause(monkeypatch):
    """Missing server credentials should identify the variable and local file."""
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SERPER_API_KEY") as exc:
        await _toolset(env_file=None).search("q")

    assert ".env" in str(exc.value)


@respx.mock
async def test_provider_key_can_be_loaded_by_the_server_from_an_env_file(
    monkeypatch, tmp_path, load_fixture
):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SERPER_API_KEY=server-only-key\n")
    route = respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_search.json"))
    )

    await _toolset(env_file=env_file).search("q")

    assert route.calls.last.request.headers["X-API-KEY"] == "server-only-key"


@respx.mock
async def test_provider_error_returns_a_recoverable_message_not_an_exception(
    monkeypatch,
):
    """An agent must be able to recover from a bad call. Spec section 7:
    invalid arguments and empty results return standard tool output."""
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    respx.post(SERPER_SEARCH_URL).mock(return_value=httpx.Response(503))

    out = await _toolset().search("q")
    assert isinstance(out, str)
    assert out.startswith("SEARCH_PROVIDER_UNAVAILABLE:")


@respx.mock
async def test_rate_limit_has_a_trace_visible_failure_class(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    respx.post(SERPER_SEARCH_URL).mock(return_value=httpx.Response(429))

    out = await _toolset().search("q")

    assert out.startswith("SEARCH_PROVIDER_RATE_LIMITED:")


@respx.mock
async def test_permanent_provider_error_is_not_hidden_from_the_evaluator(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "bad")
    route = respx.post(SERPER_SEARCH_URL).mock(return_value=httpx.Response(403))

    with pytest.raises(ProviderError):
        await _toolset().search("q")

    assert route.call_count == 1


@respx.mock
async def test_tool_call_budget_is_enforced_per_toolset(monkeypatch, load_fixture):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("serper_search.json"))
    )
    toolset = _toolset(max_tool_calls=2)

    assert "[1]" in await toolset.search("a")
    assert "[1]" in await toolset.search("b")
    third = await toolset.search("c")
    assert "budget" in third.lower()


async def test_fetch_rejects_a_non_http_url(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    out = await _toolset().fetch("file:///etc/passwd")

    assert "http" in out.lower()


def test_unknown_provider_name_raises():
    from searchforge.servers.tool import build_provider

    with pytest.raises(ValueError, match="unknown provider"):
        build_provider(WebToolsetConfig(provider="altavista"))


def test_exa_provider_loads_its_own_key(monkeypatch):
    from searchforge.providers.exa import ExaProvider
    from searchforge.servers.tool import build_provider

    monkeypatch.setenv("EXA_API_KEY", "e")

    assert isinstance(build_provider(WebToolsetConfig(provider="exa")), ExaProvider)


@respx.mock
async def test_fetch_logs_the_result_size(monkeypatch, load_fixture, caplog):
    """`chars` is why this log exists: pages are uncapped by decision, so the size
    of a fetch is the signal that it blew the sequence budget. Assert it, or the
    field quietly disappears in a refactor and the next 161KB page is invisible."""
    monkeypatch.setenv("SERPER_API_KEY", "k")
    payload = load_fixture("serper_scrape.json")
    payload["text"] = "word " * 5000
    respx.post(SERPER_SCRAPE_URL).mock(return_value=httpx.Response(200, json=payload))

    with caplog.at_level("INFO", logger="searchforge.tools"):
        out = await _toolset().fetch("https://en.wikipedia.org/wiki/Nintendo")

    record = next(r for r in caplog.records if r.name == "searchforge.tools")
    assert f"chars={len(out)}" in record.getMessage()
    assert "provider=serper" in record.getMessage()


@respx.mock
async def test_provider_failure_logs_a_warning(monkeypatch, caplog):
    """A retried-out provider is the thing an operator most needs to see live."""
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setattr(transport, "RETRY_WAIT", wait_none())
    respx.post(SERPER_SEARCH_URL).mock(return_value=httpx.Response(503))

    with caplog.at_level("WARNING", logger="searchforge.tools"):
        await _toolset().search("anything")

    assert any(
        "status=503" in r.getMessage() and r.levelname == "WARNING"
        for r in caplog.records
    )


async def test_parallel_calls_are_refused_by_default(monkeypatch):
    """The budget already caps total spend, so parallelism cannot widen the
    search space -- it changes the incentive toward shotgunning broad queries
    instead of composing a precise one."""
    import asyncio

    from searchforge.servers.tool import PARALLEL_REFUSED, WebToolset, WebToolsetConfig

    released = asyncio.Event()

    class SlowProvider:
        name = "serper"

        async def search(self, query, num_results):
            await released.wait()
            return []

        async def fetch(self, url):
            return None

    monkeypatch.setattr(
        "searchforge.servers.tool.build_provider", lambda config: SlowProvider()
    )
    toolset = WebToolset(WebToolsetConfig(provider="serper", env_file=None))

    first = asyncio.create_task(toolset.search("slow query"))
    await asyncio.sleep(0)  # let the first call take the in-flight slot
    second = await toolset.search("concurrent query")

    assert second == PARALLEL_REFUSED
    released.set()
    await first
    # The refusal is not charged: the agent is told how to call, not punished.
    assert toolset._calls == 1


async def test_parallel_calls_are_served_when_enabled(monkeypatch):
    import asyncio

    from searchforge.servers.tool import PARALLEL_REFUSED, WebToolset, WebToolsetConfig

    class Provider:
        name = "serper"

        async def search(self, query, num_results):
            await asyncio.sleep(0)
            return []

    monkeypatch.setattr(
        "searchforge.servers.tool.build_provider", lambda config: Provider()
    )
    toolset = WebToolset(
        WebToolsetConfig(
            provider="serper", env_file=None, allow_parallel_tool_calls=True
        )
    )

    both = await asyncio.gather(toolset.search("a"), toolset.search("b"))

    assert PARALLEL_REFUSED not in both
    assert toolset._calls == 2
