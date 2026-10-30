"""The optional date window on `web_search`: validation, per-provider mapping,
emulated filtering, and the fairness refusals that keep arms comparable."""

from __future__ import annotations

import inspect
import json
from datetime import date

import httpx
import pytest
import respx

from searchforge.dates import (
    check_window,
    google_tbs,
    parse_bound,
    parse_result_date,
    within,
)
from searchforge.providers.base import SearchResult
from searchforge.providers.exa import EXA_SEARCH_URL, ExaProvider
from searchforge.providers.firecrawl import FIRECRAWL_SEARCH_URL, FirecrawlProvider
from searchforge.providers.keenable import KEENABLE_SEARCH_URL, KeenableProvider
from searchforge.providers.parallel import ParallelProvider
from searchforge.providers.serper import SERPER_SEARCH_URL, SerperProvider
from searchforge.providers.tavily import TAVILY_SEARCH_URL, TavilyProvider
from searchforge.providers.tinyfish import TINYFISH_SEARCH_URL, TinyFishProvider
from searchforge.servers.tool import (
    DATE_FILTER_OFF,
    PROVIDERS,
    WebToolset,
    WebToolsetConfig,
    check_date_filter,
)
from searchforge.taskset import SearchForgeConfig, SearchForgeTaskset

AFTER = date(2024, 3, 5)
BEFORE = date(2024, 11, 20)


# --- pure date maths -------------------------------------------------------


def test_bounds_accept_only_iso_days():
    assert parse_bound("2024-03-05", "after") == (AFTER, None)
    assert parse_bound(None, "after") == (None, None)
    for bad in ("20240305", "03/05/2024", "2024-3-5", "last week"):
        value, problem = parse_bound(bad, "after")
        assert value is None
        assert "YYYY-MM-DD" in problem and "`after`" in problem


def test_an_impossible_day_is_named_as_such():
    value, problem = parse_bound("2024-02-30", "before")
    assert value is None
    assert "not a real calendar day" in problem


def test_inverted_window_is_rejected():
    assert check_window(AFTER, BEFORE) is None
    assert "later than" in check_window(BEFORE, AFTER)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-06-01", date(2024, 6, 1)),
        ("2024-06-01T23:30:00Z", date(2024, 6, 1)),
        ("2024-06-01T23:30:00-05:00", date(2024, 6, 2)),  # normalised to UTC
        ("Jun 1, 2024", date(2024, 6, 1)),
        ("1 June 2024", date(2024, 6, 1)),
        (1717200000, date(2024, 6, 1)),
    ],
)
def test_result_dates_parse_from_every_shape_providers_send(raw, expected):
    assert parse_result_date(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "3 days ago", "sometime in 2024", True])
def test_unreadable_result_dates_are_none_not_a_guess(raw):
    assert parse_result_date(raw) is None


def test_window_is_inclusive_and_undated_is_outside():
    assert within(AFTER, AFTER, BEFORE)
    assert within(BEFORE, AFTER, BEFORE)
    assert not within(date(2024, 3, 4), AFTER, BEFORE)
    assert within(date(2030, 1, 1), AFTER, None)
    assert not within(None, AFTER, BEFORE)


def test_google_tbs_is_unpadded_and_omits_an_open_end():
    assert google_tbs(AFTER, BEFORE) == "cdr:1,cd_min:3/5/2024,cd_max:11/20/2024"
    assert google_tbs(AFTER, None) == "cdr:1,cd_min:3/5/2024"


# --- native mapping, one request shape per provider ------------------------


def _body(route: respx.Route) -> dict:
    return json.loads(route.calls.last.request.content)


@respx.mock
async def test_serper_sends_a_google_date_range():
    route = respx.post(SERPER_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"organic": []})
    )
    await SerperProvider("k").search("q", 5, after=AFTER, before=BEFORE)
    assert _body(route)["tbs"] == "cdr:1,cd_min:3/5/2024,cd_max:11/20/2024"


@respx.mock
async def test_firecrawl_forwards_the_same_google_range():
    route = respx.post(FIRECRAWL_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"web": []}})
    )
    await FirecrawlProvider("k").search("q", 5, after=AFTER, before=BEFORE)
    assert _body(route)["tbs"] == "cdr:1,cd_min:3/5/2024,cd_max:11/20/2024"


@respx.mock
async def test_exa_sends_inclusive_published_date_timestamps():
    route = respx.post(EXA_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await ExaProvider("k").search("q", 5, after=AFTER, before=BEFORE)
    body = _body(route)
    assert body["startPublishedDate"] == "2024-03-05T00:00:00.000Z"
    assert body["endPublishedDate"] == "2024-11-20T23:59:59.999Z"


@respx.mock
async def test_tavily_sends_start_and_end_dates():
    route = respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await TavilyProvider("k").search("q", 5, after=AFTER, before=BEFORE)
    body = _body(route)
    assert (body["start_date"], body["end_date"]) == ("2024-03-05", "2024-11-20")


@respx.mock
async def test_tinyfish_sends_date_query_parameters():
    route = respx.get(url__startswith=TINYFISH_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await TinyFishProvider("k").search("q", 5, after=AFTER, before=BEFORE)
    params = route.calls.last.request.url.params
    assert (params["after_date"], params["before_date"]) == ("2024-03-05", "2024-11-20")


@respx.mock
async def test_keenable_sends_published_bounds():
    route = respx.post(KEENABLE_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await KeenableProvider("k").search("q", 5, after=AFTER, before=BEFORE)
    body = _body(route)
    assert (body["published_after"], body["published_before"]) == (
        "2024-03-05",
        "2024-11-20",
    )


@pytest.mark.parametrize(
    ("provider", "url", "keys"),
    [
        (SerperProvider, SERPER_SEARCH_URL, {"tbs"}),
        (ExaProvider, EXA_SEARCH_URL, {"startPublishedDate", "endPublishedDate"}),
        (TavilyProvider, TAVILY_SEARCH_URL, {"start_date", "end_date"}),
        (KeenableProvider, KEENABLE_SEARCH_URL, {"published_after", "published_before"}),
    ],
)
@respx.mock
async def test_no_window_sends_no_date_fields(provider, url, keys):
    route = respx.post(url).mock(
        return_value=httpx.Response(200, json={"organic": [], "results": []})
    )
    await provider("k").search("q", 5)
    assert not keys & _body(route).keys()


async def test_parallel_refuses_a_native_window_rather_than_dropping_it():
    with pytest.raises(ValueError, match="no native date filter"):
        await ParallelProvider("k").search("q", 5, after=AFTER)


def test_capability_table():
    supported = {n for n, (_, c) in PROVIDERS.items() if c.supports_date_filter}
    assert supported == {"serper", "exa", "firecrawl", "tavily", "tinyfish", "keenable"}


# --- fairness refusals at load ---------------------------------------------


def test_native_mode_refuses_an_unsupported_provider():
    with pytest.raises(ValueError, match="'parallel' has no native filter"):
        check_date_filter("parallel", "native")
    check_date_filter("parallel", "emulated")
    check_date_filter("parallel", "off")


def _tasks(date_filter="off", **kwargs):
    config = SearchForgeConfig(**kwargs)
    config.task.tools.date_filter = date_filter
    return list(SearchForgeTaskset(config).load())


def test_taskset_load_fails_before_any_rollout_on_a_native_mismatch():
    with pytest.raises(ValueError, match="native"):
        _tasks("native", providers=["serper", "parallel"])
    assert _tasks("emulated", providers=["serper", "parallel"])


# --- the tool server -------------------------------------------------------


class _Recorder:
    """A provider double that records what it was asked for."""

    name = "serper"
    supports_date_filter = True

    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    async def search(self, query, num_results, **dates):
        self.calls.append(dates)
        return self.results


def _toolset(monkeypatch, provider, date_filter):
    monkeypatch.setattr("searchforge.servers.tool.build_provider", lambda c: provider)
    return WebToolset(WebToolsetConfig(env_file=None, date_filter=date_filter))


def _hit(rank, day):
    return SearchResult(f"t{rank}", f"https://e.com/{rank}", "x", day, rank)


async def test_native_mode_passes_the_window_to_the_provider(monkeypatch):
    provider = _Recorder([_hit(1, None)])
    out = await _toolset(monkeypatch, provider, "native").search_dated(
        "q", after="2024-03-05", before="2024-11-20"
    )
    assert provider.calls == [{"after": AFTER, "before": BEFORE}]
    assert out.startswith("Date filter (native): published 2024-03-05 to 2024-11-20")
    assert "https://e.com/1" in out  # native results are not re-filtered here


async def test_emulated_mode_filters_every_arm_on_result_dates(monkeypatch):
    provider = _Recorder(
        [
            _hit(1, "2024-06-01"),
            _hit(2, "2023-01-01"),  # outside
            _hit(3, None),  # undated: dropped
            _hit(4, "Nov 20, 2024"),  # inclusive end
        ]
    )
    out = await _toolset(monkeypatch, provider, "emulated").search_dated(
        "q", after="2024-03-05", before="2024-11-20"
    )
    assert provider.calls == [{}]  # the provider never sees the dates
    assert out.startswith("Date filter (emulated)")
    assert "e.com/1" in out and "e.com/4" in out
    assert "e.com/2" not in out and "e.com/3" not in out


async def test_a_dated_tool_without_dates_behaves_like_plain_search(monkeypatch):
    provider = _Recorder([_hit(1, None)])
    out = await _toolset(monkeypatch, provider, "emulated").search_dated("q")
    assert provider.calls == [{}]
    assert "Date filter" not in out
    assert "e.com/1" in out


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"after": "March 2024"}, "YYYY-MM-DD"),
        ({"before": "2024-13-01"}, "not a real calendar day"),
        ({"after": "2024-12-01", "before": "2024-01-01"}, "later than"),
    ],
)
async def test_bad_dates_return_a_message_and_cost_nothing(monkeypatch, kwargs, message):
    provider = _Recorder([])
    toolset = _toolset(monkeypatch, provider, "native")
    out = await toolset.search_dated("q", **kwargs)
    assert message in out
    assert provider.calls == []
    assert toolset._calls == 0


async def test_dated_search_is_refused_when_the_filter_is_off(monkeypatch):
    provider = _Recorder([])
    out = await _toolset(monkeypatch, provider, "off").search_dated("q", after="2024-01-01")
    assert out == DATE_FILTER_OFF
    assert provider.calls == []


class _FakeMCP:
    def __init__(self):
        self.tools: dict[str, inspect.Signature] = {}

    def add_tool(self, fn, name, description):
        assert name not in self.tools, f"{name} registered twice"
        self.tools[name] = inspect.signature(fn)


@pytest.mark.parametrize(
    ("mode", "params"),
    [
        ("off", ["query"]),
        ("native", ["query", "after", "before"]),
        ("emulated", ["query", "after", "before"]),
    ],
)
def test_only_one_search_schema_is_advertised(mode, params):
    mcp = _FakeMCP()
    WebToolset(WebToolsetConfig(env_file=None, date_filter=mode)).register(mcp)
    assert set(mcp.tools) == {"search", "fetch"}
    assert list(mcp.tools["search"].parameters) == params


@pytest.mark.parametrize(
    ("mode", "params"),
    [("off", ["query"]), ("native", ["query", "after", "before"])],
)
async def test_real_mcp_server_advertises_the_schema_the_model_sees(mode, params):
    """The fake above checks signatures; this checks the JSON schema a real MCP
    server builds from them. `register` repeats a verifiers-internal loop, so this
    is the test that fails if a verifiers or mcp upgrade changes how tools land."""
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("searchforge-test")
    WebToolset(WebToolsetConfig(env_file=None, date_filter=mode)).register(mcp)
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    assert set(tools) == {"search", "fetch"}
    assert list(tools["search"].input_schema["properties"]) == params


# --- the prompt -------------------------------------------------------------


def test_prompt_states_todays_date_and_can_be_pinned():
    pinned = _tasks(providers=["serper"], current_date=date(2025, 1, 2))[0]
    assert "Today's date is 2025-01-02." in pinned.data.system_prompt
    default = _tasks(providers=["serper"])[0]
    assert f"Today's date is {date.today().isoformat()}." in default.data.system_prompt


def test_prompt_mentions_the_filter_only_when_it_exists():
    assert "`after`" not in _tasks("off", providers=["serper"])[0].data.system_prompt
    assert "`after`" in _tasks("emulated", providers=["serper"])[0].data.system_prompt
