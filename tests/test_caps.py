"""Page and excerpt caps, fetch pagination, the fetch switch, and the metrics read
back from cap markers."""

from types import SimpleNamespace

import pytest

from searchforge.caps import (
    EXCERPT_ELLIPSIS,
    cap_excerpts,
    cap_page,
    parse_fetch_marker,
    slice_page,
)
from searchforge.prompts import tool_lines
from searchforge.providers.base import PageContent, SearchResult
from searchforge.render import render_page
from searchforge.servers.tool import (
    BUDGET_EXHAUSTED,
    FETCH_DISABLED,
    PAGINATION_OFF,
    WebToolset,
    WebToolsetConfig,
)
from searchforge.taskset import (
    SearchForgeConfig,
    SearchForgeTask,
    SearchForgeTaskConfig,
    SearchForgeTaskset,
)
from searchforge.trace_reader import fetch_page_stats

PAGE = "".join(chr(ord("a") + i % 26) for i in range(20_500))


# --- pure caps ----------------------------------------------------------------


def test_uncapped_and_fitting_pages_come_back_byte_identical():
    assert cap_page(PAGE, None) == PAGE
    assert cap_page("short", 8000) == "short"


def test_cap_keeps_the_head_and_names_both_lengths():
    out = cap_page(PAGE, 8000)
    assert out.startswith(PAGE[:8000])
    assert out.split("\n\n")[0] == PAGE[:8000]
    assert out.endswith("[truncated: showing 8,000 of 20,500 chars]")


def test_pages_cover_the_text_exactly_once_and_say_how_to_continue():
    first, second, third = (slice_page(PAGE, 8000, n) for n in (1, 2, 3))
    assert first.startswith(PAGE[:8000]) and "page=2" in first
    assert second.startswith(PAGE[8000:16000]) and "page 2 of 3" in second
    assert third.startswith(PAGE[16000:]) and "last page" in third
    assert slice_page(PAGE, 8000, 4) is None
    assert slice_page(PAGE, 8000, 0) is None


def test_a_page_that_fits_is_served_whole_without_a_marker():
    assert slice_page("short", 8000, 1) == "short"


def test_excerpts_are_capped_and_the_rest_left_alone():
    long = SearchResult("t", "https://a.com", "x" * 2500, None, 1)
    short = SearchResult("t", "https://b.com", "fine", None, 2)
    capped = cap_excerpts([long, short], 2000)
    assert capped[0].excerpt == "x" * 2000 + EXCERPT_ELLIPSIS
    assert capped[1] is short
    assert cap_excerpts([long], None)[0] is long


def test_markers_parse_back_to_raw_length_and_truncation():
    assert parse_fetch_marker(cap_page(PAGE, 8000)) == (20_500, True)
    assert parse_fetch_marker(slice_page(PAGE, 8000, 3)) == (20_500, True)
    assert parse_fetch_marker("no marker here") is None


# --- the tool server ----------------------------------------------------------


class _Pages:
    """A provider double serving one long page and one long excerpt."""

    name = "serper"
    supports_date_filter = True

    def __init__(self):
        self.fetches = 0

    async def fetch(self, url):
        self.fetches += 1
        return PageContent(url=url, title="T", text=PAGE, date=None)

    async def search(self, query, num_results, **dates):
        return [SearchResult("t", "https://a.com", "y" * 3000, None, 1)]


def _toolset(monkeypatch, **config):
    provider = _Pages()
    monkeypatch.setattr("searchforge.servers.tool.build_provider", lambda c: provider)
    return WebToolset(WebToolsetConfig(env_file=None, **config)), provider


async def test_fetch_is_capped_by_default_after_rendering_the_header(monkeypatch):
    toolset, _ = _toolset(monkeypatch)
    out = await toolset.fetch("https://a.com/x")
    assert out == render_page(
        PageContent("https://a.com/x", "T", cap_page(PAGE, 8000), None)
    )
    assert out.startswith("T\nhttps://a.com/x\n\n" + PAGE[:8000])
    assert "[truncated: showing 8,000 of 20,500 chars]" in out


async def test_uncapped_fetch_matches_the_pre_cap_rendering(monkeypatch):
    toolset, _ = _toolset(monkeypatch, max_page_chars=None)
    out = await toolset.fetch("https://a.com/x")
    assert out == render_page(PageContent("https://a.com/x", "T", PAGE, None))


async def test_search_excerpts_are_capped_in_the_tool_server(monkeypatch):
    toolset, _ = _toolset(monkeypatch)
    out = await toolset.search("q")
    assert "y" * 2000 + EXCERPT_ELLIPSIS in out
    assert "y" * 2001 not in out


async def test_each_page_is_one_charged_call(monkeypatch):
    toolset, provider = _toolset(
        monkeypatch, fetch_pagination=True, max_tool_calls=2
    )
    assert "page 2 of 3" in await toolset.fetch_paged("https://a.com/x", page=2)
    assert "page 3 of 3" in await toolset.fetch_paged("https://a.com/x", page=3)
    assert await toolset.fetch_paged("https://a.com/x", page=1) == BUDGET_EXHAUSTED
    assert provider.fetches == 2


async def test_a_missing_page_says_how_many_exist(monkeypatch):
    toolset, _ = _toolset(monkeypatch, fetch_pagination=True)
    out = await toolset.fetch_paged("https://a.com/x", page=9)
    assert "has 3 page(s); page 9 does not exist" in out


async def test_a_page_below_one_is_refused_without_charge(monkeypatch):
    toolset, provider = _toolset(monkeypatch, fetch_pagination=True)
    assert "1 or greater" in await toolset.fetch_paged("https://a.com/x", page=0)
    assert provider.fetches == 0 and toolset._calls == 0


async def test_paged_fetch_is_refused_when_pagination_is_off(monkeypatch):
    toolset, provider = _toolset(monkeypatch)
    assert await toolset.fetch_paged("https://a.com/x", page=2) == PAGINATION_OFF
    assert provider.fetches == 0


async def test_disabled_fetch_is_refused_without_charge(monkeypatch):
    toolset, provider = _toolset(monkeypatch, enable_fetch=False)
    assert await toolset.fetch("https://a.com/x") == FETCH_DISABLED
    assert provider.fetches == 0 and toolset._calls == 0


def test_pagination_without_a_page_size_is_a_config_error():
    with pytest.raises(ValueError, match="page size"):
        WebToolsetConfig(fetch_pagination=True, max_page_chars=None)


# --- what the model is offered ------------------------------------------------


@pytest.mark.parametrize(
    ("config", "tools"),
    [
        ({}, {"search": ["query"], "fetch": ["url"]}),
        ({"fetch_pagination": True}, {"search": ["query"], "fetch": ["url", "page"]}),
        ({"enable_fetch": False}, {"search": ["query"]}),
    ],
)
async def test_real_mcp_server_advertises_only_the_enabled_fetch(config, tools):
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("searchforge-test")
    WebToolset(WebToolsetConfig(env_file=None, **config)).register(mcp)
    advertised = {tool.name: tool for tool in await mcp.list_tools()}
    assert set(advertised) == set(tools)
    for name, params in tools.items():
        assert list(advertised[name].input_schema["properties"]) == params


def _prompt(**tools) -> str:
    config = SearchForgeConfig(
        providers=["serper"],
        task=SearchForgeTaskConfig(tools=WebToolsetConfig(**tools)),
    )
    return SearchForgeTaskset(config).load()[0].data.system_prompt


def test_search_only_prompt_never_names_fetch():
    prompt = _prompt(enable_fetch=False, max_tool_calls=9)
    assert "web_fetch" not in prompt
    assert "9 provider-backed `web_search` calls" in prompt


def test_default_prompt_names_both_tools_and_states_no_cap():
    prompt = _prompt()
    assert "web_fetch" in prompt
    assert "8000" not in prompt and "8,000" not in prompt
    assert tool_lines(10, True) in prompt


# --- metrics read back from the trace -----------------------------------------


def _trace(*messages):
    return SimpleNamespace(
        tool_messages=[SimpleNamespace(name=n, content=c) for n, c in messages]
    )


def _render(text: str) -> str:
    return render_page(PageContent("https://a.com", "T", text, None))


def test_fetch_stats_read_markers_and_uncut_bodies_and_skip_refusals():
    trace = _trace(
        ("web_fetch", _render(cap_page(PAGE, 8000))),
        ("web_fetch", _render("a" * 300)),
        ("web_fetch", BUDGET_EXHAUSTED),
        ("web_search", _render(cap_page(PAGE, 8000))),
    )
    assert fetch_page_stats(trace) == [(20_500, True), (300, False)]


async def test_fetch_metrics_are_a_mean_length_and_a_truncation_count():
    trace = _trace(
        ("web_fetch", _render(cap_page(PAGE, 8000))),
        ("web_fetch", _render("a" * 500)),
    )
    task = object.__new__(SearchForgeTask)
    assert await task.fetch_chars_raw(trace) == (20_500 + 500) / 2
    assert await task.fetch_truncated(trace) == 1.0
    assert await task.fetch_chars_raw(_trace()) == 0.0


def test_fetch_metrics_carry_no_reward_weight():
    for name in ("fetch_chars_raw", "fetch_truncated"):
        assert not hasattr(getattr(SearchForgeTask, name), "_vf_weight")
