from searchforge.providers.base import PageContent, SearchResult
from searchforge.render import render_page, render_results

RESULTS = [
    SearchResult(
        title="Nintendo",
        url="https://en.wikipedia.org/wiki/Nintendo",
        excerpt="Nintendo Co., Ltd. is a Japanese video game company founded in 1889.",
        date="2026-08-14",
        rank=1,
    ),
    SearchResult(
        title="Company History",
        url="https://www.nintendo.co.jp/corporate/history/",
        excerpt="Founded by Fusajiro Yamauchi.",
        date=None,
        rank=2,
    ),
]


def test_markdown_lists_every_result_with_rank_title_url():
    out = render_results(RESULTS, "markdown")
    assert "[1] Nintendo" in out
    assert "https://en.wikipedia.org/wiki/Nintendo" in out
    assert "[2] Company History" in out


def test_markdown_includes_date_when_present_and_omits_when_absent():
    out = render_results(RESULTS, "markdown")
    assert "(2026-08-14)" in out
    # the second result has no date, so no empty parens anywhere
    assert "()" not in out


def test_markdown_is_more_compact_than_json():
    """JSON keys and quoting repeat per result and cost sequence budget that
    policy updates pay for 3-5x. See spec section 8."""
    assert len(render_results(RESULTS, "markdown")) < len(render_results(RESULTS, "json"))


def test_json_format_is_parseable():
    import json

    parsed = json.loads(render_results(RESULTS, "json"))
    assert [r["url"] for r in parsed] == [r.url for r in RESULTS]


def test_empty_results_render_as_an_explicit_message_not_empty_string():
    out = render_results([], "markdown")
    assert out.strip()
    assert "no results" in out.lower()


def test_unknown_format_raises():
    import pytest

    with pytest.raises(ValueError, match="result_format"):
        render_results(RESULTS, "yaml")


def test_render_page_includes_title_url_and_text():
    page = PageContent(
        url="https://example.com/a",
        title="Example",
        text="Body text here.",
        date="2026-01-02",
    )
    out = render_page(page)
    assert "Example" in out
    assert "https://example.com/a" in out
    assert "Body text here." in out
