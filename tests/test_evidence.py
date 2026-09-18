from searchforge.evidence import (
    answer_present,
    grounded_fraction,
    normalize_url,
    urls_in_text,
)


def test_normalize_strips_tracking_parameters():
    assert normalize_url("https://a.com/p?utm_source=x&utm_medium=y") == "https://a.com/p"


def test_normalize_keeps_meaningful_query_parameters():
    assert normalize_url("https://a.com/s?id=42") == "https://a.com/s?id=42"


def test_normalize_unifies_scheme_case_and_trailing_slash():
    assert normalize_url("HTTP://A.com/Path/") == normalize_url("http://a.com/Path")


def test_normalize_drops_a_fragment():
    assert normalize_url("https://a.com/p#section") == "https://a.com/p"


def test_normalize_is_path_case_sensitive():
    """Hosts are case-insensitive; paths are not."""
    assert normalize_url("https://a.com/Path") != normalize_url("https://a.com/path")


def test_urls_in_text_finds_bare_and_markdown_links():
    text = "See https://a.com/one and [two](https://b.com/two)."
    assert urls_in_text(text) == {"https://a.com/one", "https://b.com/two"}


def test_urls_in_text_strips_trailing_sentence_punctuation():
    assert urls_in_text("Source: https://a.com/p.") == {"https://a.com/p"}


def test_urls_in_text_returns_empty_set_for_no_urls():
    assert urls_in_text("no links at all") == set()


def test_grounded_fraction_is_one_when_every_citation_was_retrieved():
    assert grounded_fraction({"https://a.com"}, {"https://a.com", "https://b.com"}) == 1.0


def test_grounded_fraction_is_zero_for_a_fabricated_citation():
    assert grounded_fraction({"https://fake.com"}, {"https://a.com"}) == 0.0


def test_grounded_fraction_is_partial_for_a_mix():
    result = grounded_fraction({"https://a.com", "https://fake.com"}, {"https://a.com"})
    assert result == 0.5


def test_grounded_fraction_of_no_citations_is_one_not_zero():
    """No citation is not a fabricated citation. Answering without citing is a
    separate behaviour and must not be scored as dishonesty."""
    assert grounded_fraction(set(), {"https://a.com"}) == 1.0


def test_grounded_fraction_matches_across_tracking_parameters():
    cited = {"https://a.com/p?utm_source=chat"}
    assert grounded_fraction(cited, {"https://a.com/p"}) == 1.0


def test_answer_present_is_case_insensitive():
    assert answer_present("Kyoto", "headquartered in kyoto, japan")


def test_answer_present_ignores_thousands_separators():
    assert answer_present("56000", "revenue was 56,000 units")


def test_answer_present_requires_every_item_of_a_list_answer():
    assert answer_present(["a", "b"], "both a and b appear")
    assert not answer_present(["a", "b"], "only a appears")


def test_answer_present_is_false_for_an_absent_answer():
    assert not answer_present("Osaka", "headquartered in Kyoto")
