"""Reward and metric wiring, exercised through the pure helpers each one calls.

Constructing a real vf.Trace is the job of the live smoke test in Task 10; here
we assert the contract each signal must satisfy and the weights they carry.
"""

from searchforge.taskset import SearchForgeTask


def _weight(name: str) -> float:
    fn = getattr(SearchForgeTask, name)
    return fn._vf_weight


def test_context_overflow_is_the_only_penalty():
    assert _weight("context_overflow") == -0.25


def test_candidate_signals_ship_at_weight_zero():
    """Visible in every trace, promoted later by a metadata-only config entry."""
    assert _weight("citations_grounded") == 0.0
    assert _weight("citation_presence") == 0.0
    assert _weight("answer_in_results") == 0.0


def test_efficiency_signals_are_metrics_not_rewards():
    """A metric cannot be promoted by a weight-only config entry, which is the
    point: rewarding tool-call counts trains the model to stop searching."""
    for name in (
        "num_tool_calls",
        "num_search_calls",
        "num_fetch_calls",
        "tokens_consumed",
        "native_search_calls",
    ):
        fn = getattr(SearchForgeTask, name)
        assert not hasattr(fn, "_vf_weight"), f"{name} must be a metric, not a reward"


def test_no_reward_reads_a_gold_url_list():
    """Gold URLs encode the dataset builder's provider. Comparing against them
    would measure provider agreement, not answer quality."""
    import inspect

    import searchforge.taskset as module

    source = inspect.getsource(module)
    assert "gold_url" not in source
    assert "expected_url" not in source


def _trace(*provider_states):
    """The canary only reads `trace.assistant_messages`. Real AssistantMessage
    objects keep the provider_state shape honest — a hand-rolled stub is what let
    this function read a `Trace.model_calls` attribute that does not exist."""
    from types import SimpleNamespace

    from verifiers.v1.types import AssistantMessage

    return SimpleNamespace(
        assistant_messages=[
            AssistantMessage(content="x", provider_state=state)
            for state in provider_states
        ]
    )


def test_canary_is_silent_when_the_model_used_only_our_tools():
    from searchforge.trace_reader import native_search_citations

    assert native_search_citations(_trace(None, [{"type": "reasoning"}])) == []


def test_canary_fires_on_a_provider_side_search_call():
    from searchforge.trace_reader import native_search_citations

    assert native_search_citations(_trace([{"type": "web_search_call"}])) == [
        "web_search_call"
    ]


def test_canary_reports_url_citations_from_a_native_answer():
    from searchforge.trace_reader import native_search_citations

    item = {
        "type": "message",
        "content": [
            {
                "type": "output_text",
                "annotations": [{"type": "url_citation", "url": "https://a.com/p"}],
            }
        ],
    }
    assert native_search_citations(_trace([item])) == ["https://a.com/p"]


def test_canary_survives_string_content():
    """Anthropic sends content as a str; iterating it yields characters."""
    from searchforge.trace_reader import native_search_citations

    assert native_search_citations(_trace([{"type": "text", "content": "hello"}])) == []


def test_tool_counts_use_serialized_mcp_tool_names():
    from types import SimpleNamespace

    from searchforge.trace_reader import tool_call_counts

    trace = SimpleNamespace(
        tool_messages=[
            SimpleNamespace(name="web_search"),
            SimpleNamespace(name="web_fetch"),
            SimpleNamespace(name="web_search"),
        ]
    )

    assert tool_call_counts(trace) == {"web_search": 2, "web_fetch": 1}
