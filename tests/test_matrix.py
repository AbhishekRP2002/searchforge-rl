import pytest

from scripts.matrix import (
    VoidedRunError,
    compare,
    pair,
    render_comparisons,
    render_table,
    summarise,
)


def trace(
    source_id: str,
    provider: str,
    correct: float,
    *,
    answer_in_results: float = 1.0,
    citations_grounded: float = 1.0,
    citation_presence: float = 1.0,
    native_search_calls: float = 0.0,
    num_tool_calls: float = 2.0,
    ok: bool = True,
    error_type: str | None = None,
) -> dict:
    return {
        "ok": ok,
        "task": {
            "key": f"{source_id}::{provider}",
            "data": {"source_id": source_id, "provider": provider},
        },
        "rewards": {
            "searchforge": {"score": correct, "weight": 1.0},
            "answer_in_results": {"score": answer_in_results, "weight": 0.0},
            "citations_grounded": {"score": citations_grounded, "weight": 0.0},
            "citation_presence": {"score": citation_presence, "weight": 0.0},
        },
        "metrics": {
            "native_search_calls": native_search_calls,
            "num_tool_calls": num_tool_calls,
            "num_search_calls": 1.0,
            "num_fetch_calls": 1.0,
            "tokens_consumed": 250.0,
        },
        "errors": [{"type": error_type, "message": "failed"}]
        if error_type
        else [],
    }


def test_summarise_reads_scores_from_serialized_verifiers_rewards():
    summary = summarise([trace("a", "serper", 1.0), trace("b", "serper", 0.0)])

    assert summary["accuracy"] == 0.5
    assert summary["answer_in_results"] == 1.0
    assert summary["extraction_gap"] == 0.5
    assert summary["accuracy_ci95"][0] < summary["accuracy"]
    assert summary["accuracy_ci95"][1] > summary["accuracy"]


def test_summarise_reports_failures_without_scoring_them():
    traces = [
        trace("a", "serper", 1.0),
        trace("b", "serper", 0.0, ok=False, error_type="ProviderError"),
    ]

    summary = summarise(traces)

    assert summary["n"] == 2
    assert summary["n_scored"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["failures"] == {"ProviderError": 1}


def test_summarise_voids_a_run_with_native_search():
    with pytest.raises(VoidedRunError, match="native"):
        summarise([trace("a", "serper", 1.0, native_search_calls=1.0)])


def test_pair_reads_source_ids_and_scores_from_real_trace_shape():
    cells = {
        "serper": [trace("a", "serper", 1.0), trace("b", "serper", 1.0)],
        "exa": [trace("a", "exa", 0.0)],
    }

    result = pair(cells)

    assert result == {
        "paired_source_ids": ["a"],
        "dropped": ["b"],
        "discordant": 1,
    }


def test_render_table_includes_failure_and_retrieval_diagnostics():
    summary = summarise([trace("a", "serper", 1.0)])

    rendered = render_table({("model", "serper"): summary})

    assert "n_scored" in rendered
    assert "failure_rate" in rendered
    assert "answer_in_results" in rendered
    assert "citation_presence" in rendered
    assert "mean_search_calls" in rendered
    assert "mean_tokens" in rendered
    assert "model | serper" in rendered


def test_compare_reports_paired_delta_and_discordant_direction():
    cells = {
        "serper": [
            trace("a", "serper", 1.0),
            trace("b", "serper", 1.0),
            trace("c", "serper", 0.0),
        ],
        "exa": [
            trace("a", "exa", 0.0),
            trace("b", "exa", 1.0),
            trace("c", "exa", 1.0),
        ],
    }

    result = compare(cells, baseline="serper", challenger="exa")

    assert result["n_pairs"] == 3
    assert result["challenger_minus_baseline"] == 0.0
    assert result["challenger_wins"] == 1
    assert result["baseline_wins"] == 1
    assert result["ties"] == 1
    assert result["mcnemar_exact_p"] == 1.0


def test_render_comparisons_makes_the_pairing_visible():
    cells = {
        "serper": [trace("a", "serper", 1.0)],
        "exa": [trace("a", "exa", 0.0)],
    }

    rendered = render_comparisons({"model": cells})

    assert "paired" in rendered.lower()
    assert "serper" in rendered
    assert "exa" in rendered
    assert "challenger_minus_baseline" in rendered
