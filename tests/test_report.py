import copy
import json
from pathlib import Path

import pytest

from searchforge.analysis import build_report, load_run, render_markdown
from searchforge.analysis.cli import main
from searchforge.analysis.fingerprint import flatten
from searchforge.analysis.runs import read_traces, split_arms
from searchforge.analysis.stats import holm, paired, sign_test

FIXTURE = Path(__file__).parent / "fixtures" / "traces_episodes.jsonl"

CONFIG = {
    "model": "model-a",
    "sampling": {"temperature": None},
    "run": {"name": "x", "dir": "x"},
    "output_dir": "outputs",
    "env": {
        "taskset": {
            "id": "searchforge",
            "data_path": "/nowhere/tasks_v1.jsonl",
            "providers": ["serper", "exa"],
            "current_date": None,
            "task": {
                "tools": {"provider": "serper", "num_results": 5, "max_tool_calls": 10},
                "judges": [{"id": "searchforge", "model": "judge-a", "api_key_var": "K"}],
            },
        },
        "agent": {"harness": {"id": "null"}},
    },
}


def trace(
    source_id: str,
    provider: str,
    correct: float,
    *,
    model: str = "model-a",
    ok: bool = True,
    error_type: str | None = "ProviderError",
    native: float = 0.0,
    judged: bool = True,
    source_dataset: str = "simpleqa",
    hops: int = 1,
    tools: tuple[str, ...] = ("web_search", "web_fetch"),
) -> dict:
    return {
        "ok": ok,
        "verifiers": {"version": "0.4.0"},
        "agent": {"config": {"model": model, "sampling": {}, "harness": {"id": "null"}}},
        "task": {
            "key": f"{source_id}::{provider}",
            "data": {
                "source_id": source_id,
                "provider": provider,
                "source_dataset": source_dataset,
                "hops": hops,
                "answer_type": "entity",
                "system_prompt": "prompt",
            },
        },
        "tools": [{"name": name} for name in tools],
        "nodes": [{"message": {"role": "tool", "name": name}} for name in tools],
        "rewards": {
            "searchforge": {"score": correct, "weight": 1.0},
            "answer_in_results": {"score": 1.0, "weight": 0.0},
            "citations_grounded": {"score": 1.0, "weight": 0.0},
            "citation_presence": {"score": 1.0, "weight": 0.0},
            "context_overflow": {"score": 0.0, "weight": -0.25},
        },
        "metrics": {
            "native_search_calls": native,
            "num_tool_calls": 2.0,
            "num_search_calls": 1.0,
            "num_fetch_calls": 1.0,
            "tokens_consumed": 250.0,
        },
        "timing": {"agent": {"start": 0.0, "end": 4.0}},
        "info": {"judge_calls": [{"name": "searchforge"}]} if judged else {},
        "errors": [] if ok or error_type is None else [{"type": error_type}],
    }


def write_run(root: Path, name: str, traces: list[dict], config: dict | None = CONFIG) -> Path:
    run = root / name
    (run / "configs" / "resolved").mkdir(parents=True)
    (run / "traces.jsonl").write_text(
        "\n".join(json.dumps({"id": "ep", "traces": [t]}) for t in traces)
    )
    if config is not None:
        (run / "configs" / "resolved" / "eval.json").write_text(json.dumps(config))
    return run


def scores(provider: str, values: list[float], **kwargs) -> list[dict]:
    return [trace(f"q{i}", provider, v, **kwargs) for i, v in enumerate(values)]


# ---- reading -------------------------------------------------------------------


def test_read_traces_unwraps_episodes_from_a_real_artifact():
    """traces.jsonl holds one Episode per line. Reading a line as a Trace yields
    empty fields rather than an error, so this pins the real shape."""
    traces = read_traces(FIXTURE)

    assert len(traces) == 2
    assert {t["task"]["data"]["provider"] for t in traces} == {"serper", "exa"}


def test_one_run_splits_into_one_arm_per_provider(tmp_path):
    run = write_run(tmp_path, "r", scores("serper", [1, 0]) + scores("exa", [1, 1]))

    arms = split_arms([load_run(run)])

    assert [arm.name for arm in arms] == ["exa", "serper"]
    assert {arm.model for arm in arms} == {"model-a"}
    assert all(len(arm.traces) == 2 for arm in arms)


def test_a_traces_path_works_as_well_as_a_run_directory(tmp_path):
    run = write_run(tmp_path, "r", scores("serper", [1]))

    assert load_run(run / "traces.jsonl").config == load_run(run).config


def test_same_provider_in_two_runs_gets_distinct_arm_names(tmp_path):
    one = write_run(tmp_path, "monday", scores("serper", [1]))
    two = write_run(tmp_path, "tuesday", scores("serper", [0]))

    names = [arm.name for arm in split_arms([load_run(one), load_run(two)])]

    assert names == ["serper@monday", "serper@tuesday"]


def test_the_real_artifact_reports_cleanly():
    report = build_report([load_run(FIXTURE)], expected=["serper", "exa"])

    assert report["status"] == "ok"
    assert report["validity"] == [], "a 0.3.1 run (info.judge) still passes"
    assert [arm["summary"]["accuracy"] for arm in report["arms"]] == [1.0, 1.0]


# ---- statistics ----------------------------------------------------------------


def test_sign_test_matches_exact_mcnemar():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 0) == pytest.approx(2 / 32)
    assert sign_test(3, 3) == 1.0


def test_holm_adjusts_in_input_order_and_stays_monotone():
    adjusted = holm([0.04, 0.01, 0.03])

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_paired_joins_on_source_id_and_reports_dropped_ids():
    result = paired({"a": 1.0, "b": 0.0, "c": 1.0}, {"a": 1.0, "b": 1.0, "d": 0.0})

    assert result["n_pairs"] == 2
    assert result["dropped"] == ["c", "d"]
    assert (result["challenger_wins"], result["baseline_wins"], result["ties"]) == (1, 0, 1)
    assert result["delta"] == 0.5


def test_repeated_rollouts_are_averaged_per_question(tmp_path):
    traces = [trace("q0", "serper", 1.0), trace("q0", "serper", 0.0), trace("q0", "exa", 1.0)]
    report = build_report([load_run(write_run(tmp_path, "r", traces))])

    (result,) = report["comparisons"]
    assert result["n_pairs"] == 1
    assert result["delta"] == pytest.approx(0.5)


# ---- comparisons ---------------------------------------------------------------


def test_n_providers_get_every_pair_with_holm_correction(tmp_path):
    traces = (
        scores("serper", [0] * 40)
        + scores("exa", [1] * 40)
        + scores("tavily", [0] * 40)
    )
    report = build_report([load_run(write_run(tmp_path, "r", traces))])

    pairs = [(r["baseline"], r["challenger"]) for r in report["comparisons"]]
    assert pairs == [("serper", "exa"), ("serper", "tavily"), ("exa", "tavily")]
    for r in report["comparisons"]:
        assert r["p_holm"] >= r["p"]
    assert "significant at 5%" in report["comparisons"][0]["verdict"]
    assert report["comparisons"][1]["verdict"] == "not significant"


def test_baseline_compares_every_arm_against_one(tmp_path):
    traces = scores("serper", [0, 1]) + scores("exa", [1, 1]) + scores("tavily", [1, 0])
    report = build_report([load_run(write_run(tmp_path, "r", traces))], baseline="serper")

    assert [(r["baseline"], r["challenger"]) for r in report["comparisons"]] == [
        ("serper", "exa"),
        ("serper", "tavily"),
    ]


def test_headline_reads_in_points_and_flags_small_samples(tmp_path):
    traces = scores("serper", [0, 1]) + scores("exa", [1, 1])
    report = build_report([load_run(write_run(tmp_path, "r", traces))], baseline="serper")

    headline = report["comparisons"][0]["headline"]
    assert headline.startswith("exa vs serper: +50.0 pts")
    assert "n=2 pairs -- too few pairs to conclude" in headline


# ---- fingerprint ---------------------------------------------------------------


def test_runs_differing_only_in_provider_are_comparable(tmp_path):
    exa_config = copy.deepcopy(CONFIG)
    exa_config["env"]["taskset"]["task"]["tools"]["provider"] = "exa"
    exa_config["env"]["taskset"]["providers"] = ["exa"]
    exa_config["run"] = {"name": "y", "dir": "y"}
    one = write_run(tmp_path, "a", scores("serper", [1]))
    two = write_run(tmp_path, "b", scores("exa", [1]), exa_config)

    report = build_report([load_run(one), load_run(two)])

    assert report["status"] == "ok"
    assert len({arm["fingerprint"] for arm in report["arms"]}) == 1


def test_a_second_difference_refuses_the_comparison_and_names_the_key(tmp_path):
    other = copy.deepcopy(CONFIG)
    other["env"]["taskset"]["task"]["tools"]["max_tool_calls"] = 20
    one = write_run(tmp_path, "a", scores("serper", [1]))
    two = write_run(tmp_path, "b", scores("exa", [1]), other)

    report = build_report([load_run(one), load_run(two)])

    assert report["status"] == "refused"
    assert list(report["comparability"]["blocking"]) == [
        "env.taskset.task.tools.max_tool_calls"
    ]
    assert report["comparisons"] == []
    assert "REFUSED" in render_markdown(report)


def test_trace_level_differences_are_caught_even_with_identical_configs(tmp_path):
    one = write_run(tmp_path, "a", scores("serper", [1]))
    two = write_run(tmp_path, "b", scores("exa", [1], model="model-b"))

    report = build_report([load_run(one), load_run(two)])

    assert report["status"] == "refused"
    assert "trace.model" in report["comparability"]["blocking"]


def test_allow_diff_waives_a_difference_and_stamps_the_report(tmp_path):
    one = write_run(tmp_path, "a", scores("serper", [1]))
    two = write_run(tmp_path, "b", scores("exa", [1], model="model-b"))

    report = build_report([load_run(one), load_run(two)], allow_diff=["trace.model"])

    assert report["status"] == "ok"
    assert "trace.model" in report["comparability"]["waived"]
    assert "NOT A PROVIDER-ONLY COMPARISON: differs in trace.model" in render_markdown(report)


def test_credentials_and_bookkeeping_never_count_as_differences(tmp_path):
    other = copy.deepcopy(CONFIG)
    other["env"]["taskset"]["task"]["judges"][0]["api_key_var"] = "OTHER"
    other["output_dir"] = "elsewhere"
    one = write_run(tmp_path, "a", scores("serper", [1]))
    two = write_run(tmp_path, "b", scores("exa", [1]), other)

    assert build_report([load_run(one), load_run(two)])["status"] == "ok"


def test_flatten_indexes_lists_of_objects_only():
    assert flatten({"a": [{"b": 1}], "c": [1, 2]}) == {"a.0.b": 1, "c": [1, 2]}


# ---- validity ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "level", "message"),
    [
        (lambda t: t["metrics"].update(native_search_calls=1.0), "error", "voided"),
        (lambda t: t.update(info={}), "error", "judge did not run"),
        (lambda t: t.update(tools=[{"name": "web_fetch"}]), "error", "web_search was never advertised"),
        (lambda t: t.update(nodes=[]), "warning", "never called"),
        (lambda t: t.update(ok=False, errors=[]), "error", "no error class"),
        (lambda t: t.update(ok=False, errors=[{"type": "X"}]), "warning", "failure rate"),
    ],
)
def test_validity_checks(tmp_path, mutate, level, message):
    bad = trace("q0", "exa", 1.0)
    mutate(bad)
    run = write_run(tmp_path, "r", [trace("q0", "serper", 1.0), bad])

    report = build_report([load_run(run)])

    (check,) = [c for c in report["validity"] if message in c["message"]]
    assert (check["arm"], check["level"]) == ("exa", level)


def test_an_invalid_arm_is_left_out_of_comparisons(tmp_path):
    traces = scores("serper", [1]) + scores("exa", [1], native=1.0)
    report = build_report([load_run(write_run(tmp_path, "r", traces))])

    assert report["comparisons"] == []
    assert [arm["valid"] for arm in report["arms"]] == [False, True]


def test_legacy_judge_key_counts_as_judge_proof(tmp_path):
    legacy = trace("q0", "exa", 1.0)
    legacy["info"] = {"judge": [{"parsed": 1.0}]}
    report = build_report([load_run(write_run(tmp_path, "r", [legacy]))])

    assert report["validity"] == []


def test_expected_provider_missing_is_an_error(tmp_path):
    report = build_report(
        [load_run(write_run(tmp_path, "r", scores("serper", [1])))], expected=["exa"]
    )

    assert report["validity"] == [
        {"arm": "exa", "level": "error", "message": "expected provider has no traces"}
    ]


# ---- slices --------------------------------------------------------------------


def test_slices_break_down_by_dataset_and_hops_with_small_n_flags(tmp_path):
    traces = [
        trace("q0", "serper", 1, source_dataset="simpleqa", hops=1),
        trace("q1", "serper", 0, source_dataset="hotpotqa", hops=2),
        trace("q0", "exa", 1, source_dataset="simpleqa", hops=1),
        trace("q1", "exa", 1, source_dataset="hotpotqa", hops=2),
    ]
    report = build_report([load_run(write_run(tmp_path, "r", traces))])

    fields = {(s["field"], s["value"]) for s in report["slices"]}
    assert fields == {
        ("source_dataset", "simpleqa"),
        ("source_dataset", "hotpotqa"),
        ("hops", "1"),
        ("hops", "2"),
    }, "answer_type has one value and is skipped"
    hotpot = next(s for s in report["slices"] if s["value"] == "hotpotqa")
    assert abs(hotpot["pairs"][0]["delta"]) == 1.0
    assert hotpot["arms"]["exa"]["small"] is True


# ---- CLI -----------------------------------------------------------------------


def test_cli_writes_markdown_and_json(tmp_path, capsys):
    run = write_run(tmp_path, "r", scores("serper", [0, 1]) + scores("exa", [1, 1]))

    assert main([str(run)]) == 0
    assert "## Verdict" in capsys.readouterr().out

    out = tmp_path / "report.json"
    main([str(run), "--format", "json", "--out", str(out)])
    assert json.loads(out.read_text())["status"] == "ok"


def test_cli_gate_fails_closed_on_any_check(tmp_path):
    run = write_run(tmp_path, "r", scores("serper", [1]) + scores("exa", [1], native=1.0))

    with pytest.raises(SystemExit) as exit_info:
        main([str(run), "--gate"])
    assert exit_info.value.code == 1


def test_cli_exits_2_when_the_comparison_is_refused(tmp_path):
    other = copy.deepcopy(CONFIG)
    other["model"] = "model-b"
    one = write_run(tmp_path, "a", scores("serper", [1]))
    two = write_run(tmp_path, "b", scores("exa", [1]), other)

    with pytest.raises(SystemExit) as exit_info:
        main([str(one), str(two)])
    assert exit_info.value.code == 2
