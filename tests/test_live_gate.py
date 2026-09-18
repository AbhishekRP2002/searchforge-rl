import json

import pytest

from scripts.live_gate import GateFailure, validate_traces


def _trace(provider: str, *, native: float = 0.0, judge: bool = True) -> dict:
    return {
        "ok": True,
        "task": {"data": {"provider": provider}},
        "tools": [{"name": "web_search"}, {"name": "web_fetch"}],
        "nodes": [
            {"message": {"role": "tool", "name": "web_search", "content": "x"}},
            {"message": {"role": "tool", "name": "web_fetch", "content": "y"}},
        ],
        "metrics": {"native_search_calls": native},
        "info": {"judge": [{"result": "yes"}]} if judge else {},
        "errors": [],
    }


def _episode(*traces: dict) -> str:
    """One JSONL line as verifiers actually writes it: an Episode wrapping traces.

    These fixtures previously wrote bare traces, which matched the reader's bug
    rather than the format, so the suite stayed green while the gate rejected
    every real artifact.
    """
    return json.dumps({"id": "ep", "ok": True, "errors": [], "traces": list(traces)})


def test_live_gate_accepts_both_arms_across_episodes(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(_episode(_trace(p)) for p in ("serper", "exa")))

    report = validate_traces(path, expected_providers={"serper", "exa"})

    assert report["providers"] == ["exa", "serper"]
    assert report["advertised_tools"] == ["web_fetch", "web_search"]
    assert report["issued_tools"] == ["web_fetch", "web_search"]
    assert report["judge_executions"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda trace: trace.update(tools=[]), "advertise"),
        (lambda trace: trace.update(nodes=[]), "issue"),
        (lambda trace: trace.update(info={}), "judge"),
        (
            lambda trace: trace.update(metrics={"native_search_calls": 1.0}),
            "native",
        ),
        (lambda trace: trace.update(ok=False), "failed"),
    ],
)
def test_live_gate_fails_closed_when_proof_is_missing(tmp_path, mutation, message):
    trace = _trace("serper")
    mutation(trace)
    path = tmp_path / "traces.jsonl"
    path.write_text(_episode(trace))

    with pytest.raises(GateFailure, match=message):
        validate_traces(path, expected_providers={"serper"})


def test_gate_passes_a_real_run_artifact():
    """The gate is fail-closed, so a false negative is as damaging as a false
    positive: it blocked a run that had actually met every condition. This pins it
    against a real traces.jsonl rather than a hand-built one."""
    from pathlib import Path

    from scripts.live_gate import validate_traces

    fixture = Path(__file__).parent / "fixtures" / "traces_episodes.jsonl"
    report = validate_traces(fixture, expected_providers={"serper", "exa"})

    assert report["traces"] == 2
    assert report["providers"] == ["exa", "serper"]
    assert report["advertised_tools"] == ["web_fetch", "web_search"]
    assert report["issued_tools"] == ["web_fetch", "web_search"]
    assert report["judge_executions"] == 2
    assert report["native_search_calls"] == 0.0
