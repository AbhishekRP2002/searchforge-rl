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


def test_live_gate_accepts_comparative_real_trace_shape(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(_trace(p)) for p in ("serper", "exa")))

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
    path.write_text(json.dumps(trace))

    with pytest.raises(GateFailure, match=message):
        validate_traces(path, expected_providers={"serper"})
