"""Fail-closed validation for a completed comparative live eval."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.matrix import read_traces


class GateFailure(RuntimeError):
    pass


def _records(path: Path) -> list[dict]:
    records = read_traces(path)
    if not records:
        raise GateFailure("trace serialization produced no records")
    return records


def validate_traces(path: Path, *, expected_providers: set[str]) -> dict:
    traces = _records(path)
    providers = {trace["task"]["data"]["provider"] for trace in traces}
    missing_providers = expected_providers - providers
    if missing_providers:
        raise GateFailure(f"missing provider traces: {sorted(missing_providers)}")

    advertised = {
        tool.get("name")
        for trace in traces
        for tool in trace.get("tools", [])
        if tool.get("name")
    }
    required_tools = {"web_search", "web_fetch"}
    if missing := required_tools - advertised:
        raise GateFailure(f"MCP did not advertise tools: {sorted(missing)}")

    issued = {
        message.get("name")
        for trace in traces
        for node in trace.get("nodes", [])
        if (message := node.get("message", {})).get("role") == "tool"
        and message.get("name")
    }
    if missing := required_tools - issued:
        raise GateFailure(f"model did not issue tools: {sorted(missing)}")

    native = sum(
        float(trace.get("metrics", {}).get("native_search_calls") or 0.0)
        for trace in traces
    )
    if native:
        raise GateFailure(f"native search canary fired {native:g} time(s)")

    judge_executions = sum(
        bool(trace.get("info", {}).get("judge")) for trace in traces
    )
    if judge_executions != len(traces):
        raise GateFailure(
            f"judge proof missing for {len(traces) - judge_executions} trace(s)"
        )

    failures = [trace for trace in traces if not trace.get("ok", False)]
    unclassified = [trace for trace in failures if not trace.get("errors")]
    if unclassified:
        raise GateFailure(f"{len(unclassified)} failed trace(s) were unclassified")
    if failures:
        raise GateFailure(f"{len(failures)} trace(s) failed")

    return {
        "traces": len(traces),
        "providers": sorted(providers),
        "advertised_tools": sorted(advertised),
        "issued_tools": sorted(issued),
        "judge_executions": judge_executions,
        "native_search_calls": native,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("providers", nargs="+", help="expected comparative arms")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_traces(args.traces, expected_providers=set(args.providers)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
