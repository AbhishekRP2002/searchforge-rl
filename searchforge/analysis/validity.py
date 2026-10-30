"""Checks that decide whether an arm's numbers mean what they claim.

An error makes an arm unusable: it is reported but left out of every paired
comparison. A warning is reported and the arm still counts. In gate mode the
report fails closed on either -- the old `live_gate.py` contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from searchforge.analysis.runs import Arm

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Check:
    arm: str
    level: str
    message: str


def _judge_ran(trace: dict) -> bool:
    # verifiers 0.4 records each judge exchange under `judge_calls`; 0.3.x wrote
    # bare responses under `judge`. Both count, so older run artifacts still read.
    info = trace.get("info") or {}
    return bool(info.get("judge_calls") or info.get("judge"))


def check_arm(arm: Arm, *, max_failure_rate: float) -> list[Check]:
    checks: list[Check] = []

    def add(level: str, message: str) -> None:
        checks.append(Check(arm.name, level, message))

    traces = arm.traces
    if not traces:
        add(ERROR, "no traces")
        return checks

    native = sum(
        float((trace.get("metrics") or {}).get("native_search_calls") or 0.0)
        for trace in traces
    )
    if native:
        add(
            ERROR,
            f"voided: {native:g} native provider search call(s) bypassed the "
            "SearchForge tools",
        )

    advertised = {
        tool.get("name") for trace in traces for tool in trace.get("tools") or [] if tool.get("name")
    }
    if "web_search" not in advertised:
        add(ERROR, "web_search was never advertised to the model")
    issued = {
        message.get("name")
        for trace in traces
        for node in trace.get("nodes") or []
        if (message := node.get("message") or {}).get("role") == "tool"
    }
    if unused := sorted(name for name in advertised - issued if name):
        add(WARNING, f"advertised but never called: {', '.join(unused)}")

    scored = arm.scored
    unjudged = sum(not _judge_ran(trace) for trace in scored)
    if unjudged:
        add(ERROR, f"judge did not run on {unjudged} of {len(scored)} scored trace(s)")

    failed = [trace for trace in traces if not trace.get("ok", False)]
    if unclassified := sum(not trace.get("errors") for trace in failed):
        add(ERROR, f"{unclassified} failed trace(s) carry no error class")
    rate = len(failed) / len(traces)
    if rate > max_failure_rate:
        add(
            WARNING,
            f"failure rate {rate:.1%} ({len(failed)}/{len(traces)}) exceeds "
            f"{max_failure_rate:.1%}",
        )
    return checks


def check_expected(arms: list[Arm], expected: list[str]) -> list[Check]:
    present = {arm.provider for arm in arms}
    return [
        Check(provider, ERROR, "expected provider has no traces")
        for provider in expected
        if provider not in present
    ]
