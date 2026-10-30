"""Build the comparison report: validity, comparability, arms, pairs, slices.

`build_report` returns plain data (it is what `--format json` prints), and
`render_markdown` turns that data into the human report. Nothing here reads
files; `runs.load_run` does that.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from searchforge.analysis import fingerprint as fp
from searchforge.analysis.runs import Arm, Run, source_id, split_arms
from searchforge.analysis.stats import holm, mean, paired, wilson
from searchforge.analysis.validity import ERROR, check_arm, check_expected

SLICES = ("source_dataset", "hops", "answer_type", "recency")
"""Task fields a result can be broken down by. A field with a single value, or
only `unknown`, says nothing and is skipped."""

SIGNIFICANCE = 0.05

STATUS_OK = "ok"
STATUS_REFUSED = "refused"


def score_name(arms: list[Arm]) -> str:
    """The correctness reward: the one reward with positive weight.

    The judge's reward is named after the judge id, which a user can replace,
    so the name is read from the traces rather than assumed.
    """
    positive = {
        name
        for arm in arms
        for trace in arm.scored
        for name, reward in (trace.get("rewards") or {}).items()
        if (reward or {}).get("weight", 0) > 0
    }
    if len(positive) == 1:
        return positive.pop()
    if "searchforge" in positive or not positive:
        return "searchforge"
    raise ValueError(
        f"several positive-weight rewards {sorted(positive)}; choose one with --score"
    )


def _reward(trace: dict, name: str) -> float:
    reward = (trace.get("rewards") or {}).get(name)
    if reward is None:
        return 0.0
    if not isinstance(reward, dict) or "score" not in reward:
        raise ValueError(f"reward {name!r} does not match the verifiers v1 trace schema")
    return float(reward["score"])


def _metric(trace: dict, name: str) -> float:
    return float((trace.get("metrics") or {}).get(name) or 0.0)


def _agent_seconds(trace: dict) -> float | None:
    agent = (trace.get("timing") or {}).get("agent") or {}
    if "start" in agent and "end" in agent:
        return float(agent["end"]) - float(agent["start"])
    return None


def summarise(traces: list[dict], score: str) -> dict:
    """One arm (or one slice of an arm) as a row of numbers."""
    scored = [trace for trace in traces if trace.get("ok", False)]
    correct = [_reward(trace, score) for trace in scored]
    available = [_reward(trace, "answer_in_results") for trace in scored]
    seconds = [s for trace in scored if (s := _agent_seconds(trace)) is not None]
    failures = Counter(
        ((trace.get("errors") or [{"type": "UnclassifiedError"}])[-1].get("type") or "UnclassifiedError")
        for trace in traces
        if not trace.get("ok", False)
    )
    return {
        "n": len(traces),
        "n_scored": len(scored),
        "accuracy": mean(correct),
        "accuracy_ci95": wilson(correct),
        "answer_in_results": mean(available),
        "extraction_gap": mean([a - c for a, c in zip(available, correct)]),
        "citation_presence": mean([_reward(t, "citation_presence") for t in scored]),
        "citations_grounded": mean([_reward(t, "citations_grounded") for t in scored]),
        "mean_tool_calls": mean([_metric(t, "num_tool_calls") for t in scored]),
        "mean_search_calls": mean([_metric(t, "num_search_calls") for t in scored]),
        "mean_fetch_calls": mean([_metric(t, "num_fetch_calls") for t in scored]),
        "mean_tokens": mean([_metric(t, "tokens_consumed") for t in scored]),
        "mean_agent_seconds": mean(seconds) if seconds else None,
        "failure_rate": (len(traces) - len(scored)) / len(traces) if traces else 0.0,
        "failures": dict(sorted(failures.items())),
    }


def per_question(traces: list[dict], score: str) -> dict[str, float]:
    """Score per source_id, averaged when a question ran more than once (`-r`)."""
    grouped: dict[str, list[float]] = {}
    for trace in traces:
        if trace.get("ok", False):
            grouped.setdefault(source_id(trace), []).append(_reward(trace, score))
    return {key: mean(values) for key, values in grouped.items()}


CONTROL = "serper"
"""Google SERP is the control arm in Exa's study, so in all-pairs mode it sits on
the baseline side of every pair it is in: deltas then read as "X vs Google"."""


def _pairs(names: list[str], baseline: str | None) -> list[tuple[str, str]]:
    """(baseline, challenger) pairs: every arm against `baseline`, or all pairs."""
    if baseline:
        return [(baseline, name) for name in names if name != baseline]
    ordered = sorted(names, key=lambda name: (name != CONTROL, name))
    return list(combinations(ordered, 2))


def _verdict(result: dict) -> str:
    if result["n_pairs"] < 30:
        return "too few pairs to conclude"
    if result["p_holm"] < SIGNIFICANCE:
        return "significant at 5% after Holm correction"
    return "not significant"


def headline(result: dict) -> str:
    low, high = result["delta_ci95"]
    return (
        f"{result['challenger']} vs {result['baseline']}: "
        f"{result['delta'] * 100:+.1f} pts (95% CI {low * 100:+.1f} to {high * 100:+.1f}), "
        f"p={result['p']:.3f} (Holm {result['p_holm']:.3f}), "
        f"n={result['n_pairs']} pairs -- {result['verdict']}"
    )


def _slice_values(arms: list[Arm]) -> dict[str, dict[str, str]]:
    """field -> source_id -> value, for every slice field worth showing."""
    by_field: dict[str, dict[str, str]] = {}
    for name in SLICES:
        values = {
            source_id(trace): str(trace["task"]["data"][name])
            for arm in arms
            for trace in arm.traces
            if name in trace["task"]["data"]
        }
        distinct = set(values.values()) - {"unknown", ""}
        if len(set(values.values())) > 1 and distinct:
            by_field[name] = values
    return by_field


def _slices(arms: list[Arm], pairs: list[tuple[str, str]], score: str, min_n: int) -> list[dict]:
    by_name = {arm.name: arm for arm in arms}
    out = []
    for name, values in _slice_values(arms).items():
        for value in sorted(set(values.values())):
            ids = {sid for sid, v in values.items() if v == value}
            rows = {
                arm.name: summarise([t for t in arm.traces if source_id(t) in ids], score)
                for arm in arms
            }
            comparisons = []
            for baseline, challenger in pairs:
                result = paired(
                    {k: v for k, v in per_question(by_name[baseline].traces, score).items() if k in ids},
                    {k: v for k, v in per_question(by_name[challenger].traces, score).items() if k in ids},
                )
                result.update(baseline=baseline, challenger=challenger, small=result["n_pairs"] < min_n)
                comparisons.append(result)
            out.append(
                {
                    "field": name,
                    "value": value,
                    "arms": {
                        arm: {"n": row["n"], "accuracy": row["accuracy"], "small": row["n"] < min_n}
                        for arm, row in rows.items()
                    },
                    "pairs": comparisons,
                }
            )
    return out


def build_report(
    runs: list[Run],
    *,
    baseline: str | None = None,
    allow_diff: list[str] | None = None,
    expected: list[str] | None = None,
    score: str | None = None,
    max_failure_rate: float = 0.0,
    min_slice_n: int = 30,
) -> dict:
    arms = split_arms(runs)
    names = [arm.name for arm in arms]
    if baseline and baseline not in names:
        raise ValueError(f"baseline {baseline!r} is not an arm; arms are {names}")
    score = score or score_name(arms)
    allow = list(allow_diff or [])

    checks = [c for arm in arms for c in check_arm(arm, max_failure_rate=max_failure_rate)]
    checks += check_expected(arms, list(expected or []))
    invalid = {c.arm for c in checks if c.level == ERROR}

    diff = fp.differences(arms)
    blocking = {key: values for key, values in diff.items() if not fp.allowed(key, allow)}
    waived = {key: values for key, values in diff.items() if fp.allowed(key, allow)}

    report: dict = {
        "status": STATUS_REFUSED if blocking else STATUS_OK,
        "score": score,
        "runs": [str(run.path) for run in runs],
        "arms": [
            {
                "name": arm.name,
                "provider": arm.provider,
                "model": arm.model,
                "run": arm.run.name,
                "fingerprint": fp.fingerprint(fp.arm_fields(arm)),
                "valid": arm.name not in invalid,
                "summary": summarise(arm.traces, score),
            }
            for arm in arms
        ],
        "validity": [c.__dict__ for c in checks],
        "comparability": {"blocking": blocking, "waived": waived},
        "comparisons": [],
        "slices": [],
    }
    if blocking:
        return report

    usable = [arm for arm in arms if arm.name not in invalid]
    pairs = [
        (b, c)
        for b, c in _pairs([arm.name for arm in usable], baseline)
        if b not in invalid and c not in invalid
    ]
    by_name = {arm.name: arm for arm in usable}
    comparisons = []
    for b, c in pairs:
        result = paired(per_question(by_name[b].traces, score), per_question(by_name[c].traces, score))
        result.update(baseline=b, challenger=c)
        comparisons.append(result)
    for result, adjusted in zip(comparisons, holm([r["p"] for r in comparisons])):
        result["p_holm"] = adjusted
        result["verdict"] = _verdict(result)
        result["headline"] = headline(result)
    report["comparisons"] = comparisons
    report["slices"] = _slices(usable, pairs, score, min_slice_n)
    return report


# ---- markdown -----------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}"


def _num(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_markdown(report: dict) -> str:
    lines = ["# SearchForge comparison report", ""]
    lines += [f"Runs: {', '.join(f'`{r}`' for r in report['runs'])}  "]
    lines += [f"Score: `{report['score']}` (1 = judged correct)", ""]

    blocking = report["comparability"]["blocking"]
    waived = report["comparability"]["waived"]
    if blocking:
        lines += [
            "## REFUSED: the arms differ in more than the provider",
            "",
            ("A provider comparison asserts that nothing else changed. These settings "
            "differ, so no comparison is printed. Re-run with matching settings, or "
            "pass `--allow-diff <key>` to accept a difference knowingly."),
            "",
        ]
        lines += _diff_table(blocking)
    elif waived:
        lines += [
            f"> **NOT A PROVIDER-ONLY COMPARISON: differs in {', '.join(sorted(waived))}.**",
            "> Any gap below mixes the provider effect with these differences.",
            "",
        ]
        lines += _diff_table(waived)

    if report["comparisons"]:
        lines += ["## Verdict", ""]
        lines += [f"- {result['headline']}" for result in report["comparisons"]]
        lines += [
            "",
            ("Delta is challenger minus baseline accuracy on the same questions. "
            "p is the exact McNemar (sign) test on questions where the arms disagree; "
            "Holm adjusts it for the number of pairs compared."),
            "",
        ]

    lines += ["## Validity", ""]
    if report["validity"]:
        lines += [
            f"- **{c['level']}** `{c['arm']}`: {c['message']}" for c in report["validity"]
        ]
        lines += ["", "Arms with an error are left out of every comparison.", ""]
    else:
        lines += ["All checks passed.", ""]

    lines += ["## Arms", ""]
    header = [
        "arm", "model", "n", "scored", "accuracy %", "95% CI", "answer in results %",
        "extraction gap", "cites %", "grounded %", "tools", "search", "fetch",
        "tokens", "agent s", "failed %", "fingerprint",
    ]
    rows = []
    for arm in report["arms"]:
        s = arm["summary"]
        low, high = s["accuracy_ci95"]
        rows.append([
            arm["name"] + ("" if arm["valid"] else " (invalid)"), arm["model"],
            str(s["n"]), str(s["n_scored"]), _pct(s["accuracy"]),
            f"{_pct(low)}-{_pct(high)}", _pct(s["answer_in_results"]),
            _pct(s["extraction_gap"]), _pct(s["citation_presence"]),
            _pct(s["citations_grounded"]), _num(s["mean_tool_calls"]),
            _num(s["mean_search_calls"]), _num(s["mean_fetch_calls"]),
            _num(s["mean_tokens"], 0), _num(s["mean_agent_seconds"]),
            _pct(s["failure_rate"]), f"`{arm['fingerprint']}`",
        ])
    lines += _table(header, rows)
    lines += [
        "",
        ("Arms with the same *fingerprint* ran with the same settings apart from "
        "the provider. *answer in results*: the gold answer appeared somewhere in "
        "what the tools returned. *extraction gap*: answer in results minus accuracy -- the provider "
        "found it, the model did not use it."),
        "",
    ]

    if report["comparisons"]:
        lines += ["## Paired comparisons", ""]
        header = ["baseline", "challenger", "pairs", "delta pts", "95% CI", "wins", "losses", "ties", "p", "p (Holm)", "dropped"]
        lines += _table(header, [
            [
                r["baseline"], r["challenger"], str(r["n_pairs"]),
                f"{r['delta'] * 100:+.1f}",
                f"{r['delta_ci95'][0] * 100:+.1f} to {r['delta_ci95'][1] * 100:+.1f}",
                str(r["challenger_wins"]), str(r["baseline_wins"]), str(r["ties"]),
                f"{r['p']:.3f}", f"{r['p_holm']:.3f}", str(len(r["dropped"])),
            ]
            for r in report["comparisons"]
        ])
        lines += [""]
        dropped = sorted({d for r in report["comparisons"] for d in r["dropped"]})
        if dropped:
            lines += [f"Dropped (scored in one arm only): {', '.join(dropped)}", ""]

    if report["slices"]:
        lines += [
            "## Slices",
            "",
            ("Exploratory: p-values here are not corrected across slices. "
            "Cells marked `*` have fewer questions than the small-n threshold."),
            "",
        ]
        current = None
        for item in report["slices"]:
            if item["field"] != current:
                current = item["field"]
                lines += [f"### by `{current}`", ""]
            arms = ", ".join(
                f"{name} {_pct(cell['accuracy'])}% (n={cell['n']}{'*' if cell['small'] else ''})"
                for name, cell in item["arms"].items()
            )
            lines += [f"- **{item['value']}**: {arms}"]
            for r in item["pairs"]:
                lines += [
                    (f"  - {r['challenger']} vs {r['baseline']}: {r['delta'] * 100:+.1f} pts, "
                    f"p={r['p']:.3f}, n={r['n_pairs']}{'*' if r['small'] else ''}")
                ]
        lines += [""]
    return "\n".join(lines)


def _diff_table(diff: dict[str, dict[str, object]]) -> list[str]:
    arms = sorted({arm for values in diff.values() for arm in values})
    rows = [[f"`{key}`", *(f"`{values.get(arm)}`" for arm in arms)] for key, values in diff.items()]
    return [*_table(["setting", *arms], rows), ""]
