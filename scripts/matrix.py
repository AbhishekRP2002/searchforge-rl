"""Summarise Verifiers traces into comparable model/provider cells."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


class VoidedRunError(Exception):
    """The run cannot be used for a provider comparison."""


def read_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _wilson(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for binary outcomes, stable at 0% and 100%."""
    if not values:
        return (0.0, 0.0)
    n = len(values)
    proportion = _mean(values)
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _reward_score(trace: dict, name: str) -> float:
    reward = trace.get("rewards", {}).get(name)
    if reward is None:
        return 0.0
    if not isinstance(reward, dict) or "score" not in reward:
        raise ValueError(f"reward {name!r} does not match the Verifiers v1 trace schema")
    return float(reward["score"])


def _source_id(trace: dict) -> str:
    return str(trace["task"]["data"]["source_id"])


def summarise(traces: list[dict]) -> dict:
    native = sum(
        float(trace.get("metrics", {}).get("native_search_calls") or 0.0)
        for trace in traces
    )
    if native:
        raise VoidedRunError(
            f"{native:.0f} native provider search calls bypassed the SearchForge toolset"
        )

    scored = [trace for trace in traces if trace.get("ok", False)]
    correct = [_reward_score(trace, "searchforge") for trace in scored]
    available = [_reward_score(trace, "answer_in_results") for trace in scored]
    failures = Counter(
        (trace.get("errors") or [{"type": "UnknownError"}])[-1]["type"]
        for trace in traces
        if not trace.get("ok", False)
    )
    return {
        "n": len(traces),
        "n_scored": len(scored),
        "accuracy": _mean(correct),
        "accuracy_ci95": _wilson(correct),
        "answer_in_results": _mean(available),
        "extraction_gap": _mean([a - c for a, c in zip(available, correct)]),
        "citations_grounded": _mean(
            [_reward_score(trace, "citations_grounded") for trace in scored]
        ),
        "citation_presence": _mean(
            [_reward_score(trace, "citation_presence") for trace in scored]
        ),
        "mean_tool_calls": _mean(
            [float(trace.get("metrics", {}).get("num_tool_calls") or 0.0) for trace in scored]
        ),
        "mean_search_calls": _mean(
            [
                float(trace.get("metrics", {}).get("num_search_calls") or 0.0)
                for trace in scored
            ]
        ),
        "mean_fetch_calls": _mean(
            [
                float(trace.get("metrics", {}).get("num_fetch_calls") or 0.0)
                for trace in scored
            ]
        ),
        "mean_tokens": _mean(
            [
                float(trace.get("metrics", {}).get("tokens_consumed") or 0.0)
                for trace in scored
            ]
        ),
        "failure_rate": (len(traces) - len(scored)) / len(traces) if traces else 0.0,
        "failures": dict(sorted(failures.items())),
    }


def pair(cells: dict[str, list[dict]]) -> dict:
    by_arm = {
        arm: {
            _source_id(trace): _reward_score(trace, "searchforge")
            for trace in traces
            if trace.get("ok", False)
        }
        for arm, traces in cells.items()
    }
    common = set.intersection(*(set(rows) for rows in by_arm.values())) if by_arm else set()
    every = set().union(*(set(rows) for rows in by_arm.values())) if by_arm else set()
    return {
        "paired_source_ids": sorted(common),
        "dropped": sorted(every - common),
        "discordant": sum(
            len({by_arm[arm][source_id] for arm in by_arm}) > 1
            for source_id in common
        ),
    }


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def compare(
    cells: dict[str, list[dict]], *, baseline: str, challenger: str
) -> dict:
    paired = pair(cells)
    by_arm = {
        arm: {
            _source_id(trace): _reward_score(trace, "searchforge")
            for trace in traces
            if trace.get("ok", False)
        }
        for arm, traces in cells.items()
    }
    source_ids = paired["paired_source_ids"]
    differences = [
        by_arm[challenger][source_id] - by_arm[baseline][source_id]
        for source_id in source_ids
    ]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    ties = sum(value == 0 for value in differences)
    delta = _mean(differences)
    if len(differences) > 1:
        variance = sum((value - delta) ** 2 for value in differences) / (
            len(differences) - 1
        )
        margin = 1.96 * math.sqrt(variance / len(differences))
    else:
        margin = 0.0
    return {
        "baseline": baseline,
        "challenger": challenger,
        "n_pairs": len(source_ids),
        "dropped": paired["dropped"],
        "challenger_minus_baseline": delta,
        "delta_ci95": (delta - margin, delta + margin),
        "challenger_wins": wins,
        "baseline_wins": losses,
        "ties": ties,
        "mcnemar_exact_p": _mcnemar_exact(wins, losses),
    }


COLUMNS = (
    "n",
    "n_scored",
    "accuracy",
    "accuracy_ci95",
    "answer_in_results",
    "extraction_gap",
    "citation_presence",
    "citations_grounded",
    "mean_tool_calls",
    "mean_search_calls",
    "mean_fetch_calls",
    "mean_tokens",
    "failure_rate",
)


def render_table(summaries: dict[tuple[str, str], dict]) -> str:
    rows = [" | ".join(("model", "provider", *COLUMNS))]
    for (model, provider), summary in sorted(summaries.items()):
        values = [
            (
                f"[{summary[column][0]:.3f}, {summary[column][1]:.3f}]"
                if column == "accuracy_ci95"
                else f"{summary[column]:.3f}"
                if isinstance(summary[column], float)
                else str(summary[column])
            )
            for column in COLUMNS
        ]
        rows.append(" | ".join((model, provider, *values)))
    return "\n".join(rows)


def render_comparisons(models: dict[str, dict[str, list[dict]]]) -> str:
    lines = [
        "model | paired providers | n_pairs | challenger_minus_baseline | "
        "delta_ci95 | challenger_wins | baseline_wins | ties | mcnemar_exact_p | dropped"
    ]
    for model, cells in sorted(models.items()):
        if len(cells) != 2:
            raise ValueError(
                f"paired comparison for {model!r} needs exactly two providers"
            )
        providers = sorted(cells)
        baseline = "serper" if "serper" in cells else providers[0]
        challenger = next(provider for provider in providers if provider != baseline)
        result = compare(cells, baseline=baseline, challenger=challenger)
        interval = result["delta_ci95"]
        lines.append(
            " | ".join(
                (
                    model,
                    f"{baseline} -> {challenger}",
                    str(result["n_pairs"]),
                    f"{result['challenger_minus_baseline']:.3f}",
                    f"[{interval[0]:.3f}, {interval[1]:.3f}]",
                    str(result["challenger_wins"]),
                    str(result["baseline_wins"]),
                    str(result["ties"]),
                    f"{result['mcnemar_exact_p']:.3f}",
                    str(len(result["dropped"])),
                )
            )
        )
    return "\n".join(lines)


def main() -> None:
    import sys

    summaries = {}
    cells: dict[str, dict[str, list[dict]]] = {}
    for argument in sys.argv[1:]:
        model, provider, path = argument.split("=", 2)
        traces = read_traces(Path(path))
        summaries[(model, provider)] = summarise(traces)
        cells.setdefault(model, {})[provider] = traces
    print(render_table(summaries))
    print()
    print(render_comparisons(cells))


if __name__ == "__main__":
    main()
