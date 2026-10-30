"""Small, dependency-free statistics for paired provider comparisons."""

from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def wilson(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a proportion, stable at 0% and 100%."""
    if not values:
        return (0.0, 0.0)
    n = len(values)
    proportion = mean(values)
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact test on the discordant pairs.

    With one binary rollout per question this is exactly McNemar's exact test.
    With several rollouts averaged per question it is the sign test, its direct
    generalisation: ties drop out, and the question is only which side won more.
    """
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in input order.

    Seven providers make 21 pairs; at 5% each, one spurious "win" is expected
    by chance alone. Holm controls that without Bonferroni's full penalty.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def paired(baseline: dict[str, float], challenger: dict[str, float]) -> dict:
    """Join two arms on source_id and test challenger minus baseline."""
    common = sorted(set(baseline) & set(challenger))
    differences = [challenger[key] - baseline[key] for key in common]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    delta = mean(differences)
    if len(differences) > 1:
        variance = sum((value - delta) ** 2 for value in differences) / (
            len(differences) - 1
        )
        margin = 1.96 * math.sqrt(variance / len(differences))
    else:
        margin = 0.0
    return {
        "n_pairs": len(common),
        "dropped": sorted(set(baseline) ^ set(challenger)),
        "delta": delta,
        "delta_ci95": (delta - margin, delta + margin),
        "challenger_wins": wins,
        "baseline_wins": losses,
        "ties": len(common) - wins - losses,
        "p": sign_test(wins, losses),
    }
