from __future__ import annotations

from typing import Sequence

AdjustmentMethod = str

METHODS = ("bonferroni", "holm", "benjamini-hochberg")


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def bonferroni(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    return [_clip(m * p) for p in p_values]


def holm(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, _clip((m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running = min(running, _clip(p_values[index] * m / (rank + 1)))
        adjusted[index] = running
    return adjusted


def adjust_p_values(p_values: Sequence[float], method: AdjustmentMethod) -> list[float]:
    if method == "bonferroni":
        return bonferroni(p_values)
    if method == "holm":
        return holm(p_values)
    if method == "benjamini-hochberg":
        return benjamini_hochberg(p_values)
    raise ValueError(f"unknown adjustment method: {method}")


def adjust_intervals(
    p_values: Sequence[float],
    intervals: Sequence[tuple[float, float] | None],
    adjusted: Sequence[float],
) -> list[tuple[float, float] | None]:
    """Propagate each p-value's adjustment factor to its uncertainty range."""
    scaled: list[tuple[float, float] | None] = []
    for p_value, interval, adjusted_p in zip(p_values, intervals, adjusted):
        if interval is None:
            scaled.append(None)
            continue
        factor = adjusted_p / p_value if p_value > 0 else 1.0
        scaled.append((_clip(interval[0] * factor), _clip(interval[1] * factor)))
    return scaled
