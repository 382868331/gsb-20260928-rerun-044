from __future__ import annotations

from typing import Sequence


def bonferroni(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def holm(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def adjust(p_values: Sequence[float], rule: str) -> list[float]:
    if rule == "none":
        return [float(p) for p in p_values]
    if rule == "bonferroni":
        return bonferroni(p_values)
    if rule == "holm":
        return holm(p_values)
    raise ValueError(f"unknown adjustment rule: {rule}")
