from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Alternative = Literal["two-sided", "greater", "less"]
RunMethod = Literal["exact", "monte-carlo"]


@dataclass(frozen=True)
class Dataset:
    name: str
    records: tuple[dict[str, Any], ...]
    version: str
    group_column: str = "group"
    condition_column: str = "condition"
    value_column: str = "value"

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        name: str = "dataset",
        version: str | None = None,
        group_column: str = "group",
        condition_column: str = "condition",
        value_column: str = "value",
    ) -> "Dataset":
        return cls(
            name=name,
            records=tuple(records),
            version=version or "working",
            group_column=group_column,
            condition_column=condition_column,
            value_column=value_column,
        )


@dataclass(frozen=True)
class TestSpec:
    test_type: str
    null_hypothesis: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PermutationResult:
    observed_statistic: float
    p_value: float
    method: RunMethod
    total_assignments: int
    evaluated_assignments: int
    monte_carlo_draws: int
    standard_error: float | None = None
    confidence_level: float | None = None
    confidence_interval: tuple[float, float] | None = None
    as_extreme_count: int = 0
    seed: int | None = None
    stopping_rule: str = "fixed"
    adaptive: bool = False
    looks: int = 0
    stopped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "observed_statistic": self.observed_statistic,
            "p_value": self.p_value,
            "method": self.method,
            "total_assignments": self.total_assignments,
            "evaluated_assignments": self.evaluated_assignments,
            "monte_carlo_draws": self.monte_carlo_draws,
            "as_extreme_count": self.as_extreme_count,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "confidence_interval": list(self.confidence_interval)
            if self.confidence_interval is not None
            else None,
            "seed": self.seed,
            "stopping_rule": self.stopping_rule,
            "adaptive": self.adaptive,
            "looks": self.looks,
            "stopped_reason": self.stopped_reason,
        }
        return result


@dataclass(frozen=True)
class TestResult:
    test_id: int
    batch_id: int
    null_hypothesis: str
    result: PermutationResult
