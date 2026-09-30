from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from .models import Alternative, Dataset, PermutationResult, RunMethod, TestSpec


def _multinomial_arrangements(labels: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Yield distinct arrangements while preserving each group's label counts."""
    unique_labels = tuple(dict.fromkeys(labels))
    counts = [labels.count(label) for label in unique_labels]
    size = len(labels)

    def build(index: int, remaining: tuple[int, ...], arrangement: list[str | None]):
        if index == len(unique_labels) - 1:
            for position in remaining:
                arrangement[position] = unique_labels[index]
            yield tuple(arrangement)
            return

        for chosen in itertools.combinations(remaining, counts[index]):
            chosen_set = set(chosen)
            next_arrangement = arrangement.copy()
            for position in chosen:
                next_arrangement[position] = unique_labels[index]
            yield from build(
                index + 1,
                tuple(position for position in remaining if position not in chosen_set),
                next_arrangement,
            )

    yield from build(0, tuple(range(size)), [None] * size)


@dataclass(frozen=True)
class AssignmentRecord:
    assignment: tuple[str | None, ...]
    statistic: float
    as_extreme: bool


@dataclass
class PermutationRun:
    result: PermutationResult | None
    records: tuple[AssignmentRecord, ...]
    rng_state: tuple[Any, ...] | None
    random_draws: int
    evaluated_assignments: int
    as_extreme_count: int
    complete: bool


class GroupedPermutationTest:
    test_type = "grouped_permutation_mean_difference"

    def __init__(self, dataset: Dataset, spec: TestSpec | None = None):
        self.dataset = dataset
        self.spec = spec or TestSpec(
            test_type=self.test_type,
            null_hypothesis="Exchangeable condition labels within groups; condition means are equal.",
        )
        if self.spec.test_type != self.test_type:
            raise ValueError(f"Unsupported test type: {self.spec.test_type}")
        self.alternative: Alternative = self.spec.settings.get("alternative", "two-sided")
        if self.alternative not in {"two-sided", "greater", "less"}:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        self._prepare()

    def _prepare(self) -> None:
        records = self.dataset.records
        required = {
            self.dataset.group_column,
            self.dataset.condition_column,
            self.dataset.value_column,
        }
        if not records:
            raise ValueError("dataset must contain at least one record")
        missing = required.difference(records[0])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        values: list[float] = []
        observed_labels: list[str] = []
        groups: dict[Any, list[int]] = defaultdict(list)
        all_labels: set[str] = set()

        for index, row in enumerate(records):
            group = row[self.dataset.group_column]
            label = str(row[self.dataset.condition_column])
            value = float(row[self.dataset.value_column])
            values.append(value)
            observed_labels.append(label)
            groups[group].append(index)
            all_labels.add(label)

        configured = self.spec.settings.get("conditions")
        self.conditions = tuple(configured or sorted(all_labels))
        if len(self.conditions) != 2:
            raise ValueError("this test requires exactly two condition labels")
        if not all_labels.issubset(self.conditions):
            raise ValueError("records contain condition labels absent from settings.conditions")
        if any(len(indexes) < 2 for indexes in groups.values()):
            raise ValueError("every group must contain at least two linked observations")
        if any(len({observed_labels[i] for i in indexes}) == 1 for indexes in groups.values()):
            raise ValueError("every group must contain both condition labels")

        self.values = tuple(values)
        self.observed_labels = tuple(observed_labels)
        self.group_indexes = tuple(tuple(indexes) for indexes in groups.values())
        self.group_label_sets = tuple(
            tuple(observed_labels[i] for i in indexes) for indexes in self.group_indexes
        )
        self.observed_statistic = self._statistic(self.observed_labels)
        self.total_assignments = math.prod(
            math.factorial(len(labels))
            // math.prod(math.factorial(labels.count(level)) for level in set(labels))
            for labels in self.group_label_sets
        )

    def _statistic(self, labels: Sequence[str]) -> float:
        sums = {label: 0.0 for label in self.conditions}
        counts = {label: 0 for label in self.conditions}
        for label, value in zip(labels, self.values):
            sums[label] += value
            counts[label] += 1
        if counts[self.conditions[0]] == 0 or counts[self.conditions[1]] == 0:
            raise ValueError("both conditions must remain present after permutation")
        return sums[self.conditions[0]] / counts[self.conditions[0]] - (
            sums[self.conditions[1]] / counts[self.conditions[1]]
        )

    def _as_extreme(self, statistic: float) -> bool:
        if self.alternative == "greater":
            return statistic >= self.observed_statistic
        if self.alternative == "less":
            return statistic <= self.observed_statistic
        return abs(statistic) >= abs(self.observed_statistic)

    def _assignment(self, group_arrangements: Sequence[Sequence[str]]) -> tuple[str, ...]:
        labels = [None] * len(self.values)
        for indexes, arrangement in zip(self.group_indexes, group_arrangements):
            for index, label in zip(indexes, arrangement):
                labels[index] = label
        return tuple(labels)

    def _all_arrangements(self) -> Iterator[tuple[str, ...]]:
        group_iterators = [
            _multinomial_arrangements(labels) for labels in self.group_label_sets
        ]
        for group_arrangements in itertools.product(*group_iterators):
            yield self._assignment(group_arrangements)

    def _record(self, assignment: Sequence[str]) -> AssignmentRecord:
        statistic = self._statistic(assignment)
        return AssignmentRecord(tuple(assignment), statistic, self._as_extreme(statistic))

    def random_assignment(self, rng: random.Random) -> tuple[str, ...]:
        return self._assignment(
            [rng.sample(labels, k=len(labels)) for labels in self.group_label_sets]
        )

    def evaluate(self, assignment: Sequence[str]) -> AssignmentRecord:
        return self._record(assignment)

    def stopping_settings(self) -> dict[str, Any]:
        return dict(self.spec.settings.get("stopping") or {"type": "fixed"})

    def run(
        self,
        *,
        samples: int = 10_000,
        max_exact_assignments: int = 100_000,
        seed: int | None = None,
        existing_records: Iterable[AssignmentRecord] = (),
        evaluated_assignments: int = 0,
        as_extreme_count: int | None = None,
        random_draws: int = 0,
        rng_state: tuple[Any, ...] | None = None,
        max_new_samples: int | None = None,
    ) -> PermutationRun:
        if samples < 1:
            raise ValueError("samples must be positive")
        use_exact = self.total_assignments <= max_exact_assignments
        method: RunMethod = "exact" if use_exact else "monte-carlo"

        existing = tuple(existing_records)
        hits = sum(record.as_extreme for record in existing)
        if as_extreme_count is not None:
            hits = as_extreme_count
        evaluated = evaluated_assignments if evaluated_assignments else len(existing)

        if use_exact:
            return self._run_exact(evaluated, hits, max_new_samples)

        target = min(samples, self.total_assignments)
        return self._run_monte_carlo(
            target=target,
            seed=seed,
            existing=existing,
            evaluated=evaluated,
            hits=hits,
            random_draws=random_draws,
            rng_state=rng_state,
            max_new_samples=max_new_samples,
        )

    def _run_exact(
        self,
        evaluated: int,
        hits: int,
        max_new_samples: int | None,
    ) -> PermutationRun:
        records: list[AssignmentRecord] = []
        assignments = itertools.islice(self._all_arrangements(), evaluated, None)
        if max_new_samples is not None:
            assignments = itertools.islice(assignments, max_new_samples)
        for assignment in assignments:
            record = self._record(assignment)
            records.append(record)
            if record.as_extreme:
                hits += 1
            evaluated += 1
        complete = evaluated == self.total_assignments
        result = None
        if complete:
            result = PermutationResult(
                observed_statistic=self.observed_statistic,
                p_value=hits / self.total_assignments,
                method="exact",
                total_assignments=self.total_assignments,
                evaluated_assignments=evaluated,
                monte_carlo_draws=0,
                as_extreme_count=hits,
            )
        return PermutationRun(
            result,
            tuple(records),
            None,
            0,
            evaluated,
            hits,
            complete,
        )

    def _run_monte_carlo(
        self,
        *,
        target: int,
        seed: int | None,
        existing: Sequence[AssignmentRecord],
        evaluated: int,
        hits: int,
        random_draws: int,
        rng_state: tuple[Any, ...] | None,
        max_new_samples: int | None,
    ) -> PermutationRun:
        rng = random.Random(seed)
        if rng_state is not None:
            rng.setstate(rng_state)
        seen = {record.assignment for record in existing}
        records: list[AssignmentRecord] = []

        produced = 0
        while len(seen) < target and (
            max_new_samples is None or produced < max_new_samples
        ):
            group_arrangements = [
                rng.sample(labels, k=len(labels)) for labels in self.group_label_sets
            ]
            random_draws += 1
            assignment = self._assignment(group_arrangements)
            if assignment in seen:
                continue
            record = self._record(assignment)
            seen.add(assignment)
            produced += 1
            records.append(record)
            if record.as_extreme:
                hits += 1

        evaluated = len(seen)
        if evaluated < target:
            return PermutationRun(
                None,
                tuple(records),
                rng.getstate(),
                random_draws,
                evaluated,
                hits,
                False,
            )
        p_value = (hits + 1) / (evaluated + 1)
        standard_error = math.sqrt(max(0.0, p_value * (1 - p_value) / (evaluated + 2)))
        z = 1.959963984540054
        confidence_interval = (
            min(1.0, max(0.0, p_value - z * standard_error)),
            min(1.0, max(0.0, p_value + z * standard_error)),
        )
        result = PermutationResult(
            observed_statistic=self.observed_statistic,
            p_value=p_value,
            method="monte-carlo",
            total_assignments=self.total_assignments,
            evaluated_assignments=evaluated,
            monte_carlo_draws=random_draws,
            standard_error=standard_error,
            confidence_level=0.95,
            confidence_interval=confidence_interval,
            as_extreme_count=hits,
            seed=seed,
        )
        return PermutationRun(
            result,
            tuple(records),
            rng.getstate(),
            random_draws,
            evaluated,
            hits,
            True,
        )
