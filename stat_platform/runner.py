from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any, Sequence

from .adjustment import adjust_intervals, adjust_p_values
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import GroupedPermutationTest
from .storage import JsonStore, connect, dumps, loads


def _tupleize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_tupleize(item) for item in value)
    return value


class PlatformRunner:
    def __init__(self, database: str | Path = "experiments.json"):
        self.database = Path(database)
        self.store: JsonStore = connect(self.database)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "PlatformRunner":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def create_batch(
        self,
        dataset: Dataset,
        *,
        name: str | None = None,
        specs: Sequence[TestSpec] | None = None,
        samples: int = 10_000,
        max_exact_assignments: int = 100_000,
        seed: int | None = None,
    ) -> int:
        dataset_id = self._store_dataset(dataset)
        batch_id = self.store.insert(
            "batches",
            {
                "dataset_id": dataset_id,
                "name": name or f"{dataset.name}:{dataset.version}",
                "seed": seed,
                "rng_state_json": None,
                "random_draws": 0,
            },
        )
        for spec in specs or ():
            self.add_test(
                batch_id,
                spec,
                samples=samples,
                max_exact_assignments=max_exact_assignments,
                seed=seed,
            )
        self.store.commit()
        return batch_id

    def add_test(
        self,
        batch_id: int,
        spec: TestSpec,
        *,
        samples: int = 10_000,
        max_exact_assignments: int = 100_000,
        seed: int | None = None,
    ) -> int:
        test_id = self.store.insert(
            "tests",
            {
                "batch_id": batch_id,
                "test_type": spec.test_type,
                "null_hypothesis": spec.null_hypothesis,
                "settings_json": dumps(spec.settings),
                "seed": seed,
                "samples": samples,
                "max_exact_assignments": max_exact_assignments,
            },
        )
        self.store.insert(
            "run_states",
            {
                "test_id": test_id,
                "evaluated_assignments": 0,
                "as_extreme_count": 0,
                "random_draws": 0,
                "rng_state_json": None,
                "looks": 0,
                "stopping_reason": None,
            },
        )
        self.store.commit()
        return test_id

    def _engine_for_test(self, test: dict[str, Any]) -> tuple[GroupedPermutationTest, dict[str, Any]]:
        batch = self.store.find_one("batches", id=int(test["batch_id"]))
        dataset_row = self.store.find_one("datasets", id=int(batch["dataset_id"]))
        dataset = self._dataset_from_row(dataset_row)
        settings = loads(test["settings_json"])
        spec = TestSpec(test["test_type"], test["null_hypothesis"], settings)
        return self._engine(dataset, spec), batch

    def _is_exact(self, test: dict[str, Any], engine: GroupedPermutationTest) -> bool:
        return engine.total_assignments <= int(test["max_exact_assignments"])

    def _target_samples(self, test: dict[str, Any], engine: GroupedPermutationTest) -> int:
        if self._is_exact(test, engine):
            return engine.total_assignments
        return min(int(test["samples"]), engine.total_assignments)

    def _batch_permutation_rows(self, batch_id: int) -> list[dict[str, Any]]:
        return sorted(
            self.store.find_many("batch_permutations", batch_id=batch_id),
            key=lambda row: int(row["ordinal"]),
        )

    def _ensure_batch_permutations(
        self,
        batch: dict[str, Any],
        engine: GroupedPermutationTest,
        count: int,
        *,
        exact: bool,
    ) -> None:
        """Grow the shared permutation pool; never re-generate existing ones."""
        batch_id = int(batch["id"])
        existing = self._batch_permutation_rows(batch_id)
        missing = count - len(existing)
        if missing <= 0:
            return
        ordinal = len(existing)
        if exact:
            assignments = itertools.islice(
                engine.iter_all_assignments(), ordinal, ordinal + missing
            )
            for assignment in assignments:
                ordinal += 1
                self.store.insert(
                    "batch_permutations",
                    {
                        "batch_id": batch_id,
                        "ordinal": ordinal,
                        "assignment_key": dumps(list(assignment)),
                    },
                )
            return
        rng = random.Random(batch.get("seed"))
        if batch.get("rng_state_json") is not None:
            rng.setstate(_tupleize(loads(batch["rng_state_json"])))
        seen = {row["assignment_key"] for row in existing}
        draws = int(batch.get("random_draws") or 0)
        while missing > 0:
            assignment = engine.draw_random_assignment(rng)
            draws += 1
            key = dumps(list(assignment))
            if key in seen:
                continue
            seen.add(key)
            ordinal += 1
            missing -= 1
            self.store.insert(
                "batch_permutations",
                {"batch_id": batch_id, "ordinal": ordinal, "assignment_key": key},
            )
        batch["rng_state_json"] = dumps(rng.getstate())
        batch["random_draws"] = draws

    def run_batch(
        self,
        batch_id: int,
        *,
        max_new_samples: int | None = None,
    ) -> list[TestResult]:
        rows = sorted(self.store.find_many("tests", batch_id=batch_id), key=lambda row: int(row["id"]))
        return [
            self.run_test(int(row["id"]), max_new_samples=max_new_samples)
            for row in rows
        ]

    def run_test(
        self,
        test_id: int,
        *,
        max_new_samples: int | None = None,
    ) -> TestResult:
        if max_new_samples is None:
            while self.advance_test(test_id) is None:
                pass
            return self.get_result(test_id)
        result = self.advance_test(test_id, max_new_samples=max_new_samples)
        if result is None:
            raise RuntimeError("test made one persisted chunk but is not complete; resume it later")
        return result

    def advance_test(
        self,
        test_id: int,
        *,
        max_new_samples: int | None = None,
    ) -> TestResult | None:
        completed_row = self.store.find_one("test_results", test_id=test_id)
        if completed_row is not None:
            result = self._result_from_dict(loads(completed_row["result_json"]))
            return self._test_result(test_id, result)

        test = self.store.find_one("tests", id=test_id)
        if test is None:
            raise KeyError(f"unknown test {test_id}")
        engine, batch = self._engine_for_test(test)
        exact = self._is_exact(test, engine)
        target = self._target_samples(test, engine)
        state = self.store.find_one("run_states", test_id=test_id)
        evaluated = int(state["evaluated_assignments"])
        hits = int(state["as_extreme_count"])
        looks = int(state.get("looks") or 0)

        budget = target - evaluated
        if max_new_samples is not None:
            budget = min(budget, max_new_samples)
        if budget > 0:
            self._ensure_batch_permutations(
                batch, engine, evaluated + budget, exact=exact
            )
            shared_rows = self._batch_permutation_rows(int(batch["id"]))
            ordinal = len(self.store.find_many("permutation_samples", test_id=test_id))
            for row in shared_rows[evaluated : evaluated + budget]:
                assignment = tuple(json.loads(row["assignment_key"]))
                record = engine.evaluate_assignment(assignment)
                ordinal += 1
                hits += int(record.as_extreme)
                self.store.insert(
                    "permutation_samples",
                    {
                        "test_id": test_id,
                        "ordinal": ordinal,
                        "batch_ordinal": int(row["ordinal"]),
                        "assignment_key": row["assignment_key"],
                        "statistic": record.statistic,
                        "as_extreme": int(record.as_extreme),
                    },
                )
            evaluated += budget
            looks += 1

        settings = loads(test["settings_json"])
        adaptive = settings.get("adaptive") or {}
        stopping_reason: str | None = None
        complete = False
        if evaluated >= target:
            complete = True
            stopping_reason = "exact_enumeration" if exact else "target_samples"
        elif not exact and adaptive:
            min_samples = int(adaptive.get("min_samples", 1))
            half_width = adaptive.get("ci_half_width")
            if half_width is not None and evaluated >= min_samples:
                inference = engine.monte_carlo_inference(hits, evaluated)
                width = inference["confidence_interval"][1] - inference["confidence_interval"][0]
                if width / 2.0 <= float(half_width):
                    complete = True
                    stopping_reason = "precision_target"

        state.update(
            {
                "evaluated_assignments": evaluated,
                "as_extreme_count": hits,
                "looks": looks,
                "stopping_reason": stopping_reason,
            }
        )
        self.store.commit()
        if not complete:
            return None
        if exact:
            result = PermutationResult(
                observed_statistic=engine.observed_statistic,
                p_value=hits / engine.total_assignments,
                method="exact",
                total_assignments=engine.total_assignments,
                evaluated_assignments=evaluated,
                monte_carlo_draws=0,
                as_extreme_count=hits,
                looks=looks,
                stopping_reason=stopping_reason,
            )
        else:
            inference = engine.monte_carlo_inference(hits, evaluated, looks=looks)
            result = PermutationResult(
                observed_statistic=engine.observed_statistic,
                p_value=inference["p_value"],
                method="monte-carlo",
                total_assignments=engine.total_assignments,
                evaluated_assignments=evaluated,
                monte_carlo_draws=int(batch.get("random_draws") or 0),
                standard_error=inference["standard_error"],
                confidence_level=inference["confidence_level"],
                confidence_interval=inference["confidence_interval"],
                unadjusted_confidence_interval=inference[
                    "unadjusted_confidence_interval"
                ],
                as_extreme_count=hits,
                seed=batch.get("seed"),
                looks=looks,
                stopping_reason=stopping_reason,
            )
        self.store.insert(
            "test_results",
            {"test_id": test_id, "result_json": dumps(result.to_dict())},
        )
        self.store.commit()
        return self._test_result(test_id, result)

    def compare_batch(
        self,
        batch_id: int,
        *,
        method: str = "holm",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Unadjusted and multiplicity-adjusted conclusions for one batch."""
        rows = sorted(
            self.store.find_many("tests", batch_id=batch_id),
            key=lambda row: int(row["id"]),
        )
        completed: list[TestResult] = []
        pending: list[int] = []
        for row in rows:
            result_row = self.store.find_one("test_results", test_id=int(row["id"]))
            if result_row is None:
                pending.append(int(row["id"]))
                continue
            completed.append(
                self._test_result(int(row["id"]), self._result_from_dict(loads(result_row["result_json"])))
            )
        p_values = [item.result.p_value for item in completed]
        intervals = [item.result.confidence_interval for item in completed]
        adjusted = adjust_p_values(p_values, method) if p_values else []
        adjusted_intervals = adjust_intervals(p_values, intervals, adjusted)
        comparisons = []
        for item, adjusted_p, adjusted_interval in zip(
            completed, adjusted, adjusted_intervals
        ):
            comparisons.append(
                {
                    "test_id": item.test_id,
                    "null_hypothesis": item.null_hypothesis,
                    "method": item.result.method,
                    "looks": item.result.looks,
                    "stopping_reason": item.result.stopping_reason,
                    "p_value": item.result.p_value,
                    "confidence_interval": item.result.confidence_interval,
                    "confidence_level": item.result.confidence_level,
                    "adjusted_p_value": adjusted_p,
                    "adjusted_confidence_interval": adjusted_interval,
                    "reject_unadjusted": item.result.p_value <= alpha,
                    "reject_adjusted": adjusted_p <= alpha,
                }
            )
        return {
            "batch_id": batch_id,
            "adjustment_method": method,
            "alpha": alpha,
            "pending_tests": pending,
            "comparisons": comparisons,
        }

    def get_result(self, test_id: int) -> TestResult:
        row = self.store.find_one("test_results", test_id=test_id)
        if row is None:
            raise KeyError(f"test {test_id} has not completed")
        return self._test_result(test_id, self._result_from_dict(loads(row["result_json"])))

    def _store_dataset(self, dataset: Dataset) -> int:
        payload = {
            "name": dataset.name,
            "version": dataset.version,
            "records": dataset.records,
            "group_column": dataset.group_column,
            "condition_column": dataset.condition_column,
            "value_column": dataset.value_column,
        }
        data_hash = hashlib.sha256(dumps(payload).encode("utf-8")).hexdigest()
        previous = self.store.find_one("datasets", data_hash=data_hash)
        if previous is not None:
            return int(previous["id"])
        return self.store.insert(
            "datasets",
            {
                "name": dataset.name,
                "version": dataset.version,
                "data_hash": data_hash,
                "group_column": dataset.group_column,
                "condition_column": dataset.condition_column,
                "value_column": dataset.value_column,
                "records_json": dumps(dataset.records),
            },
        )

    @staticmethod
    def _engine(dataset: Dataset, spec: TestSpec) -> GroupedPermutationTest:
        if spec.test_type == GroupedPermutationTest.test_type:
            return GroupedPermutationTest(dataset, spec)
        raise ValueError(f"unknown test type: {spec.test_type}")

    @staticmethod
    def _dataset_from_row(row: dict[str, Any]) -> Dataset:
        return Dataset(
            name=row["name"],
            records=tuple(loads(row["records_json"])),
            version=row["version"],
            group_column=row["group_column"],
            condition_column=row["condition_column"],
            value_column=row["value_column"],
        )

    @staticmethod
    def _result_from_dict(value: object) -> PermutationResult:
        data = dict(value)
        confidence_interval = data.get("confidence_interval")
        if confidence_interval is not None:
            data["confidence_interval"] = tuple(confidence_interval)
        unadjusted = data.get("unadjusted_confidence_interval")
        if unadjusted is not None:
            data["unadjusted_confidence_interval"] = tuple(unadjusted)
        return PermutationResult(**data)

    def _test_result(self, test_id: int, result: PermutationResult) -> TestResult:
        row = self.store.find_one("tests", id=test_id)
        return TestResult(
            test_id=test_id,
            batch_id=int(row["batch_id"]),
            null_hypothesis=row["null_hypothesis"],
            result=result,
        )
