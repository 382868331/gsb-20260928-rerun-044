from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

from .adjustment import adjust
from .models import Dataset, PermutationResult, TestResult, TestSpec
from .permutation import AssignmentRecord, GroupedPermutationTest
from .storage import JsonStore, connect, dumps, loads


def _restore_rng_state(value: object) -> tuple:
    version, internal, gauss = value
    return (version, tuple(internal), gauss)


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
            },
        )
        self.store.insert(
            "batch_states",
            {"batch_id": batch_id, "rng_state_json": None, "random_draws": 0},
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
            },
        )
        self.store.commit()
        return test_id

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
        batch = self.store.find_one("batches", id=int(test["batch_id"]))
        dataset_row = self.store.find_one("datasets", id=int(batch["dataset_id"]))
        dataset = self._dataset_from_row(dataset_row)
        settings = loads(test["settings_json"])
        spec = TestSpec(test["test_type"], test["null_hypothesis"], settings)
        engine = self._engine(dataset, spec)
        existing_rows = sorted(
            self.store.find_many("permutation_samples", test_id=test_id),
            key=lambda row: int(row["ordinal"]),
        )
        state = self.store.find_one("run_states", test_id=test_id)
        if engine.total_assignments > int(test["max_exact_assignments"]):
            return self._advance_monte_carlo(
                test, engine, spec, state, max_new_samples
            )
        existing = tuple(
            AssignmentRecord(
                tuple(json.loads(row["assignment_key"])),
                float(row["statistic"]),
                bool(row["as_extreme"]),
            )
            for row in existing_rows
        )
        rng_state = (
            _restore_rng_state(loads(state["rng_state_json"]))
            if state["rng_state_json"] is not None
            else None
        )
        run = engine.run(
            samples=int(test["samples"]),
            max_exact_assignments=int(test["max_exact_assignments"]),
            seed=test["seed"],
            existing_records=existing,
            evaluated_assignments=int(state["evaluated_assignments"]),
            as_extreme_count=int(state["as_extreme_count"]),
            random_draws=int(state["random_draws"]),
            rng_state=rng_state,
            max_new_samples=max_new_samples,
        )
        self._append_samples(test_id, len(existing), run.records)
        state.update(
            {
                "evaluated_assignments": run.evaluated_assignments,
                "as_extreme_count": run.as_extreme_count,
                "random_draws": run.random_draws,
                "rng_state_json": dumps(run.rng_state) if run.rng_state is not None else None,
            }
        )
        self.store.commit()
        if not run.complete:
            return None
        self.store.insert(
            "test_results",
            {"test_id": test_id, "result_json": dumps(run.result.to_dict())},
        )
        self.store.commit()
        return self._test_result(test_id, run.result)

    def _advance_monte_carlo(
        self,
        test: dict[str, Any],
        engine: GroupedPermutationTest,
        spec: TestSpec,
        state: dict[str, Any],
        max_new_samples: int | None,
    ) -> TestResult | None:
        test_id = int(test["id"])
        batch_id = int(test["batch_id"])
        stopping = engine.stopping_settings()
        adaptive = stopping.get("type", "fixed") != "fixed"
        cap = stopping.get("max_samples") if adaptive else None
        target = min(
            int(cap) if cap is not None else int(test["samples"]),
            engine.total_assignments,
        )
        evaluated = int(state["evaluated_assignments"])
        hits = int(state["as_extreme_count"])
        looks = int(state.get("looks", 0))

        want = target - evaluated
        if max_new_samples is not None:
            want = min(want, max_new_samples)
        pool = self._ensure_pool(batch_id, engine, evaluated + max(want, 0))

        records: list[AssignmentRecord] = []
        produced = 0
        while produced < want and evaluated < len(pool):
            record = engine.evaluate(pool[evaluated])
            records.append(record)
            if record.as_extreme:
                hits += 1
            evaluated += 1
            produced += 1

        self._append_samples(test_id, evaluated - produced, records)

        stopped_reason = engine.check_stopping(evaluated, hits) if adaptive else None
        if adaptive:
            looks += 1
        complete = evaluated >= target or stopped_reason is not None

        batch_state = self.store.find_one("batch_states", batch_id=batch_id)
        state.update(
            {
                "evaluated_assignments": evaluated,
                "as_extreme_count": hits,
                "random_draws": int(batch_state["random_draws"]),
                "looks": looks,
            }
        )
        self.store.commit()
        if not complete:
            return None
        p_value = (hits + 1) / (evaluated + 1)
        standard_error = math.sqrt(max(0.0, p_value * (1 - p_value) / (evaluated + 2)))
        z = 1.959963984540054
        result = PermutationResult(
            observed_statistic=engine.observed_statistic,
            p_value=p_value,
            method="monte-carlo",
            total_assignments=engine.total_assignments,
            evaluated_assignments=evaluated,
            monte_carlo_draws=int(batch_state["random_draws"]),
            standard_error=standard_error,
            confidence_level=0.95,
            confidence_interval=(
                min(1.0, max(0.0, p_value - z * standard_error)),
                min(1.0, max(0.0, p_value + z * standard_error)),
            ),
            as_extreme_count=hits,
            seed=self._pool_seed(batch_id),
            stopping_rule=stopping.get("type", "fixed"),
            adaptive=adaptive,
            looks=looks,
            stopped_reason=stopped_reason or ("max-samples" if evaluated >= target and adaptive else None),
        )
        self.store.insert(
            "test_results",
            {"test_id": test_id, "result_json": dumps(result.to_dict())},
        )
        self.store.commit()
        return self._test_result(test_id, result)

    def _pool_seed(self, batch_id: int) -> int | None:
        batch = self.store.find_one("batches", id=batch_id)
        if batch is not None and batch.get("seed") is not None:
            return batch["seed"]
        seeds = [
            row["seed"]
            for row in self.store.find_many("tests", batch_id=batch_id)
            if row.get("seed") is not None
        ]
        return min(seeds) if seeds else None

    def _ensure_pool(
        self, batch_id: int, engine: GroupedPermutationTest, needed: int
    ) -> list[tuple[str, ...]]:
        rows = sorted(
            self.store.find_many("batch_assignments", batch_id=batch_id),
            key=lambda row: int(row["ordinal"]),
        )
        if len(rows) < needed:
            state = self.store.find_one("batch_states", batch_id=batch_id)
            seed = self._pool_seed(batch_id)
            rng = random.Random(seed)
            if state["rng_state_json"] is not None:
                rng.setstate(_restore_rng_state(loads(state["rng_state_json"])))
            keys = {row["assignment_key"] for row in rows}
            ordinal = len(rows)
            draws = int(state["random_draws"])
            while len(rows) < needed and ordinal < engine.total_assignments:
                assignment = engine.random_assignment(rng)
                draws += 1
                key = dumps(list(assignment))
                if key in keys:
                    continue
                keys.add(key)
                ordinal += 1
                row = {
                    "id": self.store.next_id("batch_assignments"),
                    "batch_id": batch_id,
                    "ordinal": ordinal,
                    "assignment_key": key,
                }
                self.store.data["batch_assignments"].append(row)
                rows.append(row)
            state.update(
                {"rng_state_json": dumps(rng.getstate()), "random_draws": draws}
            )
            self.store.commit()
        return [tuple(json.loads(row["assignment_key"])) for row in rows[:needed]]

    def compare_batch(
        self,
        batch_id: int,
        *,
        rule: str = "holm",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        rows = sorted(
            self.store.find_many("tests", batch_id=batch_id),
            key=lambda row: int(row["id"]),
        )
        if not rows:
            raise KeyError(f"unknown batch {batch_id}")
        results = [self.get_result(int(row["id"])) for row in rows]
        p_values = [item.result.p_value for item in results]
        adjusted = adjust(p_values, rule)
        m = len(results)
        entries = []
        for item, p_adj in zip(results, adjusted):
            result = item.result
            adjusted_interval = None
            adjusted_level = None
            if result.standard_error is not None:
                base_level = result.confidence_level or 0.95
                adjusted_level = 1.0 - (1.0 - base_level) / m
                z_adj = NormalDist().inv_cdf((1.0 + adjusted_level) / 2.0)
                half = z_adj * result.standard_error
                adjusted_interval = [
                    min(1.0, max(0.0, result.p_value - half)),
                    min(1.0, max(0.0, result.p_value + half)),
                ]
            entries.append(
                {
                    "test_id": item.test_id,
                    "null_hypothesis": item.null_hypothesis,
                    "method": result.method,
                    "adaptive": result.adaptive,
                    "looks": result.looks,
                    "stopping_rule": result.stopping_rule,
                    "stopped_reason": result.stopped_reason,
                    "unadjusted_p_value": result.p_value,
                    "unadjusted_reject": result.p_value <= alpha,
                    "adjusted_p_value": p_adj,
                    "adjusted_reject": p_adj <= alpha,
                    "confidence_level": result.confidence_level,
                    "confidence_interval": list(result.confidence_interval)
                    if result.confidence_interval is not None
                    else None,
                    "adjusted_confidence_level": adjusted_level,
                    "adjusted_confidence_interval": adjusted_interval,
                }
            )
        return {
            "batch_id": batch_id,
            "rule": rule,
            "alpha": alpha,
            "tests": entries,
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

    def _load_samples(self, test_id: int) -> tuple[AssignmentRecord, ...]:
        rows = sorted(
            self.store.find_many("permutation_samples", test_id=test_id),
            key=lambda row: int(row["ordinal"]),
        )
        return tuple(
            AssignmentRecord(
                tuple(json.loads(row["assignment_key"])),
                float(row["statistic"]),
                bool(row["as_extreme"]),
            )
            for row in rows
        )

    def _append_samples(
        self,
        test_id: int,
        existing_count: int,
        records: Sequence[AssignmentRecord],
    ) -> None:
        existing_keys = {
            record["assignment_key"]
            for record in self.store.find_many("permutation_samples", test_id=test_id)
        }
        ordinal = existing_count
        for record in records:
            assignment_key = dumps(list(record.assignment))
            if assignment_key in existing_keys:
                continue
            ordinal += 1
            existing_keys.add(assignment_key)
            self.store.insert(
                "permutation_samples",
                {
                    "test_id": test_id,
                    "ordinal": ordinal,
                    "assignment_key": assignment_key,
                    "statistic": record.statistic,
                    "as_extreme": int(record.as_extreme),
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
        return PermutationResult(**data)

    def _test_result(self, test_id: int, result: PermutationResult) -> TestResult:
        row = self.store.find_one("tests", id=test_id)
        return TestResult(
            test_id=test_id,
            batch_id=int(row["batch_id"]),
            null_hypothesis=row["null_hypothesis"],
            result=result,
        )
