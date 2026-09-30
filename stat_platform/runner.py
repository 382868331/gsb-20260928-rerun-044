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
from .sequential import SequentialDesign, build_design, default_looks
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
            # Each advance runs through one chunk or one planned look; for a
            # sequential test this keeps advancing between looks transparently
            # without creating decisions at non-look boundaries.
            while True:
                result = self.advance_test(test_id)
                if result is not None:
                    break
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
        rule = stopping.get("type", "fixed")
        adaptive = rule != "fixed"
        design = self._load_design(state, stopping, engine.total_assignments)
        look_schedule = design.looks if design is not None else None
        cap = design.max_samples if design is not None else stopping.get("max_samples")
        target = min(
            int(cap) if cap is not None else int(test["samples"]),
            engine.total_assignments,
        )
        evaluated = int(state["evaluated_assignments"])
        hits = int(state["as_extreme_count"])
        looks = int(state.get("looks", 0))

        # With a pre-planned design the stopping rule is evaluated only at a
        # planned look; chunks between looks are merely persisted, so they can
        # never create an unplanned (invalid) decision.
        advance_limit = target
        if look_schedule is not None:
            next_look = next(
                (count for count in look_schedule if count > evaluated),
                look_schedule[-1],
            )
            advance_limit = min(target, next_look)
        want = advance_limit - evaluated
        if max_new_samples is not None:
            want = min(want, max_new_samples)
        else:
            want = advance_limit - evaluated
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

        stopped_reason = None
        decision = None
        look_log = list(loads(state.get("look_log_json") or "[]"))
        at_look = look_schedule is not None and evaluated in set(look_schedule)
        if at_look:
            looks += 1
            if rule == "ci-half-width":
                # Estimation-precision target: this produces a narrow interval
                # but is never a significance decision and carries no decision
                # error guarantee across looks.
                entry = self._ci_look_entry(evaluated, hits, stopping)
                entry["look"] = looks
                look_log.append(entry)
                if entry["precision_target_met"]:
                    stopped_reason = "ci-half-width"
            elif design is not None:
                decision = design.evaluate(evaluated, hits)
                entry = design.progress(evaluated, hits)
                entry["look"] = looks
                look_log.append(entry)
                if decision is not None:
                    stopped_reason = f"decided-{decision}"

        complete = evaluated >= target or stopped_reason is not None
        if complete and adaptive and stopped_reason is None:
            # Budget exhausted while still inside the continuation region: the
            # decision is honestly reported as inconclusive, never forced.
            stopped_reason = "inconclusive-max-samples"

        batch_state = self.store.find_one("batch_states", batch_id=batch_id)
        state.update(
            {
                "evaluated_assignments": evaluated,
                "as_extreme_count": hits,
                "random_draws": int(batch_state["random_draws"]),
                "looks": looks,
                "look_log_json": dumps(look_log),
                "design_json": dumps(design.to_dict()) if design is not None else None,
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
            stopping_rule=rule,
            adaptive=adaptive,
            looks=looks,
            stopped_reason=(
                stopped_reason
                if stopped_reason is not None
                else ("max-samples" if evaluated >= target and rule == "ci-half-width" else None)
            ),
            decision=decision if design is not None else None,
            design=design.to_dict() if design is not None else None,
            look_log=tuple(look_log),
        )
        self.store.insert(
            "test_results",
            {"test_id": test_id, "result_json": dumps(result.to_dict())},
        )
        self.store.commit()
        return self._test_result(test_id, result)

    @staticmethod
    def _load_design(
        state: dict[str, Any],
        stopping: dict[str, Any],
        total_assignments: int | None = None,
    ) -> SequentialDesign | None:
        if stopping.get("type") not in {"decision", "ci-half-width"}:
            return None
        stored = state.get("design_json")
        if stored:
            return SequentialDesign.from_dict(loads(stored))
        looks = stopping.get("looks")
        max_samples = int(stopping.get("max_samples") or stopping.get("min_samples") or 1000)
        if not looks:
            looks = default_looks(
                max_samples,
                int(stopping.get("number_of_looks", 5)),
                min_samples=int(stopping.get("min_samples", 1) or 1),
            )
        looks = tuple(int(count) for count in looks)
        if total_assignments is not None and looks[-1] > total_assignments:
            # Project the pre-planned schedule onto the smaller feasible space;
            # the last permitted check sits at full coverage.
            looks = tuple(sorted({min(count, total_assignments) for count in looks}))
        design = build_design(
            looks,
            q0=float(stopping.get("alpha", stopping.get("decision_alpha", 0.05))),
            q_sig=float(stopping.get("q_sig", stopping.get("indifference_p", 0.01))),
            q_nonsig=float(stopping.get("q_nonsig", stopping.get("indifference_p_high", 0.2))),
            alpha_significant=float(stopping.get("risk_significant", 0.05)),
            alpha_not_significant=float(stopping.get("risk_not_significant", 0.05)),
        )
        state["design_json"] = dumps(design.to_dict())
        return design

    @staticmethod
    def _ci_look_entry(evaluated: int, hits: int, stopping: dict[str, Any]) -> dict[str, Any]:
        p_value = (hits + 1) / (evaluated + 1)
        standard_error = math.sqrt(max(0.0, p_value * (1 - p_value) / (evaluated + 2)))
        half = 1.959963984540054 * standard_error
        return {
            "evaluated": evaluated,
            "as_extreme_count": hits,
            "current_p_estimate": p_value,
            "ci_half_width": half,
            "target_half_width": stopping.get("max_half_width"),
            "precision_target_met": stopping.get("max_half_width") is not None
            and half <= float(stopping["max_half_width"]),
        }

    def peek_test(self, test_id: int) -> dict[str, Any]:
        """Current evidence and unmet stopping conditions without consuming draws."""
        test = self.store.find_one("tests", id=test_id)
        if test is None:
            raise KeyError(f"unknown test {test_id}")
        completed_row = self.store.find_one("test_results", test_id=test_id)
        state = self.store.find_one("run_states", test_id=test_id)
        evaluated = int(state["evaluated_assignments"])
        hits = int(state["as_extreme_count"])
        settings = loads(test["settings_json"])
        stopping = settings.get("stopping") or {"type": "fixed"}
        design = self._load_design(state, stopping)
        entry: dict[str, Any] = {
            "test_id": test_id,
            "batch_id": int(test["batch_id"]),
            "status": "completed" if completed_row is not None else "running",
            "evaluated": evaluated,
            "as_extreme_count": hits,
            "current_p_estimate": (hits + 1) / (evaluated + 1) if evaluated else None,
            "looks_recorded": int(state.get("looks", 0)),
            "look_log": list(loads(state.get("look_log_json") or "[]")),
        }
        if design is not None:
            entry["design"] = design.to_dict()
            entry["next_conditions"] = design.progress(evaluated, hits)
        return entry

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
                    "individual_decision": item.result.decision,
                    "individual_meaning": self._individual_meaning(item.result),
                }
            )
        family_rejected = [entry for entry in entries if entry["adjusted_reject"]]
        decided = [entry for entry in entries if entry["individual_decision"] is not None]
        family = {
            "number_of_tests": m,
            "rule": rule,
            "family_wise_alpha": alpha,
            "family_rejected_count": len(family_rejected),
            "family_rejected_test_ids": [entry["test_id"] for entry in family_rejected],
            "all_individual_decisions_made": len(decided) == m,
            "undecided_test_ids": [
                entry["test_id"]
                for entry in entries
                if entry["individual_decision"] is None and entry["adaptive"]
            ],
            "meaning": (
                "Per-test entries judge each null hypothesis separately at its own "
                "pre-planned decision risk; adjusted_reject additionally controls the "
                f"family-wise error over all {m} tests under shared (possibly dependent) "
                "permutations using the " + rule + " rule. A family claim that at least "
                "one null is false requires an adjusted rejection; it does not imply the "
                "other hypotheses are false."
            ),
        }
        return {
            "batch_id": batch_id,
            "rule": rule,
            "alpha": alpha,
            "tests": entries,
            "family": family,
        }

    @staticmethod
    def _individual_meaning(result: PermutationResult) -> str:
        if not result.adaptive or result.decision is None:
            return (
                "No risk-controlled individual decision: the value is a fixed-budget "
                "Monte Carlo estimate and its interval cannot be interpreted as valid "
                "at arbitrary inspection times."
            )
        if result.decision == "significant":
            return (
                "Reject this null under the pre-planned look schedule; the exact "
                "probability of this declaration at the decision boundary is bounded "
                "by risk_significant_at_q0 in design."
            )
        return (
            "Do not reject this null under the pre-planned look schedule; the exact "
            "probability of this declaration at the decision boundary is bounded by "
            "risk_not_significant_at_q0 in design."
        )

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
