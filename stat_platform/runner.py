from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from dataclasses import replace
from statistics import NormalDist
from typing import Any, Sequence

from .adjustment import adjust
from .hierarchy import (
    HypothesisGate,
    gates_for_specs,
    gate_is_opened_by,
    path_risk_summaries,
    reallocation_edges,
    validate_gates,
    validate_reallocation_routes,
)
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
        family_alpha: float = 0.05,
    ) -> int:
        spec_list = tuple(specs or ())
        gates = gates_for_specs(spec_list)
        # Validate the whole confirmation family before persisting anything, so
        # a rejected plan leaves existing batches and exploratory results intact.
        validate_gates(gates, family_alpha)
        dataset_id = self._store_dataset(dataset)
        routes = reallocation_edges(gates)
        batch_id = self.store.insert(
            "batches",
            {
                "dataset_id": dataset_id,
                "name": name or f"{dataset.name}:{dataset.version}",
                "seed": seed,
                "family_alpha": family_alpha,
                "reallocation_routes_json": dumps(routes),
                "route_version": 1,
            },
        )
        self.store.insert(
            "batch_states",
            {"batch_id": batch_id, "rng_state_json": None, "random_draws": 0},
        )
        for index, spec in enumerate(spec_list):
            self.add_test(
                batch_id,
                spec,
                samples=samples,
                max_exact_assignments=max_exact_assignments,
                seed=seed,
                gate=gates[index],
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
        gate: HypothesisGate | None = None,
    ) -> int:
        if gate is None:
            gate = HypothesisGate.from_settings(dict(spec.settings), 0)
        if gate is not None and gate.reallocate_to is not None:
            self._extend_reallocation_routes(batch_id, gate)
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
                "hypothesis_key": gate.key if gate is not None else None,
                "parent_hypothesis": gate.parent if gate is not None else None,
                "parent_hypotheses_json": dumps(list(gate.parents))
                if gate is not None
                else None,
                "parent_mode": gate.parent_mode if gate is not None else None,
                "reallocate_to": gate.reallocate_to if gate is not None else None,
                "risk_share": gate.risk_share if gate is not None else None,
            },
        )
        initial_status = "open" if gate is None or not gate.parents else "pending"
        self.store.insert(
            "run_states",
            {
                "test_id": test_id,
                "evaluated_assignments": 0,
                "as_extreme_count": 0,
                "random_draws": 0,
                "rng_state_json": None,
                "gate_status": initial_status,
                "gate_opened_at": 0,
                "gate_opened_via_json": None,
                "formal_evaluated_start": 0,
                "exploratory_evaluated": 0,
                "exploratory_hits": 0,
            },
        )
        self.store.commit()
        return test_id

    def _batch_routes(self, batch_id: int) -> tuple[dict[str, str], int]:
        """Persisted routing plan and its version for a batch.

        The plan is stored when the batch is created and only ever extended
        through validated additions; batches saved before routing versions
        existed fall back to the edges recorded on their test rows.
        """
        batch = self.store.find_one("batches", id=batch_id)
        stored = batch.get("reallocation_routes_json")
        if stored is not None:
            return dict(loads(stored)), int(batch.get("route_version") or 1)
        routes = {
            row["hypothesis_key"]: row["reallocate_to"]
            for row in self.store.find_many("tests", batch_id=batch_id)
            if row.get("hypothesis_key") is not None and row.get("reallocate_to")
        }
        return routes, 1

    def _extend_reallocation_routes(self, batch_id: int, gate: HypothesisGate) -> None:
        routes, version = self._batch_routes(batch_id)
        existing_target = routes.get(gate.key)
        if existing_target is not None:
            if existing_target == gate.reallocate_to:
                return
            raise ValueError(
                f"reallocation route for {gate.key} is already recorded as "
                f"{gate.key}->{existing_target} (route version {version}); a "
                "saved plan cannot be retroactively rewritten"
            )
        known_keys = {
            row["hypothesis_key"]
            for row in self.store.find_many("tests", batch_id=batch_id)
            if row.get("hypothesis_key") is not None
        }
        known_keys.add(gate.key)
        if gate.reallocate_to not in known_keys:
            raise ValueError(f"unknown reallocation target {gate.reallocate_to}")
        candidate = {**routes, gate.key: gate.reallocate_to}
        validate_reallocation_routes(candidate)
        batch = self.store.find_one("batches", id=batch_id)
        batch["reallocation_routes_json"] = dumps(candidate)
        batch["route_version"] = version + 1
        self.store.commit()

    def run_batch(
        self,
        batch_id: int,
        *,
        max_new_samples: int | None = None,
    ) -> list[TestResult]:
        rows = sorted(self.store.find_many("tests", batch_id=batch_id), key=lambda row: int(row["id"]))
        results = []
        completed = set()
        while len(completed) < len(rows):
            progressed = False
            for row in rows:
                test_id = int(row["id"])
                if test_id in completed:
                    continue
                state = self.store.find_one("run_states", test_id=test_id)
                status = state.get("gate_status", "open")
                if status == "pending":
                    if self._open_dependent(test_id):
                        progressed = True
                    continue
                if status == "closed":
                    completed.add(test_id)
                    continue
                result = self.run_test(test_id, max_new_samples=max_new_samples)
                results.append(result)
                completed.add(test_id)
                progressed = True
                self._update_dependent_gates(batch_id)
            if not progressed:
                break
        return results

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
        state = self.store.find_one("run_states", test_id=test_id)
        if state.get("gate_status") == "pending":
            raise RuntimeError("this hypothesis is not open; its information is exploratory only")
        if state.get("gate_status") == "closed":
            raise RuntimeError("this hypothesis remained unopened because its gate was not met")
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
        if run.result is not None:
            run = replace(run, result=self._with_gate_info(test, state, run.result))
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
        risk_share = test.get("risk_share")
        if risk_share is not None and rule == "decision":
            stopping = dict(stopping)
            stopping.setdefault("risk_significant", float(risk_share))
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
        result = self._with_gate_info(test, state, result)
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
        if test.get("hypothesis_key") is not None:
            entry.update(
                {
                    "hypothesis": test.get("hypothesis_key"),
                    "parent": test.get("parent_hypothesis"),
                    "gate_status": state.get("gate_status", "open"),
                    "formal_status": (
                        "exploratory-only" if state.get("gate_status") != "open" else "formal"
                    ),
                    "risk_share": test.get("risk_share"),
                }
            )
            if state.get("gate_status") == "pending":
                batch = self.store.find_one("batches", id=int(test["batch_id"]))
                dataset = self._dataset_from_row(
                    self.store.find_one("datasets", id=int(batch["dataset_id"]))
                )
                engine = self._engine(
                    dataset,
                    TestSpec(test["test_type"], test["null_hypothesis"], settings),
                )
                preview_limit = min(int(test["samples"]), engine.total_assignments)
                pool = self._ensure_pool(int(test["batch_id"]), engine, preview_limit)
                preview_hits = sum(engine.evaluate(assignment).as_extreme for assignment in pool)
                entry["exploratory_preview"] = {
                    "evaluated": len(pool),
                    "as_extreme_count": preview_hits,
                    "p_estimate": (preview_hits + 1) / (len(pool) + 1),
                    "persisted_to_test": False,
                    "formal_status": "exploratory-only",
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

    def _with_gate_info(
        self,
        test: dict[str, Any],
        state: dict[str, Any],
        result: PermutationResult,
    ) -> PermutationResult:
        if test.get("hypothesis_key") is None:
            return result
        return replace(
            result,
            gate_info={
                "hypothesis": test["hypothesis_key"],
                "parent": test.get("parent_hypothesis"),
                "status": state.get("gate_status", "open"),
                "opened_after_assignment": state.get("gate_opened_at", 0),
                "risk_share": test.get("risk_share"),
                "formal_status": "formal" if state.get("gate_status", "open") == "open" else "exploratory",
            },
        )

    def _open_dependent(self, test_id: int) -> bool:
        self._update_dependent_gates(
            int(self.store.find_one("tests", id=test_id)["batch_id"])
        )
        state = self.store.find_one("run_states", test_id=test_id)
        return state.get("gate_status") == "open"

    def _update_dependent_gates(self, batch_id: int) -> None:
        tests = {
            row["hypothesis_key"]: row
            for row in self.store.find_many("tests", batch_id=batch_id)
            if row.get("hypothesis_key") is not None
        }
        changed = False
        for test in tests.values():
            parent_keys = self._parent_keys(test)
            if not parent_keys:
                continue
            state = self.store.find_one("run_states", test_id=int(test["id"]))
            if state.get("gate_status") != "pending":
                continue
            settings = loads(test["settings_json"])
            gate = HypothesisGate.from_settings(settings, int(test["id"]) - 1)
            outcomes: list[bool | None] = []
            for parent_key in parent_keys:
                parent = tests[parent_key]
                completed = self.store.find_one("test_results", test_id=int(parent["id"]))
                if completed is None:
                    outcomes.append(None)
                    continue
                parent_result = self._result_from_dict(loads(completed["result_json"]))
                outcomes.append(
                    gate_is_opened_by(gate, parent_result, p_value=parent_result.p_value)
                )
            mode = gate.parent_mode if gate is not None else "all"
            if mode == "any":
                if any(outcome is True for outcome in outcomes):
                    new_status = "open"
                elif all(outcome is not None for outcome in outcomes):
                    new_status = "closed"
                else:
                    continue
            else:
                if any(outcome is False for outcome in outcomes):
                    new_status = "closed"
                elif all(outcome is True for outcome in outcomes):
                    new_status = "open"
                else:
                    continue
            opened_via = [
                parent_key
                for parent_key, outcome in zip(parent_keys, outcomes)
                if outcome is True
            ]
            state.update(
                {
                    "gate_status": new_status,
                    "gate_opened_via_json": dumps(opened_via)
                    if new_status == "open"
                    else None,
                    "gate_opened_at": int(
                        self.store.find_one("run_states", test_id=int(parent["id"]))[
                            "evaluated_assignments"
                        ]
                    )
                    if new_status == "open"
                    else 0,
                }
            )
            changed = True
        if changed:
            self.store.commit()
        self._apply_share_transfers(batch_id)

    @staticmethod
    def _parent_keys(test: dict[str, Any]) -> list[str]:
        stored = test.get("parent_hypotheses_json")
        if stored:
            return list(loads(stored))
        single = test.get("parent_hypothesis")
        return [single] if single is not None else []

    @staticmethod
    def _used_share(
        test: dict[str, Any],
        result: PermutationResult,
        share: float | None = None,
    ) -> float:
        # Share already consumed by a formal decision; it never becomes
        # reusable, even if a display is withdrawn or a test is reopened.
        if share is None:
            share = float(test.get("risk_share") or 0.0)
        decision = result.decision
        if decision is None:
            decision = "significant" if result.p_value <= share else "not-significant"
        if decision == "significant":
            design = result.design or {}
            return float(design.get("risk_significant_at_q0", share))
        return 0.0

    def _apply_share_transfers(self, batch_id: int) -> None:
        # Move predeclared unused shares of failed/closed hypotheses exactly
        # once. A hypothesis may declare reallocate_to in its hierarchy
        # settings; when it closes without opening or finishes with a formal
        # not-significant decision, the unused part of its share moves to the
        # declared target. Shares spent on a formal significant decision are
        # never released again.
        routes, route_version = self._batch_routes(batch_id)
        changed = False
        for row in self.store.find_many("tests", batch_id=batch_id):
            source_key = row.get("hypothesis_key")
            target_key = routes.get(source_key) if source_key is not None else None
            if source_key is None or target_key is None:
                continue
            existing = self.store.find_one(
                "share_transfers", batch_id=batch_id, source_key=source_key
            )
            if existing is not None:
                if existing["target_key"] != target_key:
                    raise RuntimeError(
                        f"recorded transfer {source_key}->"
                        f"{existing['target_key']} conflicts with route version "
                        f"{route_version} ({source_key}->{target_key}); saved "
                        "transfer evidence is never rewritten"
                    )
                continue
            state = self.store.find_one("run_states", test_id=int(row["id"]))
            status = state.get("gate_status", "open")
            share = float(row.get("risk_share") or 0.0)
            completed = self.store.find_one("test_results", test_id=int(row["id"]))
            if status == "closed":
                used = 0.0
                reason = "gate-closed"
            elif completed is not None:
                result = self._result_from_dict(loads(completed["result_json"]))
                received = self._transfers_by_target(batch_id).get(source_key, [])
                effective = share + sum(float(item["amount"]) for item in received)
                used = self._used_share(row, result, effective)
                reason = "decision-not-significant"
            else:
                continue
            amount = share - used
            if amount <= 1e-12:
                continue
            self.store.insert(
                "share_transfers",
                {
                    "batch_id": batch_id,
                    "source_key": source_key,
                    "target_key": target_key,
                    "amount": amount,
                    "reason": reason,
                    "route_version": route_version,
                },
            )
            changed = True
        if changed:
            self.store.commit()

    def _transfers_by_target(self, batch_id: int) -> dict[str, list[dict[str, Any]]]:
        received: dict[str, list[dict[str, Any]]] = {}
        for transfer in self.store.find_many("share_transfers", batch_id=batch_id):
            received.setdefault(transfer["target_key"], []).append(transfer)
        return received

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
        batch_row = self.store.find_one("batches", id=batch_id)
        self._apply_share_transfers(batch_id)
        routes, route_version = self._batch_routes(batch_id)
        transfers = self.store.find_many("share_transfers", batch_id=batch_id)
        transfers_in: dict[str, list[dict[str, Any]]] = {}
        transfers_out: dict[str, list[dict[str, Any]]] = {}
        for transfer in transfers:
            transfers_in.setdefault(transfer["target_key"], []).append(transfer)
            transfers_out.setdefault(transfer["source_key"], []).append(transfer)
        effective_share: dict[str, float] = {}
        for row in rows:
            key = row.get("hypothesis_key")
            if key is None:
                continue
            effective_share[key] = float(row.get("risk_share") or 0.0) + sum(
                float(item["amount"]) for item in transfers_in.get(key, [])
            )
        all_items = []
        for row in rows:
            test_id = int(row["id"])
            completed_row = self.store.find_one("test_results", test_id=test_id)
            if completed_row is not None:
                item = self.get_result(test_id)
                all_items.append((row, item, item.result.p_value))
            else:
                all_items.append((row, None, None))
        formal_items = [item for item in all_items if item[1] is not None]
        regular_items = [
            item
            for item in formal_items
            if item[0].get("hypothesis_key") is None
        ]
        adjusted = adjust([p_value for _, _, p_value in regular_items], rule)
        m = len(regular_items)
        entries = []
        adjusted_by_id = {
            item.test_id: p_adj
            for (_, item, _), p_adj in zip(regular_items, adjusted)
        }
        for row, item, _ in all_items:
            state = self.store.find_one("run_states", test_id=int(row["id"]))
            gate_status = state.get("gate_status", "open")
            if item is None:
                entries.append(
                    {
                        "test_id": int(row["id"]),
                        "null_hypothesis": row["null_hypothesis"],
                        "hypothesis": row.get("hypothesis_key"),
                        "parent": row.get("parent_hypothesis"),
                        "gate_status": gate_status,
                        "formal_status": "not-open",
                        "risk_share": row.get("risk_share"),
                        "adjusted_reject": False,
                        "formal_conclusion": None,
                        "exploratory_evidence": None,
                        "individual_meaning": (
                            "The gate did not open; an unopened observation is not a formal conclusion."
                            if gate_status == "closed"
                            else "The gate is pending; current information is exploratory only."
                        ),
                    }
                )
                continue
            result = item.result
            p_adj = adjusted_by_id.get(item.test_id, result.p_value)
            is_hierarchical = row.get("hypothesis_key") is not None
            allocated_share = row.get("risk_share")
            if is_hierarchical:
                allocated_share = effective_share.get(
                    row["hypothesis_key"],
                    float(allocated_share or 0.0),
                )
            adjusted_reject = (
                result.p_value <= float(allocated_share)
                if is_hierarchical and allocated_share is not None
                else p_adj <= alpha
            )
            reported_adjusted_p = result.p_value if is_hierarchical else p_adj
            adjusted_interval = None
            adjusted_level = None
            if result.standard_error is not None:
                base_level = result.confidence_level or 0.95
                adjusted_level = 1.0 - (1.0 - base_level) / max(1, m)
                z_adj = NormalDist().inv_cdf((1.0 + adjusted_level) / 2.0)
                half = z_adj * result.standard_error
                adjusted_interval = [
                    min(1.0, max(0.0, result.p_value - half)),
                    min(1.0, max(0.0, result.p_value + half)),
                ]
            if is_hierarchical:
                adjusted_interval = None
                adjusted_level = None
            entries.append(
                {
                    "test_id": item.test_id,
                    "hypothesis": row.get("hypothesis_key"),
                    "parent": row.get("parent_hypothesis"),
                    "gate_status": gate_status,
                    "formal_status": "formal",
                    "risk_share": row.get("risk_share"),
                    "allocated_risk_share": allocated_share,
                    "effective_risk_share": allocated_share if is_hierarchical else None,
                    "null_hypothesis": item.null_hypothesis,
                    "method": result.method,
                    "adaptive": result.adaptive,
                    "looks": result.looks,
                    "stopping_rule": result.stopping_rule,
                    "stopped_reason": result.stopped_reason,
                    "unadjusted_p_value": result.p_value,
                    "unadjusted_reject": result.p_value <= alpha,
                    "adjusted_p_value": reported_adjusted_p,
                    "adjusted_reject": adjusted_reject,
                    "adjustment_basis": (
                        "preallocated hierarchical risk share"
                        if is_hierarchical
                        else rule
                    ),
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
        decided = [entry for entry in entries if entry.get("individual_decision") is not None]
        gate_records = []
        outcomes = {}
        for row, item, _ in all_items:
            key = row.get("hypothesis_key")
            if key is None:
                continue
            state = self.store.find_one("run_states", test_id=int(row["id"]))
            status = state.get("gate_status", "open")
            gate_records.append(
                (
                    int(row["id"]),
                    HypothesisGate.from_settings(
                        loads(row["settings_json"]), int(row["id"]) - 1
                    ),
                )
            )
            outcomes[key] = {"status": status, "result": item.result if item is not None else None}
        family_alpha = float(batch_row.get("family_alpha", alpha))
        gate_paths = path_risk_summaries(gate_records, outcomes, family_alpha)
        risk_ledger = []
        for row, item, _ in all_items:
            key = row.get("hypothesis_key")
            if key is None:
                continue
            state = self.store.find_one("run_states", test_id=int(row["id"]))
            initial_share = float(row.get("risk_share") or 0.0)
            received = transfers_in.get(key, [])
            sent = transfers_out.get(key, [])
            result = item.result if item is not None else None
            effective = effective_share[key]
            used = (
                self._used_share(row, result, effective) if result is not None else 0.0
            )
            released = sum(float(transfer["amount"]) for transfer in sent)
            risk_ledger.append(
                {
                    "hypothesis": key,
                    "test_id": int(row["id"]),
                    "gate_status": state.get("gate_status", "open"),
                    "initial_allocation": initial_share,
                    "transfers_in": [
                        {
                            "from": transfer["source_key"],
                            "amount": float(transfer["amount"]),
                            "reason": transfer["reason"],
                        }
                        for transfer in received
                    ],
                    "transfers_out": [
                        {
                            "to": transfer["target_key"],
                            "amount": float(transfer["amount"]),
                            "reason": transfer["reason"],
                        }
                        for transfer in sent
                    ],
                    "effective_allocation": effective,
                    "used_risk_share": used,
                    "remaining_risk_share": max(0.0, effective - used - released),
                }
            )
        family = {
            "number_of_tests": len(rows),
            "regular_test_count": m,
            "planned_family_alpha": family_alpha,
            "rule": rule,
            "family_wise_alpha": alpha,
            "formally_open_count": m,
            "pending_test_ids": [
                e["test_id"]
                for e in entries
                if e.get("gate_status") == "pending"
            ],
            "closed_test_ids": [e["test_id"] for e in entries if e.get("gate_status") == "closed"],
            "gate_paths": gate_paths,
            "risk_ledger": risk_ledger,
            "reallocation_routes": {
                "version": route_version,
                "edges": [
                    {"from": source, "to": target}
                    for source, target in sorted(routes.items())
                ],
            },
            "family_rejected_count": len(family_rejected),
            "family_rejected_test_ids": [entry["test_id"] for entry in family_rejected],
            "all_individual_decisions_made": len(decided) == len(formal_items),
            "undecided_test_ids": [
                entry["test_id"]
                for entry in entries
                if entry.get("individual_decision") is None and entry.get("adaptive")
            ],
            "meaning": (
                "Per-test entries judge each null hypothesis separately at its own "
                "pre-planned decision risk; adjusted_reject additionally controls the "
                f"family-wise error over the {m} regular tests under shared (possibly "
                "dependent) permutations using the " + rule + " rule. Hierarchical tests "
                "use their preallocated risk shares. A family claim that at least "
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
