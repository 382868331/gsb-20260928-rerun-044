from __future__ import annotations

import random
from pathlib import Path

import pytest

from stat_platform import (
    Dataset,
    GroupedPermutationTest,
    PlatformRunner,
    TestSpec,
    build_design,
    crossing_probabilities,
    default_looks,
)


def triple_dataset() -> Dataset:
    records = []
    values = {"A": (1, 4), "B": (2, 5), "C": (3, 3)}
    for group, (control, treatment) in values.items():
        records.append({"group": group, "condition": "control", "value": control})
        records.append({"group": group, "condition": "treatment", "value": treatment})
    return Dataset.from_records(records, name="triples", version="v1")


def spec(stopping: dict | None = None) -> TestSpec:
    settings = {"alternative": "two-sided", "conditions": ("control", "treatment")}
    if stopping is not None:
        settings["stopping"] = stopping
    return TestSpec(GroupedPermutationTest.test_type, "equal means", settings)


def simulate(design, q, trials, seed):
    rng = random.Random(seed)
    decisions = {"significant": 0, "not-significant": 0, None: 0}
    for _ in range(trials):
        hits = 0
        evaluated = 0
        outcome = None
        previous = 0
        for n_k in design.looks:
            hits += sum(rng.random() < q for _ in range(n_k - previous))
            evaluated = n_k
            outcome = design.evaluate(evaluated, hits)
            if outcome is not None:
                break
            previous = n_k
        decisions[outcome] += 1
    return {key: value / trials for key, value in decisions.items()}


def test_exact_boundaries_control_error_across_all_looks():
    # The naive fixed-sample reuse at every look overspends; the exact DP
    # boundaries must respect the pre-chosen risks at the boundary q0.
    looks = default_looks(1200, 4)
    design = build_design(
        looks, q0=0.05, q_sig=0.01, q_nonsig=0.1,
        alpha_significant=0.05, alpha_not_significant=0.05,
    )
    risk_sig, risk_not = crossing_probabilities(
        design.looks, design.lower, design.upper, design.q0
    )
    assert risk_sig <= 0.05
    assert risk_not <= 0.05
    assert design.feasible is True

    empirical = simulate(design, 0.05, trials=20_000, seed=3)
    assert empirical["significant"] <= 0.06
    assert empirical["not-significant"] <= 0.06

    # Far below alpha the true p-value is correctly ruled out most of the time.
    powered = build_design(default_looks(20_000, 4), q0=0.05, q_sig=0.01, q_nonsig=0.1)
    assert powered.power_not_significant_at_qnonsig >= 0.8
    assert powered.power_significant_at_qsig >= 0.8


def test_single_look_reduces_to_one_shot_binomial_rule():
    design = build_design((1000,), q0=0.05)
    assert len(design.looks) == 1
    risk_sig, risk_not = crossing_probabilities(
        design.looks, design.lower, design.upper, 0.05
    )
    assert risk_sig <= 0.05
    assert risk_not <= 0.05
    assert design.evaluate(500, 0) is None


def test_decisions_only_happen_at_planned_looks(tmp_path: Path):
    stopping = {
        "type": "decision",
        "alpha": 0.05,
        "risk_significant": 0.05,
        "risk_not_significant": 0.05,
        "q_sig": 0.01,
        "q_nonsig": 0.2,
        "looks": [4, 8],
        "max_samples": 8,
    }
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            triple_dataset(),
            specs=[spec(stopping)],
            samples=8,
            max_exact_assignments=4,
            seed=5,
        )
        test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
        with pytest.raises(RuntimeError):
            runner.run_test(test_id, max_new_samples=1)
        mid = runner.peek_test(test_id)
        assert mid["status"] == "running"
        # A non-look checkpoint must not create a decision or a look.
        assert mid["looks_recorded"] == 0
        assert mid["next_conditions"]["samples_until_next_look"] == 3
        try:
            runner.run_test(test_id, max_new_samples=3)
        except RuntimeError:
            pass
        first_look = runner.peek_test(test_id)
        assert first_look["looks_recorded"] == 1
        if first_look["next_conditions"]["decision"] is not None:
            result = runner.get_result(test_id)
        else:
            result = runner.run_test(test_id)

    assert result.result.looks >= 1
    assert result.result.design["looks"] == [4, 8]
    assert result.result.look_log[0]["evaluated"] == 4
    assert result.result.stopped_reason in {
        "decided-significant",
        "decided-not-significant",
        "inconclusive-max-samples",
    }


def test_chunking_and_resume_never_reset_looks_or_reuse_permutations(tmp_path: Path):
    stopping = {
        "type": "decision",
        "alpha": 0.05,
        "looks": [4, 8],
        "max_samples": 8,
    }
    with PlatformRunner(tmp_path / "chunked.json") as runner:
        batch_id = runner.create_batch(
            triple_dataset(),
            specs=[spec(stopping)],
            samples=8,
            max_exact_assignments=4,
            seed=9,
        )
        test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
        for _ in range(8):
            try:
                runner.run_test(test_id, max_new_samples=1)
                break
            except RuntimeError:
                pass
        chunked = runner.get_result(test_id)
        rows = [
            row["assignment_key"]
            for row in runner.store.find_many("permutation_samples", test_id=test_id)
        ]
        keys = rows

    with PlatformRunner(tmp_path / "oneshot.json") as runner:
        batch_id = runner.create_batch(
            triple_dataset(),
            specs=[spec(stopping)],
            samples=8,
            max_exact_assignments=4,
            seed=9,
        )
        one_shot = runner.run_batch(batch_id)[0]

    assert len(keys) == len(set(keys))
    assert len(keys) <= 8
    # No permutation is consumed twice after a stop or a resume/reopen.
    assert chunked.result.looks == one_shot.result.looks
    assert chunked.result.decision == one_shot.result.decision
    assert chunked.result.stopped_reason == one_shot.result.stopped_reason
    assert chunked.result.as_extreme_count == one_shot.result.as_extreme_count


def test_exhaustive_enumeration_agrees_with_full_monte_carlo_cover(tmp_path: Path):
    exact = GroupedPermutationTest(triple_dataset(), spec()).run().result
    assert exact.method == "exact"
    assert exact.total_assignments == 8
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            triple_dataset(),
            specs=[spec()],
            samples=8,
            max_exact_assignments=4,
            seed=1,
        )
        covered = runner.run_batch(batch_id)[0].result
    assert covered.method == "monte-carlo"
    assert covered.evaluated_assignments == 8
    assert covered.as_extreme_count == exact.as_extreme_count


def test_family_report_distinguishes_individual_and_family_claims(tmp_path: Path):
    stopping = {
        "type": "decision",
        "alpha": 0.5,
        "risk_significant": 0.5,
        "risk_not_significant": 0.5,
        "q_sig": 0.1,
        "q_nonsig": 0.8,
        "looks": [4, 8],
        "max_samples": 8,
    }
    second = {"alternative": "less", "conditions": ("control", "treatment"), "stopping": stopping}
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            triple_dataset(),
            specs=[spec(stopping), TestSpec(GroupedPermutationTest.test_type, "less means", second)],
            samples=8,
            max_exact_assignments=4,
            seed=4,
        )
        runner.run_batch(batch_id)
        report = runner.compare_batch(batch_id, rule="holm", alpha=0.05)

    assert report["family"]["number_of_tests"] == 2
    assert "family claim" in report["family"]["meaning"]
    for entry in report["tests"]:
        assert entry["individual_decision"] in {
            "significant", "not-significant", None,
        }
        assert entry["individual_meaning"]


def test_different_chunk_sizes_share_stopping_record(tmp_path: Path):
    stopping = {
        "type": "decision",
        "alpha": 0.05,
        "looks": [2, 4, 6, 8],
        "max_samples": 8,
    }
    records = []
    for chunk in (1, 2, 3, 8):
        path = tmp_path / f"chunk{chunk}.json"
        with PlatformRunner(path) as runner:
            batch_id = runner.create_batch(
                triple_dataset(),
                specs=[spec(stopping)],
                samples=8,
                max_exact_assignments=4,
                seed=13,
            )
            test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
            while True:
                try:
                    result = runner.run_test(test_id, max_new_samples=chunk)
                    break
                except RuntimeError:
                    pass
            records.append(
                (
                    result.result.looks,
                    result.result.stopped_reason,
                    [entry["evaluated"] for entry in result.result.look_log],
                )
            )
    assert len({(looks_, reason, tuple(log)) for looks_, reason, log in records}) == 1
    assert records[0][2] == [count for count in (2, 4, 6, 8) if count <= records[0][2][-1]]
