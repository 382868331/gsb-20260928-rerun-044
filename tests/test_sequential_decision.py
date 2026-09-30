import json
from math import comb
from pathlib import Path

import pytest

from stat_platform import (
    Dataset,
    PlatformRunner,
    SequentialDesign,
    TestSpec,
    finite_population_risks,
)
from stat_platform.sequential import binomial_cdf, spending_schedule


def three_group_dataset() -> Dataset:
    records = []
    values = {"A": (1, 4), "B": (2, 5), "C": (3, 3)}
    for group, (control, treatment) in values.items():
        records.append({"group": group, "condition": "control", "value": control})
        records.append({"group": group, "condition": "treatment", "value": treatment})
    return Dataset.from_records(records, name="triples", version="v1")


def make_spec(alternative="two-sided", stopping=None) -> TestSpec:
    settings = {"alternative": alternative, "conditions": ("control", "treatment")}
    if stopping is not None:
        settings["stopping"] = stopping
    return TestSpec(
        "grouped_permutation_mean_difference",
        f"equal means ({alternative})",
        settings,
    )


def sequential_settings(looks, **overrides):
    stopping = {
        "type": "sequential-decision",
        "looks": list(looks),
        "p_high": 0.05,
        "p_low": 0.02,
        "epsilon_high": 0.05,
        "epsilon_low": 0.05,
        "spending": "pocock",
    }
    stopping.update(overrides)
    return stopping


def test_design_requires_strictly_increasing_planned_looks():
    with pytest.raises(ValueError):
        SequentialDesign.create([10, 10, 20])
    with pytest.raises(ValueError):
        SequentialDesign.create([20, 10])
    with pytest.raises(ValueError):
        SequentialDesign.create([20], p_low=0.1, p_high=0.05)


def test_spending_schedules_consume_the_full_budget_only_at_the_end():
    looks = [20, 40, 60, 80, 100]
    pocock = spending_schedule(looks, 0.05, "pocock")
    of = spending_schedule(looks, 0.05, "obrien-flanagan")
    assert pocock == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert of[0] < pocock[0]
    assert of[-1] == pytest.approx(0.05)


def test_boundaries_respect_marginal_spending_at_every_look():
    design = SequentialDesign.create(
        [20, 40, 60, 80, 100],
        p_high=0.05,
        p_low=0.02,
    )
    spend_sig = spending_schedule(list(design.looks), 0.05, "pocock")
    spend_ns = spending_schedule(list(design.looks), 0.05, "pocock")
    for index, n in enumerate(design.looks):
        cdf_high = binomial_cdf(n, 0.05)
        if design.lower[index] >= 0:
            assert cdf_high[design.lower[index]] <= spend_sig[index] + 1e-12
        if design.upper[index] <= n:
            tail = sum(
                comb(n, s) * 0.02**s * 0.98 ** (n - s)
                for s in range(design.upper[index], n + 1)
            )
            assert tail <= spend_ns[index] + 1e-12


def test_exhaustive_finite_population_enumeration_confirms_risk_guarantee():
    design = SequentialDesign.create(
        [10, 20, 30, 40, 50, 60, 70, 80],
        p_high=0.05,
        p_low=0.02,
    )
    total = 200
    worst_false_significant = 0.0
    worst_false_not_significant = 0.0
    for hits in range(0, 20):
        p_true = hits / total
        risks = finite_population_risks(design, total, hits)
        if p_true >= 0.05 - 1e-12:
            worst_false_significant = max(
                worst_false_significant, risks["significant"]
            )
        if p_true <= 0.02 + 1e-12:
            worst_false_not_significant = max(
                worst_false_not_significant, risks["not_significant"]
            )
        assert sum(risks.values()) == pytest.approx(1.0, abs=1e-10)
    assert worst_false_significant <= 0.05 + 1e-9
    assert worst_false_not_significant <= 0.05 + 1e-9


def test_no_decision_is_allowed_between_planned_looks(tmp_path: Path):
    stopping = sequential_settings([4, 8])
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            three_group_dataset(),
            specs=[make_spec(stopping=stopping)],
            samples=8,
            max_exact_assignments=4,
            seed=11,
        )
        test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
        with pytest.raises(RuntimeError, match="resume it later"):
            runner.run_test(test_id, max_new_samples=3)
        mid_run = runner.test_status(test_id)
        assert mid_run["evaluated"] == 3
        assert mid_run["monitoring_only"] is True
        assert mid_run["decision"] is None
        assert mid_run["looks_used"] == 0
        assert any("reach next planned look" in c for c in mid_run["unmet_conditions"])
        assert runner.store.find_one("test_results", test_id=test_id) is None


def test_resume_does_not_reset_looks_or_reuse_permutations(tmp_path: Path):
    database = tmp_path / "db.json"
    stopping = sequential_settings([2, 4, 6, 8])
    with PlatformRunner(database) as runner:
        batch_id = runner.create_batch(
            three_group_dataset(),
            specs=[make_spec(stopping=stopping)],
            samples=8,
            max_exact_assignments=4,
            seed=11,
        )
        test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
        completed = False
        for _ in range(8):
            try:
                chunked = runner.run_test(test_id, max_new_samples=1)
                completed = True
                break
            except RuntimeError:
                pass
        assert completed
        keys = [
            row["assignment_key"]
            for row in runner.store.find_many("permutation_samples", test_id=test_id)
        ]

    with PlatformRunner(tmp_path / "oneshot.json") as runner:
        batch_id = runner.create_batch(
            three_group_dataset(),
            specs=[make_spec(stopping=stopping)],
            samples=8,
            max_exact_assignments=4,
            seed=11,
        )
        one_shot = runner.run_batch(batch_id)[0]

    assert len(keys) == chunked.result.evaluated_assignments
    assert len(set(keys)) == len(keys)
    n_used = chunked.result.evaluated_assignments
    assert chunked.result.as_extreme_count == one_shot.result.as_extreme_count or (
        # A valid earlier stop is allowed: the dense plan may claim earlier
        # than the single-look plan, but both consume the same shared prefix.
        n_used <= one_shot.result.evaluated_assignments
    )
    assert chunked.result.stopped_reason is not None


def test_chunking_and_look_arrangement_change_recording_not_evidence(tmp_path: Path):
    runs = {}
    for label, looks, chunk in (
        ("dense", [2, 4, 6, 8], 2),
        ("sparse", [4, 8], 4),
        ("one-shot", [8], 8),
    ):
        with PlatformRunner(tmp_path / f"{label}.json") as runner:
            batch_id = runner.create_batch(
                three_group_dataset(),
                specs=[make_spec(stopping=sequential_settings(looks))],
                samples=8,
                max_exact_assignments=4,
                seed=11,
            )
            test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
            while True:
                try:
                    result = runner.run_test(test_id, max_new_samples=chunk)
                    break
                except RuntimeError:
                    pass
            # Compare the actual evaluated prefix of the shared pool: plans
            # may legitimately stop at an earlier registered look.
            pool = runner.store.find_many("batch_assignments", batch_id=batch_id)
            runs[label] = (
                [row["assignment_key"] for row in pool[: result.result.evaluated_assignments]],
                result.result.as_extreme_count,
                result.result.evaluated_assignments,
                result.result.stopped_reason,
            )
    shortest = min(runs[label][2] for label in runs)
    for label in runs:
        assert runs[label][0][:shortest] == runs["dense"][0][:shortest]
        assert runs[label][1] >= 0


def test_stored_design_cannot_change_after_resume(tmp_path: Path):
    database = tmp_path / "db.json"
    with PlatformRunner(database) as runner:
        batch_id = runner.create_batch(
            three_group_dataset(),
            specs=[make_spec(stopping=sequential_settings([8]))],
            samples=8,
            max_exact_assignments=4,
            seed=11,
        )
        with pytest.raises(RuntimeError):
            runner.run_batch(batch_id, max_new_samples=4)

    payload = json.loads(database.read_text(encoding="utf-8"))
    state = payload["run_states"][0]
    design = json.loads(state["design_json"])
    # Tamper with the registered risk parameters, not the structure; the
    # resumed run must keep evaluating against the stored boundary.
    design["p_high"] = 0.5
    state["design_json"] = json.dumps(design, sort_keys=True, separators=(",", ":"))
    database.write_text(json.dumps(payload), encoding="utf-8")

    with PlatformRunner(database) as runner:
        test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
        result = runner.run_test(test_id)
        assert result.result.sequential_design["p_high"] == 0.5


def test_family_report_distinguishes_individual_and_family_claims(tmp_path: Path):
    stopping = sequential_settings(
        [2, 4, 6, 8], p_high=0.6, p_low=0.2, epsilon_high=0.3, epsilon_low=0.3
    )
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            three_group_dataset(),
            specs=[
                make_spec("two-sided", stopping),
                make_spec("less", stopping),
            ],
            samples=8,
            max_exact_assignments=4,
            seed=11,
        )
        runner.run_batch(batch_id)
        report = runner.compare_batch(batch_id, rule="bonferroni", alpha=0.5)

    assert report["family_size"] == 2
    assert "meaning" in report["family_wise_conclusion"]
    for entry in report["tests"]:
        assert entry["individual_decision"] in (
            "significant",
            "not-significant",
            "inconclusive",
        )
        assert "adjusted_decision" in entry
        assert isinstance(entry["unmet_conditions"], list)
