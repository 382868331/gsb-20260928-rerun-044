from pathlib import Path

import pytest

from stat_platform import Dataset, GroupedPermutationTest, PlatformRunner, TestSpec
from stat_platform.adjustment import adjust, bonferroni, holm


def three_group_dataset() -> Dataset:
    records = []
    values = {"A": (1, 4), "B": (2, 5), "C": (3, 3)}
    for group, (control, treatment) in values.items():
        records.append({"group": group, "condition": "control", "value": control})
        records.append({"group": group, "condition": "treatment", "value": treatment})
    return Dataset.from_records(records, name="triples", version="v1")


def make_spec(alternative: str, stopping: dict | None = None) -> TestSpec:
    settings = {"alternative": alternative, "conditions": ("control", "treatment")}
    if stopping is not None:
        settings["stopping"] = stopping
    return TestSpec(
        GroupedPermutationTest.test_type,
        f"equal means ({alternative})",
        settings,
    )


def create_mc_batch(runner: PlatformRunner, specs, **kwargs) -> int:
    options = {"samples": 8, "max_exact_assignments": 4, "seed": 11}
    options.update(kwargs)
    return runner.create_batch(three_group_dataset(), specs=specs, **options)


def sample_keys(runner: PlatformRunner, test_id: int) -> list[str]:
    rows = runner.store.find_many("permutation_samples", test_id=test_id)
    return sorted(row["assignment_key"] for row in rows)


def test_tests_in_one_batch_share_the_same_legal_permutations(tmp_path: Path):
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = create_mc_batch(runner, [make_spec("two-sided"), make_spec("less")])
        results = runner.run_batch(batch_id)

        first_keys = sample_keys(runner, results[0].test_id)
        second_keys = sample_keys(runner, results[1].test_id)
        pool = runner.store.find_many("batch_assignments", batch_id=batch_id)

    assert len(pool) == 8
    assert first_keys == second_keys
    assert len(set(first_keys)) == 8
    assert results[0].null_hypothesis != results[1].null_hypothesis
    assert results[0].result.as_extreme_count != results[1].result.as_extreme_count


def test_monte_carlo_full_cover_matches_exact_enumeration(tmp_path: Path):
    dataset = three_group_dataset()
    exact_two_sided = GroupedPermutationTest(dataset, make_spec("two-sided")).run().result
    exact_less = GroupedPermutationTest(dataset, make_spec("less")).run().result

    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = create_mc_batch(runner, [make_spec("two-sided"), make_spec("less")])
        sampled = runner.run_batch(batch_id)

    assert exact_two_sided.total_assignments == 8
    assert sampled[0].result.as_extreme_count == exact_two_sided.as_extreme_count
    assert sampled[1].result.as_extreme_count == exact_less.as_extreme_count


def test_resume_at_any_boundary_never_recounts_used_permutations(tmp_path: Path):
    database = tmp_path / "db.json"
    with PlatformRunner(database) as runner:
        batch_id = create_mc_batch(runner, [make_spec("two-sided")])
        test_id = runner.store.find_many("tests", batch_id=batch_id)[0]["id"]
        for _ in range(3):
            with pytest.raises(RuntimeError, match="resume it later"):
                runner.run_test(test_id, max_new_samples=2)
        chunked = runner.run_test(test_id)

    with PlatformRunner(tmp_path / "one-shot.json") as runner:
        batch_id = create_mc_batch(runner, [make_spec("two-sided")])
        one_shot = runner.run_batch(batch_id)[0]

    assert chunked.result.p_value == one_shot.result.p_value
    assert chunked.result.as_extreme_count == one_shot.result.as_extreme_count
    with PlatformRunner(database) as runner:
        keys = sample_keys(runner, test_id)
    assert len(keys) == 8
    assert len(set(keys)) == 8


def test_run_order_does_not_change_results_for_same_configuration(tmp_path: Path):
    outcomes = {}
    for label, order in (("forward", (0, 1)), ("reverse", (1, 0))):
        with PlatformRunner(tmp_path / f"{label}.json") as runner:
            batch_id = create_mc_batch(
                runner, [make_spec("two-sided"), make_spec("less")]
            )
            test_ids = [row["id"] for row in runner.store.find_many("tests", batch_id=batch_id)]
            results = [None, None]
            for position in order:
                results[position] = runner.run_test(test_ids[position])
            outcomes[label] = [
                (item.result.p_value, item.result.as_extreme_count) for item in results
            ]
    assert outcomes["forward"] == outcomes["reverse"]


def test_interleaved_chunking_matches_sequential_runs(tmp_path: Path):
    with PlatformRunner(tmp_path / "interleaved.json") as runner:
        batch_id = create_mc_batch(runner, [make_spec("two-sided"), make_spec("less")])
        test_ids = [row["id"] for row in runner.store.find_many("tests", batch_id=batch_id)]
        finished = {}
        pending = list(test_ids)
        while pending:
            for test_id in list(pending):
                try:
                    finished[test_id] = runner.run_test(test_id, max_new_samples=2)
                    pending.remove(test_id)
                except RuntimeError:
                    pass
        interleaved = [finished[tid].result.p_value for tid in test_ids]

    with PlatformRunner(tmp_path / "sequential.json") as runner:
        batch_id = create_mc_batch(runner, [make_spec("two-sided"), make_spec("less")])
        sequential = [item.result.p_value for item in runner.run_batch(batch_id)]

    assert interleaved == sequential


def test_adaptive_stopping_is_recorded_not_treated_as_fixed(tmp_path: Path):
    stopping = {"type": "decision", "alpha": 0.05, "min_samples": 2, "max_samples": 8}
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = create_mc_batch(
            runner,
            [make_spec("two-sided", stopping), make_spec("two-sided")],
        )
        adaptive_result, fixed_result = runner.run_batch(batch_id)

    assert adaptive_result.result.adaptive is True
    assert adaptive_result.result.stopping_rule == "decision"
    assert adaptive_result.result.looks >= 1
    assert adaptive_result.result.stopped_reason is not None
    assert fixed_result.result.adaptive is False
    assert fixed_result.result.stopped_reason is None


def test_adjustment_rules():
    p_values = [0.01, 0.04, 0.20]
    assert bonferroni(p_values) == pytest.approx([0.03, 0.12, 0.6])
    assert holm(p_values) == pytest.approx([0.03, 0.08, 0.20])
    assert adjust(p_values, "none") == p_values
    with pytest.raises(ValueError):
        adjust(p_values, "sidak")


def test_compare_batch_reports_unadjusted_and_adjusted_conclusions(tmp_path: Path):
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = create_mc_batch(runner, [make_spec("two-sided"), make_spec("less")])
        runner.run_batch(batch_id)
        report = runner.compare_batch(batch_id, rule="holm", alpha=0.5)

    assert report["rule"] == "holm"
    assert len(report["tests"]) == 2
    raw = [entry["unadjusted_p_value"] for entry in report["tests"]]
    adjusted = [entry["adjusted_p_value"] for entry in report["tests"]]
    assert adjusted == holm(raw)
    for entry in report["tests"]:
        assert entry["unadjusted_reject"] == (entry["unadjusted_p_value"] <= 0.5)
        assert entry["adjusted_reject"] == (entry["adjusted_p_value"] <= 0.5)
        assert entry["confidence_interval"] is not None
        assert entry["adjusted_confidence_interval"] is not None
        low, high = entry["adjusted_confidence_interval"]
        assert low <= entry["adjusted_p_value"] <= high or low == 0.0
        assert entry["adjusted_confidence_level"] > entry["confidence_level"]
