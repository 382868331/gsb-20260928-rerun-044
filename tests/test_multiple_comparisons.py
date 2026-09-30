from pathlib import Path

import pytest

from stat_platform import (
    Dataset,
    GroupedPermutationTest,
    PlatformRunner,
    TestSpec,
    adjust_p_values,
)


def paired_dataset() -> Dataset:
    return Dataset.from_records(
        [
            {"group": "A", "condition": "control", "value": 1},
            {"group": "A", "condition": "treatment", "value": 3},
            {"group": "B", "condition": "control", "value": 2},
            {"group": "B", "condition": "treatment", "value": 4},
        ],
        name="paired",
        version="v1",
    )


def make_spec(alternative: str, null: str) -> TestSpec:
    return TestSpec(
        GroupedPermutationTest.test_type,
        null,
        {"alternative": alternative, "conditions": ("control", "treatment")},
    )


def wide_dataset() -> Dataset:
    records = []
    for group in ("A", "B", "C", "D", "E"):
        records.append({"group": group, "condition": "control", "value": 1})
        records.append({"group": group, "condition": "treatment", "value": 2})
    return Dataset.from_records(records, name="wide", version="v1")


def test_shared_permutations_are_stored_once_per_batch(tmp_path: Path):
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            paired_dataset(),
            specs=[
                make_spec("two-sided", "equal means"),
                make_spec("less", "control not greater"),
                make_spec("greater", "control not smaller"),
            ],
        )
        results = runner.run_batch(batch_id)
        shared = runner.store.find_many("batch_permutations", batch_id=batch_id)
        assert len(shared) == 4
        assert len({row["assignment_key"] for row in shared}) == 4
        for result in results:
            samples = runner.store.find_many("permutation_samples", test_id=result.test_id)
            assert len(samples) == 4
            assert {row["batch_ordinal"] for row in samples} == {1, 2, 3, 4}
        assert [item.result.p_value for item in results] == [0.5, 0.25, 1.0]


def test_compare_batch_reports_unadjusted_and_adjusted(tmp_path: Path):
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            paired_dataset(),
            specs=[
                make_spec("two-sided", "equal means"),
                make_spec("less", "control not greater"),
                make_spec("greater", "control not smaller"),
            ],
        )
        runner.run_batch(batch_id)
        report = runner.compare_batch(batch_id, method="bonferroni")

    assert report["adjustment_method"] == "bonferroni"
    assert report["pending_tests"] == []
    raw = [item["p_value"] for item in report["comparisons"]]
    adjusted = [item["adjusted_p_value"] for item in report["comparisons"]]
    assert raw == [0.5, 0.25, 1.0]
    assert adjusted == [1.0, 0.75, 1.0]
    holm = adjust_p_values(raw, "holm")
    assert holm == [1.0, 0.75, 1.0]
    bonf = adjust_p_values([0.01, 0.04, 0.05], "bonferroni")
    assert bonf == pytest.approx([0.03, 0.12, 0.15])
    holm_unit = adjust_p_values([0.01, 0.04, 0.05], "holm")
    assert holm_unit == pytest.approx([0.03, 0.08, 0.08])
    bh = adjust_p_values([0.01, 0.04, 0.05], "benjamini-hochberg")
    assert bh == pytest.approx([0.03, 0.05, 0.05])


def test_monte_carlo_matches_full_enumeration_on_tiny_space(tmp_path: Path):
    dataset = wide_dataset()
    spec = make_spec("two-sided", "equal means")
    engine = GroupedPermutationTest(dataset, spec)
    exact = engine.run().result
    assert exact.method == "exact"

    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            dataset,
            specs=[spec],
            samples=engine.total_assignments,
            max_exact_assignments=0,
            seed=99,
        )
        (result,) = runner.run_batch(batch_id)
        shared = runner.store.find_many("batch_permutations", batch_id=batch_id)

    assert result.result.method == "monte-carlo"
    assert len(shared) == engine.total_assignments
    assert len({row["assignment_key"] for row in shared}) == engine.total_assignments
    assert result.result.as_extreme_count == exact.as_extreme_count


def run_config(tmp_path: Path, order: tuple[int, ...]) -> dict[int, tuple[float, int]]:
    with PlatformRunner(tmp_path) as runner:
        batch_id = runner.create_batch(
            wide_dataset(),
            specs=[
                make_spec("two-sided", "equal means"),
                make_spec("greater", "control not smaller"),
            ],
            samples=20,
            max_exact_assignments=0,
            seed=20260930,
        )
        for test_id in order:
            runner.run_test(test_id)
        return {
            test_id: (
                runner.get_result(test_id).result.p_value,
                runner.get_result(test_id).result.as_extreme_count,
            )
            for test_id in (1, 2)
        }


def test_run_order_does_not_change_results(tmp_path: Path):
    forward = run_config(tmp_path / "forward.json", (1, 2))
    backward = run_config(tmp_path / "backward.json", (2, 1))
    assert forward == backward


def test_checkpoint_resume_never_double_counts(tmp_path: Path):
    database = tmp_path / "db.json"
    with PlatformRunner(database) as runner:
        batch_id = runner.create_batch(
            wide_dataset(),
            specs=[make_spec("two-sided", "equal means")],
            samples=25,
            max_exact_assignments=0,
            seed=7,
        )
        with pytest.raises(RuntimeError):
            runner.run_batch(batch_id, max_new_samples=10)
        with pytest.raises(RuntimeError):
            runner.run_batch(batch_id, max_new_samples=10)
        (result,) = runner.run_batch(batch_id)
        shared = runner.store.find_many("batch_permutations", batch_id=batch_id)
        samples = runner.store.find_many("permutation_samples", test_id=result.test_id)

    assert len(shared) == 25
    assert len({row["assignment_key"] for row in shared}) == 25
    assert len(samples) == 25
    assert len({row["assignment_key"] for row in samples}) == 25
    assert result.result.evaluated_assignments == 25
    assert result.result.looks == 3
    assert result.result.stopping_reason == "target_samples"
    assert result.result.unadjusted_confidence_interval is not None
    assert result.result.confidence_level > 0.95

    with PlatformRunner(tmp_path / "oneshot.json") as runner:
        batch_id = runner.create_batch(
            wide_dataset(),
            specs=[make_spec("two-sided", "equal means")],
            samples=25,
            max_exact_assignments=0,
            seed=7,
        )
        (oneshot,) = runner.run_batch(batch_id)
    assert oneshot.result.p_value == result.result.p_value
    assert oneshot.result.as_extreme_count == result.result.as_extreme_count


def test_adaptive_precision_stop_is_recorded_not_fixed_horizon(tmp_path: Path):
    spec = TestSpec(
        GroupedPermutationTest.test_type,
        "equal means",
        {
            "alternative": "two-sided",
            "conditions": ("control", "treatment"),
            "adaptive": {"min_samples": 25, "ci_half_width": 1.0},
        },
    )
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            wide_dataset(),
            specs=[spec],
            samples=32,
            max_exact_assignments=0,
            seed=3,
        )
        with pytest.raises(RuntimeError):
            runner.run_batch(batch_id, max_new_samples=10)
        with pytest.raises(RuntimeError):
            runner.run_batch(batch_id, max_new_samples=10)
        (result,) = runner.run_batch(batch_id, max_new_samples=10)

    assert result.result.method == "monte-carlo"
    assert result.result.stopping_reason == "precision_target"
    assert result.result.evaluated_assignments == 30
    assert result.result.looks == 3
    assert result.result.confidence_level == pytest.approx(1 - 0.05 / 3)
    naive = result.result.unadjusted_confidence_interval
    adjusted = result.result.confidence_interval
    assert naive is not None
    assert (adjusted[1] - adjusted[0]) >= (naive[1] - naive[0])
