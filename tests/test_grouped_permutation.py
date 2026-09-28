from pathlib import Path

import pytest

from stat_platform import Dataset, GroupedPermutationTest, PlatformRunner, TestSpec


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


def test_exact_enumeration_keeps_linked_observations_within_groups():
    spec = TestSpec(
        GroupedPermutationTest.test_type,
        "equal paired means",
        {"alternative": "two-sided", "conditions": ("control", "treatment")},
    )
    run = GroupedPermutationTest(paired_dataset(), spec).run()

    assert run.result.method == "exact"
    assert run.result.total_assignments == 4
    assert run.result.evaluated_assignments == 4
    assert run.result.p_value == 0.5
    assert run.result.standard_error is None


def test_one_sided_exact_probability_is_not_treated_as_random_estimate():
    spec = TestSpec(
        GroupedPermutationTest.test_type,
        "treatment not less than control",
        {"alternative": "less", "conditions": ("control", "treatment")},
    )
    result = GroupedPermutationTest(paired_dataset(), spec).run().result

    assert result.method == "exact"
    assert result.p_value == 0.25
    assert result.as_extreme_count == 1


def test_database_resume_completes_without_recomputing_existing_samples(tmp_path: Path):
    database = tmp_path / "experiments.db"
    spec = TestSpec(
        GroupedPermutationTest.test_type,
        "equal paired means",
        {"alternative": "two-sided", "conditions": ("control", "treatment")},
    )

    with PlatformRunner(database) as runner:
        batch_id = runner.create_batch(paired_dataset(), specs=[spec])
        with pytest.raises(RuntimeError, match="resume it later"):
            runner.run_batch(batch_id, max_new_samples=2)
        result = runner.run_batch(batch_id)[0]
        resumed = runner.run_batch(batch_id)[0]

    assert result.test_id == resumed.test_id
    assert result.null_hypothesis == "equal paired means"
    assert resumed.result.method == "exact"
    assert resumed.result.p_value == 0.5

    with PlatformRunner(database) as runner:
        count = len(runner.store.find_many("permutation_samples", test_id=result.test_id))
    assert count == 4


def test_monte_carlo_reports_standard_error_and_reproducible_seed(tmp_path: Path):
    dataset = Dataset.from_records(
        [
            {"group": "g", "condition": "a", "value": 1},
            {"group": "g", "condition": "b", "value": 2},
        ],
        name="small",
        version="v1",
    )
    spec = TestSpec(
        GroupedPermutationTest.test_type,
        "equal means",
        {"alternative": "two-sided", "conditions": ("a", "b")},
    )
    first = GroupedPermutationTest(dataset, spec).run(
        samples=1,
        max_exact_assignments=1,
        seed=7,
    ).result
    second = GroupedPermutationTest(dataset, spec).run(
        samples=1,
        max_exact_assignments=1,
        seed=7,
    ).result

    assert first.method == "monte-carlo"
    assert first.standard_error is not None
    assert first.confidence_interval is not None
    assert first.p_value == second.p_value
