from pathlib import Path

import pytest

from stat_platform import Dataset, GroupedPermutationTest, PlatformRunner, TestSpec


def triple_dataset() -> Dataset:
    records = []
    values = {"A": (1, 4), "B": (2, 5), "C": (3, 3)}
    for group, (control, treatment) in values.items():
        records.append({"group": group, "condition": "control", "value": control})
        records.append({"group": group, "condition": "treatment", "value": treatment})
    return Dataset.from_records(records, name="triples", version="v1")


def gated_spec(
    key: str,
    alternative: str,
    *,
    parent: str | None = None,
    share: float,
    threshold: float | None = None,
    allocation: dict | None = None,
) -> TestSpec:
    hierarchy = {"key": key, "risk_share": share}
    if parent is not None:
        hierarchy["parent"] = parent
        hierarchy["open_when"] = {"p_value_at_most": threshold}
    if allocation is not None:
        hierarchy["allocation"] = allocation
    return TestSpec(
        GroupedPermutationTest.test_type,
        f"hypothesis {key}",
        {
            "alternative": alternative,
            "conditions": ("control", "treatment"),
            "hierarchy": hierarchy,
        },
    )


def plan() -> list[TestSpec]:
    return [
        gated_spec("P1", "less", share=0.35, allocation={"formal_samples": 8}),
        gated_spec(
            "F1a",
            "less",
            parent="P1",
            share=0.05,
            threshold=0.4,
            allocation={"formal_samples_after_open": 8},
        ),
        gated_spec(
            "F1b",
            "greater",
            parent="P1",
            share=0.05,
            threshold=0.4,
            allocation={"formal_samples_after_open": 8},
        ),
        gated_spec("P2", "two-sided", share=0.1, allocation={"formal_samples": 8}),
        gated_spec(
            "F2",
            "two-sided",
            parent="P2",
            share=0.05,
            threshold=0.4,
            allocation={"formal_samples_after_open": 8},
        ),
    ]


def run_plan(path: Path, chunk: int | None = None) -> tuple[PlatformRunner, int]:
    runner = PlatformRunner(path)
    batch_id = runner.create_batch(
        triple_dataset(),
        specs=plan(),
        samples=8,
        max_exact_assignments=4,
        seed=21,
        family_alpha=0.6,
    )
    if chunk is None:
        runner.run_batch(batch_id)
        return runner, batch_id
    while True:
        try:
            runner.run_batch(batch_id, max_new_samples=chunk)
            return runner, batch_id
        except RuntimeError:
            pass


def test_dependent_hypotheses_open_only_after_primary_gate(tmp_path: Path):
    with PlatformRunner(tmp_path / "db.json") as runner:
        batch_id = runner.create_batch(
            triple_dataset(),
            specs=plan(),
            samples=8,
            max_exact_assignments=4,
            seed=21,
            family_alpha=0.6,
        )
        test_ids = {
            row["hypothesis_key"]: row["id"]
            for row in runner.store.find_many("tests", batch_id=batch_id)
        }
        peek = runner.peek_test(test_ids["F1a"])
        assert peek["gate_status"] == "pending"
        assert peek["formal_status"] == "exploratory-only"
        assert runner.store.find_many("permutation_samples", test_id=test_ids["F1a"]) == []
        assert peek["exploratory_preview"]["persisted_to_test"] is False
        assert peek["exploratory_preview"]["evaluated"] == 8
        with pytest.raises(RuntimeError, match="not open"):
            runner.run_test(test_ids["F1a"])

    with run_plan(tmp_path / "finished.json")[0] as runner:
        batch_id = runner.store.find_many("batches", name="triples:v1")[0]["id"]
        tests = {
            row["hypothesis_key"]: row
            for row in runner.store.find_many("tests", batch_id=batch_id)
        }
        assert runner.get_result(tests["P1"]["id"]).result.p_value == pytest.approx(1 / 3)
        assert runner.get_result(tests["F1a"]["id"]).result.gate_info["formal_status"] == "formal"
        assert runner.get_result(tests["F1b"]["id"]).result.gate_info["formal_status"] == "formal"
        assert runner.store.find_one("run_states", test_id=tests["F2"]["id"])["gate_status"] == "closed"
        with pytest.raises(RuntimeError, match="remained unopened"):
            runner.run_test(tests["F2"]["id"])

        report = runner.compare_batch(batch_id, alpha=0.05)
        by_key = {entry["hypothesis"]: entry for entry in report["tests"]}
        assert by_key["F2"]["formal_status"] == "not-open"
        assert by_key["F2"]["adjusted_reject"] is False
        assert "not a formal conclusion" in by_key["F2"]["individual_meaning"]
        assert report["family"]["closed_test_ids"] == [tests["F2"]["id"]]
        assert report["family"]["planned_family_alpha"] == 0.6
        paths = {tuple(item["path"]): item for item in report["family"]["gate_paths"]}
        assert paths[("P1", "F1a")]["planned_risk_share"] == pytest.approx(0.4)
        assert paths[("P1", "F1a")]["used_risk_share"] == pytest.approx(0.35)
        assert paths[("P1", "F1b")]["formal_claims"][-1]["decision"] == "not-significant"
        assert paths[("P2", "F2")]["status"] == "closed"
        assert paths[("P1", "F1a")]["preplanned_allocation"][1] == {
            "hypothesis": "F1a",
            "formal_samples_after_open": 8,
        }


def test_hierarchical_chunking_and_separate_storage_preserve_gate_records(tmp_path: Path):
    records = []
    for label, chunk in (("one", 1), ("two", 2), ("all", 8)):
        runner, batch_id = run_plan(tmp_path / f"{label}.json", chunk)
        with runner:
            states = []
            for row in sorted(runner.store.find_many("tests", batch_id=batch_id), key=lambda item: item["id"]):
                state = runner.store.find_one("run_states", test_id=row["id"])
                completed = runner.store.find_one("test_results", test_id=row["id"])
                states.append(
                    (
                        row["hypothesis_key"],
                        state["gate_status"],
                        state["evaluated_assignments"],
                        completed is not None,
                    )
                )
            records.append(tuple(states))
    assert len(set(records)) == 1
    assert records[0] == (
        ("P1", "open", 8, True),
        ("F1a", "open", 8, True),
        ("F1b", "open", 8, True),
        ("P2", "open", 8, True),
        ("F2", "closed", 0, False),
    )


def test_every_root_to_leaf_path_must_fit_inside_family_budget(tmp_path: Path):
    specs = plan()
    specs[1].settings["hierarchy"]["risk_share"] = 0.31
    with PlatformRunner(tmp_path / "invalid.json") as runner:
        with pytest.raises(ValueError, match="risk shares"):
            runner.create_batch(
                triple_dataset(),
                specs=specs,
                samples=8,
                max_exact_assignments=4,
                family_alpha=0.6,
            )
