from pathlib import Path

import pytest

from stat_platform import Dataset, GroupedPermutationTest, PlatformRunner, TestSpec


def paired_dataset() -> Dataset:
    records = []
    for group, (control, treatment) in {"A": (1, 4), "B": (2, 5)}.items():
        records.append({"group": group, "condition": "control", "value": control})
        records.append({"group": group, "condition": "treatment", "value": treatment})
    return Dataset.from_records(records, name="paired", version="v1")


def routed_spec(key: str, share: float, reallocate_to: str | None = None) -> TestSpec:
    hierarchy: dict = {"key": key, "risk_share": share}
    if reallocate_to is not None:
        hierarchy["reallocate_to"] = reallocate_to
    return TestSpec(
        GroupedPermutationTest.test_type,
        f"hypothesis {key}",
        {
            "alternative": "two-sided",
            "conditions": ("control", "treatment"),
            "hierarchy": hierarchy,
        },
    )


def create(
    runner: PlatformRunner, specs: list[TestSpec], family_alpha: float = 0.05
) -> int:
    return runner.create_batch(
        paired_dataset(),
        specs=specs,
        samples=4,
        max_exact_assignments=4,
        seed=7,
        family_alpha=family_alpha,
    )


def transfers(runner: PlatformRunner, batch_id: int) -> list[dict]:
    return runner.store.find_many("share_transfers", batch_id=batch_id)


def ledger_by_key(report: dict) -> dict:
    return {
        entry["hypothesis"]: entry for entry in report["family"]["risk_ledger"]
    }


class CheckCycleRejection:
    def test_two_node_cycle_names_nodes_and_edges(self, tmp_path: Path):
        specs = [
            routed_spec("R1", 0.02, reallocate_to="R2"),
            routed_spec("R2", 0.02, reallocate_to="R1"),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="R1 -> R2 -> R1") as excinfo:
                create(runner, specs)
            message = str(excinfo.value)
            assert "R1->R2" in message and "R2->R1" in message
            assert "regardless of how small" in message
            # Rejected before anything is persisted.
            assert runner.store.find_many("batches") == []
            assert runner.store.find_many("share_transfers") == []

    def test_three_node_cycle_names_all_edges(self, tmp_path: Path):
        specs = [
            routed_spec("R1", 0.01, reallocate_to="R2"),
            routed_spec("R2", 0.01, reallocate_to="R3"),
            routed_spec("R3", 0.01, reallocate_to="R1"),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="reallocation cycle") as excinfo:
                create(runner, specs)
            message = str(excinfo.value)
            for edge in ("R1->R2", "R2->R3", "R3->R1"):
                assert edge in message
            assert runner.store.find_many("batches") == []

    def test_self_loop_is_a_cycle(self, tmp_path: Path):
        specs = [routed_spec("R1", 0.02, reallocate_to="R1")]
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="reallocation|itself"):
                create(runner, specs)


class CheckAcyclicRouting:
    def test_shared_successor_receives_each_share_exactly_once(self, tmp_path: Path):
        specs = [
            routed_spec("R1", 0.02, reallocate_to="S"),
            routed_spec("R2", 0.02, reallocate_to="S"),
            routed_spec("S", 0.01),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, specs)
            runner.run_batch(batch_id)
            first = runner.compare_batch(batch_id)
            second = runner.compare_batch(batch_id)
        rows = transfers(runner, batch_id)
        assert len(rows) == 2
        assert {(row["source_key"], row["target_key"]) for row in rows} == {
            ("R1", "S"),
            ("R2", "S"),
        }
        assert all(row["amount"] == pytest.approx(0.02) for row in rows)
        by_key = ledger_by_key(first)
        assert len(by_key["S"]["transfers_in"]) == 2
        assert by_key["S"]["effective_allocation"] == pytest.approx(0.05)
        # Repeated report reads never add allocations.
        assert second["family"]["risk_ledger"] == first["family"]["risk_ledger"]

    def test_acyclic_chain_and_branch_transfer_own_share_once(
        self, tmp_path: Path
    ):
        specs = [
            routed_spec("R1", 0.02, reallocate_to="R2"),
            routed_spec("R2", 0.02, reallocate_to="R3"),
            routed_spec("R4", 0.01, reallocate_to="R3"),
            routed_spec("R3", 0.005),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, specs, family_alpha=0.055)
            runner.run_batch(batch_id)
            report = runner.compare_batch(batch_id)
            report_again = runner.compare_batch(batch_id)
        rows = {
            (row["source_key"], row["target_key"]): row
            for row in transfers(runner, batch_id)
        }
        assert set(rows) == {("R1", "R2"), ("R2", "R3"), ("R4", "R3")}
        # Each edge moves the source's own unused share exactly once; shares
        # received by R2 do not silently propagate further along the chain.
        assert rows[("R1", "R2")]["amount"] == pytest.approx(0.02)
        assert rows[("R2", "R3")]["amount"] == pytest.approx(0.02)
        assert rows[("R4", "R3")]["amount"] == pytest.approx(0.01)
        by_key = ledger_by_key(report)
        assert by_key["R2"]["effective_allocation"] == pytest.approx(0.04)
        assert by_key["R3"]["effective_allocation"] == pytest.approx(0.035)
        assert report_again["family"]["risk_ledger"] == report["family"]["risk_ledger"]
        routes = report["family"]["reallocation_routes"]
        assert routes["version"] == 1
        assert routes["edges"] == [
            {"from": "R1", "to": "R2"},
            {"from": "R2", "to": "R3"},
            {"from": "R4", "to": "R3"},
        ]


class CheckResumeAndImmutability:
    def test_partial_run_resume_restores_routes_and_transfer_history(
        self, tmp_path: Path
    ):
        specs = [
            routed_spec("R1", 0.02, reallocate_to="R2"),
            routed_spec("R2", 0.02),
        ]
        path = tmp_path / "db.json"
        with PlatformRunner(path) as runner:
            batch_id = create(runner, specs)
            test_ids = [
                row["id"] for row in runner.store.find_many("tests", batch_id=batch_id)
            ]
            # Partial execution: one persisted chunk, then stop.
            assert runner.advance_test(test_ids[0], max_new_samples=2) is None
        with PlatformRunner(path) as runner:
            batch = runner.store.find_one("batches", id=batch_id)
            assert batch["route_version"] == 1
            assert batch["reallocation_routes_json"] is not None
            runner.run_batch(batch_id)
            runner.compare_batch(batch_id)
            runner.compare_batch(batch_id)
            rows = transfers(runner, batch_id)
        assert len(rows) == 1
        assert rows[0]["source_key"] == "R1"
        assert rows[0]["target_key"] == "R2"
        assert rows[0]["amount"] == pytest.approx(0.02)
        assert rows[0]["route_version"] == 1

    def test_plan_change_cannot_rewrite_saved_routes_or_transfers(
        self, tmp_path: Path
    ):
        specs = [
            routed_spec("R1", 0.02, reallocate_to="R2"),
            routed_spec("R2", 0.02),
            routed_spec("R3", 0.01),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, specs)
            runner.run_batch(batch_id)
            runner.compare_batch(batch_id)
            before = list(transfers(runner, batch_id))
            # R1's route is already recorded; pointing it elsewhere would be a
            # retroactive rewrite and is refused.
            rewrite = routed_spec("R1", 0.02, reallocate_to="R3")
            with pytest.raises(ValueError, match="retroactively rewritten"):
                runner.add_test(batch_id, rewrite, samples=4)
            # A brand-new edge is validated against the persisted routes and
            # bumps the route version without touching saved transfers.
            cycle_attempt = routed_spec("R3", 0.01, reallocate_to="R1")
            runner.add_test(batch_id, cycle_attempt, samples=4)
            assert transfers(runner, batch_id) == before
            batch = runner.store.find_one("batches", id=batch_id)
            assert batch["route_version"] == 2
            # The recorded transfer evidence is untouched by the plan change.
            assert before[0]["route_version"] == 1
