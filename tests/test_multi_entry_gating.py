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
    share: float,
    parents: list[str] | None = None,
    threshold: float | None = None,
    parent_mode: str | None = None,
    reallocate_to: str | None = None,
) -> TestSpec:
    hierarchy: dict = {"key": key, "risk_share": share}
    if parents:
        hierarchy["parent"] = list(parents)
        hierarchy["open_when"] = {"p_value_at_most": threshold}
    if parent_mode is not None:
        hierarchy["parent_mode"] = parent_mode
    if reallocate_to is not None:
        hierarchy["reallocate_to"] = reallocate_to
    return TestSpec(
        GroupedPermutationTest.test_type,
        f"hypothesis {key}",
        {
            "alternative": alternative,
            "conditions": ("control", "treatment"),
            "hierarchy": hierarchy,
        },
    )


def create(
    runner: PlatformRunner,
    specs: list[TestSpec],
    family_alpha: float = 0.6,
    samples: int = 8,
) -> int:
    return runner.create_batch(
        triple_dataset(),
        specs=specs,
        samples=samples,
        max_exact_assignments=4,
        seed=21,
        family_alpha=family_alpha,
    )


def run_fully(runner: PlatformRunner, batch_id: int, chunk: int | None = None) -> None:
    if chunk is None:
        runner.run_batch(batch_id)
        return
    while True:
        try:
            runner.run_batch(batch_id, max_new_samples=chunk)
            return
        except RuntimeError:
            pass


def states_by_key(runner: PlatformRunner, batch_id: int) -> dict:
    states = {}
    for row in runner.store.find_many("tests", batch_id=batch_id):
        state = runner.store.find_one("run_states", test_id=row["id"])
        completed = runner.store.find_one("test_results", test_id=row["id"])
        states[row["hypothesis_key"]] = (
            state["gate_status"],
            state["evaluated_assignments"],
            completed is not None,
        )
    return states


class CheckFamilyBudget:
    def test_parallel_primaries_must_fit_the_family_budget(self, tmp_path: Path):
        db = tmp_path / "db.json"
        with PlatformRunner(db) as runner:
            keep_id = create(
                runner,
                [gated_spec("P1", "less", share=0.02)],
                family_alpha=0.05,
            )
            run_fully(runner, keep_id)
            snapshot = {table: list(rows) for table, rows in runner.store.data.items()}
            specs = [
                gated_spec("Q1", "less", share=0.04),
                gated_spec("Q2", "greater", share=0.04),
            ]
            with pytest.raises(ValueError, match="risk shares") as excinfo:
                create(runner, specs, family_alpha=0.05)
        message = str(excinfo.value)
        assert "Q1" in message and "Q2" in message
        assert "0.04" in message and "0.05" in message
        with PlatformRunner(db) as runner:
            # The rejected plan persisted nothing; earlier results are intact.
            assert runner.store.data == snapshot
            assert runner.get_result(keep_id).result.p_value is not None

    def test_path_check_alone_would_accept_this_plan(self, tmp_path: Path):
        # Each root-to-leaf path is just one root at 0.04 <= 0.05, so only the
        # family-wide budget check can reject the combined 0.08 allocation.
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="parallel primary"):
                create(
                    runner,
                    [
                        gated_spec("Q1", "less", share=0.04),
                        gated_spec("Q2", "greater", share=0.04),
                    ],
                    family_alpha=0.05,
                )


class CheckMultiParentGates:
    def plan(self, mode: str | None) -> list[TestSpec]:
        return [
            gated_spec("P1", "less", share=0.35),
            gated_spec("P2", "two-sided", share=0.2),
            gated_spec(
                "F",
                "less",
                share=0.2,
                parents=["P1", "P2"],
                threshold=0.4,
                parent_mode=mode,
            ),
        ]

    def test_multiple_parents_require_a_predeclared_mode(self, tmp_path: Path):
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="parent_mode"):
                create(runner, self.plan(None))

    def test_any_mode_opens_when_one_parent_satisfies(self, tmp_path: Path):
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, self.plan("any"))
            run_fully(runner, batch_id)
            tests = {
                row["hypothesis_key"]: row
                for row in runner.store.find_many("tests", batch_id=batch_id)
            }
            state = runner.store.find_one("run_states", test_id=tests["F"]["id"])
            assert state["gate_status"] == "open"
            assert state["gate_opened_via_json"] == '["P1"]'
            # The shared follow-up receives exactly one formal allocation:
            # two opening paths never let it draw its budget twice.
            assert state["evaluated_assignments"] == 8
            assert runner.get_result(tests["F"]["id"]).result.evaluated_assignments == 8

    def test_all_mode_closes_when_one_parent_fails(self, tmp_path: Path):
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, self.plan("all"))
            run_fully(runner, batch_id)
            tests = {
                row["hypothesis_key"]: row
                for row in runner.store.find_many("tests", batch_id=batch_id)
            }
            state = runner.store.find_one("run_states", test_id=tests["F"]["id"])
            assert state["gate_status"] == "closed"
            with pytest.raises(RuntimeError, match="remained unopened"):
                runner.run_test(tests["F"]["id"])

    def test_chunked_resume_matches_uninterrupted_run(self, tmp_path: Path):
        records = []
        for label, chunk in (("one", 1), ("three", 3), ("all", None)):
            with PlatformRunner(tmp_path / f"{label}.json") as runner:
                batch_id = create(runner, self.plan("any"))
                run_fully(runner, batch_id, chunk)
                records.append(states_by_key(runner, batch_id))
                report = runner.compare_batch(batch_id)
                records.append(
                    {
                        tuple(item["path"]): (
                            item["planned_risk_share"],
                            item["used_risk_share"],
                            item["status"],
                        )
                        for item in report["family"]["gate_paths"]
                    }
                )
        assert records[0] == records[2] == records[4]
        assert records[1] == records[3] == records[5]


class CheckDuplicateParentDeclarations:
    @pytest.mark.parametrize(
        "parents",
        [
            ["P1", "P1"],
            ["P1", "P2", "P1"],
            ["P2", "P1", "P2", "P1"],
        ],
    )
    def test_repeated_parent_for_one_node_is_rejected_and_located(
        self, tmp_path: Path, parents: list[str]
    ):
        db = tmp_path / "db.json"
        specs = [
            gated_spec("P1", "less", share=0.2),
            gated_spec("P2", "two-sided", share=0.2),
            gated_spec(
                "F",
                "less",
                share=0.2,
                parents=parents,
                threshold=0.4,
                parent_mode="all",
            ),
        ]
        with PlatformRunner(db) as runner:
            with pytest.raises(ValueError, match="duplicate parent") as excinfo:
                create(runner, specs, family_alpha=0.6)
        message = str(excinfo.value)
        # The error names the offending hypothesis node and the repeated source.
        assert "F" in message
        assert "P1" in message
        with PlatformRunner(db) as runner:
            # A rejected plan persists nothing; the error cannot reach any
            # later allocation computation.
            assert runner.store.find_many("batches") == []

    def test_distinct_parents_sharing_a_successor_remain_legal(self, tmp_path: Path):
        specs = [
            gated_spec("P1", "less", share=0.2),
            gated_spec("P2", "two-sided", share=0.2),
            gated_spec(
                "F",
                "less",
                share=0.2,
                parents=["P1", "P2"],
                threshold=0.4,
                parent_mode="any",
            ),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, specs, family_alpha=0.6)
            run_fully(runner, batch_id)
            report = runner.compare_batch(batch_id)
            tests = {
                row["hypothesis_key"]: row
                for row in runner.store.find_many("tests", batch_id=batch_id)
            }
            state = runner.store.find_one("run_states", test_id=tests["F"]["id"])
        assert state["gate_status"] == "open"
        # Two distinct parents contribute one shared gate path each, not two
        # copies of the same parent edge.
        assert sorted(
            tuple(item["path"]) for item in report["family"]["gate_paths"]
        ) == [("P1", "F"), ("P2", "F")]

    def test_two_node_gate_cycle_is_rejected(self, tmp_path: Path):
        specs = [
            gated_spec(
                "A",
                "less",
                share=0.2,
                parents=["B"],
                threshold=0.4,
            ),
            gated_spec(
                "B",
                "less",
                share=0.2,
                parents=["A"],
                threshold=0.4,
            ),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="hypothesis gates") as excinfo:
                create(runner, specs, family_alpha=0.6)
        message = str(excinfo.value)
        assert "A -> B -> A" in message
        assert "A->B" in message and "B->A" in message

    def test_legal_chain_history_is_not_inflated_by_repeat_reads_or_resume(
        self, tmp_path: Path
    ):
        db = tmp_path / "db.json"
        specs = [
            gated_spec("A", "two-sided", share=0.2, reallocate_to="B"),
            gated_spec("B", "two-sided", share=0.2, reallocate_to="C"),
            gated_spec("C", "less", share=0.2),
        ]
        with PlatformRunner(db) as runner:
            batch_id = create(runner, specs, family_alpha=0.6)
            run_fully(runner, batch_id)
            first = runner.compare_batch(batch_id)
            second = runner.compare_batch(batch_id)
            transfers = [
                (item["source_key"], item["target_key"], item["amount"])
                for item in runner.store.find_many("share_transfers", batch_id=batch_id)
            ]
        assert first["family"]["risk_ledger"] == second["family"]["risk_ledger"]
        assert transfers == [("A", "B", pytest.approx(0.2)), ("B", "C", pytest.approx(0.4))]

        # Reopening the database (recovery) must not re-read or re-insert any
        # already persisted allocation.
        with PlatformRunner(db) as runner:
            third = runner.compare_batch(batch_id)
            recovered = [
                (item["source_key"], item["target_key"], item["amount"])
                for item in runner.store.find_many("share_transfers", batch_id=batch_id)
            ]
        assert third["family"]["risk_ledger"] == first["family"]["risk_ledger"]
        assert recovered == transfers


class CheckShareReallocation:
    def plan(self) -> list[TestSpec]:
        return [
            gated_spec("P1", "less", share=0.35, reallocate_to="F"),
            gated_spec("P2", "two-sided", share=0.2, reallocate_to="F"),
            gated_spec("F", "less", share=0.2, parents=["P1"], threshold=0.4),
        ]

    def test_failed_parent_share_moves_once_and_used_share_never_returns(
        self, tmp_path: Path
    ):
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, self.plan())
            run_fully(runner, batch_id)
            first = runner.compare_batch(batch_id)
            second = runner.compare_batch(batch_id)
        ledger = first["family"]["risk_ledger"]
        by_key = {entry["hypothesis"]: entry for entry in ledger}
        # P2 finished not-significant: its unused 0.2 moved to F exactly once.
        assert by_key["P2"]["transfers_out"] == [
            {
                "to": "F",
                "amount": pytest.approx(0.2),
                "reason": "decision-not-significant",
            }
        ]
        assert by_key["P2"]["remaining_risk_share"] == pytest.approx(0.0)
        # P1 was formally significant: its share is used and never reallocates.
        assert by_key["P1"]["transfers_out"] == []
        assert by_key["P1"]["used_risk_share"] == pytest.approx(0.35)
        # F holds one initial allocation plus one transferred share.
        assert by_key["F"]["initial_allocation"] == pytest.approx(0.2)
        assert by_key["F"]["transfers_in"] == [
            {
                "from": "P2",
                "amount": pytest.approx(0.2),
                "reason": "decision-not-significant",
            }
        ]
        assert by_key["F"]["effective_allocation"] == pytest.approx(0.4)
        # Repeated reports never duplicate the transfer.
        assert second["family"]["risk_ledger"] == ledger
        entries = {item["hypothesis"]: item for item in first["tests"]}
        assert entries["F"]["effective_risk_share"] == pytest.approx(0.4)
        assert entries["F"]["adjusted_reject"] is True

    def test_closed_follow_up_share_moves_to_declared_target(self, tmp_path: Path):
        specs = [
            gated_spec("P1", "two-sided", share=0.2),
            gated_spec("S", "less", share=0.2),
            gated_spec(
                "F",
                "less",
                share=0.2,
                parents=["P1"],
                threshold=0.4,
                reallocate_to="S",
            ),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, specs)
            run_fully(runner, batch_id)
            report = runner.compare_batch(batch_id)
        by_key = {
            entry["hypothesis"]: entry for entry in report["family"]["risk_ledger"]
        }
        assert by_key["F"]["gate_status"] == "closed"
        assert by_key["F"]["transfers_out"] == [
            {"to": "S", "amount": pytest.approx(0.2), "reason": "gate-closed"}
        ]
        assert by_key["S"]["effective_allocation"] == pytest.approx(0.4)

    def test_unknown_reallocation_target_is_rejected(self, tmp_path: Path):
        specs = [gated_spec("P1", "less", share=0.2, reallocate_to="MISSING")]
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="reallocation target"):
                create(runner, specs)


class CheckReallocationRoutes:
    def test_two_node_reallocation_cycle_reports_nodes_and_edges(self, tmp_path: Path):
        specs = [
            gated_spec("R1", "less", share=0.01, reallocate_to="R2"),
            gated_spec("R2", "less", share=0.01, reallocate_to="R1"),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="reallocation routes") as excinfo:
                create(runner, specs, family_alpha=0.05)
        message = str(excinfo.value)
        assert "R1 -> R2 -> R1" in message
        assert "R1->R2" in message and "R2->R1" in message
        assert runner.store.find_many("batches") == []

    def test_three_node_reallocation_cycle_is_rejected_even_with_small_shares(
        self, tmp_path: Path
    ):
        specs = [
            gated_spec("R1", "less", share=0.001, reallocate_to="R2"),
            gated_spec("R2", "less", share=0.001, reallocate_to="R3"),
            gated_spec("R3", "less", share=0.001, reallocate_to="R1"),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            with pytest.raises(ValueError, match="R1 -> R2 -> R3 -> R1") as excinfo:
                create(runner, specs, family_alpha=0.05)
        message = str(excinfo.value)
        assert "R3->R1" in message
        assert runner.store.find_many("batches") == []

    def test_shared_successor_and_legal_acyclic_branch_transfer_once(self, tmp_path: Path):
        specs = [
            gated_spec("P1", "two-sided", share=0.2, reallocate_to="SHARED"),
            gated_spec("P2", "two-sided", share=0.2, reallocate_to="SHARED"),
            gated_spec("P3", "two-sided", share=0.2, reallocate_to="BRANCH"),
            gated_spec("SHARED", "less", share=0.1),
            gated_spec("BRANCH", "less", share=0.1),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, specs, family_alpha=0.8)
            run_fully(runner, batch_id)
            report = runner.compare_batch(batch_id)
            transfers = runner.store.find_many("share_transfers", batch_id=batch_id)
        by_key = {entry["hypothesis"]: entry for entry in report["family"]["risk_ledger"]}
        incoming = {
            item["from"] for item in by_key["SHARED"]["transfers_in"]
        }
        assert incoming == {"P1", "P2"}
        assert by_key["BRANCH"]["transfers_in"] == [
            {"from": "P3", "amount": pytest.approx(0.2), "reason": "decision-not-significant"}
        ]
        assert [(item["source_key"], item["target_key"]) for item in transfers] == [
            ("P1", "SHARED"),
            ("P2", "SHARED"),
            ("P3", "BRANCH"),
        ]
        assert len(transfers) == 3

    def test_legal_acyclic_chain_forwards_unused_received_share_once(self, tmp_path: Path):
        specs = [
            gated_spec("A", "two-sided", share=0.2, reallocate_to="B"),
            gated_spec("B", "two-sided", share=0.2, reallocate_to="C"),
            gated_spec("C", "less", share=0.2),
        ]
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = create(runner, specs, family_alpha=0.6)
            run_fully(runner, batch_id)
            report = runner.compare_batch(batch_id)
            transfers = runner.store.find_many("share_transfers", batch_id=batch_id)
        by_key = {entry["hypothesis"]: entry for entry in report["family"]["risk_ledger"]}
        assert by_key["A"]["transfers_out"][0]["amount"] == pytest.approx(0.2)
        assert by_key["B"]["transfers_out"][0]["amount"] == pytest.approx(0.4)
        assert by_key["C"]["effective_allocation"] == pytest.approx(0.6)
        assert [(item["source_key"], item["target_key"]) for item in transfers] == [
            ("A", "B"),
            ("B", "C"),
        ]

    def test_partial_report_and_resume_preserve_route_version_and_transfer_history(
        self, tmp_path: Path
    ):
        db = tmp_path / "db.json"
        specs = [
            gated_spec("P1", "less", share=0.35, reallocate_to="F"),
            gated_spec("P2", "two-sided", share=0.2, reallocate_to="F"),
            gated_spec("F", "less", share=0.2, parents=["P1"], threshold=0.4),
        ]
        with PlatformRunner(db) as runner:
            batch_id = create(runner, specs)
            tests = {
                row["hypothesis_key"]: row
                for row in runner.store.find_many("tests", batch_id=batch_id)
            }
            runner.run_test(tests["P1"]["id"])
            runner.run_test(tests["P2"]["id"])
            partial = runner.compare_batch(batch_id)
            historical = list(runner.store.find_many("share_transfers", batch_id=batch_id))
            assert partial["family"]["route_version"] == 1
            assert [(item["source_key"], item["target_key"], item["route_version"]) for item in historical] == [
                ("P2", "F", 1)
            ]
            with pytest.raises(ValueError, match="R1 -> R2 -> R1"):
                create(
                    runner,
                    [
                        gated_spec("R1", "less", share=0.01, reallocate_to="R2"),
                        gated_spec("R2", "less", share=0.01, reallocate_to="R1"),
                    ],
                )
            assert runner.store.find_many("share_transfers", batch_id=batch_id) == historical

        with PlatformRunner(db) as runner:
            run_fully(runner, batch_id)
            first = runner.compare_batch(batch_id)
            second = runner.compare_batch(batch_id)
            transfers = runner.store.find_many("share_transfers", batch_id=batch_id)
        assert first["family"]["route_version"] == 1
        assert second["family"]["risk_ledger"] == first["family"]["risk_ledger"]
        assert [(item["source_key"], item["target_key"], item["route_version"]) for item in transfers] == [
            ("P2", "F", 1)
        ]


class CheckComputeBudgetKeepsAllocation:
    def test_larger_sample_budget_does_not_change_risk_allocation(self, tmp_path: Path):
        specs = [
            gated_spec("P1", "less", share=0.35),
            gated_spec("P2", "two-sided", share=0.2, reallocate_to="F"),
            gated_spec("F", "less", share=0.2, parents=["P1"], threshold=0.4),
        ]
        ledgers = []
        for label, samples in (("small", 8), ("large", 500)):
            with PlatformRunner(tmp_path / f"{label}.json") as runner:
                batch_id = create(runner, specs, samples=samples)
                run_fully(runner, batch_id)
                report = runner.compare_batch(batch_id)
                ledgers.append(report["family"]["risk_ledger"])
                ledgers.append(
                    [
                        {
                            "path": item["path"],
                            "planned_risk_share": item["planned_risk_share"],
                        }
                        for item in report["family"]["gate_paths"]
                    ]
                )
        assert ledgers[0] == ledgers[2]
        assert ledgers[1] == ledgers[3]
