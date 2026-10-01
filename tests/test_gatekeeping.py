from __future__ import annotations

from pathlib import Path

import pytest

from stat_platform import (
    Dataset,
    GateNode,
    GatePlan,
    GroupedPermutationTest,
    PlatformRunner,
    TestSpec,
    plan_from_positions,
)


def paired_dataset() -> Dataset:
    records = []
    for group, (control, treatment) in {"A": (1, 4), "B": (2, 5)}.items():
        records.append({"group": group, "condition": "control", "value": control})
        records.append({"group": group, "condition": "treatment", "value": treatment})
    return Dataset.from_records(records, name="paired", version="v1")


def triple_dataset() -> Dataset:
    records = []
    for group, (control, treatment) in {"A": (1, 4), "B": (2, 5), "C": (3, 3)}.items():
        records.append({"group": group, "condition": "control", "value": control})
        records.append({"group": group, "condition": "treatment", "value": treatment})
    return Dataset.from_records(records, name="triples", version="v1")


def make_spec(alternative: str) -> TestSpec:
    return TestSpec(
        GroupedPermutationTest.test_type,
        f"equal means ({alternative})",
        {"alternative": alternative, "conditions": ("control", "treatment")},
    )


def p_values_per_assignment(dataset: Dataset, spec: TestSpec) -> dict[tuple, float]:
    """Exact p-value of the test when each legal assignment is the observed one."""
    engine = GroupedPermutationTest(dataset, spec)
    assignments = list(engine._all_arrangements())
    statistics = [engine._statistic(assignment) for assignment in assignments]
    observed = engine.observed_statistic
    p_values = {}
    for assignment, statistic in zip(assignments, statistics):
        if spec.settings["alternative"] == "less":
            hits = sum(other <= statistic for other in statistics)
        elif spec.settings["alternative"] == "greater":
            hits = sum(other >= statistic for other in statistics)
        else:
            hits = sum(abs(other) >= abs(statistic) for other in statistics)
        p_values[assignment] = hits / len(assignments)
    assert engine.observed_statistic == observed
    return p_values


def build_plan(test_ids, alpha=0.3):
    return GatePlan(
        alpha=alpha,
        nodes=(
            GateNode(test_id=test_ids[0], share=1.0),
            GateNode(test_id=test_ids[1], depends_on=(test_ids[0],)),
        ),
    )


class CheckPlanValidation:
    def test_initial_shares_must_fit_the_budget(self):
        with pytest.raises(ValueError, match="family-wise"):
            GatePlan(alpha=0.05, nodes=(GateNode(1, 0.6), GateNode(2, 0.6)))

    def test_dependencies_must_be_acyclic(self):
        with pytest.raises(ValueError, match="acyclic"):
            GatePlan(
                alpha=0.05,
                nodes=(GateNode(1, 0.5, (2,)), GateNode(2, 0.5, (1,))),
            )

    def test_every_opening_path_stays_within_budget(self):
        plan = GatePlan(
            alpha=0.05,
            nodes=(
                GateNode(1, 0.4),
                GateNode(2, 0.6),
                GateNode(3, depends_on=(1,)),
                GateNode(4, depends_on=(1, 2)),
            ),
        )
        paths = plan.compare_paths()
        assert paths  # enumeration is non-empty
        assert all(path["within_budget"] for path in paths)
        assert plan.max_open_risk_share() <= 1.0 + 1e-9
        # The path where both primaries fail opens nothing downstream.
        empty = next(path for path in paths if path["rejected"] == [])
        assert empty["unopened"] == [3, 4]
        # The full path opens everything and can conclude on all hypotheses.
        full = paths[-1]
        assert full["open"] == [1, 2, 3, 4]

    def test_inherited_share_follows_pre_specified_weights(self):
        plan = GatePlan(
            alpha=0.5,
            nodes=(
                GateNode(1, 1.0),
                GateNode(2, depends_on=(1,)),
                GateNode(3, depends_on=(1,)),
            ),
            edge_weights={(1, 2): 3.0, (1, 3): 1.0},
        )
        states = plan.evaluate({1: 0.01, 2: 0.2, 3: 0.2})
        assert states[2]["available_share"] == pytest.approx(0.75)
        assert states[3]["available_share"] == pytest.approx(0.25)
        assert states[2]["alpha_threshold"] == pytest.approx(0.375)
        assert states[2]["formal_reject"]
        assert not states[3]["formal_reject"]

    def test_unopened_nodes_consume_no_risk_and_stay_exploratory(self):
        plan = build_plan([1, 2], alpha=0.3)
        states = plan.evaluate({1: 0.5, 2: 0.01})
        assert states[1]["status"] == "not-rejected"
        assert states[2]["status"] == "unopened"
        assert states[2]["exploratory"]
        assert states[2]["exploratory_p_value"] == 0.01
        assert not states[2]["formal_reject"]
        assert states[2]["risk_share_used"] == 0.0


class CheckExhaustiveFamilyError:
    """Fully enumerable small example: 2 groups give exactly 4 assignments."""

    def test_familywise_error_over_full_enumeration(self):
        dataset = paired_dataset()
        primary_spec = make_spec("less")
        followup_spec = make_spec("less")
        primary_p = p_values_per_assignment(dataset, primary_spec)
        followup_p = p_values_per_assignment(dataset, followup_spec)
        plan = build_plan([1, 2], alpha=0.3)

        rejections = 0
        assignments = list(primary_p)
        for assignment in assignments:
            states = plan.evaluate(
                {1: primary_p[assignment], 2: followup_p[assignment]}
            )
            # Both nulls are true under the permutation regime: any formal
            # rejection is a family-wise error.
            if any(state["formal_reject"] for state in states.values()):
                rejections += 1
            # The follow-up may only be judged when the primary rejected.
            if states[2]["status"] in {"rejected", "not-rejected"}:
                assert states[1]["formal_reject"]
        assert rejections / len(assignments) <= plan.alpha
        assert rejections > 0  # the check is non-vacuous

    def test_two_primaries_share_the_budget_exactly(self):
        dataset = paired_dataset()
        less_p = p_values_per_assignment(dataset, make_spec("less"))
        greater_p = p_values_per_assignment(dataset, make_spec("greater"))
        plan = GatePlan(
            alpha=0.5,
            nodes=(GateNode(1, 0.5), GateNode(2, 0.5), GateNode(3, depends_on=(1,))),
        )
        errors = 0
        for assignment in less_p:
            states = plan.evaluate(
                {1: less_p[assignment], 2: greater_p[assignment], 3: less_p[assignment]}
            )
            if any(state["formal_reject"] for state in states.values()):
                errors += 1
            if states[3]["open"]:
                assert states[1]["formal_reject"]
        # Each side rejects on exactly one of four assignments at p = 0.25.
        assert errors / len(less_p) == 0.5 <= plan.alpha


class CheckGateReport:
    def make_batch(self, runner, gate, **kwargs):
        options = {"samples": 8, "max_exact_assignments": 4, "seed": 11}
        options.update(kwargs)
        return runner.create_batch(
            triple_dataset(),
            specs=[make_spec("less"), make_spec("less")],
            gate=gate,
            **options,
        )

    def test_report_marks_unopened_followup_exploratory(self, tmp_path: Path):
        gate = {"alpha": 0.01, "nodes": [{"test": 0, "share": 1.0}, {"test": 1, "depends_on": [0]}]}
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.make_batch(runner, gate)
            runner.run_batch(batch_id)
            report = runner.gate_report(batch_id)

        primary, followup = report["tests"]
        assert primary["open"] and not primary["formal_reject"]
        assert followup["status"] == "unopened"
        assert followup["exploratory"]
        assert followup["exploratory_p_value"] is not None
        assert followup["risk_share_used"] == 0.0
        assert report["family"]["unopened_test_ids"] == [followup["test_id"]]
        assert report["paths"]

    def test_failed_primary_cannot_be_bypassed(self, tmp_path: Path):
        gate = {"alpha": 0.01, "nodes": [{"test": 0, "share": 1.0}, {"test": 1, "depends_on": [0]}]}
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.make_batch(runner, gate)
            runner.run_batch(batch_id)
            before = runner.gate_report(batch_id)
            # Re-running or resuming the finished batch cannot reopen anything.
            runner.run_batch(batch_id)
            runner.run_test(2)
            after = runner.gate_report(batch_id)
            assert before == after
            assert after["tests"][1]["status"] == "unopened"
            # The fixed plan cannot be redefined once results exist.
            with pytest.raises(ValueError, match="already fixed"):
                runner.define_gate(batch_id, build_plan([1, 2]))
            with pytest.raises(ValueError, match="already fixed"):
                runner.define_gate(
                    batch_id, GatePlan(alpha=0.9, nodes=(GateNode(2, 1.0),))
                )

    def test_plan_must_be_fixed_before_results(self, tmp_path: Path):
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = runner.create_batch(
                triple_dataset(),
                specs=[make_spec("less")],
                samples=8,
                max_exact_assignments=4,
                seed=11,
            )
            runner.run_batch(batch_id)
            with pytest.raises(ValueError, match="before any test"):
                runner.define_gate(batch_id, GatePlan(alpha=0.05, nodes=(GateNode(1, 1.0),)))

    def test_open_pending_until_test_completes(self, tmp_path: Path):
        gate = {"alpha": 0.3, "nodes": [{"test": 0, "share": 1.0}, {"test": 1, "depends_on": [0]}]}
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.make_batch(runner, gate)
            report = runner.gate_report(batch_id)
            assert report["tests"][0]["status"] == "open-pending"
            assert report["tests"][1]["status"] == "unopened"

    def test_chunking_and_save_location_do_not_change_outcome(self, tmp_path: Path):
        gate = {"alpha": 0.3, "nodes": [{"test": 0, "share": 1.0}, {"test": 1, "depends_on": [0]}]}
        reports = []
        look_logs = []
        for directory, chunk in ((tmp_path / "flat", None), (tmp_path / "deep" / "nested", 3)):
            with PlatformRunner(directory / "db.json") as runner:
                batch_id = self.make_batch(runner, gate)
                while True:
                    try:
                        runner.run_batch(batch_id, max_new_samples=chunk)
                        break
                    except RuntimeError:
                        continue
                reports.append(runner.gate_report(batch_id))
                look_logs.append(
                    [
                        runner.get_result(test_id).result.look_log
                        for test_id in (1, 2)
                    ]
                )
        assert reports[0] == reports[1]
        assert look_logs[0] == look_logs[1]

    def test_resume_preserves_evidence_openings_and_allocation(self, tmp_path: Path):
        gate = {"alpha": 0.3, "nodes": [{"test": 0, "share": 1.0}, {"test": 1, "depends_on": [0]}]}
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.make_batch(runner, gate)
            with pytest.raises(RuntimeError):
                runner.run_batch(batch_id, max_new_samples=4)
            mid = runner.gate_report(batch_id)
            mid_evaluated = [
                runner.store.find_one("run_states", test_id=test_id)[
                    "evaluated_assignments"
                ]
                for test_id in (1, 2)
            ]
        # Reopen the same database (new save location object, same file) and finish.
        with PlatformRunner(tmp_path / "db.json") as runner:
            runner.run_batch(batch_id)
            final = runner.gate_report(batch_id)
            plan = runner.gate_plan(batch_id)

        assert mid["plan"] == final["plan"]
        assert plan.alpha == 0.3
        # Evidence was kept, not restarted: the mid-run counts are a strict
        # prefix of the final evaluation.
        assert mid_evaluated[0] == 4  # first test checkpointed, evidence kept
        assert all(
            entry["p_value"] is not None for entry in final["tests"]
        )
        # Extra budget after completion changes neither shares nor openings.
        with PlatformRunner(tmp_path / "db.json") as runner:
            runner.run_batch(batch_id)
            again = runner.gate_report(batch_id)
        assert again == final

    def test_compare_includes_gate_state(self, tmp_path: Path):
        gate = {"alpha": 0.01, "nodes": [{"test": 0, "share": 1.0}, {"test": 1, "depends_on": [0]}]}
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.make_batch(runner, gate)
            runner.run_batch(batch_id)
            comparison = runner.compare_batch(batch_id, rule="holm", alpha=0.05)
        gates = [entry["gate"] for entry in comparison["tests"]]
        assert gates[0]["open"]
        assert gates[1]["status"] == "unopened"
        assert gates[1]["exploratory"]
        assert gates[1]["risk_share_used"] == 0.0


class CheckCli:
    def test_gate_command_sets_plan_and_reports(self, tmp_path: Path, capsys):
        from stat_platform.cli import main

        database = tmp_path / "db.json"
        spec_file = tmp_path / "gate.json"
        spec_file.write_text(
            '{"alpha": 0.3, "nodes": [{"test": 0, "share": 1.0},'
            ' {"test": 1, "depends_on": [0]}]}',
            encoding="utf-8",
        )
        with PlatformRunner(database) as runner:
            runner.create_batch(
                paired_dataset(),
                specs=[make_spec("less"), make_spec("less")],
                samples=8,
                max_exact_assignments=4,
                seed=7,
            )
        capsys.readouterr()
        main(["--database", str(database), "gate", "1", "--set", str(spec_file)])
        output = capsys.readouterr().out
        assert '"family_wise_alpha": 0.3' in output
        assert '"paths"' in output
        assert '"status": "open-pending"' in output
