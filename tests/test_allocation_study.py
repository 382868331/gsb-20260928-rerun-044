from pathlib import Path

import pytest

from stat_platform import (
    AllocationTarget,
    Dataset,
    GroupedPermutationTest,
    PlatformRunner,
    TestSpec,
    study_allocations,
)
from stat_platform.storage import dumps, loads


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


def diamond_targets(**overrides) -> list[AllocationTarget]:
    base = {
        "P": AllocationTarget(key="P"),
        "A": AllocationTarget(key="A", parents=("P",)),
        "B": AllocationTarget(key="B", parents=("P",)),
        "C": AllocationTarget(
            key="C", parents=("A", "B"), transfer_targets=()
        ),
    }
    base.update(overrides)
    return list(base.values())


class CheckPureStudy:
    def test_every_scheme_respects_path_and_root_budgets(self):
        study = study_allocations(diamond_targets(), 0.05)
        assert len(study.schemes) == 24
        for scheme in study.schemes:
            assert scheme["feasible"]
            shares = scheme["shares"]
            assert shares["P"] <= 0.05 + 1e-12
            assert shares["P"] + shares["A"] + shares["C"] <= 0.05 + 1e-12
            assert shares["P"] + shares["B"] + shares["C"] <= 0.05 + 1e-12

    def test_shared_successor_merges_real_per_edge_amounts(self):
        targets = diamond_targets(
            A=AllocationTarget(
                key="A", parents=("P",), used_share=0.01, transfer_targets=("C",)
            ),
            B=AllocationTarget(
                key="B", parents=("P",), used_share=0.02, transfer_targets=("C",)
            ),
        )
        study = study_allocations(
            targets, 0.05, priorities=[("A", "B", "C", "P")]
        )
        (scheme,) = study.schemes
        incoming = [t for t in scheme["transfers"] if t["to"] == "C"]
        assert {(t["from"], round(t["amount"], 6)) for t in incoming} == {
            ("A", 0.04),
            ("B", 0.03),
        }
        # The shared successor merges what both parents actually sent; the
        # merged total is the sum of the real ledger amounts, nothing more.
        assert scheme["effective"]["C"] == pytest.approx(0.07)
        assert study.max_effective_share["C"]["amount"] == pytest.approx(0.07)
        assert study.max_effective_share["C"]["attained_by"] == [0]

    def test_used_shares_are_locked_minimums_in_every_scheme(self):
        targets = diamond_targets(
            P=AllocationTarget(key="P", used_share=0.03),
        )
        study = study_allocations(targets, 0.05)
        assert study.used_shares["P"] == 0.03
        for scheme in study.schemes:
            assert scheme["shares"]["P"] >= 0.03 - 1e-12

    def test_max_share_derivation_per_target(self):
        study = study_allocations(diamond_targets(), 0.05)
        for key in ("P", "A", "C"):
            assert study.max_effective_share[key]["amount"] == pytest.approx(0.05)

    def test_duplicate_transfer_route_is_rejected(self):
        targets = diamond_targets(
            A=AllocationTarget(
                key="A", parents=("P",), transfer_targets=("C", "C")
            ),
        )
        with pytest.raises(ValueError, match="more than once"):
            study_allocations(targets, 0.05)

    def test_cyclic_routes_are_rejected(self):
        targets = diamond_targets(
            A=AllocationTarget(key="A", parents=("P",), transfer_targets=("B",)),
            B=AllocationTarget(key="B", parents=("P",), transfer_targets=("A",)),
        )
        with pytest.raises(ValueError, match="must not contain cycles"):
            study_allocations(targets, 0.05)

    def test_priority_orders_must_cover_every_target(self):
        with pytest.raises(ValueError, match="every target"):
            study_allocations(
                diamond_targets(), 0.05, priorities=[("P", "A", "B")]
            )


class CheckBatchStudy:
    def plan(self) -> list[TestSpec]:
        return [
            gated_spec("P1", "less", share=0.35),
            gated_spec("P2", "two-sided", share=0.2),
            gated_spec(
                "F1", "less", share=0.2, parents=["P2"], threshold=0.4
            ),
        ]

    def create(self, runner: PlatformRunner) -> int:
        return runner.create_batch(
            triple_dataset(),
            specs=self.plan(),
            samples=8,
            max_exact_assignments=4,
            seed=21,
            family_alpha=0.6,
        )

    def test_study_consumes_nothing_and_old_conclusions_stay_traceable(
        self, tmp_path: Path
    ):
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.create(runner)
            runner.run_batch(batch_id)
            before_data = loads(dumps(runner.store.data))
            before_report = runner.compare_batch(batch_id)
            study = runner.study_allocation(
                batch_id, priorities=[("P1", "P2", "F1")]
            )
            # Mid-course priority adjustment: a new candidate order is just
            # another read-only comparison over the same locked ledger.
            reprioritized = runner.study_allocation(
                batch_id, priorities=[("F1", "P2", "P1")]
            )
            assert loads(dumps(runner.store.data)) == before_data
            assert runner.compare_batch(batch_id) == before_report
            assert study["consumes_budget"] is False
            assert reprioritized["used_shares"] == study["used_shares"]

    def test_partial_closure_and_used_share_locking(self, tmp_path: Path):
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.create(runner)
            runner.run_batch(batch_id)
            study = runner.study_allocation(batch_id)
            # P1 (less, p=0.25) is formally significant at its 0.35 share, so
            # that spent quota is locked; P2 fails and F1 stays closed.
            assert study["used_shares"]["P1"] == pytest.approx(0.35)
            assert study["used_shares"]["P2"] == 0.0
            assert study["gate_status"]["F1"] == "closed"
            for scheme in study["schemes"]:
                assert scheme["feasible"]
                assert scheme["shares"]["P1"] >= 0.35 - 1e-12
                assert scheme["total_used_share"] == pytest.approx(0.35)

    def test_extra_routes_are_validated_like_declared_ones(self, tmp_path: Path):
        with PlatformRunner(tmp_path / "db.json") as runner:
            batch_id = self.create(runner)
            runner.run_batch(batch_id)
            with pytest.raises(ValueError, match="cycles"):
                runner.study_allocation(
                    batch_id, extra_routes=[("P1", "F1"), ("F1", "P1")]
                )
            study = runner.study_allocation(
                batch_id, extra_routes=[("P2", "F1")]
            )
            assert study["schemes"]
            assert study["route_version"] == 1

