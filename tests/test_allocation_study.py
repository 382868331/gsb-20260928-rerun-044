import json
from pathlib import Path

import pytest

from stat_platform import Dataset, GroupedPermutationTest, PlatformRunner, TestSpec
from stat_platform.allocation import (
    gates_from_plan,
    simulate_scheme,
    study_schemes,
)


def diamond_plan() -> list[dict]:
    return [
        {"key": "A", "risk_share": 0.02, "reallocate_to": "C"},
        {"key": "B", "risk_share": 0.02, "reallocate_to": "C"},
        {
            "key": "C",
            "risk_share": 0.01,
            "parents": ["A", "B"],
            "parent_mode": "any",
            "open_when": {"p_value_at_most": 0.05},
        },
    ]


def test_shared_successor_merges_transfers_once_per_edge():
    gates = gates_from_plan(diamond_plan())
    simulation = simulate_scheme(
        gates, 0.05, outcomes={"A": "not-significant", "B": "not-significant"}
    )
    ledger = {entry["hypothesis"]: entry for entry in simulation["ledger"]}
    assert ledger["C"]["transfers_in"] == [
        {"from": "A", "amount": pytest.approx(0.02)},
        {"from": "B", "amount": pytest.approx(0.02)},
    ]
    assert ledger["C"]["effective_allocation"] == pytest.approx(0.05)
    assert simulation["budget_respected"]


def test_partial_closure_only_releases_decided_nodes():
    gates = gates_from_plan(diamond_plan())
    simulation = simulate_scheme(gates, 0.05, outcomes={"A": "closed"})
    ledger = {entry["hypothesis"]: entry for entry in simulation["ledger"]}
    assert ledger["C"]["transfers_in"] == [
        {"from": "A", "amount": pytest.approx(0.02)}
    ]
    assert ledger["C"]["effective_allocation"] == pytest.approx(0.03)
    assert ledger["B"]["remaining_risk_share"] == pytest.approx(0.02)


def test_study_reports_max_share_per_target_across_scenarios():
    study = study_schemes(
        [{"name": "diamond", "plan": diamond_plan()}], 0.05
    )
    (scheme,) = study["schemes"]
    assert scheme["status"] == "ok"
    assert scheme["scenarios_evaluated"] == 27
    max_c = scheme["max_shares"]["C"]
    assert max_c["max_effective_allocation"] == pytest.approx(0.05)
    assert max_c["scenario"]["A"] in {"not-significant", "closed"}
    assert max_c["scenario"]["B"] in {"not-significant", "closed"}
    assert scheme["worst_case_total_used_risk_share"] <= 0.05 + 1e-12


def test_cycles_and_unknown_routes_reject_the_scheme_not_the_study():
    cyclic = [
        {"key": "A", "risk_share": 0.02, "reallocate_to": "B"},
        {"key": "B", "risk_share": 0.02, "reallocate_to": "A"},
    ]
    study = study_schemes([{"name": "loop", "plan": cyclic}], 0.05)
    (scheme,) = study["schemes"]
    assert scheme["status"] == "rejected"
    assert "cycle" in scheme["reason"]

    dangling = [
        {"key": "A", "risk_share": 0.02, "reallocate_to": "MISSING"},
    ]
    study = study_schemes([{"name": "dangling", "plan": dangling}], 0.05)
    assert study["schemes"][0]["status"] == "rejected"


def test_used_shares_are_locked_and_never_reallocated():
    decided = {"A": "significant"}
    used = {"A": 0.02}
    study = study_schemes(
        [{"name": "diamond", "plan": diamond_plan()}],
        0.05,
        used_shares=used,
        decided=decided,
    )
    (scheme,) = study["schemes"]
    assert scheme["status"] == "ok"
    # A already spent its share on a formal decision: C can only ever merge
    # B's release, never A's spent budget.
    assert scheme["max_shares"]["C"]["max_effective_allocation"] == pytest.approx(0.03)

    # A scheme that drops the hypothesis with locked used budget is infeasible.
    shrunken = [
        {"key": "B", "risk_share": 0.02, "reallocate_to": "C"},
        {
            "key": "C",
            "risk_share": 0.01,
            "parents": ["B"],
            "open_when": {"p_value_at_most": 0.05},
        },
    ]
    study = study_schemes(
        [{"name": "shrunken", "plan": shrunken}],
        0.05,
        used_shares=used,
        decided=decided,
    )
    (scheme,) = study["schemes"]
    assert scheme["status"] == "infeasible"
    assert "A" in scheme["reason"]


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


def diamond_specs() -> list[TestSpec]:
    return [
        gated_spec("A", "less", share=0.2, reallocate_to="C"),
        gated_spec("B", "less", share=0.2, reallocate_to="C"),
        gated_spec(
            "C",
            "two-sided",
            share=0.2,
            parents=["A", "B"],
            threshold=0.4,
            parent_mode="any",
        ),
    ]


def test_runner_study_locks_used_budget_and_traces_old_conclusions(tmp_path: Path):
    db_path = tmp_path / "study.json"
    with PlatformRunner(db_path) as runner:
        batch_id = runner.create_batch(
            triple_dataset(),
            specs=diamond_specs(),
            samples=8,
            max_exact_assignments=4,
            seed=21,
            family_alpha=0.8,
        )
        runner.run_batch(batch_id)
        before = runner.compare_batch(batch_id)
        snapshot = db_path.read_bytes()

        # Mid-course priority adjustment: keep the diamond but add a new
        # follow-up D below C and try to route C's release to D.
        adjusted = {
            "name": "priority-shift",
            "plan": diamond_plan_runner() + [
                {
                    "key": "D",
                    "risk_share": 0.2,
                    "parents": ["C"],
                    "open_when": {"p_value_at_most": 0.6},
                }
            ],
        }
        # The same extension is illegal once the family budget is already
        # fully consumed: with family_alpha = 0.6 the locked 0.6 used by C
        # leaves nothing for D, and the study must say so.
        over_budget = study_schemes(
            [dict(adjusted)],
            0.6,
            used_shares={"C": 0.6},
            decided={"A": "not-significant", "B": "not-significant", "C": "significant"},
        )
        # A scheme that simply drops C (whose decision already consumed
        # budget) must be rejected as infeasible.
        dropped = {
            "name": "drop-C",
            "plan": [
                {"key": "A", "risk_share": 0.2},
                {"key": "B", "risk_share": 0.2},
            ],
        }
        study = runner.study_allocations(batch_id, [adjusted, dropped])

        # The study is a pure what-if: nothing was persisted or consumed.
        assert db_path.read_bytes() == snapshot
        after = runner.compare_batch(batch_id)
        assert after == before

    assert study["route_version"] == before["family"]["route_version"]
    assert over_budget["schemes"][0]["status"] == "infeasible"
    assert over_budget["schemes"][0]["budget_respected"] is False

    assert study["family_alpha"] == pytest.approx(0.8)
    # A and B finished not-significant (exact one-sided p = 0.25 > 0.2) and
    # released their shares; C merged both transfers and decided at 0.6.
    assert study["decided_outcomes"] == {
        "A": "not-significant",
        "B": "not-significant",
        "C": "significant",
    }
    assert study["locked_used_shares"]["C"] == pytest.approx(0.6)

    schemes = {entry["name"]: entry for entry in study["study"]["schemes"]}
    current = schemes["current-plan"]
    assert current["status"] == "ok"
    assert current["max_shares"]["C"]["max_effective_allocation"] == pytest.approx(0.6)
    assert current["worst_case_total_used_risk_share"] <= 0.8 + 1e-12

    shifted = schemes["priority-shift"]
    assert shifted["status"] == "ok"
    # C's significant decision already consumed 0.6; the new priority route
    # can never hand D more than its own 0.2 share, and the worst-case total
    # stays inside the 0.8 family budget.
    assert shifted["max_shares"]["D"]["max_effective_allocation"] == pytest.approx(0.2)
    assert shifted["worst_case_total_used_risk_share"] <= 0.8 + 1e-12

    assert schemes["drop-C"]["status"] == "infeasible"
    assert "C" in schemes["drop-C"]["reason"]


def diamond_plan_runner() -> list[dict]:
    return [
        {"key": "A", "risk_share": 0.2, "reallocate_to": "C"},
        {"key": "B", "risk_share": 0.2, "reallocate_to": "C"},
        {
            "key": "C",
            "risk_share": 0.2,
            "parents": ["A", "B"],
            "parent_mode": "any",
            "open_when": {"p_value_at_most": 0.4},
        },
    ]
