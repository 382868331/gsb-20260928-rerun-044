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


def primary_spec(key: str, alternative: str, share: float, **extra) -> TestSpec:
    hierarchy = {"key": key, "risk_share": share}
    hierarchy.update(extra)
    return TestSpec(
        GroupedPermutationTest.test_type,
        f"hypothesis {key}",
        {
            "alternative": alternative,
            "conditions": ("control", "treatment"),
            "hierarchy": hierarchy,
        },
    )


def follow_spec(
    key: str,
    alternative: str,
    share: float,
    *,
    parents: list,
    threshold: float = 0.4,
    mode: str | None = None,
    on_close: str | None = None,
) -> TestSpec:
    hierarchy = {
        "key": key,
        "risk_share": share,
        "parents": list(parents),
        "open_when": {"p_value_at_most": threshold},
    }
    if mode is not None:
        hierarchy["parent_mode"] = mode
    if on_close is not None:
        hierarchy["share_on_close"] = on_close
    return TestSpec(
        GroupedPermutationTest.test_type,
        f"hypothesis {key}",
        {
            "alternative": alternative,
            "conditions": ("control", "treatment"),
            "hierarchy": hierarchy,
        },
    )


def shared_plan(mode: str = "any", on_close: str | None = None) -> list:
    return [
        primary_spec("P1", "less", 0.35),
        primary_spec("P2", "two-sided", 0.1),
        follow_spec("F", "less", 0.1, parents=["P1", "P2"], mode=mode, on_close=on_close),
    ]


def create_and_run(path: Path, specs: list, family_alpha: float = 0.6,
                   samples: int = 8, chunk: int | None = None,
                   reopen: bool = False) -> tuple:
    runner = PlatformRunner(path)
    batch_id = runner.create_batch(
        triple_dataset(),
        specs=specs,
        samples=samples,
        max_exact_assignments=4,
        seed=21,
        family_alpha=family_alpha,
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
        if reopen:
            runner.close()
            runner = PlatformRunner(path)


def ledger_by_key(report: dict) -> dict:
    return {
        entry["hypothesis"]: entry
        for entry in report["family"]["budget_ledger"]["hypotheses"]
    }


def comparable_ledger(ledger: dict) -> dict:
    normalized = dict(ledger)
    normalized["hypotheses"] = sorted(
        (
            {key: value for key, value in entry.items() if key != "test_id"}
            for entry in ledger["hypotheses"]
        ),
        key=lambda entry: entry["hypothesis"],
    )
    return normalized


def test_parallel_primaries_exceeding_family_budget_are_rejected(tmp_path: Path):
    path = tmp_path / "db.json"
    with PlatformRunner(path) as runner:
        ok_batch = runner.create_batch(
            triple_dataset(),
            specs=[primary_spec("P0", "less", 0.04)],
            samples=8,
            max_exact_assignments=4,
            seed=21,
            family_alpha=0.05,
        )
        runner.run_batch(ok_batch)
        ok_test_id = runner.store.find_many("tests", batch_id=ok_batch)[0]["id"]
        ok_result = runner.get_result(ok_test_id).result.p_value

    specs = [primary_spec("P1", "less", 0.04), primary_spec("P2", "two-sided", 0.04)]
    with PlatformRunner(path) as runner:
        with pytest.raises(ValueError, match="risk shares conflict") as excinfo:
            runner.create_batch(
                triple_dataset(),
                specs=specs,
                samples=8,
                max_exact_assignments=4,
                family_alpha=0.05,
            )
        message = str(excinfo.value)
        assert "P1=0.04" in message and "P2=0.04" in message
        assert "0.08" in message and "0.05" in message
        rejected_batch = runner.store.find_many("batches")[-1]
        assert rejected_batch["family_alpha"] == 0.05
        assert runner.store.find_many("tests", batch_id=rejected_batch["id"]) == []
        # The rejected plan and earlier exploratory results are retained.
        assert runner.get_result(ok_test_id).result.p_value == ok_result
        peek = runner.peek_test(ok_test_id)
        assert peek["status"] == "completed"


def test_shared_followup_requires_predeclared_parent_mode(tmp_path: Path):
    specs = shared_plan(mode=None)
    with PlatformRunner(tmp_path / "db.json") as runner:
        with pytest.raises(ValueError, match="parent_mode"):
            runner.create_batch(
                triple_dataset(), specs=specs, samples=8,
                max_exact_assignments=4, family_alpha=0.6,
            )


def test_shared_followup_any_mode_receives_one_share(tmp_path: Path):
    runner, batch_id = create_and_run(tmp_path / "db.json", shared_plan(mode="any"))
    with runner:
        tests = {
            row["hypothesis_key"]: row
            for row in runner.store.find_many("tests", batch_id=batch_id)
        }
        state_f = runner.store.find_one("run_states", test_id=tests["F"]["id"])
        assert state_f["gate_status"] == "open"
        assert state_f["gate_opened_by"] == "P1"
        # The shared follow-up is evaluated once; no double draw via two parents.
        assert state_f["evaluated_assignments"] == 8
        report = runner.compare_batch(batch_id, alpha=0.05)
        ledger = ledger_by_key(report)
        assert ledger["F"]["initial_share"] == pytest.approx(0.1)
        assert ledger["F"]["effective_share"] == pytest.approx(0.1)
        assert ledger["F"]["transferred_in"] == []
        assert ledger["F"]["opened_by"] == "P1"
        # P1 was formally judged significant; its used share stays spent.
        assert ledger["P1"]["used_share"] == pytest.approx(0.35)
        assert ledger["P1"]["remaining_share"] == pytest.approx(0.0)
        # P2 failed; its unused share is reserved by default.
        assert ledger["P2"]["formal_decision"] == "not-significant"
        assert ledger["P2"]["transferred_out"] == [{"to": "reserved", "amount": 0.1}]
        assert report["family"]["budget_ledger"]["reserved_share"] == pytest.approx(0.2)
        assert report["family"]["budget_ledger"]["total_used_share"] == pytest.approx(0.35)


def test_shared_followup_all_mode_closes_when_one_parent_fails(tmp_path: Path):
    runner, batch_id = create_and_run(
        tmp_path / "db.json", shared_plan(mode="all", on_close="family")
    )
    with runner:
        tests = {
            row["hypothesis_key"]: row
            for row in runner.store.find_many("tests", batch_id=batch_id)
        }
        state_f = runner.store.find_one("run_states", test_id=tests["F"]["id"])
        assert state_f["gate_status"] == "closed"
        assert state_f["evaluated_assignments"] == 0
        with pytest.raises(RuntimeError, match="remained unopened"):
            runner.run_test(tests["F"]["id"])
        report = runner.compare_batch(batch_id, alpha=0.05)
        ledger = ledger_by_key(report)
        assert ledger["F"]["status"] == "closed"
        assert ledger["F"]["transferred_out"] == [{"to": "family", "amount": 0.1}]
        assert report["family"]["budget_ledger"]["returned_to_family"] == pytest.approx(0.1)
        assert report["family"]["closed_test_ids"] == [tests["F"]["id"]]


def test_failed_parent_share_transfers_to_declared_followup(tmp_path: Path):
    specs = [
        primary_spec("P1", "less", 0.35),
        primary_spec("P2", "two-sided", 0.1, share_on_close="F"),
        follow_spec("F", "less", 0.1, parents=["P1"]),
    ]
    runner, batch_id = create_and_run(tmp_path / "db.json", specs)
    with runner:
        report = runner.compare_batch(batch_id, alpha=0.05)
        ledger = ledger_by_key(report)
        assert ledger["P2"]["transferred_out"] == [{"to": "F", "amount": 0.1}]
        assert ledger["F"]["transferred_in"] == [{"from": "P2", "amount": 0.1}]
        assert ledger["F"]["initial_share"] == pytest.approx(0.1)
        assert ledger["F"]["effective_share"] == pytest.approx(0.2)
        entry_f = next(e for e in report["tests"] if e["hypothesis"] == "F")
        assert entry_f["effective_risk_share"] == pytest.approx(0.2)
        assert entry_f["transferred_in_share"] == pytest.approx(0.1)


def test_unknown_or_cyclic_share_destination_rejected(tmp_path: Path):
    specs = [
        primary_spec("P1", "less", 0.3, share_on_close="GHOST"),
        primary_spec("P2", "two-sided", 0.1),
    ]
    with PlatformRunner(tmp_path / "db.json") as runner:
        with pytest.raises(ValueError, match="share_on_close"):
            runner.create_batch(
                triple_dataset(), specs=specs, samples=8,
                max_exact_assignments=4, family_alpha=0.6,
            )
    cyclic = [
        primary_spec("P1", "less", 0.3, share_on_close="P2"),
        primary_spec("P2", "two-sided", 0.1, share_on_close="P1"),
    ]
    with PlatformRunner(tmp_path / "db2.json") as runner:
        with pytest.raises(ValueError, match="cycles"):
            runner.create_batch(
                triple_dataset(), specs=cyclic, samples=8,
                max_exact_assignments=4, family_alpha=0.6,
            )


def snapshot(runner: PlatformRunner, batch_id: int) -> tuple:
    report = runner.compare_batch(batch_id, alpha=0.05)
    states = []
    for row in sorted(runner.store.find_many("tests", batch_id=batch_id), key=lambda r: r["id"]):
        state = runner.store.find_one("run_states", test_id=row["id"])
        completed = runner.store.find_one("test_results", test_id=row["id"])
        states.append(
            (
                row["hypothesis_key"],
                state["gate_status"],
                state.get("gate_opened_by"),
                state["evaluated_assignments"],
                completed is not None,
            )
        )
    return tuple(states), comparable_ledger(report["family"]["budget_ledger"])


def test_open_order_and_resume_do_not_change_budget_ledger(tmp_path: Path):
    runner_a, batch_a = create_and_run(tmp_path / "one_shot.json", shared_plan(mode="any"))
    with runner_a:
        states_a, ledger_a = snapshot(runner_a, batch_a)

    plan = shared_plan(mode="any")
    reversed_specs = [plan[1], plan[0], plan[2]]
    runner_b, batch_b = create_and_run(tmp_path / "reversed.json", reversed_specs)
    with runner_b:
        states_b, ledger_b = snapshot(runner_b, batch_b)

    runner_c, batch_c = create_and_run(
        tmp_path / "resumed.json", shared_plan(mode="any"), chunk=2, reopen=True
    )
    with runner_c:
        states_c, ledger_c = snapshot(runner_c, batch_c)

    key_states = lambda states: sorted(states)
    assert key_states(states_a) == key_states(states_b) == key_states(states_c)
    assert ledger_a == ledger_b == ledger_c


def test_used_share_stays_spent_after_peek_and_reopen(tmp_path: Path):
    path = tmp_path / "db.json"
    runner, batch_id = create_and_run(path, shared_plan(mode="any"))
    tests = {
        row["hypothesis_key"]: row
        for row in runner.store.find_many("tests", batch_id=batch_id)
    }
    _, ledger_before = snapshot(runner, batch_id)
    runner.peek_test(tests["P1"]["id"])
    runner.peek_test(tests["F"]["id"])
    runner.close()

    with PlatformRunner(path) as reopened:
        # Re-running a completed test returns the stored result; nothing resets.
        reopened.run_test(tests["P1"]["id"])
        _, ledger_after = snapshot(reopened, batch_id)
    assert ledger_before == ledger_after
    p1 = next(e for e in ledger_after["hypotheses"] if e["hypothesis"] == "P1")
    assert p1["used_share"] == pytest.approx(0.35)
    assert ledger_after["family_remaining_share"] == pytest.approx(0.6 - 0.35)


def test_larger_compute_budget_keeps_original_risk_allocation(tmp_path: Path):
    runner_small, batch_small = create_and_run(
        tmp_path / "small.json", shared_plan(mode="any"), samples=8
    )
    with runner_small:
        _, ledger_small = snapshot(runner_small, batch_small)
    runner_large, batch_large = create_and_run(
        tmp_path / "large.json", shared_plan(mode="any"), samples=64
    )
    with runner_large:
        _, ledger_large = snapshot(runner_large, batch_large)
    assert ledger_small == ledger_large
