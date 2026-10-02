from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class HypothesisGate:
    key: str
    parents: tuple[str, ...]
    open_when: dict[str, Any] | None
    risk_share: float | None
    allocation: dict[str, Any] = field(default_factory=dict)
    parent_mode: str | None = None
    share_on_close: str = "reserved"

    @property
    def parent(self) -> str | None:
        return self.parents[0] if self.parents else None

    @classmethod
    def from_settings(cls, settings: dict[str, Any], index: int) -> "HypothesisGate | None":
        value = settings.get("hierarchy")
        if not value:
            return None
        key = str(value.get("key") or f"H{index + 1}")
        raw_parents = value.get("parents")
        if raw_parents is None:
            single = value.get("parent")
            parents = (str(single),) if single is not None else ()
        else:
            parents = tuple(str(item) for item in raw_parents)
        condition = value.get("open_when") or (
            {"decision": "significant"} if parents else None
        )
        risk_share = value.get("risk_share")
        parent_mode = value.get("parent_mode")
        share_on_close = str(value.get("share_on_close", "reserved"))
        return cls(
            key=key,
            parents=parents,
            open_when=dict(condition) if condition is not None else None,
            risk_share=float(risk_share) if risk_share is not None else None,
            allocation=dict(value.get("allocation") or {}),
            parent_mode=str(parent_mode) if parent_mode is not None else None,
            share_on_close=share_on_close,
        )


def gates_for_specs(specs: Sequence[Any]) -> list[HypothesisGate | None]:
    return [
        HypothesisGate.from_settings(dict(spec.settings), index)
        for index, spec in enumerate(specs)
    ]


def validate_gates(gates: Sequence[HypothesisGate | None], family_alpha: float = 0.05) -> None:
    declared = [gate for gate in gates if gate is not None]
    keys = [gate.key for gate in declared]
    if len(keys) != len(set(keys)):
        raise ValueError("hypothesis keys must be unique")
    key_set = set(keys)
    for gate in declared:
        if gate.risk_share is None or not 0.0 < gate.risk_share <= family_alpha + 1e-12:
            raise ValueError("risk_share must lie in (0, family_alpha]")
        if gate.parents:
            if len(set(gate.parents)) != len(gate.parents):
                raise ValueError(f"hypothesis {gate.key} lists a parent more than once")
            for parent in gate.parents:
                if parent not in key_set:
                    raise ValueError(f"unknown parent hypothesis {parent}")
            if gate.open_when is None:
                raise ValueError("a dependent hypothesis needs open_when")
        if len(gate.parents) > 1 and gate.parent_mode not in {"any", "all"}:
            raise ValueError(
                f"shared follow-up {gate.key} has multiple parents; pre-specify "
                "parent_mode 'any' (one satisfied parent opens it) or 'all' "
                "(every parent must be satisfied)"
            )
        if gate.parent_mode is not None and gate.parent_mode not in {"any", "all"}:
            raise ValueError("parent_mode must be 'any' or 'all'")
        if gate.share_on_close not in {"reserved", "family"}:
            if gate.share_on_close not in key_set:
                raise ValueError(
                    f"unknown share_on_close target {gate.share_on_close}"
                )
            if gate.share_on_close == gate.key:
                raise ValueError("share_on_close cannot route a share to itself")

    by_key = {gate.key: gate for gate in declared}
    for gate in declared:
        seen: set[str] = set()
        target = gate.share_on_close
        while target in by_key:
            if target in seen or target == gate.key:
                raise ValueError("share_on_close routing must not contain cycles")
            seen.add(target)
            target = by_key[target].share_on_close

    def path_chains(gate: HypothesisGate, trail: tuple[str, ...]) -> Iterator[list[HypothesisGate]]:
        if gate.key in trail:
            raise ValueError("hypothesis gates must not contain cycles")
        if not gate.parents:
            yield [gate]
            return
        for parent in gate.parents:
            for chain in path_chains(by_key[parent], trail + (gate.key,)):
                yield chain + [gate]

    for gate in declared:
        for chain in path_chains(gate, ()):
            shares = [node.risk_share for node in chain if node.risk_share is not None]
            if sum(shares) > family_alpha + 1e-12:
                raise ValueError(
                    "risk shares on every root-to-leaf path must not exceed family alpha"
                )

    total_share = sum(gate.risk_share or 0.0 for gate in declared)
    if total_share > family_alpha + 1e-12:
        detail = ", ".join(f"{gate.key}={gate.risk_share:g}" for gate in declared)
        raise ValueError(
            "risk shares conflict with the confirmation family budget: "
            f"{detail} sum to {total_share:g} but the declared family alpha is "
            f"{family_alpha:g}; the conflicting allocations listed above must be "
            "reduced so their combined share fits the family budget"
        )


def gate_is_opened_by(
    gate: HypothesisGate, result: Any | None, p_value: float | None = None
) -> bool:
    if gate.open_when is None or result is None:
        return False
    expected_decision = gate.open_when.get("decision")
    if expected_decision is not None:
        return result.decision == expected_decision
    threshold = gate.open_when.get("p_value_at_most")
    return threshold is not None and p_value is not None and p_value <= float(threshold)


def path_risk_summaries(
    gates: Sequence[tuple[int, HypothesisGate]],
    outcomes: dict[str, dict[str, Any]],
    family_alpha: float,
) -> list[dict[str, Any]]:
    by_key = {gate.key: (test_id, gate) for test_id, gate in gates}
    leaves = [
        gate
        for _, gate in gates
        if gate.key not in {parent for _, node in gates for parent in node.parents}
    ]
    summaries = []
    for leaf in leaves:
        chain: list[HypothesisGate] = []
        current: HypothesisGate | None = leaf
        while current is not None:
            chain.append(current)
            current = by_key[current.parent][1] if current.parent is not None else None
        chain.reverse()
        planned_share = sum(node.risk_share or 0.0 for node in chain)
        used_share = 0.0
        status = "pending"
        claims = []
        for node in chain:
            test_id, _ = by_key[node.key]
            outcome = outcomes.get(node.key)
            if outcome is None or outcome.get("status") in {"pending", "closed"}:
                status = outcome.get("status") if outcome else "pending"
                break
            if outcome.get("status") != "open":
                status = "closed"
                break
            result = outcome.get("result")
            decision = result.decision if result is not None else None
            if result is not None and node.risk_share is not None:
                decision = "significant" if result.p_value <= node.risk_share else "not-significant"
            if decision is not None:
                design = result.design or {}
                if decision == "significant":
                    used_share += float(design.get("risk_significant_at_q0", node.risk_share or 0.0))
                claims.append({"hypothesis": node.key, "test_id": test_id, "decision": decision})
            if node is leaf:
                status = "open"
        summaries.append(
            {
                "path": [node.key for node in chain],
                "test_ids": [by_key[node.key][0] for node in chain],
                "planned_risk_share": planned_share,
                "used_risk_share": used_share,
                "remaining_risk_share": max(0.0, family_alpha - used_share),
                "status": status,
                "formal_claims": claims,
                "preplanned_allocation": [
                    {"hypothesis": node.key, **node.allocation} for node in chain
                ],
            }
        )
    return summaries


def _formal_decision(
    gate: HypothesisGate, result: Any | None, effective_share: float
) -> str | None:
    if result is None:
        return None
    decision = result.decision
    if gate.risk_share is not None:
        decision = (
            "significant" if result.p_value <= effective_share else "not-significant"
        )
    return decision


def budget_ledger(
    gates: Sequence[tuple[int, HypothesisGate]],
    outcomes: dict[str, dict[str, Any]],
    family_alpha: float,
) -> dict[str, Any]:
    """Per-hypothesis risk budget ledger for one confirmation family.

    Every hypothesis starts with its pre-declared initial share. A hypothesis
    that closes unopened or finishes without a formal rejection routes its
    unused initial share to its pre-declared ``share_on_close`` destination
    (another hypothesis key, the family pool, or reserved). A shared follow-up
    opened through several parents still receives exactly one usable share.
    Shares already spent on a formal judgment stay spent: they are never
    returned by peeking, reopening, or re-running a completed test.
    """
    declared = [(test_id, gate) for test_id, gate in gates]
    by_key = {gate.key: gate for _, gate in declared}

    order: list[str] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(key: str) -> None:
        if key in permanent or key not in by_key:
            return
        if key in temporary:
            raise ValueError("share_on_close routing must not contain cycles")
        temporary.add(key)
        visit(by_key[key].share_on_close)
        temporary.discard(key)
        permanent.add(key)
        order.append(key)

    for _, gate in declared:
        visit(gate.key)

    effective: dict[str, float] = {
        gate.key: float(gate.risk_share or 0.0) for _, gate in declared
    }
    transferred_in: dict[str, list[dict[str, Any]]] = {gate.key: [] for _, gate in declared}
    transferred_out: dict[str, list[dict[str, Any]]] = {gate.key: [] for _, gate in declared}
    used: dict[str, float] = {gate.key: 0.0 for _, gate in declared}
    decisions: dict[str, str | None] = {}
    returned_to_family = 0.0
    reserved = 0.0

    for key in order:
        gate = by_key[key]
        outcome = outcomes.get(key) or {}
        status = outcome.get("status", "pending")
        result = outcome.get("result")
        share = float(gate.risk_share or 0.0)
        decision = None
        if status == "open" and result is not None:
            decision = _formal_decision(gate, result, effective[key])
            if decision == "significant":
                design = result.design or {}
                used[key] = float(design.get("risk_significant_at_q0", effective[key]))
        decisions[key] = decision
        failed = status == "closed" or (
            status == "open" and result is not None and decision != "significant"
        )
        leftover = max(0.0, share - used[key])
        if leftover <= 0.0:
            continue
        if failed:
            target = gate.share_on_close
            if target == "family":
                returned_to_family += leftover
                transferred_out[key].append({"to": "family", "amount": leftover})
            elif target == "reserved":
                reserved += leftover
                transferred_out[key].append({"to": "reserved", "amount": leftover})
            else:
                transferred_in[target].append({"from": key, "amount": leftover})
                transferred_out[key].append({"to": target, "amount": leftover})
                effective[target] += leftover
        elif decision == "significant":
            reserved += leftover

    entries = []
    for test_id, gate in declared:
        outcome = outcomes.get(gate.key) or {}
        entries.append(
            {
                "hypothesis": gate.key,
                "test_id": test_id,
                "status": outcome.get("status", "pending"),
                "opened_by": outcome.get("opened_by"),
                "initial_share": float(gate.risk_share or 0.0),
                "transferred_in": transferred_in[gate.key],
                "transferred_out": transferred_out[gate.key],
                "effective_share": effective[gate.key],
                "used_share": used[gate.key],
                "remaining_share": max(0.0, effective[gate.key] - used[gate.key]),
                "formal_decision": decisions[gate.key],
            }
        )
    total_used = sum(used.values())
    return {
        "hypotheses": entries,
        "family_alpha": family_alpha,
        "total_initial_share": sum(
            float(gate.risk_share or 0.0) for _, gate in declared
        ),
        "total_used_share": total_used,
        "returned_to_family": returned_to_family,
        "reserved_share": reserved,
        "family_remaining_share": max(0.0, family_alpha - total_used),
    }
