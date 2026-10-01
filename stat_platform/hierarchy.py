from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class HypothesisGate:
    key: str
    parent: str | None
    open_when: dict[str, Any] | None
    risk_share: float | None
    allocation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: dict[str, Any], index: int) -> "HypothesisGate | None":
        value = settings.get("hierarchy")
        if not value:
            return None
        key = str(value.get("key") or f"H{index + 1}")
        parent = value.get("parent")
        condition = value.get("open_when") or (
            {"decision": "significant"} if parent is not None else None
        )
        risk_share = value.get("risk_share")
        return cls(
            key=key,
            parent=str(parent) if parent is not None else None,
            open_when=dict(condition) if condition is not None else None,
            risk_share=float(risk_share) if risk_share is not None else None,
            allocation=dict(value.get("allocation") or {}),
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
        if gate.parent is not None:
            if gate.parent not in key_set:
                raise ValueError(f"unknown parent hypothesis {gate.parent}")
            if gate.open_when is None:
                raise ValueError("a dependent hypothesis needs open_when")

    by_key = {gate.key: gate for gate in declared}
    for gate in declared:
        path: list[str] = []
        current = gate
        while current.parent is not None:
            if current.key in path:
                raise ValueError("hypothesis gates must not contain cycles")
            path.append(current.key)
            current = by_key[current.parent]
        shares = [
            node.risk_share
            for node in [by_key[key] for key in reversed(path)] + [gate]
            if node.risk_share is not None
        ]
        if sum(shares) > family_alpha + 1e-12:
            raise ValueError("risk shares on every root-to-leaf path must not exceed family alpha")


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
        if gate.key not in {node.parent for _, node in gates if node.parent is not None}
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
