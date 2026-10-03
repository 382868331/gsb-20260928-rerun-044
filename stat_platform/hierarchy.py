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
    parents: tuple[str, ...] = ()
    parent_mode: str = "all"
    parent_mode_explicit: bool = False
    reallocate_to: str | None = None

    @classmethod
    def from_settings(cls, settings: dict[str, Any], index: int) -> "HypothesisGate | None":
        value = settings.get("hierarchy")
        if not value:
            return None
        key = str(value.get("key") or f"H{index + 1}")
        raw_parents = value.get("parents", value.get("parent"))
        if raw_parents is None:
            parents: tuple[str, ...] = ()
        elif isinstance(raw_parents, str):
            parents = (raw_parents,)
        else:
            parents = tuple(str(item) for item in raw_parents)
        parent = parents[0] if parents else None
        condition = value.get("open_when") or (
            {"decision": "significant"} if parent is not None else None
        )
        risk_share = value.get("risk_share")
        reallocate_to = value.get("reallocate_to")
        return cls(
            key=key,
            parent=parent,
            open_when=dict(condition) if condition is not None else None,
            risk_share=float(risk_share) if risk_share is not None else None,
            allocation=dict(value.get("allocation") or {}),
            parents=parents,
            parent_mode=str(value.get("parent_mode") or "all"),
            parent_mode_explicit="parent_mode" in value,
            reallocate_to=str(reallocate_to) if reallocate_to is not None else None,
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
            for parent in gate.parents:
                if parent not in key_set:
                    raise ValueError(f"unknown parent hypothesis {parent}")
            if gate.open_when is None:
                raise ValueError("a dependent hypothesis needs open_when")
            if len(gate.parents) > 1 and not gate.parent_mode_explicit:
                raise ValueError(
                    "a hypothesis with multiple parents must predeclare parent_mode "
                    "('all' parents must satisfy the gate, or 'any' single one)"
                )
            if gate.parent_mode not in {"all", "any"}:
                raise ValueError("parent_mode must be 'all' or 'any'")
        if gate.reallocate_to is not None:
            if gate.reallocate_to not in key_set:
                raise ValueError(f"unknown reallocation target {gate.reallocate_to}")
            if gate.reallocate_to == gate.key:
                raise ValueError("a hypothesis cannot reallocate its share to itself")

    by_key = {gate.key: gate for gate in declared}
    for path in enumerate_root_paths(declared):
        shares = [by_key[key].risk_share or 0.0 for key in path]
        if sum(shares) > family_alpha + 1e-12:
            raise ValueError(
                "risk shares on every root-to-leaf path must not exceed family alpha"
            )

    roots = [gate for gate in declared if not gate.parents]
    root_total = sum(gate.risk_share or 0.0 for gate in roots)
    if root_total > family_alpha + 1e-12:
        detail = ", ".join(
            f"{gate.key}={gate.risk_share:g}" for gate in roots
        )
        raise ValueError(
            "risk shares of parallel primary hypotheses exceed the family budget: "
            f"{detail} sum to {root_total:g} > family alpha {family_alpha:g}"
        )


def enumerate_root_paths(declared: Sequence[HypothesisGate]) -> list[list[str]]:
    """Every root-to-leaf path of the (possibly multi-parent) gate graph."""
    by_key = {gate.key: gate for gate in declared}
    memo: dict[str, list[list[str]]] = {}

    def paths_to(key: str, active: frozenset[str]) -> list[list[str]]:
        if key in active:
            raise ValueError("hypothesis gates must not contain cycles")
        if key in memo:
            return memo[key]
        gate = by_key[key]
        if not gate.parents:
            paths = [[key]]
        else:
            paths = []
            for parent in gate.parents:
                for prefix in paths_to(parent, active | {key}):
                    paths.append(prefix + [key])
        memo[key] = paths
        return paths

    children = {parent for gate in declared for parent in gate.parents}
    leaves = [gate.key for gate in declared if gate.key not in children]
    return [path for leaf in leaves for path in paths_to(leaf, frozenset())]


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
    declared = [gate for _, gate in gates]
    summaries = []
    for path in enumerate_root_paths(declared):
        chain = [by_key[key][1] for key in path]
        leaf = chain[-1]
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
