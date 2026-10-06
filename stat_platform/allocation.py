"""Risk-budget allocation studies across hierarchical hypotheses.

A study is a pure what-if analysis: the researcher fixes the total family
budget, declares the hypotheses to test (with parent open conditions) and the
allowed share-transfer routes, and the platform compares legal allocation
schemes without consuming any budget or persisting any evidence. Shares
already spent on formal decisions are locked and can never be reassigned by
a candidate scheme.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Sequence

from .hierarchy import HypothesisGate, validate_gates

# Outcomes that release the unused part of a share along reallocate_to edges.
_RELEASING_OUTCOMES = {"not-significant", "closed"}
# Outcomes enumerated for hypotheses that are not yet decided.
_UNDECIDED_OUTCOMES = ("significant", "not-significant", "untested")
_KNOWN_OUTCOMES = {"significant", "not-significant", "closed", "untested"}

_EPS = 1e-12


def gates_from_plan(plan: Sequence[dict[str, Any]]) -> list[HypothesisGate]:
    """Parse hierarchy declarations (same shape as settings['hierarchy'])."""
    gates = []
    for index, item in enumerate(plan):
        gate = HypothesisGate.from_settings({"hierarchy": dict(item)}, index)
        if gate is None:
            raise ValueError(f"plan entry {index} does not declare a hierarchy")
        gates.append(gate)
    return gates


def simulate_scheme(
    gates: Sequence[HypothesisGate],
    family_alpha: float,
    *,
    outcomes: dict[str, str] | None = None,
    used_shares: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Propagate shares along declared transfer routes for one scenario.

    ``outcomes`` maps hypothesis key to one of ``significant``,
    ``not-significant``, ``closed`` or ``untested``; missing keys are treated
    as ``untested`` and neither consume nor release anything beyond their
    locked used share. ``used_shares`` locks budget already spent on formal
    decisions: it is charged to the node again here and can never be
    transferred away. Every transfer edge fires at most once and shared
    successors merge incoming amounts by straightforward addition.
    """
    declared = [gate for gate in gates if gate is not None]
    validate_gates(declared, family_alpha)
    outcomes = dict(outcomes or {})
    used_shares = {key: float(value) for key, value in (used_shares or {}).items()}
    for key, outcome in outcomes.items():
        if outcome not in _KNOWN_OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r} for hypothesis {key}")

    by_key = {gate.key: gate for gate in declared}
    edges = [
        (gate.key, gate.reallocate_to) for gate in declared if gate.reallocate_to
    ]
    order = _topological_order([gate.key for gate in declared], edges)

    effective = {gate.key: float(gate.risk_share or 0.0) for gate in declared}
    transfers_in: dict[str, list[dict[str, Any]]] = {gate.key: [] for gate in declared}
    transfers_out: dict[str, list[dict[str, Any]]] = {gate.key: [] for gate in declared}
    used: dict[str, float] = {}
    violations: list[str] = []

    for key in order:
        gate = by_key[key]
        locked = used_shares.get(key, 0.0)
        if locked > effective[key] + _EPS:
            violations.append(
                f"hypothesis {key} already used {locked:g} but the scheme only "
                f"leaves it {effective[key]:g}; used budget cannot be reallocated"
            )
        outcome = outcomes.get(key, "untested")
        if outcome == "significant":
            # A formal significant decision consumes the full effective share.
            charge = max(locked, effective[key])
        else:
            charge = locked
        used[key] = charge
        releasable = effective[key] - charge
        if (
            gate.reallocate_to is not None
            and outcome in _RELEASING_OUTCOMES | {"significant"}
            and releasable > _EPS
        ):
            transfer = {"to": gate.reallocate_to, "amount": releasable}
            transfers_out[key].append(transfer)
            transfers_in[gate.reallocate_to].append(
                {"from": key, "amount": releasable}
            )
            effective[gate.reallocate_to] += releasable

    ledger = []
    for gate in declared:
        key = gate.key
        released = sum(item["amount"] for item in transfers_out[key])
        ledger.append(
            {
                "hypothesis": key,
                "outcome": outcomes.get(key, "untested"),
                "initial_allocation": float(gate.risk_share or 0.0),
                "transfers_in": transfers_in[key],
                "transfers_out": transfers_out[key],
                "effective_allocation": effective[key],
                "used_risk_share": used[key],
                "remaining_risk_share": max(0.0, effective[key] - used[key] - released),
            }
        )
    total_used = sum(used.values())
    return {
        "family_alpha": float(family_alpha),
        "outcomes": {key: outcomes.get(key, "untested") for key in by_key},
        "ledger": ledger,
        "total_used_risk_share": total_used,
        "budget_respected": total_used <= float(family_alpha) + _EPS,
        "violations": violations,
    }


def _topological_order(
    keys: Sequence[str], edges: Sequence[tuple[str, str]]
) -> list[str]:
    incoming: dict[str, int] = {key: 0 for key in keys}
    outgoing: dict[str, list[str]] = {key: [] for key in keys}
    for source, target in edges:
        outgoing[source].append(target)
        incoming[target] = incoming.get(target, 0) + 1
        incoming.setdefault(source, incoming.get(source, 0))
        outgoing.setdefault(target, [])
    ready = [key for key in keys if incoming.get(key, 0) == 0]
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in outgoing.get(node, []):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if len(order) != len(incoming):
        raise ValueError("reallocation routes must not contain cycles")
    return order


def study_schemes(
    schemes: Sequence[dict[str, Any]],
    family_alpha: float,
    *,
    used_shares: dict[str, float] | None = None,
    decided: dict[str, str] | None = None,
    max_scenarios: int = 4096,
) -> dict[str, Any]:
    """Compare candidate allocation schemes under one fixed family budget.

    Each scheme is ``{'name': str, 'plan': [hierarchy dicts]}``. Schemes are
    validated before any simulation (duplicate edges, cycles, path and root
    budget violations reject the scheme), then every legal outcome scenario
    for the not-yet-decided hypotheses is simulated. The study reports, per
    scheme and per hypothesis, the maximum effective share the hypothesis can
    receive and the scenario that grants it. Nothing is persisted and no
    budget is consumed; ``used_shares`` locks already-spent budget so no
    candidate scheme can assign it again.
    """
    used_shares = {key: float(value) for key, value in (used_shares or {}).items()}
    decided = dict(decided or {})
    results = []
    for scheme in schemes:
        name = str(scheme.get("name") or f"scheme-{len(results) + 1}")
        plan = scheme.get("plan", scheme.get("hypotheses"))
        if plan is None:
            results.append(
                {"name": name, "status": "rejected", "reason": "scheme has no plan"}
            )
            continue
        try:
            gates = gates_from_plan(plan)
            validate_gates(gates, family_alpha)
        except ValueError as exc:
            results.append({"name": name, "status": "rejected", "reason": str(exc)})
            continue

        keys = [gate.key for gate in gates]
        missing = [key for key in used_shares if key not in keys and used_shares[key] > _EPS]
        if missing:
            results.append(
                {
                    "name": name,
                    "status": "infeasible",
                    "reason": (
                        "scheme drops hypotheses with locked used budget: "
                        + ", ".join(missing)
                    ),
                }
            )
            continue

        undecided = [key for key in keys if key not in decided]
        if 3 ** len(undecided) > max_scenarios:
            results.append(
                {
                    "name": name,
                    "status": "rejected",
                    "reason": (
                        f"{len(undecided)} undecided hypotheses need "
                        f"{3 ** len(undecided)} scenarios > max_scenarios={max_scenarios}"
                    ),
                }
            )
            continue

        max_shares: dict[str, dict[str, Any]] = {
            key: {"max_effective_allocation": 0.0, "scenario": None} for key in keys
        }
        worst_total_used = 0.0
        budget_respected = True
        violations: list[str] = []
        scenarios_evaluated = 0
        for combo in product(_UNDECIDED_OUTCOMES, repeat=len(undecided)):
            scenario = dict(decided)
            scenario.update(zip(undecided, combo))
            simulation = simulate_scheme(
                gates,
                family_alpha,
                outcomes=scenario,
                used_shares=used_shares,
            )
            scenarios_evaluated += 1
            violations.extend(simulation["violations"])
            budget_respected = budget_respected and simulation["budget_respected"]
            worst_total_used = max(
                worst_total_used, simulation["total_used_risk_share"]
            )
            for entry in simulation["ledger"]:
                key = entry["hypothesis"]
                current = max_shares[key]["max_effective_allocation"]
                if entry["effective_allocation"] > current + _EPS:
                    max_shares[key] = {
                        "max_effective_allocation": entry["effective_allocation"],
                        "scenario": dict(scenario),
                    }

        violations = sorted(set(violations))
        status = "ok" if not violations and budget_respected else "infeasible"
        results.append(
            {
                "name": name,
                "status": status,
                "scenarios_evaluated": scenarios_evaluated,
                "violations": violations,
                "budget_respected": budget_respected,
                "worst_case_total_used_risk_share": worst_total_used,
                "max_shares": max_shares,
            }
        )
    return {
        "family_alpha": float(family_alpha),
        "locked_used_shares": used_shares,
        "decided_outcomes": decided,
        "schemes": results,
    }
