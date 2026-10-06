"""Risk-quota allocation studies across hierarchical hypotheses.

A study is a pure, read-only comparison of legal ways to distribute a fixed
family risk budget over declared targets, parent open conditions, and allowed
transfer routes. It never persists anything and never consumes budget: shares
already spent on formal decisions are locked in as immutable minimums before
any candidate scheme is evaluated, so used quota can never be reallocated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Sequence

from .hierarchy import _ensure_acyclic, _ensure_unique_edges


@dataclass(frozen=True)
class AllocationTarget:
    """One hypothesis taking part in an allocation study."""

    key: str
    parents: tuple[str, ...] = ()
    used_share: float = 0.0
    transfer_targets: tuple[str, ...] = ()
    gate_status: str = "open"


@dataclass(frozen=True)
class AllocationStudy:
    family_alpha: float
    used_shares: dict[str, float]
    schemes: tuple[dict[str, Any], ...]
    max_effective_share: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_alpha": self.family_alpha,
            "used_shares": dict(self.used_shares),
            "schemes": [dict(scheme) for scheme in self.schemes],
            "max_effective_share": {
                key: dict(value) for key, value in self.max_effective_share.items()
            },
            "consumes_budget": False,
            "meaning": (
                "Read-only comparison of legal allocation schemes under the fixed "
                "family budget. Used shares are locked minimums and cannot be "
                "reallocated; evaluating a scheme consumes nothing."
            ),
        }


def validate_targets(
    targets: Sequence[AllocationTarget], family_alpha: float
) -> None:
    keys = [target.key for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("allocation target keys must be unique")
    key_set = set(keys)
    for target in targets:
        if target.used_share < -1e-12:
            raise ValueError(f"used share of {target.key} must not be negative")
        if target.used_share > family_alpha + 1e-12:
            raise ValueError(
                f"used share of {target.key} exceeds the family budget and can "
                "never fit any legal scheme"
            )
        for parent in target.parents:
            if parent not in key_set:
                raise ValueError(f"unknown parent hypothesis {parent}")
        if len(set(target.parents)) != len(target.parents):
            raise ValueError(
                f"hypothesis {target.key} lists a parent source more than once"
            )
        if len(set(target.transfer_targets)) != len(target.transfer_targets):
            raise ValueError(
                f"hypothesis {target.key} lists a transfer target more than once"
            )
        for receiver in target.transfer_targets:
            if receiver not in key_set:
                raise ValueError(f"unknown transfer target {receiver}")
            if receiver == target.key:
                raise ValueError("a hypothesis cannot transfer quota to itself")
    gate_edges = [
        (parent, target.key) for target in targets for parent in target.parents
    ]
    _ensure_acyclic(keys, gate_edges, "hypothesis gates")
    transfer_edges = [
        (target.key, receiver)
        for target in targets
        for receiver in target.transfer_targets
    ]
    # Duplicate routes and cycles must never become a way to inflate quota:
    # reject them before any scheme is compared.
    _ensure_unique_edges(transfer_edges, "transfer route")
    _ensure_acyclic(keys, transfer_edges, "transfer routes")
    root_used = sum(target.used_share for target in targets if not target.parents)
    if root_used > family_alpha + 1e-12:
        raise ValueError(
            "used shares of parallel primary hypotheses already exceed the "
            f"family budget: {root_used:g} > {family_alpha:g}"
        )


def study_allocations(
    targets: Sequence[AllocationTarget],
    family_alpha: float,
    *,
    priorities: Sequence[Sequence[str]] | None = None,
) -> AllocationStudy:
    """Compare legal allocation schemes and derive each target's maximum share.

    Every scheme is a greedy waterfall over a priority order: each target in
    turn receives the largest own share that still fits the fixed family
    budget alongside already-assigned shares and the locked used minimums of
    the remaining targets. Unused surplus then flows once along the declared
    acyclic transfer routes, split evenly across a source's allowed targets;
    a shared successor merges what it receives from different parents by
    summing the real per-edge amounts. Used shares are locked minimums in
    every scheme, so quota spent on formal decisions is never reallocated.
    """
    targets = tuple(targets)
    validate_targets(targets, family_alpha)
    by_key = {target.key: target for target in targets}
    keys = [target.key for target in targets]
    if priorities is None:
        if len(keys) > 8:
            raise ValueError(
                "too many targets to enumerate all priority orders; pass "
                "explicit priorities"
            )
        orders = [tuple(order) for order in permutations(keys)]
    else:
        orders = [tuple(str(key) for key in order) for order in priorities]
        for order in orders:
            if sorted(order) != sorted(keys):
                raise ValueError(
                    "each priority order must list every target exactly once: "
                    f"{list(order)}"
                )
    constraint_sets = _constraint_sets(targets)
    transfer_order = _topological_order(
        keys,
        [(target.key, r) for target in targets for r in target.transfer_targets],
    )
    schemes: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {
        key: {"amount": 0.0, "attained_by": []} for key in keys
    }
    for order in orders:
        shares, feasible = _waterfall(order, by_key, constraint_sets, family_alpha)
        scheme: dict[str, Any] = {"priority": list(order), "feasible": feasible}
        if not feasible:
            schemes.append(scheme)
            continue
        effective, transfers = _route_surplus(shares, by_key, transfer_order)
        scheme.update(
            {
                "shares": {key: shares[key] for key in keys},
                "effective": {key: effective[key] for key in keys},
                "transfers": transfers,
                "total_initial_share": sum(shares.values()),
                "total_used_share": sum(by_key[key].used_share for key in keys),
            }
        )
        schemes.append(scheme)
        index = len(schemes) - 1
        for key in keys:
            amount = effective[key]
            if amount > best[key]["amount"] + 1e-12:
                best[key] = {"amount": amount, "attained_by": [index]}
            elif abs(amount - best[key]["amount"]) <= 1e-12:
                best[key]["attained_by"].append(index)
    return AllocationStudy(
        family_alpha=family_alpha,
        used_shares={key: by_key[key].used_share for key in keys},
        schemes=tuple(schemes),
        max_effective_share=best,
    )


def _constraint_sets(
    targets: Sequence[AllocationTarget],
) -> list[frozenset]:
    """Budget-limited groups: the root set and every root-to-leaf path."""
    by_key = {target.key: target for target in targets}
    sets: list[frozenset] = [
        frozenset(target.key for target in targets if not target.parents)
    ]
    memo: dict[str, list[tuple[str, ...]]] = {}

    def paths_to(key: str) -> list[tuple[str, ...]]:
        if key in memo:
            return memo[key]
        target = by_key[key]
        if not target.parents:
            paths = [(key,)]
        else:
            paths = [
                prefix + (key,)
                for parent in dict.fromkeys(target.parents)
                for prefix in paths_to(parent)
            ]
        memo[key] = paths
        return paths

    children = {parent for target in targets for parent in target.parents}
    for target in targets:
        if target.key not in children:
            for path in paths_to(target.key):
                sets.append(frozenset(path))
    deduped: list[frozenset] = []
    for group in sets:
        if group and group not in deduped:
            deduped.append(group)
    return deduped


def _waterfall(
    order: tuple[str, ...],
    by_key: dict[str, AllocationTarget],
    constraint_sets: Sequence[frozenset],
    family_alpha: float,
) -> tuple[dict[str, float], bool]:
    assigned: dict[str, float] = {}
    remaining = set(by_key)
    for key in order:
        remaining.discard(key)
        limit = family_alpha
        for group in constraint_sets:
            if key not in group:
                continue
            others_assigned = sum(
                assigned[other] for other in group if other in assigned
            )
            others_locked = sum(
                by_key[other].used_share
                for other in group
                if other in remaining
            )
            limit = min(limit, family_alpha - others_assigned - others_locked)
        limit = max(0.0, limit)
        used = by_key[key].used_share
        if used > limit + 1e-12:
            return assigned, False
        assigned[key] = max(used, limit)
    return assigned, True


def _route_surplus(
    shares: dict[str, float],
    by_key: dict[str, AllocationTarget],
    transfer_order: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    effective = dict(shares)
    transfers: list[dict[str, Any]] = []
    for key in transfer_order:
        target = by_key[key]
        if not target.transfer_targets:
            continue
        surplus = max(0.0, effective[key] - target.used_share)
        if surplus <= 1e-12:
            continue
        portion = surplus / len(target.transfer_targets)
        for receiver in target.transfer_targets:
            effective[receiver] += portion
            transfers.append({"from": key, "to": receiver, "amount": portion})
    return effective, transfers


def _topological_order(
    keys: Sequence[str], edges: Sequence[tuple[str, str]]
) -> list[str]:
    outgoing: dict[str, list[str]] = {key: [] for key in keys}
    indegree: dict[str, int] = {key: 0 for key in keys}
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    ready = [key for key in keys if indegree[key] == 0]
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for neighbor in outgoing[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)
    return order

