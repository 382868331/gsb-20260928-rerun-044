from __future__ import annotations

"""Pre-planned hierarchical (gatekeeping) follow-up hypotheses.

A batch may pre-register a set of *primary* hypotheses and *follow-up*
hypotheses.  A follow-up enters formal judgement only when every primary it
depends on has been formally rejected at its own pre-allocated risk share.
Until then its observations are reported as exploratory only and can never be
presented as a formal pass.

The family-wise risk budget ``alpha`` is split before running: each primary
owns an initial share (shares sum to at most 1), and a rejected node passes
its available share to its direct dependents along pre-specified edge
weights (serial gatekeeping).  Because every formal judgement at a node is
made at level ``alpha * available_share`` and the open shares along any
opening path sum to at most 1, the family-wise error is bounded by ``alpha``
by Bonferroni, however the permutation draws are shared between tests.

The plan is fixed before any test completes and the gate state is derived
deterministically from the persisted results, so pausing, resuming, or
adding compute budget can neither reset the openings nor inflate the risk.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GateNode:
    """One hypothesis in the gate plan.

    ``share`` is the pre-allocated initial risk share (fraction of the
    family-wise alpha); primaries carry positive shares and pure follow-ups
    usually start at 0 and only inherit share from rejected parents.
    ``depends_on`` lists the test ids that must all be formally rejected
    before this hypothesis opens for formal judgement.
    """

    test_id: int
    share: float = 0.0
    depends_on: tuple[int, ...] = ()

    @property
    def role(self) -> str:
        return "primary" if not self.depends_on else "follow-up"


@dataclass(frozen=True)
class GatePlan:
    """Fixed pre-run allocation of risk shares and opening conditions."""

    alpha: float
    nodes: tuple[GateNode, ...]
    edge_weights: Mapping[tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [node.test_id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate test_id in gate plan")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        known = set(ids)
        for node in self.nodes:
            if node.share < 0:
                raise ValueError("shares must be non-negative")
            unknown = set(node.depends_on) - known
            if unknown:
                raise ValueError(f"unknown dependencies: {sorted(unknown)}")
            if node.test_id in node.depends_on:
                raise ValueError("a node cannot depend on itself")
        if self._topological_order() is None:
            raise ValueError("gate dependencies must be acyclic")
        initial = sum(node.share for node in self.nodes)
        if initial > 1.0 + 1e-9:
            raise ValueError(
                f"initial shares sum to {initial:.6f} > 1; the family-wise "
                "risk budget would be exceeded"
            )
        for (parent, child), weight in self.edge_weights.items():
            if parent not in known or child not in known:
                raise ValueError("edge weights must reference planned tests")
            if weight <= 0:
                raise ValueError("edge weights must be positive")

    def node(self, test_id: int) -> GateNode:
        for node in self.nodes:
            if node.test_id == test_id:
                return node
        raise KeyError(f"test {test_id} is not part of the gate plan")

    def _topological_order(self) -> tuple[int, ...] | None:
        remaining = {node.test_id: set(node.depends_on) for node in self.nodes}
        order: list[int] = []
        while remaining:
            ready = sorted(
                test_id for test_id, deps in remaining.items() if not deps
            )
            if not ready:
                return None
            for test_id in ready:
                order.append(test_id)
                del remaining[test_id]
            for deps in remaining.values():
                deps.difference_update(ready)
        return tuple(order)

    def _inherit_weight(self, parent: int, child: int) -> float:
        return float(self.edge_weights.get((parent, child), 1.0))

    def _children(self, parent: int) -> list[int]:
        return [
            node.test_id
            for node in self.nodes
            if parent in node.depends_on
        ]

    def evaluate(
        self,
        p_values: Mapping[int, float | None],
    ) -> dict[int, dict[str, Any]]:
        """Propagate openings and formal judgements from observed p-values.

        ``p_values`` maps test id to its completed p-value, or ``None`` when
        the test has not finished.  The result is deterministic in the
        persisted evidence, so re-deriving it after a pause/resume reproduces
        exactly the same openings and judgements.
        """
        states: dict[int, dict[str, Any]] = {}
        for test_id in self._topological_order():
            node = self.node(test_id)
            parents = [states[parent] for parent in node.depends_on]
            opened = all(parent["formal_reject"] for parent in parents)
            inherited = 0.0
            for parent_id, parent in zip(node.depends_on, parents):
                if not parent["formal_reject"]:
                    continue
                children = self._children(parent_id)
                total_weight = sum(
                    self._inherit_weight(parent_id, child) for child in children
                )
                inherited += (
                    parent["available_share"]
                    * self._inherit_weight(parent_id, test_id)
                    / total_weight
                )
            available_share = node.share + inherited
            threshold = self.alpha * available_share
            p_value = p_values.get(test_id)
            formal_reject = bool(
                opened and p_value is not None and p_value <= threshold
            )
            judged = opened and p_value is not None
            if not opened:
                status = "unopened"
            elif p_value is None:
                status = "open-pending"
            elif formal_reject:
                status = "rejected"
            else:
                status = "not-rejected"
            states[test_id] = {
                "test_id": test_id,
                "role": node.role,
                "depends_on": list(node.depends_on),
                "initial_share": node.share,
                "inherited_share": inherited,
                "available_share": available_share,
                "alpha_threshold": threshold,
                "open": opened,
                "status": status,
                "p_value": p_value,
                "formal_reject": formal_reject,
                # A conclusion only consumes family-wise risk once it is
                # formally judged; unopened observations consume nothing and
                # are exploratory only.
                "risk_share_used": threshold if judged else 0.0,
                "exploratory": not opened,
                "exploratory_p_value": None if opened else p_value,
            }
        return states

    def compare_paths(self) -> list[dict[str, Any]]:
        """Enumerate every opening path and its achievable conclusions.

        Each scenario is a self-consistent set of rejected nodes (a rejected
        node's dependencies are rejected as well).  For every scenario the
        report lists which hypotheses would be open, which formal conclusions
        could be reached, and the total risk share in use - which never
        exceeds the family-wise budget.
        """
        ids = list(self._topological_order())
        scenarios: list[dict[str, Any]] = []
        for size in range(len(ids) + 1):
            for subset in combinations(ids, size):
                rejected = set(subset)
                if any(
                    not set(self.node(test_id).depends_on) <= rejected
                    for test_id in rejected
                ):
                    continue
                opened = {
                    test_id
                    for test_id in ids
                    if set(self.node(test_id).depends_on) <= rejected
                }
                # A rejected node has passed its share on to its children, so
                # the budget still in use is carried only by hypotheses that
                # remain open for judgement.
                judgeable = opened - rejected
                shares = self._open_shares(rejected)
                open_share = sum(shares[test_id] for test_id in judgeable)
                scenarios.append(
                    {
                        "rejected": sorted(rejected),
                        "open": sorted(opened),
                        "unopened": sorted(set(ids) - opened),
                        "achievable_conclusions": sorted(judgeable),
                        "open_risk_share": open_share,
                        "within_budget": open_share <= 1.0 + 1e-9,
                    }
                )
        scenarios.sort(key=lambda item: (len(item["rejected"]), item["rejected"]))
        return scenarios

    def _open_shares(self, rejected: set[int]) -> dict[int, float]:
        shares: dict[int, float] = {}
        for test_id in self._topological_order():
            node = self.node(test_id)
            inherited = 0.0
            for parent_id in node.depends_on:
                if parent_id not in rejected:
                    continue
                children = self._children(parent_id)
                total_weight = sum(
                    self._inherit_weight(parent_id, child) for child in children
                )
                inherited += (
                    shares[parent_id]
                    * self._inherit_weight(parent_id, test_id)
                    / total_weight
                )
            shares[test_id] = node.share + inherited
        return shares

    def max_open_risk_share(self) -> float:
        """Largest total risk share simultaneously open over all paths."""
        return max(
            (scenario["open_risk_share"] for scenario in self.compare_paths()),
            default=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nodes": [
                {
                    "test_id": node.test_id,
                    "share": node.share,
                    "depends_on": list(node.depends_on),
                }
                for node in self.nodes
            ],
            "edges": [
                {"parent": parent, "child": child, "weight": weight}
                for (parent, child), weight in sorted(self.edge_weights.items())
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GatePlan":
        nodes = tuple(
            GateNode(
                test_id=int(item["test_id"]),
                share=float(item.get("share", 0.0)),
                depends_on=tuple(int(dep) for dep in item.get("depends_on", ())),
            )
            for item in value["nodes"]
        )
        edges = {
            (int(edge["parent"]), int(edge["child"])): float(edge["weight"])
            for edge in value.get("edges", ())
        }
        return cls(alpha=float(value["alpha"]), nodes=nodes, edge_weights=edges)


def plan_from_positions(
    alpha: float,
    nodes: Sequence[Mapping[str, Any]],
    test_ids: Sequence[int],
    edges: Sequence[Mapping[str, Any]] = (),
) -> GatePlan:
    """Build a plan specified with 0-based positions into a batch's tests."""
    for item in nodes:
        position = int(item["test"])
        if not 0 <= position < len(test_ids):
            raise ValueError(
                f"gate node position {position} out of range for "
                f"{len(test_ids)} tests"
            )
    resolved_nodes = tuple(
        GateNode(
            test_id=test_ids[int(item["test"])],
            share=float(item.get("share", 0.0)),
            depends_on=tuple(
                test_ids[int(dep)] for dep in item.get("depends_on", ())
            ),
        )
        for item in nodes
    )
    resolved_edges = {
        (test_ids[int(edge["parent"])], test_ids[int(edge["child"])]): float(
            edge["weight"]
        )
        for edge in edges
    }
    return GatePlan(alpha=alpha, nodes=resolved_nodes, edge_weights=resolved_edges)
