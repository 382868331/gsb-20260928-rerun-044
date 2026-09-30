from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def binomial_pmf(n: int, p: float) -> list[float]:
    """Exact Binom(n, p) probabilities computed by forward recurrence."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    pmf = [0.0] * (n + 1)
    pmf[0] = (1.0 - p) ** n
    if p in (0.0, 1.0) or n == 0:
        if p == 1.0 and n > 0:
            pmf[0] = 0.0
            pmf[n] = 1.0
        return pmf
    odds = p / (1.0 - p)
    for s in range(n):
        pmf[s + 1] = pmf[s] * odds * (n - s) / (s + 1)
    total = math.fsum(pmf)
    if total > 0.0:
        return [value / total for value in pmf]
    return pmf


def binomial_cdf(n: int, p: float) -> list[float]:
    pmf = binomial_pmf(n, p)
    cumulative = 0.0
    out: list[float] = []
    for value in pmf:
        cumulative += value
        out.append(cumulative)
    return out


def spending_schedule(
    looks: Sequence[int], budget: float, kind: str
) -> list[float]:
    """Cumulative error budget consumed up to each planned look."""
    n_max = looks[-1]
    if kind == "pocock":
        return [budget * (k + 1) / len(looks) for k in range(len(looks))]
    if kind == "obrien-flanagan":
        # (n/n_max)^2 spending: very little early, most budget at the end.
        return [budget * (n / n_max) ** 2 for n in looks]
    raise ValueError(f"unknown spending function: {kind}")


@dataclass(frozen=True)
class SequentialDesign:
    """Pre-registered sequential decision boundary for Monte Carlo p-values.

    Each random permutation is a Bernoulli trial with success probability
    equal to the (unknown) exact permutation tail probability ``p``.  Decisions
    are taken ONLY at the pre-selected cumulative sample counts ``looks``.

    Error guarantees (i.i.d. binomial design, also conservative for sampling
    permutations without replacement):

    * significant boundary  S_k <= lower[k]  controls
      sup_{p >= p_high} P(false significant) <= epsilon_high;
    * not-significant boundary  S_k >= upper[k]  controls
      sup_{p <= p_low} P(false not-significant) <= epsilon_low.

    Between ``p_low`` and ``p_high`` is a pre-declared indifference region;
    if the final look lands there the verdict is ``inconclusive`` rather than
    a forced claim.
    """

    looks: tuple[int, ...]
    p_high: float
    p_low: float
    epsilon_high: float
    epsilon_low: float
    spending: str
    lower: tuple[int, ...]
    upper: tuple[int, ...]

    @classmethod
    def create(
        cls,
        looks: Sequence[int],
        *,
        p_high: float = 0.05,
        p_low: float | None = None,
        epsilon_high: float = 0.05,
        epsilon_low: float = 0.05,
        spending: str = "pocock",
        total_assignments: int | None = None,
    ) -> "SequentialDesign":
        normalized = cls._normalize_looks(looks, total_assignments)
        if not 0.0 < p_high <= 1.0:
            raise ValueError("p_high must lie in (0, 1]")
        if p_low is None:
            p_low = p_high / 2.0
        if not 0.0 <= p_low < p_high:
            raise ValueError("p_low must satisfy 0 <= p_low < p_high")
        if not 0.0 < epsilon_high <= 1.0 or not 0.0 < epsilon_low <= 1.0:
            raise ValueError("error budgets must lie in (0, 1]")
        if epsilon_high + epsilon_low > 1.0:
            raise ValueError("epsilon_high + epsilon_low must not exceed 1")

        sig_spend = spending_schedule(normalized, epsilon_high, spending)
        ns_spend = spending_schedule(normalized, epsilon_low, spending)

        raw_lower: list[int] = []
        raw_upper: list[int] = []
        for index, n in enumerate(normalized):
            cdf_high = binomial_cdf(n, p_high)
            # Bonferroni/alpha-spending bound: allocate the cumulative spend
            # to the unconditional marginal tail mass at this look.  Union
            # bounds over the planned looks keep the overall risk <= epsilon.
            boundary_low = -1
            for s, mass in enumerate(cdf_high):
                if mass <= sig_spend[index] + 1e-12:
                    boundary_low = s
                else:
                    break

            cdf_low = binomial_cdf(n, p_low)
            # Smallest boundary u with P_p_low(S >= u) <= spend.
            boundary_high = n + 1
            tail = 0.0
            for s in range(n, -1, -1):
                tail += cdf_low[s] - (cdf_low[s - 1] if s > 0 else 0.0)
                if tail <= ns_spend[index] + 1e-12:
                    boundary_high = s
                else:
                    break
            raw_lower.append(boundary_low)
            raw_upper.append(boundary_high)
        lower: list[int] = []
        upper: list[int] = []
        for index, n in enumerate(normalized):
            low = max(raw_lower[: index + 1])
            high = max(value for value in raw_upper[: index + 1] if value <= n)
            lower.append(low)
            upper.append(high if high > low else n + 1)
        return cls(
            looks=tuple(normalized),
            p_high=p_high,
            p_low=p_low,
            epsilon_high=epsilon_high,
            epsilon_low=epsilon_low,
            spending=spending,
            lower=tuple(lower),
            upper=tuple(upper),
        )

    @staticmethod
    def _normalize_looks(
        looks: Sequence[int], total_assignments: int | None
    ) -> list[int]:
        values = [int(n) for n in looks]
        if not values:
            raise ValueError("at least one planned look is required")
        if any(n <= 0 for n in values):
            raise ValueError("planned looks must be positive")
        if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
            raise ValueError("planned looks must be strictly increasing")
        if total_assignments is not None and values[-1] > int(total_assignments):
            # Sampling without replacement cannot pass the finite permutation
            # space; reaching it means the exact enumeration answer is known.
            values = [n for n in values if n < int(total_assignments)]
            values.append(int(total_assignments))
            values = sorted(set(values))
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "looks": list(self.looks),
            "p_high": self.p_high,
            "p_low": self.p_low,
            "epsilon_high": self.epsilon_high,
            "epsilon_low": self.epsilon_low,
            "spending": self.spending,
            "lower": list(self.lower),
            "upper": list(self.upper),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SequentialDesign":
        looks = tuple(int(n) for n in data["looks"])
        lower = tuple(int(v) for v in data["lower"])
        upper = tuple(int(v) for v in data["upper"])
        if not (len(lower) == len(upper) == len(looks)):
            raise ValueError("stored sequential design has mismatched boundary lengths")
        return cls(
            looks=looks,
            p_high=float(data["p_high"]),
            p_low=float(data["p_low"]),
            epsilon_high=float(data["epsilon_high"]),
            epsilon_low=float(data["epsilon_low"]),
            spending=str(data.get("spending", "pocock")),
            lower=lower,
            upper=upper,
        )

    def look_index(self, evaluated: int) -> int | None:
        for index, n in enumerate(self.looks):
            if evaluated == n:
                return index
        return None

    def next_look(self, evaluated: int) -> int | None:
        for n in self.looks:
            if n > evaluated:
                return n
        return None

    def evaluate(self, evaluated: int, hits: int) -> dict[str, Any]:
        """Verdict at a planned look; non-look sample counts are monitoring only."""
        index = self.look_index(evaluated)
        estimate = (hits + 1) / (evaluated + 1) if evaluated else float("nan")
        if index is None:
            return {
                "at_planned_look": False,
                "look": None,
                "evaluated": evaluated,
                "hits": hits,
                "p_estimate": estimate,
                "decision": "monitoring",
                "lower_boundary": None,
                "upper_boundary": None,
                "remaining_conditions": [
                    f"reach next planned look at n={self.next_look(evaluated)}"
                ],
            }
        low = self.lower[index]
        high = self.upper[index]
        final = index == len(self.looks) - 1
        if hits <= low:
            decision = "significant"
            remaining: list[str] = []
        elif hits >= high:
            decision = "not-significant"
            remaining = []
        elif final:
            decision = "inconclusive"
            remaining = [
                "final look landed inside the indifference region; "
                "extend the registered look schedule before claiming either side"
            ]
        else:
            decision = "continue"
            remaining = []
            if low >= 0:
                remaining.append(
                    f"hits <= {low} at a planned look for significant"
                )
            if high <= evaluated:
                remaining.append(
                    f"hits >= {high} at a planned look for not-significant"
                )
            remaining.append(
                f"continue to next planned look at n={self.looks[index + 1]}"
            )
        return {
            "at_planned_look": True,
            "look": index + 1,
            "planned_looks": len(self.looks),
            "evaluated": evaluated,
            "hits": hits,
            "p_estimate": estimate,
            "decision": decision,
            "lower_boundary": low,
            "upper_boundary": high,
            "p_high": self.p_high,
            "p_low": self.p_low,
            "final_look": final,
            "remaining_conditions": remaining,
        }


def finite_population_risks(
    design: SequentialDesign,
    total_assignments: int,
    hit_count: int,
) -> dict[str, float]:
    """Exact sequential risk when draws are a uniform ordering of a finite pool.

    The pool contains ``hit_count`` as-extreme assignments out of
    ``total_assignments``.  Enumerating the dynamic program over
    (hits remaining, misses remaining) states gives the exact probability of
    reaching each boundary; used to validate the design on exhaustible data.
    """
    state: dict[tuple[int, int], float] = {
        (hit_count, total_assignments - hit_count): 1.0
    }
    significant = 0.0
    not_significant = 0.0
    seen = 0
    for index, n in enumerate(design.looks):
        gap = n - seen
        new_state: dict[tuple[int, int], float] = {}
        low = design.lower[index]
        high = design.upper[index]
        for (hits_left, miss_left), weight in state.items():
            remaining = hits_left + miss_left
            lo_d = max(0, gap - miss_left)
            hi_d = min(gap, hits_left)
            denominator = math.comb(remaining, gap)
            for drawn_hits in range(lo_d, hi_d + 1):
                probability = (
                    math.comb(hits_left, drawn_hits)
                    * math.comb(miss_left, gap - drawn_hits)
                    / denominator
                )
                cumulative_hits = hit_count - hits_left + drawn_hits
                if cumulative_hits <= low:
                    significant += weight * probability
                elif cumulative_hits >= high:
                    not_significant += weight * probability
                else:
                    key = (hits_left - drawn_hits, miss_left - (gap - drawn_hits))
                    new_state[key] = new_state.get(key, 0.0) + weight * probability
        state = new_state
        seen = n
    alive = math.fsum(state.values())
    return {
        "significant": significant,
        "not_significant": not_significant,
        "inconclusive": alive,
    }
