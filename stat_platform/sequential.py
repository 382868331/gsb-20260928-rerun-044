from __future__ import annotations
"""Pre-planned, risk-controlled sequential stopping for Monte Carlo p-values.

The Monte Carlo permutation p-value estimates an exact tail probability ``q``
from independent uniform draws: the as-extreme count is binomial.  A decision
may be made at *looks*, and the risk of the two possible wrong decisions must
be controlled *over the whole look schedule*, not by reusing a fixed-sample
interval at arbitrary times.

This module builds Pocock-style boundaries from equal alpha spending and then
evaluates the *exact* crossing probabilities of the resulting boundaries with
a dynamic program, so the user-chosen error risks are guaranteed and the
actually achieved risks are reported.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


def binomial_log_pmf(k: int, n: int, q: float) -> float:
    if q <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if q >= 1.0:
        return 0.0 if k == n else -math.inf
    if k < 0 or k > n:
        return -math.inf
    log_combinations = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )
    return (
        log_combinations
        + k * math.log(q)
        + (n - k) * math.log1p(-q)
    )


def binomial_cdf(k: int, n: int, q: float) -> float:
    """P(X <= k) for X ~ Binomial(n, q), computed exactly in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    terms = [binomial_log_pmf(j, n, q) for j in range(k + 1)]
    return math.exp(max(terms)) * sum(math.exp(term - max(terms)) for term in terms)


def binomial_sf(k: int, n: int, q: float) -> float:
    """P(X >= k) for X ~ Binomial(n, q)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    terms = [binomial_log_pmf(j, n, q) for j in range(k, n + 1)]
    return math.exp(max(terms)) * sum(math.exp(term - max(terms)) for term in terms)


@dataclass(frozen=True)
class SequentialDesign:
    """Boundaries and exact risks for a pre-planned sequence of looks.

    ``looks`` gives the cumulative accepted-permutation count at which a look
    is permitted.  At look ``k`` (1-based) the process stops with
    ``significant`` when the as-extreme count is ``<= lower[k-1]`` and with
    ``not-significant`` when it is ``>= upper[k-1]``.  Otherwise it continues;
    the final look always forces one of the two decisions.

    Error risks are evaluated at the single decision boundary ``q0``:
    ``risk_significant_at_q0`` is P(declare significant | q = q0) and
    ``risk_not_significant_at_q0`` is P(declare not-significant | q = q0).
    Correct-decision power is reported on each side: significant power at a
    true probability ``q_sig < q0`` and not-significant power at
    ``q_nonsig > q0``.  Between these levels the design is deliberately
    inconclusive unless the budget is large enough to separate them.
    """

    looks: tuple[int, ...]
    q0: float
    q_sig: float
    q_nonsig: float
    alpha_significant: float
    alpha_not_significant: float
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    risk_significant_at_q0: float
    risk_not_significant_at_q0: float
    power_significant_at_qsig: float
    power_not_significant_at_qnonsig: float
    feasible: bool = field(default=True)

    @property
    def max_samples(self) -> int:
        return self.looks[-1]

    def look_index(self, evaluated: int) -> int | None:
        """1-based look number if ``evaluated`` is a planned look, else None."""
        for index, count in enumerate(self.looks):
            if count == evaluated:
                return index + 1
        return None

    def evaluate(self, evaluated: int, hits: int) -> str | None:
        """Return the decision only when a planned look is reached."""
        look = self.look_index(evaluated)
        if look is None:
            return None
        index = look - 1
        if hits <= self.lower[index]:
            return "significant"
        if hits >= self.upper[index]:
            return "not-significant"
        # A planned look inside the continuation region yields no decision;
        # at the final look the budget was exhausted inconclusively.
        return None

    def progress(self, evaluated: int, hits: int) -> dict[str, Any]:
        """Current evidence and the still-unmet stopping conditions."""
        next_look = next(
            (count for count in self.looks if count >= evaluated),
            self.looks[-1],
        )
        index = self.looks.index(next_look) if next_look in self.looks else len(self.looks) - 1
        needed = max(0, next_look - evaluated)
        decision = self.evaluate(evaluated, hits)
        return {
            "evaluated": evaluated,
            "as_extreme_count": hits,
            "current_p_estimate": (hits + 1) / (evaluated + 1) if evaluated else None,
            "next_look_at": next_look,
            "samples_until_next_look": needed,
            "significant_if_hits_at_most": self.lower[index],
            "not_significant_if_hits_at_least": self.upper[index],
            "decision": decision,
            "in_continuation_region": decision is None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "looks": list(self.looks),
            "q0": self.q0,
            "q_sig": self.q_sig,
            "q_nonsig": self.q_nonsig,
            "alpha_significant": self.alpha_significant,
            "alpha_not_significant": self.alpha_not_significant,
            "lower_boundaries": list(self.lower),
            "upper_boundaries": list(self.upper),
            "risk_significant_at_q0": self.risk_significant_at_q0,
            "risk_not_significant_at_q0": self.risk_not_significant_at_q0,
            "power_significant_at_qsig": self.power_significant_at_qsig,
            "power_not_significant_at_qnonsig": self.power_not_significant_at_qnonsig,
            "feasible": self.feasible,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SequentialDesign":
        return cls(
            looks=tuple(value["looks"]),
            q0=float(value["q0"]),
            q_sig=float(value.get("q_sig", value.get("q1", 0.01))),
            q_nonsig=float(value.get("q_nonsig", 0.2)),
            alpha_significant=float(value["alpha_significant"]),
            alpha_not_significant=float(value["alpha_not_significant"]),
            lower=tuple(value["lower_boundaries"]),
            upper=tuple(value["upper_boundaries"]),
            risk_significant_at_q0=float(value["risk_significant_at_q0"]),
            risk_not_significant_at_q0=float(value["risk_not_significant_at_q0"]),
            power_significant_at_qsig=float(value.get("power_significant_at_qsig", 0.0)),
            power_not_significant_at_qnonsig=float(
                value.get("power_not_significant_at_qnonsig", value.get("power_not_significant_at_q1", 0.0))
            ),
            feasible=bool(value.get("feasible", True)),
        )


def crossing_probabilities(
    looks: Sequence[int],
    lower: Sequence[int],
    upper: Sequence[int],
    q: float,
) -> tuple[float, float]:
    """Exact P(cross lower) and P(cross upper) over the look schedule."""
    # Dense distribution over hit counts; mass outside the open continuation
    # band (a_k, b_k) is absorbed at each look and never propagated.
    distribution = [0.0] * (looks[-1] + 1)
    distribution[0] = 1.0
    previous_n = 0
    lower_total = 0.0
    upper_total = 0.0
    for n_k, a_k, b_k in zip(looks, lower, upper):
        step = n_k - previous_n
        increment_pmf = [math.exp(binomial_log_pmf(add, step, q)) for add in range(step + 1)]
        advanced = [0.0] * (n_k + 1)
        for hits, probability in enumerate(distribution):
            if probability == 0.0:
                continue
            for add, increment in enumerate(increment_pmf):
                if increment:
                    advanced[hits + add] += probability * increment
        next_distribution = [0.0] * (looks[-1] + 1)
        for hits in range(n_k + 1):
            probability = advanced[hits]
            if probability == 0.0:
                continue
            if hits <= a_k:
                lower_total += probability
            elif hits >= b_k:
                upper_total += probability
            else:
                next_distribution[hits] = probability
        distribution = next_distribution
        previous_n = n_k
    return lower_total, upper_total


def build_design(
    looks: Sequence[int],
    *,
    q0: float = 0.05,
    q_sig: float = 0.01,
    q_nonsig: float = 0.2,
    alpha_significant: float = 0.05,
    alpha_not_significant: float = 0.05,
) -> SequentialDesign:
    """Build boundaries from equal (Pocock) alpha spending.

    At each look the nominal spend for each boundary is ``alpha * k / K`` of
    the one-shot binomial tail, which is conservative and unambiguous.
    Exact crossing probabilities then confirm the overall guarantee.
    """
    looks = tuple(int(n) for n in looks)
    if not looks or any(looks[index] <= looks[index - 1] for index in range(1, len(looks))):
        raise ValueError("looks must be a strictly increasing positive sequence")
    if not 0.0 < q_sig < q0 < q_nonsig < 1.0:
        raise ValueError("require 0 < q_sig < q0 < q_nonsig < 1")
    if not 0.0 < alpha_significant < 1.0 or not 0.0 < alpha_not_significant < 1.0:
        raise ValueError("error risks must lie strictly between 0 and 1")

    total_looks = len(looks)

    # Precompute the one-shot binomial tails at q0 once per look; boundary
    # selection under every spending scale then reuses these arrays.
    cdf_by_look: list[list[float]] = []
    sf_by_look: list[list[float]] = []
    for n_k in looks:
        pmf = [math.exp(binomial_log_pmf(j, n_k, q0)) for j in range(n_k + 1)]
        cdf: list[float] = []
        cumulative = 0.0
        for value in pmf:
            cumulative += value
            cdf.append(cumulative)
        sf = [0.0] * (n_k + 1)
        cumulative = 0.0
        for j in range(n_k, -1, -1):
            cumulative += pmf[j]
            sf[j] = cumulative
        cdf_by_look.append(cdf)
        sf_by_look.append(sf)

    def boundaries_for_scale(scale: float) -> tuple[list[int], list[int]]:
        lower: list[int] = []
        upper: list[int] = []
        for index, n_k in enumerate(looks, start=1):
            spend_low = scale * alpha_significant * index / total_looks
            spend_high = scale * alpha_not_significant * index / total_looks
            cdf = cdf_by_look[index - 1]
            sf = sf_by_look[index - 1]
            a_k = -1
            for candidate, value in enumerate(cdf):
                if value <= spend_low:
                    a_k = candidate
                else:
                    break
            b_k = n_k + 1
            for candidate in range(n_k, -1, -1):
                if sf[candidate] <= spend_high:
                    b_k = candidate
                else:
                    break
            lower.append(a_k)
            upper.append(b_k)
        return lower, upper

    def overspent(scale: float) -> tuple[bool, float, float, list[int], list[int]]:
        lower, upper = boundaries_for_scale(scale)
        risk_low, risk_high = crossing_probabilities(looks, lower, upper, q0)
        tolerance = 1e-9
        return (
            risk_low > alpha_significant + tolerance
            or risk_high > alpha_not_significant + tolerance,
            risk_low,
            risk_high,
            lower,
            upper,
        )

    # Repeated looks inflate the crossing probability; shrink the per-look
    # nominal spending until the *exact* risk meets the pre-chosen limits.
    scale = 1.0
    over, risk_low, risk_high, lower, upper = overspent(scale)
    if over:
        low_scale, high_scale = 0.0, 1.0
        previous_signature = None
        for _ in range(40):
            mid = (low_scale + high_scale) / 2
            lower_mid, upper_mid = boundaries_for_scale(mid)
            signature = (tuple(lower_mid), tuple(upper_mid))
            risk_low_mid, risk_high_mid = crossing_probabilities(
                looks, lower_mid, upper_mid, q0
            )
            tolerance = 1e-9
            over_mid = (
                risk_low_mid > alpha_significant + tolerance
                or risk_high_mid > alpha_not_significant + tolerance
            )
            if over_mid:
                high_scale = mid
            else:
                low_scale = mid
            if signature == previous_signature:
                break
            previous_signature = signature
        scale = low_scale
        over, risk_low, risk_high, lower, upper = overspent(scale)

    power_sig, _ = crossing_probabilities(looks, lower, upper, q_sig)
    _, power_nonsig = crossing_probabilities(looks, lower, upper, q_nonsig)
    design = SequentialDesign(
            looks=looks,
            q0=q0,
            q_sig=q_sig,
            q_nonsig=q_nonsig,
        alpha_significant=alpha_significant,
        alpha_not_significant=alpha_not_significant,
        lower=tuple(lower),
        upper=tuple(upper),
            risk_significant_at_q0=risk_low,
            risk_not_significant_at_q0=risk_high,
            power_significant_at_qsig=power_sig,
            power_not_significant_at_qnonsig=power_nonsig,
        feasible=not over,
    )
    return design


def default_looks(
    max_samples: int,
    number_of_looks: int,
    *,
    min_samples: int = 1,
) -> tuple[int, ...]:
    """Evenly spaced cumulative look counts ending exactly at the budget."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    number_of_looks = max(1, min(number_of_looks, max_samples))
    start = min(min_samples, max_samples)
    values = {
        int(round(start + (max_samples - start) * k / number_of_looks))
        for k in range(1, number_of_looks + 1)
    }
    values.add(max_samples)
    return tuple(sorted(value for value in values if value >= 1))
