"""由检验计数汇总 p 值并评估蒙特卡洛抽样误差。

精确模式：p = 极端配置数 / 配置总数。

蒙特卡洛模式采用 Phipson & Smyth (2010) 的 +1 估计：
    p_hat = (ge + 1) / (n + 1)
（ge 为抽样中与观测统计量同等或更极端的次数，n 为抽样次数）。
这保证 p 值永远大于 0，且当 n 覆盖整个配置空间时与精确概率一致。
抽样次数很少时它仍是“估计”，结果中以 ``p_kind='estimated'`` 明确标注，
并给出标准误与 95% 置信区间，绝不冒充精确概率。
"""

from __future__ import annotations

import math


def exact_p_value(extreme_count, total_configs):
    return extreme_count / total_configs


def mc_p_value(extreme_count, n_draws):
    """+1 修正的蒙特卡洛 p 值估计。"""
    return (extreme_count + 1) / (n_draws + 1)


def mc_standard_error(p_hat, n_draws):
    """抽样标准误，以 p_hat 作为二项比例的插入估计：sqrt(p(1-p)/(n+1))。"""
    return math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / (n_draws + 1))


def wilson_interval(extreme_count, n_draws, z=1.959963984540054):
    """基于 +1 后验次数的 Wilson 95% 置信区间，用于量化抽样误差。

    以 x = ge + 1、N = n + 1 计算，与 :func:`mc_p_value` 保持一致，
    即使 ge=0 也给出非退化的区间。
    """
    x = extreme_count + 1
    n = n_draws + 1
    phat = x / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def summarize_mc(extreme_count, n_draws):
    p_hat = mc_p_value(extreme_count, n_draws)
    se = mc_standard_error(p_hat, n_draws)
    ci_low, ci_high = wilson_interval(extreme_count, n_draws)
    return {
        "p_value": p_hat,
        "p_kind": "estimated",
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_evaluated": n_draws,
    }


def summarize_exact(extreme_count, total_configs):
    return {
        "p_value": exact_p_value(extreme_count, total_configs),
        "p_kind": "exact",
        "se": None,
        "ci_low": None,
        "ci_high": None,
        "n_evaluated": total_configs,
    }
