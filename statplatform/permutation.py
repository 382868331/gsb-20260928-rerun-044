"""带分组结构的置换检验核心。

关键约束：组内有关联的观测不能跨组拆散交换。每个分组（受试者/区组/层）
内部独立重排处理标签，全局配置是各组配置的笛卡尔积。

两种分配方式：
- ``fixed``（默认，区组随机化）：保持每组内各标签的观测数不变，
  配置数为各组 C(n_g, k_g) 之积；
- ``free``：允许组内标签多重集的任意分配（极端时整组归为同一处理），
  配置数为各组 2**n_g 之积。

计算量小时 ``exact`` 模式枚举全部配置；超过阈值时 ``monte_carlo`` 模式
按固定种子抽样。每个抽样序号由基础种子确定性派生，因此暂停恢复后
从同一序号继续，结果与一次连续运行完全一致，且绝不重复计算。
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from math import comb
from typing import Callable, Optional

GREATER = "greater"
LESS = "less"
TWO_SIDED = "two_sided"

EXACT = "exact"
MONTE_CARLO = "monte_carlo"


def mean_difference(label_a, label_b) -> Callable:
    """默认统计量：label_a 均值减去 label_b 均值（跨组合并）。"""

    def statistic(rows, assigned, spec, bundle):
        vals_a, vals_b = [], []
        vcol = bundle.value_col
        for row, lab in zip(rows, assigned):
            if lab == label_a:
                vals_a.append(row[vcol])
            elif lab == label_b:
                vals_b.append(row[vcol])
        if not vals_a or not vals_b:
            return float("nan")
        return sum(vals_a) / len(vals_a) - sum(vals_b) / len(vals_b)

    statistic.__name__ = f"mean_difference({label_a!r}-{label_b!r})"
    return statistic


@dataclass(frozen=True)
class PermutationSpec:
    """置换检验设定。

    labels 必须且只能有两个处理标签；组信息来自数据本身的分组列。
    """

    labels: tuple
    alternative: str = TWO_SIDED
    allocation: str = "fixed"  # fixed | free
    run_mode: Optional[str] = None  # None 时按枚举阈值自动决定
    max_exact_configs: int = 100_000
    n_draws: int = 10_000       # monte_carlo 模式抽样次数（含可能的观测配置）
    chunk_size: int = 1_000     # 每个事务提交的评估数，支持中途暂停
    statistic: Optional[Callable] = field(default=None, compare=False)

    def __post_init__(self):
        if len(self.labels) != 2:
            raise ValueError("当前置换检验支持恰好两个处理标签")
        if self.alternative not in (GREATER, LESS, TWO_SIDED):
            raise ValueError(f"未知 alternative: {self.alternative!r}")
        if self.allocation not in ("fixed", "free"):
            raise ValueError("allocation 必须是 'fixed' 或 'free'")
        if self.run_mode not in (None, EXACT, MONTE_CARLO):
            raise ValueError("run_mode 必须是 None、'exact' 或 'monte_carlo'")
        if self.n_draws <= 0:
            raise ValueError("n_draws 必须为正整数")

    def stat_fn(self):
        return self.statistic or mean_difference(self.labels[0], self.labels[1])

    def decide_mode(self, total_configs):
        if self.run_mode is not None:
            return self.run_mode
        return EXACT if total_configs <= self.max_exact_configs else MONTE_CARLO


def group_label_counts(bundle, spec):
    """每组的观测数与 label_a 的实际数量（fixed 模式需保持的处理数）。"""
    a = spec.labels[0]
    info = []
    for gid, idx in bundle.groups():
        labels = [bundle.rows[i][bundle.label_col] for i in idx]
        unknown = set(labels) - set(spec.labels)
        if unknown:
            raise ValueError(f"组 {gid!r} 中出现设定之外的标签: {sorted(unknown)}")
        info.append((gid, idx, len(idx), sum(1 for x in labels if x == a)))
    return info


def count_configs(group_info, allocation):
    total = 1
    for _, _, n, k in group_info:
        if allocation == "fixed":
            total *= comb(n, k)
        else:
            total *= 2 ** n
    return total


def _iter_group_assignments(n, k, allocation):
    """生成单组内 label_a 所在位置的集合序列。"""
    if allocation == "fixed":
        for chosen in itertools.combinations(range(n), k):
            yield chosen
    else:
        for chosen_k in range(n + 1):
            for chosen in itertools.combinations(range(n), chosen_k):
                yield chosen


def iter_exact_combos(group_info, allocation):
    """按确定性顺序枚举全部全局配置（每组一个 label_a 位置集合的笛卡尔积）。"""
    per_group = [
        list(_iter_group_assignments(n, k, allocation)) for _, _, n, k in group_info
    ]
    yield from itertools.product(*per_group)


def build_assignment(group_info, a, b, combo):
    """根据各组的 label_a 位置集合构造每行标签。"""
    assigned = {}
    for (_, idx, _, _), chosen in zip(group_info, combo):
        chosen_set = set(chosen)
        for local, row_idx in enumerate(idx):
            assigned[row_idx] = a if local in chosen_set else b
    return [assigned[i] for i in range(len(assigned))]


def sample_assignment(group_info, allocation, rng):
    """在各组内部独立抽样一个全局标签分配。"""
    a_positions = []
    for _, _, n, k in group_info:
        if allocation == "fixed":
            a_positions.append(set(rng.sample(range(n), k)))
        else:
            a_positions.append({i for i in range(n) if rng.random() < 0.5})
    return a_positions


def sampled_build_assignment(group_info, labels, a_positions):
    a, b = labels
    assigned = {}
    for (_, idx, _, _), chosen in zip(group_info, a_positions):
        for local, row_idx in enumerate(idx):
            assigned[row_idx] = a if local in chosen else b
    return [assigned[i] for i in range(len(assigned))]


def derive_seed(base_seed, draw_index):
    """由基础种子与抽样序号确定性派生独立种子。

    SHA-256 混合保证不同序号互不相关；同一 (seed, index) 永远产生同一抽样，
    这是“暂停恢复不重复计算且可复现”的基础。
    """
    digest = hashlib.sha256(f"perm::{base_seed}::{draw_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def is_extreme(candidate, observed, alternative):
    if alternative == GREATER:
        return candidate >= observed
    if alternative == LESS:
        return candidate <= observed
    return abs(candidate) >= abs(observed)
