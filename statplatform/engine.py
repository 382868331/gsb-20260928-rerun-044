"""置换检验执行引擎（纯计算，与持久化解耦）。

运行以“块（chunk）”为单位推进：每次只评估 ``budget`` 个配置/抽样，
由上层在块边界落库。进程可以在任意块之间暂停，恢复时：
- 精确模式按确定性枚举顺序跳过已评估配置；
- 蒙特卡洛模式从下一个抽样序号继续，序号由种子确定性派生。
因此已有样本不会被重复计算，最终结果与一次性运行完全一致。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

from . import permutation as perm


@dataclass(frozen=True)
class PreparedPermutation:
    bundle: object
    spec: perm.PermutationSpec
    group_info: tuple
    stat_fn: Callable
    total_configs: int
    mode: str
    observed: float


def prepare_permutation(bundle, spec: perm.PermutationSpec):
    observed_labels = tuple(row[bundle.label_col] for row in bundle.rows)
    if set(observed_labels) != set(spec.labels):
        raise ValueError(
            f"数据标签 {sorted(set(observed_labels))} 与设定标签 {list(spec.labels)} 不一致"
        )
    group_info = tuple(perm.group_label_counts(bundle, spec))
    total_configs = perm.count_configs(group_info, spec.allocation)
    mode = spec.decide_mode(total_configs)
    stat_fn = spec.stat_fn()
    observed = stat_fn(bundle.rows, list(observed_labels), spec, bundle)
    if math.isnan(observed):
        raise ValueError("观测数据上统计量无定义（某处理组没有观测）")
    return PreparedPermutation(
        bundle=bundle,
        spec=spec,
        group_info=group_info,
        stat_fn=stat_fn,
        total_configs=total_configs,
        mode=mode,
        observed=observed,
    )


def _is_counted_extreme(stat, observed, alternative):
    return not math.isnan(stat) and perm.is_extreme(stat, observed, alternative)


def run_exact_chunk(prepared: PreparedPermutation, skip: int, budget: int, record=None):
    """推进精确枚举。返回 (本次评估数, 本次新增极端数, 是否完成)。"""
    spec = prepared.spec
    a, b = spec.labels
    combos = perm.iter_exact_combos(prepared.group_info, spec.allocation)
    for _ in range(skip):
        try:
            next(combos)
        except StopIteration:
            break
    evaluated = 0
    extreme = 0
    for combo in combos:
        if evaluated >= budget:
            break
        assigned = perm.build_assignment(prepared.group_info, a, b, combo)
        stat = prepared.stat_fn(prepared.bundle.rows, assigned, spec, prepared.bundle)
        if _is_counted_extreme(stat, prepared.observed, spec.alternative):
            extreme += 1
        if record is not None:
            record(skip + evaluated, stat)
        evaluated += 1
    done = skip + evaluated
    return evaluated, extreme, done >= prepared.total_configs


def run_mc_chunk(
    prepared: PreparedPermutation,
    start_index: int,
    budget: int,
    base_seed: int,
    target_draws: int,
    record=None,
):
    """推进蒙特卡洛抽样。返回 (本次抽样数, 本次新增极端数, 是否完成)。"""
    spec = prepared.spec
    end_index = min(target_draws, start_index + budget)
    extreme = 0
    for index in range(start_index, end_index):
        rng = random.Random(perm.derive_seed(base_seed, index))
        positions = perm.sample_assignment(
            prepared.group_info, spec.allocation, rng
        )
        assigned = perm.sampled_build_assignment(
            prepared.group_info, spec.labels, positions
        )
        stat = prepared.stat_fn(prepared.bundle.rows, assigned, spec, prepared.bundle)
        if record is not None:
            record(index, stat)
        if _is_counted_extreme(stat, prepared.observed, spec.alternative):
            extreme += 1
    count = end_index - start_index
    return count, extreme, end_index >= target_draws
