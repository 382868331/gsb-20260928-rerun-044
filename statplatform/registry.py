"""检验类型注册表。

平台不把检验逻辑写死在批次流程里：每种检验提供一个执行器
（prepare / run_chunk / summarize），批次编排层统一记录数据版本、
设定、种子、逐条结果与结论。后续新增检验（如自助法、秩检验等）
只需在此注册一个新类型，自动复用同一套批次记录与暂停恢复机制。
"""

from __future__ import annotations

from . import engine, inference, permutation

PERMUTATION = "permutation"


class PermutationExecutor:
    """带分组结构置换检验的执行器。"""

    type_name = PERMUTATION

    def prepare(self, bundle, settings):
        spec = permutation.PermutationSpec(
            labels=tuple(settings["labels"]),
            alternative=settings.get("alternative", permutation.TWO_SIDED),
            allocation=settings.get("allocation", "fixed"),
            run_mode=settings.get("run_mode"),
            max_exact_configs=int(settings.get("max_exact_configs", 100_000)),
            n_draws=int(settings.get("n_draws", 10_000)),
            chunk_size=int(settings.get("chunk_size", 1_000)),
        )
        return spec, engine.prepare_permutation(bundle, spec)

    def initial_state(self, prepared, seed):
        spec = prepared.spec
        return {
            "run_mode": prepared.mode,
            "total_configs": prepared.total_configs,
            "target_draws": None if prepared.mode == permutation.EXACT else spec.n_draws,
            "observed_stat": prepared.observed,
        }

    def run_chunk(self, prepared, seed, done_count, ge_count, budget, sink):
        if prepared.mode == permutation.EXACT:
            evaluated, extreme, complete = engine.run_exact_chunk(
                prepared, done_count, budget, record=None
            )
            sink.update_progress(evaluated, extreme)
        else:
            if seed is None:
                raise ValueError("蒙特卡洛抽样必须提供整数种子")
            count, extreme, complete = engine.run_mc_chunk(
                prepared,
                start_index=done_count,
                budget=budget,
                base_seed=seed,
                target_draws=prepared.spec.n_draws,
                record=sink.record_draw,
            )
            sink.update_progress(count, extreme)
        return complete

    def summarize(self, prepared, done_count, ge_count):
        if prepared.mode == permutation.EXACT:
            return inference.summarize_exact(ge_count, prepared.total_configs)
        return inference.summarize_mc(ge_count, done_count)

    def chunk_size_for(self, prepared):
        return prepared.spec.chunk_size


EXECUTORS = {PERMUTATION: PermutationExecutor()}


def register(type_name, executor):
    if type_name in EXECUTORS:
        raise ValueError(f"检验类型已注册: {type_name!r}")
    EXECUTORS[type_name] = executor


def get_executor(type_name):
    try:
        return EXECUTORS[type_name]
    except KeyError:
        raise KeyError(f"未注册的检验类型: {type_name!r}")
