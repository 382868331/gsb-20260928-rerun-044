# 统计试验运行平台

这是一个无第三方依赖的 Python 批次平台。当前内置“组内关联样本的两条件均值差置换检验”，并把数据版本、检验设定、随机种子、已计算排列、随机数状态和结果统一记录到原子写入的 JSON 批次库。

## 统计规则

- 每个置换只在同一 `group` 内交换 `condition` 标签，关联观测不会跨组拆散。
- 每组保留原有条件标签数量；全局两条件均值差是检验统计量。
- 排列空间不大于 `--max-exact-assignments`（默认 100000）时枚举全部合法排列，输出精确 p 值。
- 排列空间较大时使用固定种子、无放回的 Monte Carlo 抽样；结果标记为 `monte-carlo`，报告 p 值、标准误和 95% 置信区间，不能解释为精确概率。
- Monte Carlo p 值使用 `(as_extreme + 1) / (n + 1)` 保守估计；随机数状态和已见排列落盘，恢复后继续固定随机流且不重复已有样本。
- 每个检验都保存自己的 `null_hypothesis`，同一批次可扩展多项检验并分别对应原假设。

## 多假设比较与共享排列

- 同一批次内的所有检验共享一张 `batch_permutations` 合法排列表：排列只在批次层面生成并保存一次，每项检验在共享排列上分别计算自己的统计量与极端计数（`permutation_samples` 按检验记录统计量和是否极端）。
- 批次随机流由批次种子驱动，与检验的创建或运行先后顺序无关；同一配置无论先运行哪项检验，结果完全一致。
- 每项检验独立记录自己的原假设、统计量设定、目标样本数、查看次数（`looks`）与停止原因（`stopping_reason`：`exact_enumeration` / `target_samples` / `precision_target`）。
- 自适应抽样：在 `settings["adaptive"]` 中给出 `min_samples` 与 `ci_half_width`，每次推进后若置信区间半宽达标则提前停止。重复查看结果不会被当作普通固定次数试验：`looks > 1` 时报告按查看次数 Bonferroni 加宽的置信区间（`confidence_level = 1 - 0.05/looks`），并在 `unadjusted_confidence_interval` 中保留名义 95% 区间。
- 批次比较报告同时给出未经调整与按所选规则（`bonferroni` / `holm` / `benjamini-hochberg`）调整后的 p 值与不确定范围：

```bash
stat-platform compare 1 --method holm --alpha 0.05
```

向已有批次追加新的假设（资料必须与批次数据一致）：

```bash
stat-platform add-test 1 data.csv --null "control not greater" --conditions control treatment --alternative less
```

## 命令行

```bash
stat-platform run examples/paired.csv \
  --name paired --version v1 \
  --conditions control treatment \
  --alternative two-sided \
  --database experiments.json
```

大规模运行时，可一次只计算并持久化一个检查点；中断或暂停后对同一批次重复恢复：

```bash
stat-platform run data.csv --seed 20260929 --samples 100000 --checkpoint-every 1000
stat-platform resume 1 --checkpoint-every 1000
stat-platform resume 1
```

查询已完成检验：

```bash
stat-platform result 1 --database experiments.json
```

## Python API

```python
from stat_platform import Dataset, GroupedPermutationTest, PlatformRunner, TestSpec

dataset = Dataset.from_records(records, name="paired", version="v1")
spec = TestSpec(
    test_type=GroupedPermutationTest.test_type,
    null_hypothesis="conditional exchangeability within groups; equal means",
    settings={
        "alternative": "two-sided",
        "conditions": ("control", "treatment"),
    },
)

with PlatformRunner("experiments.json") as platform:
    batch_id = platform.create_batch(dataset, specs=[spec], seed=20260929)
    platform.advance_test(1, max_new_samples=1000)
    results = platform.run_batch(batch_id)
```

## 校验

```bash
.venv/bin/python -m pytest -q
```

测试使用两组配对小数据：共有 4 个合法组内排列，双侧精确 p 值为 0.5，单侧精确 p 值为 0.25；同时验证检查点恢复不会重复保存已有排列。
