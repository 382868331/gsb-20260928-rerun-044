# 统计试验运行平台

这是一个无第三方依赖的 Python 批次平台。当前内置“组内关联样本的两条件均值差置换检验”，并把数据版本、检验设定、随机种子、已计算排列、随机数状态和结果统一记录到原子写入的 JSON 批次库。

## 统计规则

- 每个置换只在同一 `group` 内交换 `condition` 标签，关联观测不会跨组拆散。
- 每组保留原有条件标签数量；全局两条件均值差是检验统计量。
- 排列空间不大于 `--max-exact-assignments`（默认 100000）时枚举全部合法排列，输出精确 p 值。
- 排列空间较大时使用固定种子、无放回的 Monte Carlo 抽样；结果标记为 `monte-carlo`，报告 p 值、标准误和 95% 置信区间，不能解释为精确概率。
- Monte Carlo p 值使用 `(as_extreme + 1) / (n + 1)` 保守估计；随机数状态和已见排列落盘，恢复后继续固定随机流且不重复已有样本。
- 每个检验都保存自己的 `null_hypothesis`，同一批次可扩展多项检验并分别对应原假设。
- 同一批次内的多项检验共享同一个批次级合法排列池（`batch_assignments`），排列只按批次种子生成一次；每项检验分别记录自己的原假设、统计量、命中计数与停止条件，互不覆盖。
- 支持可中断的自适应抽样：`--stopping decision`（p 值置信区间整体落在 α 一侧即停）或 `--stopping ci-half-width`（区间半宽达标即停），配合 `--min-samples` / `--max-samples`。自适应结果会记录 `adaptive`、`looks`（重复看结果次数）、`stopping_rule` 与 `stopped_reason`，不会伪装成固定次数试验。
- 任意检查点边界保存后恢复，不会重复统计已用排列；排列池内容只取决于批次种子，同一配置下无论各检验的运行先后或分块方式，结果一致。

## 多重比较

对同一批次内的全部已完成检验给出未经调整与调整后的结论，并保留不确定范围：

```bash
stat-platform compare 1 --rule holm --alpha 0.05
```

`--rule` 可选 `none` / `bonferroni` / `holm`。输出中每项检验同时包含 `unadjusted_p_value` / `unadjusted_reject` 与 `adjusted_p_value` / `adjusted_reject`，以及未调整置信区间和按 Bonferroni 置信度 `1-(1-γ)/m` 加宽后的 `adjusted_confidence_interval`。

## 自适应抽样

```bash
stat-platform run data.csv --max-exact-assignments 1000 --seed 42 \
  --stopping decision --decision-alpha 0.05 --min-samples 200 --max-samples 5000 \
  --checkpoint-every 500
stat-platform resume 1
```

`decision` 规则在 p 值 95% 置信区间整体低于 α（`decided-significant`）或整体高于 α（`decided-not-significant`）时提前停止；`ci-half-width` 规则在区间半宽不超过 `--ci-half-width` 时停止；两者都受 `--max-samples` 上限保护。

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
