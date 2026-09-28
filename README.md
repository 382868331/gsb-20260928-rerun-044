# 统计试验运行平台

这是一个无第三方依赖的 Python 批次平台。当前内置“组内关联样本的两条件均值差置换检验”，并把数据版本、检验设定、随机种子、已计算排列、随机数状态和结果统一记录到原子写入的 JSON 批次库。

## 统计规则

- 每个置换只在同一 `group` 内交换 `condition` 标签，关联观测不会跨组拆散。
- 每组保留原有条件标签数量；全局两条件均值差是检验统计量。
- 排列空间不大于 `--max-exact-assignments`（默认 100000）时枚举全部合法排列，输出精确 p 值。
- 排列空间较大时使用固定种子、无放回的 Monte Carlo 抽样；结果标记为 `monte-carlo`，报告 p 值、标准误和 95% 置信区间，不能解释为精确概率。
- Monte Carlo p 值使用 `(as_extreme + 1) / (n + 1)` 保守估计；随机数状态和已见排列落盘，恢复后继续固定随机流且不重复已有样本。
- 每个检验都保存自己的 `null_hypothesis`，同一批次可扩展多项检验并分别对应原假设。

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
