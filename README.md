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
- 支持**预注册的序贯停止决策**：运行前必须用 `--looks n1 n2 ... nK` 选定全部计划查看点，并预先选定允许的错误风险。停止判定只在这些查看点发生；两次查看之间只展示当前证据（命中数 `hits`、保守 p 估计）和尚未满足的停止条件，监测结果标记 `monitoring_only`，不会被计作一次查看。
- 序贯设计把每个随机排列视为一次伯努利试验：在查看点 `n_k`，命中数 `S_k <= lower_k` 判 `significant`，`S_k >= upper_k` 判 `not-significant`。边界由精确二项分布配合 α/β 消耗函数（`--spending pocock` 或 `obrien-flanagan`）构造，对多次查看取并集仍有风险上界；平台无放回抽取排列，有限总体下真实风险不超过二项设计风险（超几何随机占优二项）。
- 风险按无差别区间解释：`--p-high`（默认 0.05，即拒绝阈值）与 `--p-low`（默认 p-high/2）之间是预先声明的无差别区。`sup_{p>=p-high} P(误判显著) <= --epsilon-high`，`sup_{p<=p-low} P(误判不显著) <= --epsilon-low`；末次查看若落在无差别区内，结论为 `inconclusive`，不强行二选一。
- `--stopping ci-half-width` 仅表示“估计精度达标即停”；其固定样本置信区间不能解释为任意时刻都有效的决策区间。
- 任意检查点边界保存后恢复，不会重复统计已用排列；排列池内容只取决于批次种子，同一配置下无论各检验的运行先后或分块方式，结果一致。
- 设计随首次运行落盘（`run_states.design_json`）：暂停/重开不会清零 `looks`、已查看次数或已消耗排列，恢复时以已注册设计为准，不能借重开偷换查看计划。

## 多重比较

对同一批次内的全部已完成检验给出未经调整与调整后的结论，并保留不确定范围：

```bash
stat-platform compare 1 --rule holm --alpha 0.05
```

`--rule` 可选 `none` / `bonferroni` / `holm`，输出区分两个层面：

- `individual_decision`：该单项假设在其**各自预注册错误风险**（`individual_error_risk`）下的结论；
- `adjusted_decision` / `adjusted_p_value` / `adjusted_reject`：该结论作为**整组家族的一员**在族系水平 `--alpha` 下是否仍成立；
- `family_wise_conclusion` 给出 `at_least_one_rejected` / `all_rejected` 及文字说明 `meaning`；运行中的检验只报告监测信息（`p_estimate_is_monitoring_only`），不参与整组断言。

未调整 Monte Carlo 区间与按 Bonferroni 置信度 `1-(1-γ)/m` 加宽的 `adjusted_confidence_interval` 只在固定样本完成时是覆盖区间；序贯过程的决策有效性来自停止边界，而不是这些区间。

## 预注册序贯停止

```bash
stat-platform run data.csv --max-exact-assignments 1000 --seed 42 \
  --stopping sequential-decision \
  --looks 200 500 1000 2000 5000 \
  --p-high 0.05 --p-low 0.02 \
  --epsilon-high 0.05 --epsilon-low 0.05 \
  --spending pocock --checkpoint-every 500
stat-platform resume 1
stat-platform status 1
```

判定只在 `--looks` 给定的累计样本数处做出；`status` 可任意次查看而不消耗查看次数，返回当前 `hits`、`p_estimate`、本查看点边界以及 `unmet_conditions`。`ci-half-width` 规则在区间半宽不超过 `--ci-half-width` 时停止，仅供估计用途。

## 风险核对

`finite_population_risks(design, total_assignments, hit_count)` 对有限排列空间做完整状态枚举，精确给出每种真实命中数下判为 `significant` / `not-significant` / `inconclusive` 的概率，用于在可穷举小数据上核对错误风险约束。测试中的 8 排列配对数据还验证：不同分块大小、不同查看安排消费同一排列前缀；恢复不清零查看次数；整组报告区分单项与族系结论。

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
