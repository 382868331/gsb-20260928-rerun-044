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
- 支持**预先固定查看计划与错误风险**的序贯停止：用 `--stopping decision --looks n1 n2 ...`（或 `--num-looks` 自动生成等间距计划）选定允许查看的累计排列数，用 `--risk-significant` / `--risk-not-significant` 选定在决策边界 `q0=α` 处允许的两类错误概率。平台用精确二项动态规划反推每次查看的命中数边界，并报告在 `q0` 处**实际达到的精确错误概率**以及在 `--q-sig`、`--q-nonsig` 处的正确决策把握度（power）。决策只在计划查看点发生；预算耗尽仍落在继续区内时如实记为 `inconclusive-max-samples`，绝不强行下结论。
- 固定样本的 p 值置信区间只是一次性有效，**不能**在任意时刻反复拿来当停止判据：`--stopping ci-half-width` 仅用于把估计精度做窄，同样只在计划查看点检查，结果不会给出显著性决策保证。
- 自适应结果记录 `adaptive`、`looks`、`stopping_rule`、`stopped_reason`、`decision`、完整 `design`（边界与精确风险）和每次查看的 `look_log`；用 `stat-platform peek <test_id>` 可在不消耗排列的情况下查看当前证据与“尚未满足的停止条件”。
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

Monte Carlo 接受排列中的“命中”次数服从二项分布，平台据此构造成组序贯边界：在第 k 次查看，命中数 `hits <= lower_k` 判 `significant`，`hits >= upper_k` 判 `not-significant`，否则继续。边界先按 Pocock 式等比例花费初算，再整体缩放并用精确 DP 核算，使跨全部查看的越界概率不超过用户允许风险（直接重复套用固定样本区间会把风险放大到名义值以上，例如 4 次查看时约 6.4% > 5%）。

```bash
stat-platform run data.csv --max-exact-assignments 1000 --seed 42 \
  --stopping decision --decision-alpha 0.05 \
  --risk-significant 0.05 --risk-not-significant 0.05 \
  --q-sig 0.01 --q-nonsig 0.15 \
  --looks 2000 5000 10000 20000 --max-samples 20000 \
  --checkpoint-every 700
stat-platform peek 1
stat-platform resume 1 --checkpoint-every 700
```

暂停/恢复只按预算切块持久化：重开不会清零 `looks` 或 `look_log`，共享排列池也不会重复消耗同一排列；分块大小只影响切块，不影响查看点、决策与停止记录。

## 单项与整组结论

同一批次的多项检验共享排列（彼此可能相关），`compare` 同时给出两层含义：

- **单项结论**：每项检验按自己预先固定的计划与风险得到 `individual_decision`（`significant` / `not-significant` / 计划结束仍不确定），只对应该项的原假设。
- **整组结论**：`adjusted_p_value` / `adjusted_reject` 用 Bonferroni 或 Holm 在共享排列下控制全组族系错误率；“至少一个原假设为假”必须基于调整后拒绝，且不代表其余假设也被拒绝。报告中的 `family` 字段汇总整组拒绝情况、尚未做出单项决策的检验，并明确两层含义。

## 小数据穷举核对

排列空间不超过上限时直接完整枚举并给出精确 p 值；小空间 Monte Carlo 在无放回覆盖全部合法排列后，命中数必须与完整枚举一致。测试用可穷举数据（如 2 组 4 个排列、3 组 8 个排列）核对枚举结果，并用精确 DP 与二项模拟比较不同查看安排下的风险约束，验证不同分块/查看安排产生相同的停止记录。

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
