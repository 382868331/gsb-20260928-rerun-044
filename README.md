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

### 主要假设与后续假设门控

多项假设可以在运行前放进一个层级计划：每项声明 `hierarchy.key`、`risk_share`，后续项再声明 `parent`、开放条件（如 `p_value_at_most`）和 `allocation`。`PlatformRunner.create_batch(..., family_alpha=...)` 会先检查每条根到叶路径的风险份额总和，不合法则整组拒绝运行。

同一个确认家族里，全部并列主要假设（没有 `parent` 的项）的 `risk_share` 之和也不能超过 `family_alpha`：不能只因每条根到叶路径单独看不超额，就接受合计超出的分配（例如两个各 0.04 的并列主要假设对 0.05 的整组预算会被拒绝）。校验在任何落盘之前完成，被拒绝的计划不会写入批次，库中已有批次与探索结果原样保留；错误信息会列出冲突的各项及其份额。

一个后续项也可以声明多个父假设（`parent` 为列表）。此时必须预先声明 `parent_mode`：`"all"` 表示全部父假设都满足开放条件才打开，任一父假设失败即关闭；`"any"` 表示任一父假设满足即打开，全部父假设都失败才关闭。无论经由几条路径打开，同一后续项只获得自己声明的一份 `risk_share` 与一份正式额度，不会因多条路径先后打开而重复领用；打开记录（`gate_opened_via`）只写一次。

同一个后续项在 `parents`（或 `parent` 列表）中把同一父来源重复列出属于配置错误：`validate_gates` 在任何批次数据落盘前拒绝该计划，错误信息给出涉及的假设节点、重复来源与原始父来源顺序，重复项无论排在什么位置都会被发现。这与多个**不同**父节点共同指向同一后续项不同——共享后续节点仍是合法配置，每个父来源只构成一条门控边。对在校验加固前已落库的历史批次，恢复执行与重复 `compare` 会按声明顺序去重读取父来源，门控边、打开记录与风险账本都不会因重复条目而翻倍。

每项假设还可以预先声明 `reallocate_to`，规定自己失败（正式不显著）或关闭后未用份额的去向。创建批次前会把所有转移声明视为独立有向图检查：自环、重复边和有向环都会在任何批次证据写入前拒绝，错误信息列出环上节点与具体边；额度很小也不能让不明确的循环传播通过。合法无环图中多个来源可以指向同一个后续项，但每个来源关闭时只转移一次。转移落盘到 `share_transfers`，并记录创建计划时的 `route_version`；部分执行后恢复会沿用该版本与既有历史转移，重复读取 `compare` 不新增分配，后续计划校验失败也不会追溯改写已有证据。已经用于正式显著判断的份额永远不可回收，撤回显示或重新打开试验都不会让它再次可用。接收方的可用额度为自有份额加上全部转入份额。

后续假设在父假设正式满足条件前保持 `pending`，`peek` 只显示探索状态，直接运行未开放假设会报错；父假设失败时依赖项变为不可变的 `closed`，暂停恢复或重开试验都不能绕过该条件。已经开放的假设才保存并正式判断自己的证据，未开放阶段的观察不会计入调整结论。

`compare` 的 `family.gate_paths` 对每条开放路径列出：运行前 `preplanned_allocation`、`planned_risk_share`、已经由正式拒绝使用的 `used_risk_share`、剩余份额以及最终可作出的正式结论。追加计算预算只延长证据收集，不会改变预先固定的门控、分配或错误风险。

`compare` 的 `family.route_version` 报告该批次的路由版本，`family.risk_ledger` 逐项列出每个假设的 `initial_allocation`、`transfers_in` / `transfers_out`（来源、去向、金额、原因）、`effective_allocation`、`used_risk_share` 与 `remaining_risk_share`，用于逐项核对不同开放顺序与共享后续项配置下的初始分配、转移、已用和剩余额度；层级假设的正式判断（`adjusted_reject`）按转移后的有效份额作出。

Python 示例：

```python
primary = TestSpec(..., settings={
    "alternative": "less",
    "hierarchy": {"key": "P1", "risk_share": 0.03,
                   "allocation": {"formal_samples": 10000}},
})
follow_up = TestSpec(..., settings={
    "alternative": "two-sided",
    "hierarchy": {
        "key": "F1", "parent": "P1", "risk_share": 0.02,
        "open_when": {"p_value_at_most": 0.05},
        "allocation": {"formal_samples_after_open": 5000},
    },
})
with PlatformRunner("experiments.json") as platform:
    batch_id = platform.create_batch(dataset, specs=[primary, follow_up],
                                    family_alpha=0.05)
```

命令行可用 `--hypothesis-plan plan.json --family-alpha 0.05` 提交同一组预声明计划。

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
