# 统计试验运行平台

用 Python 标准库搭建的统计试验运行平台。第一项能力是**带分组结构人工数据的
置换检验**；批次记录、数据版本、种子与结果持久化机制对后续检验类型通用。

## 能力与设计原则

- **组内（区组/层）置换**：有关联的观测不跨组拆散交换。每个分组内部独立
  重排处理标签，全局配置是各组配置的笛卡尔积。
  - `allocation="fixed"`（默认，区组随机化）：保持每组内各标签的观测数，
    配置数为各组 C(n_g, k_g) 之积；
  - `allocation="free"`：允许组内标签多重集任意分配，配置数为各组 2^n_g 之积。
- **小则精确、大则抽样**：配置总数不超过阈值（默认 100 000）时枚举全部
  排列，给出精确 p 值；超过阈值或显式指定时用固定种子的蒙特卡洛抽样。
- **抽样误差透明**：蒙特卡洛结果一律标注 `estimated`（绝不冒充精确概率），
  p 值用 Phipson–Smyth 的 `(ge+1)/(n+1)` 估计，附标准误与 Wilson 95% 置信区间。
- **暂停/恢复不重算**：运行以 chunk 事务推进。精确模式按确定性枚举顺序
  跳过已评估配置；抽样模式从下一个序号继续，序号由 `SHA256(seed, index)`
  确定性派生。逐条结果落库且序号唯一，恢复结果与一次连续运行逐条一致。
- **批次与原假设对应**：一个批次可登记多项检验，每项检验显式记录原假设
  文本、设定、种子与数据版本，结果可一一回溯到 H0。
- **数据版本不可变**：每个数据版本保存规范化内容哈希（SHA-256）；同名同
  版本内容不同会被拒绝，任何结果都能追溯到确切数据。
- **可扩展**：新增检验只需在 `statplatform.registry` 注册执行器
  （`prepare` / `run_chunk` / `summarize`），自动复用同一套批次记录。

## 目录结构

- `statplatform/data.py`：带版本的人工数据与内容哈希
- `statplatform/permutation.py`：分组结构、层内枚举/抽样、检验统计量
- `statplatform/engine.py`：纯计算执行引擎（分块推进，可恢复）
- `statplatform/inference.py`：精确/蒙特卡洛 p 值、抽样误差与置信区间
- `statplatform/storage.py`：SQLite 表结构与事务
- `statplatform/registry.py`：检验类型注册表（扩展点）
- `statplatform/batch.py`：批次编排（登记、运行、暂停恢复、查询）
- `statplatform/cli.py`：命令行演示
- `tests/test_platform.py`：标准库 unittest 测试（含手算小数据核对）

## 使用

```python
from statplatform import Platform, DataBundle, PermutationSpec

rows = [
    {"subject": "s1", "condition": "treat", "y": 5.0},
    {"subject": "s1", "condition": "ctrl",  "y": 1.0},
    {"subject": "s2", "condition": "treat", "y": 4.0},
    {"subject": "s2", "condition": "ctrl",  "y": 2.0},
]
bundle = DataBundle.create(
    "demo", "1.0.0", rows,
    group_col="subject", label_col="condition", value_col="y",
)

with Platform("work.db") as pf:
    batch = pf.create_batch("week-1", "同批多项检验")
    tid = pf.add_test(
        batch, bundle,
        null_hypothesis="H0: 配对内处理与对照无差异（双侧）",
        settings={"labels": ["treat", "ctrl"], "alternative": "two_sided"},
        seed=20260929,  # 抽样模式种子；精确模式不需要
    )
    pf.run(tid)                       # 一次跑完
    # pf.run(tid, max_work=500)       # 或只推进 500 个配置后暂停
    # pf.run(tid)                     # 再次调用即恢复，已有样本不重算
    print(pf.get_result(tid))
    print(pf.list_tests(batch))       # 多项检验结果连同各自 H0
```

命令行演示（精确与抽样对照，内含分块暂停恢复）：

```bash
python -m statplatform.cli demo --db work.db --draws 2000 --seed 20260929
```

## 手算核对基准

对上述 2 个配对组、每组处理/对照各 1 个观测的数据，层内置换只有 4 种，
合并均值差统计量恰为 `{3, -1, 1, -3}`，观测为 3：

- 双侧 p = 2/4 = 0.5；右尾 `T>=3`：p = 1/4 = 0.25；左尾 `T<=3`：p = 1.0；
- 翻转标签顺序使观测为 -3 后，左尾 `T<=-3`：p = 1/4 = 0.25。

这些基准在测试中逐项断言；另含 20 000 次抽样估计收敛到 0.5 的检查。

## 测试

```bash
python -m unittest discover -s tests -v
```

无第三方依赖（仅标准库 `sqlite3`、`random`、`hashlib` 等）。
