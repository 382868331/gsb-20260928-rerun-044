"""带版本的人工数据。

数据按长表组织，每行一个观测，包含：
- 分组（层/区组）标识：置换只允许在同一组内部进行；
- 处理标签：参与交换的处理条件；
- 响应值：用于计算检验统计量。

保存时计算规范化的内容哈希（SHA-256），同名不同版本各自留档，
保证任何检验结果都能回溯到确切的数据版本。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


def _canonical_payload(rows, group_col, label_col, value_col):
    normalized = [
        {
            group_col: row[group_col],
            label_col: row[label_col],
            value_col: row[value_col],
        }
        for row in rows
    ]
    return json.dumps(
        {
            "columns": [group_col, label_col, value_col],
            "rows": normalized,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class DataBundle:
    """一个不可变的数据版本。"""

    name: str
    version: str
    rows: tuple
    group_col: str
    label_col: str
    value_col: str
    content_hash: str

    @classmethod
    def create(
        cls,
        name,
        version,
        rows,
        *,
        group_col="group",
        label_col="label",
        value_col="value",
    ):
        rows = tuple(dict(r) for r in rows)
        if not rows:
            raise ValueError("数据不能为空")
        for i, row in enumerate(rows):
            for col in (group_col, label_col, value_col):
                if col not in row:
                    raise KeyError(f"第 {i} 行缺少列 {col!r}")
        payload = _canonical_payload(rows, group_col, label_col, value_col)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return cls(
            name=name,
            version=str(version),
            rows=rows,
            group_col=group_col,
            label_col=label_col,
            value_col=value_col,
            content_hash=digest,
        )

    def labels(self):
        seen = []
        for row in self.rows:
            lab = row[self.label_col]
            if lab not in seen:
                seen.append(lab)
        return tuple(seen)

    def groups(self):
        order = []
        index = {}
        for i, row in enumerate(self.rows):
            g = row[self.group_col]
            if g not in index:
                index[g] = []
                order.append(g)
            index[g].append(i)
        return tuple((g, tuple(index[g])) for g in order)

    def to_payload_json(self):
        return _canonical_payload(
            self.rows, self.group_col, self.label_col, self.value_col
        )

    @classmethod
    def from_record(cls, record) -> "DataBundle":
        payload = json.loads(record["payload_json"])
        return cls.create(
            record["name"],
            record["version"],
            payload["rows"],
            group_col=record["group_col"],
            label_col=record["label_col"],
            value_col=record["value_col"],
        )


def make_paired_example():
    """核对用的极小人工数据：2 个配对组，每组处理/对照各 1 个观测。

    每组内部只有两种排列，全局共 2*2=4 种配置，可完全枚举。
    """
    rows = [
        {"subject": "s1", "condition": "treat", "y": 5.0},
        {"subject": "s1", "condition": "ctrl",  "y": 1.0},
        {"subject": "s2", "condition": "treat", "y": 4.0},
        {"subject": "s2", "condition": "ctrl",  "y": 2.0},
    ]
    return DataBundle.create(
        "paired_demo", "1.0.0", rows,
        group_col="subject", label_col="condition", value_col="y",
    )
