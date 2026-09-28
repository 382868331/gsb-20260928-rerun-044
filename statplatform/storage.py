"""SQLite 持久化层。

一个数据库文件即一个平台工作区，保留：
批次、数据版本（含内容哈希）、检验设定与种子、逐条抽样结果、
进度计数与最终结论。所有写入以事务为单位，保证暂停/恢复后状态一致。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,
    group_col     TEXT NOT NULL,
    label_col     TEXT NOT NULL,
    value_col     TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    n_rows        INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS tests (
    id            TEXT PRIMARY KEY,
    batch_id      TEXT NOT NULL REFERENCES batches(id),
    dataset_id    TEXT NOT NULL REFERENCES datasets(id),
    test_type     TEXT NOT NULL,
    null_hypothesis TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    seed          INTEGER,
    run_mode      TEXT NOT NULL,           -- exact | monte_carlo
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | complete
    total_configs INTEGER,                -- 精确模式下置换配置总数
    target_draws  INTEGER,                -- 蒙特卡洛模式计划抽样次数
    done_count    INTEGER NOT NULL DEFAULT 0,
    ge_count      INTEGER NOT NULL DEFAULT 0,
    observed_stat REAL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS draws (
    test_id    TEXT NOT NULL REFERENCES tests(id),
    draw_index INTEGER NOT NULL,
    statistic  REAL NOT NULL,
    PRIMARY KEY (test_id, draw_index)
);

CREATE TABLE IF NOT EXISTS results (
    test_id       TEXT PRIMARY KEY REFERENCES tests(id),
    p_value       REAL NOT NULL,
    p_kind        TEXT NOT NULL,           -- exact | estimated
    se            REAL,
    ci_low        REAL,
    ci_high       REAL,
    n_evaluated   INTEGER NOT NULL,
    total_configs INTEGER,
    finished_at   TEXT NOT NULL
);
"""


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def transaction(conn):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
