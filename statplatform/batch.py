"""批次编排：数据版本、检验设定、种子、逐条结果统一留档。

一个批次（batch）可包含多项检验，每项检验明确记录其原假设，
因此结果可以一一对应回各自的 H0。所有检验类型共用同一套表结构与
暂停/恢复逻辑（见 :mod:`statplatform.registry`）。
"""

from __future__ import annotations

import datetime as _dt
import json
import secrets
import uuid

from .data import DataBundle
from . import permutation
from .registry import get_executor
from .storage import connect, transaction


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


class _ChunkSink:
    """检验执行器在单个块内回写进度与逐条抽样结果。"""

    def __init__(self, conn, test_id):
        self._conn = conn
        self._test_id = test_id
        self._batch = []
        self.progress = 0
        self.extreme = 0

    def record_draw(self, draw_index, statistic):
        self._batch.append((draw_index, float(statistic)))

    def update_progress(self, evaluated, extreme):
        self.progress += evaluated
        self.extreme += extreme

    def flush(self):
        if self._batch:
            self._conn.executemany(
                "INSERT OR IGNORE INTO draws(test_id, draw_index, statistic) "
                "VALUES (?, ?, ?)",
                [(self._test_id, i, s) for i, s in self._batch],
            )


class Platform:
    def __init__(self, db_path="platform.db"):
        self.conn = connect(db_path)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 批次 -------------------------------------------------------
    def create_batch(self, name, note=""):
        batch_id = uuid.uuid4().hex
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO batches(id, name, note, created_at) VALUES (?,?,?,?)",
                (batch_id, name, note, _now()),
            )
        return batch_id

    def get_batch(self, batch_id):
        row = self.conn.execute(
            "SELECT * FROM batches WHERE id=?", (batch_id,)
        ).fetchone()
        return dict(row) if row else None

    # ---- 数据版本 ---------------------------------------------------
    def save_data(self, bundle: DataBundle):
        existing = self.conn.execute(
            "SELECT id, content_hash FROM datasets WHERE name=? AND version=?",
            (bundle.name, bundle.version),
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != bundle.content_hash:
                raise ValueError(
                    f"数据 {bundle.name} 的版本 {bundle.version} 已存在但内容不同；"
                    "版本不可变，请使用新版本号"
                )
            return existing["id"]
        dataset_id = uuid.uuid4().hex
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO datasets(id, name, version, group_col, label_col, "
                "value_col, content_hash, payload_json, n_rows, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset_id, bundle.name, bundle.version, bundle.group_col,
                    bundle.label_col, bundle.value_col, bundle.content_hash,
                    bundle.to_payload_json(), len(bundle.rows), _now(),
                ),
            )
        return dataset_id

    def _load_bundle(self, dataset_id):
        row = self.conn.execute(
            "SELECT * FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"未知数据集: {dataset_id}")
        return DataBundle.from_record(row)

    def find_dataset(self, name, version=None):
        if version is None:
            row = self.conn.execute(
                "SELECT * FROM datasets WHERE name=? ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM datasets WHERE name=? AND version=?",
                (name, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"未找到数据 {name} 版本 {version}")
        return dict(row)

    # ---- 检验登记 ---------------------------------------------------
    def add_test(
        self,
        batch_id,
        bundle: DataBundle,
        *,
        test_type="permutation",
        null_hypothesis,
        settings=None,
        seed=None,
    ):
        settings = dict(settings or {})
        executor = get_executor(test_type)
        dataset_id = self.save_data(bundle)

        # 准备阶段确定运行模式、配置空间、观测统计量，保证登记信息完整。
        spec, prepared = executor.prepare(bundle, settings)
        state = executor.initial_state(prepared, seed)

        if seed is None and state["run_mode"] == permutation.MONTE_CARLO:
            seed = secrets.randbits(63)

        test_id = uuid.uuid4().hex
        record_settings = {
            **settings,
            "labels": list(spec.labels),
            "alternative": spec.alternative,
            "allocation": spec.allocation,
        }
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO tests(id, batch_id, dataset_id, test_type, "
                "null_hypothesis, settings_json, seed, run_mode, status, "
                "total_configs, target_draws, done_count, ge_count, "
                "observed_stat, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    test_id, batch_id, dataset_id, test_type, null_hypothesis,
                    json.dumps(record_settings, ensure_ascii=False, sort_keys=True),
                    seed, state["run_mode"], "pending", state["total_configs"],
                    state["target_draws"], 0, 0, state["observed_stat"],
                    _now(), _now(),
                ),
            )
        return test_id

    def get_test(self, test_id):
        row = self.conn.execute(
            "SELECT * FROM tests WHERE id=?", (test_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"未知检验: {test_id}")
        data = dict(row)
        data["settings"] = json.loads(data.pop("settings_json"))
        return data

    def list_tests(self, batch_id):
        rows = self.conn.execute(
            "SELECT * FROM tests WHERE batch_id=? ORDER BY created_at", (batch_id,)
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["settings"] = json.loads(item.pop("settings_json"))
            item["result"] = self.get_result(row["id"])
            out.append(item)
        return out

    # ---- 运行 / 暂停 / 恢复 -----------------------------------------
    def run(self, test_id, max_work=None):
        """推进检验直至完成或累计评估 ``max_work`` 个配置（用于暂停）。

       再次调用即恢复：从 done_count 之后继续，已有样本绝不重算。
        返回当前检验记录。
        """
        test = self.get_test(test_id)
        if test["status"] == "complete":
            return test
        executor = get_executor(test["test_type"])
        bundle = self._load_bundle(test["dataset_id"])
        _, prepared = executor.prepare(bundle, test["settings"])
        chunk_size = executor.chunk_size_for(prepared)

        worked = 0
        while True:
            test = self.get_test(test_id)
            done = test["done_count"]
            ge = test["ge_count"]
            budget = chunk_size
            if max_work is not None:
                budget = min(budget, max_work - worked)
                if budget <= 0:
                    break

            sink = _ChunkSink(self.conn, test_id)
            with transaction(self.conn):
                complete = executor.run_chunk(
                    prepared, test["seed"], done, ge, budget, sink
                )
                # 以逐条记录为准做幂等核对，避免任何重复计数。
                sink.flush()
                self.conn.execute(
                    "UPDATE tests SET done_count=done_count+?, "
                    "ge_count=ge_count+?, updated_at=? WHERE id=?",
                    (sink.progress, sink.extreme, _now(), test_id),
                )
            worked += sink.progress
            if complete:
                self._finalize(test_id, prepared, executor)
                break
            if max_work is not None and worked >= max_work:
                break
        return self.get_test(test_id)

    def _finalize(self, test_id, prepared, executor):
        test = self.get_test(test_id)
        summary = executor.summarize(prepared, test["done_count"], test["ge_count"])
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO results(test_id, p_value, p_kind, se, ci_low, ci_high, "
                "n_evaluated, total_configs, finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    test_id, summary["p_value"], summary["p_kind"], summary["se"],
                    summary["ci_low"], summary["ci_high"], summary["n_evaluated"],
                    test["total_configs"], _now(),
                ),
            )
            self.conn.execute(
                "UPDATE tests SET status='complete', updated_at=? WHERE id=?",
                (_now(), test_id),
            )

    def get_result(self, test_id):
        row = self.conn.execute(
            "SELECT * FROM results WHERE test_id=?", (test_id,)
        ).fetchone()
        return dict(row) if row else None

    def progress(self, test_id):
        test = self.get_test(test_id)
        total = (
            test["target_draws"]
            if test["run_mode"] == permutation.MONTE_CARLO
            else test["total_configs"]
        )
        return {
            "test_id": test_id,
            "run_mode": test["run_mode"],
            "status": test["status"],
            "done": test["done_count"],
            "total": total,
            "seed": test["seed"],
        }
