"""平台核心测试（标准库 unittest，无需第三方依赖）。

可枚举小数据手算基准（fixed 配对，2 组各处理/对照一个观测）：
全局 4 种层内置换的合并均值差为 T ∈ {3, -1, 1, -3}，观测 T=3（最大值）。
因此：双侧 p=2/4=0.5；右尾 (T>=3) p=1/4=0.25；
对该观测左尾 (T<=3) 为 4/4=1.0。要得到 0.25 的左尾，翻转标签顺序使
观测统计量变为最小值 -3，再做 (T<=-3) 检验。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from statplatform import DataBundle, PermutationSpec, Platform
from statplatform import engine, inference


def _bundle():
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


def _new_test(pf, bundle, seed, n_draws, chunk_size, null="H0"):
    batch = pf.create_batch("b")
    return pf.add_test(
        batch, bundle,
        null_hypothesis=null,
        settings={
            "labels": ["treat", "ctrl"],
            "alternative": "two_sided",
            "run_mode": "monte_carlo",
            "n_draws": n_draws,
            "chunk_size": chunk_size,
        },
        seed=seed,
    )


def _count_extreme_from_draws(pf, tid, observed):
    rows = pf.conn.execute(
        "SELECT statistic FROM draws WHERE test_id=?", (tid,)
    ).fetchall()
    return sum(1 for r in rows if abs(r["statistic"]) >= abs(observed))


class ExactTests(unittest.TestCase):
    def test_exact_enumeration_matches_hand_calculation(self):
        bundle = _bundle()
        cases = (
            ("two_sided", ("treat", "ctrl"), 0.5),
            ("greater", ("treat", "ctrl"), 0.25),
            ("less", ("treat", "ctrl"), 1.0),
            ("less", ("ctrl", "treat"), 0.25),
        )
        for alternative, labels, expected in cases:
            with self.subTest(alternative=alternative):
                spec = PermutationSpec(
                    labels=labels,
                    alternative=alternative,
                    run_mode="exact",
                )
                prepared = engine.prepare_permutation(bundle, spec)
                self.assertEqual(prepared.total_configs, 4)
                expected_obs = 3.0 if labels[0] == "treat" else -3.0
                self.assertAlmostEqual(prepared.observed, expected_obs)
                evaluated, extreme, complete = engine.run_exact_chunk(
                    prepared, 0, 1000
                )
                self.assertTrue(complete)
                self.assertEqual(evaluated, 4)
                summary = inference.summarize_exact(
                    extreme, prepared.total_configs
                )
                self.assertAlmostEqual(summary["p_value"], expected)
                self.assertEqual(summary["p_kind"], "exact")
                self.assertIsNone(summary["se"])

    def test_permutations_never_cross_group_boundaries(self):
        bundle = _bundle()
        spec = PermutationSpec(labels=("treat", "ctrl"), run_mode="exact")
        prepared = engine.prepare_permutation(bundle, spec)
        stats = []
        engine.run_exact_chunk(
            prepared, 0, 1000, record=lambda i, value: stats.append(value)
        )
        self.assertEqual(
            sorted(stats), sorted([3.0, -1.0, 1.0, -3.0])
        )

    def test_free_allocation_count(self):
        bundle = _bundle()
        spec = PermutationSpec(labels=("treat", "ctrl"), allocation="free")
        prepared = engine.prepare_permutation(bundle, spec)
        self.assertEqual(prepared.total_configs, 16)


class MonteCarloTests(unittest.TestCase):
    def test_mc_reports_estimate_with_uncertainty_not_exact(self):
        bundle = _bundle()
        with Platform(":memory:") as pf:
            tid = _new_test(pf, bundle, seed=42, n_draws=20, chunk_size=100)
            pf.run(tid)
            result = pf.get_result(tid)
            # 抽样次数很少时必须明确是估计，并附误差，不能冒充精确概率
            self.assertEqual(result["p_kind"], "estimated")
            self.assertIsNotNone(result["se"])
            self.assertGreater(result["se"], 0)
            self.assertLessEqual(result["ci_low"], result["p_value"])
            self.assertLess(result["p_value"], result["ci_high"])
            self.assertEqual(result["total_configs"], 4)
            self.assertEqual(result["n_evaluated"], 20)

    def test_mc_converges_toward_exact_probability(self):
        bundle = _bundle()
        with Platform(":memory:") as pf:
            tid = _new_test(pf, bundle, seed=7, n_draws=20000, chunk_size=5000)
            pf.run(tid)
            self.assertAlmostEqual(pf.get_result(tid)["p_value"], 0.5, delta=0.03)

    def test_auto_mode_selection(self):
        bundle = _bundle()
        with Platform(":memory:") as pf:
            batch = pf.create_batch("b")
            tid = pf.add_test(
                batch, bundle, null_hypothesis="H0",
                settings={"labels": ["treat", "ctrl"], "max_exact_configs": 3},
            )
            test = pf.get_test(tid)
            self.assertEqual(test["run_mode"], "monte_carlo")
            self.assertIsNotNone(test["seed"])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_pause_resume_does_not_recompute_and_matches_continuous_run(self):
        bundle = _bundle()
        seed, n, chunk = 123, 17, 3

        with Platform(self.tmp / "a.db") as pf:
            tid = _new_test(pf, bundle, seed, n, chunk)
            pf.run(tid)
            continuous = pf.get_result(tid)
            continuous_draws = pf.conn.execute(
                "SELECT draw_index, statistic FROM draws ORDER BY draw_index"
            ).fetchall()

        with Platform(self.tmp / "b.db") as pf:
            tid = _new_test(pf, bundle, seed, n, chunk)
            done_marks = []
            while pf.progress(tid)["status"] != "complete":
                pf.run(tid, max_work=4)
                done_marks.append(pf.progress(tid)["done"])
            resumed = pf.get_result(tid)
            resumed_draws = pf.conn.execute(
                "SELECT draw_index, statistic FROM draws ORDER BY draw_index"
            ).fetchall()

        self.assertGreaterEqual(len(done_marks), 3)
        self.assertEqual(
            [(r[0], r[1]) for r in resumed_draws],
            [(r[0], r[1]) for r in continuous_draws],
        )
        self.assertEqual(resumed["p_value"], continuous["p_value"])
        self.assertEqual(len(resumed_draws), n)
        self.assertEqual([r[0] for r in resumed_draws], list(range(n)))

    def test_resume_after_interruption_keeps_ge_count_consistent(self):
        bundle = _bundle()
        seed, n, chunk = 99, 29, 7
        with Platform(self.tmp / "c.db") as pf:
            tid = _new_test(pf, bundle, seed, n, chunk)
            pf.run(tid, max_work=10)
            mid = pf.get_test(tid)
            stored = pf.conn.execute(
                "SELECT COUNT(*) AS c FROM draws WHERE test_id=?", (tid,)
            ).fetchone()["c"]
            self.assertEqual(stored, 10)
            self.assertEqual(
                mid["ge_count"],
                _count_extreme_from_draws(pf, tid, mid["observed_stat"]),
            )
            pf.run(tid)
            self.assertEqual(pf.get_result(tid)["n_evaluated"], 29)

    def test_running_completed_test_is_noop(self):
        bundle = _bundle()
        with Platform(self.tmp / "d.db") as pf:
            tid = _new_test(pf, bundle, 1, 11, 5)
            pf.run(tid)
            first = pf.get_result(tid)
            pf.run(tid)
            second = pf.get_result(tid)
            self.assertEqual(first["p_value"], second["p_value"])
            self.assertEqual(pf.get_test(tid)["done_count"], 11)


class PlatformTests(unittest.TestCase):
    def test_larger_blocked_design_respects_within_group_swaps(self):
        rows = []
        for subject in range(3):
            for j in range(4):
                rows.append({
                    "subject": f"s{subject}",
                    "condition": "treat" if j < 2 else "ctrl",
                    "y": float(subject * 10 + j),
                })
        bundle = DataBundle.create(
            "blocked", "1.0.0", rows,
            group_col="subject", label_col="condition", value_col="y",
        )
        spec = PermutationSpec(labels=("treat", "ctrl"), run_mode="exact")
        prepared = engine.prepare_permutation(bundle, spec)
        # 每组保持 2 个处理：C(4,2)^3 = 6^3 = 216，绝不跨组交换
        self.assertEqual(prepared.total_configs, 6 ** 3)
        seen = []
        engine.run_exact_chunk(prepared, 0, 100_000, record=lambda i, v: seen.append(v))
        self.assertEqual(len(seen), 216)

    def test_resume_across_new_platform_connection(self):
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "resume.db")
            with Platform(db) as pf:
                tid = _new_test(pf, bundle, 555, 23, 5)
                pf.run(tid, max_work=8)
                self.assertEqual(pf.progress(tid)["done"], 8)
            # 模拟进程退出后重新打开同一工作区恢复
            with Platform(db) as pf2:
                self.assertEqual(pf2.progress(tid)["done"], 8)
                pf2.run(tid)
                result = pf2.get_result(tid)
                self.assertEqual(result["n_evaluated"], 23)
                # 每条抽样只被计算一次
                count = pf2.conn.execute(
                    "SELECT COUNT(*) AS c FROM draws WHERE test_id=?", (tid,)
                ).fetchone()["c"]
                self.assertEqual(count, 23)

    def test_multiple_tests_map_to_their_own_null_hypotheses(self):
        bundle = _bundle()
        with Platform(":memory:") as pf:
            batch = pf.create_batch("multi", "同批多项检验")
            h0_two = "H0: 两组均值相等（双侧）"
            h0_gt = "H0: 处理效应不大于 0（右尾）"
            t1 = pf.add_test(
                batch, bundle, null_hypothesis=h0_two,
                settings={"labels": ["treat", "ctrl"], "run_mode": "exact"},
            )
            t2 = pf.add_test(
                batch, bundle, null_hypothesis=h0_gt,
                settings={
                    "labels": ["treat", "ctrl"], "alternative": "greater",
                    "run_mode": "exact",
                },
            )
            pf.run(t1)
            pf.run(t2)
            tests = {t["id"]: t for t in pf.list_tests(batch)}
            self.assertEqual(tests[t1]["null_hypothesis"], h0_two)
            self.assertAlmostEqual(tests[t1]["result"]["p_value"], 0.5)
            self.assertEqual(tests[t2]["null_hypothesis"], h0_gt)
            self.assertAlmostEqual(tests[t2]["result"]["p_value"], 0.25)

    def test_dataset_version_is_immutable_and_traceable(self):
        with Platform(":memory:") as pf:
            b1 = _bundle()
            dataset_id = pf.save_data(b1)
            self.assertEqual(pf.save_data(b1), dataset_id)
            rows = list(b1.rows)
            rows[0] = dict(rows[0])
            rows[0]["y"] = 500.0
            changed = DataBundle.create(
                b1.name, b1.version, rows,
                group_col="subject", label_col="condition", value_col="y",
            )
            with self.assertRaises(ValueError):
                pf.save_data(changed)
            v2 = DataBundle.create(
                b1.name, "2.0.0", rows,
                group_col="subject", label_col="condition", value_col="y",
            )
            pf.save_data(v2)
            self.assertEqual(
                pf.find_dataset("paired_demo", "1.0.0")["content_hash"],
                b1.content_hash,
            )
            self.assertEqual(
                pf.find_dataset("paired_demo", "2.0.0")["content_hash"],
                v2.content_hash,
            )


if __name__ == "__main__":
    unittest.main()
