"""命令行入口：以批次方式登记并运行检验，全部记录写入工作区数据库。"""

from __future__ import annotations

import argparse
import json

from .batch import Platform
from .data import DataBundle, make_paired_example


def _print_result(test, result):
    print(f"检验 {test['id'][:8]}  类型={test['test_type']}")
    print(f"  H0: {test['null_hypothesis']}")
    print(f"  模式: {test['run_mode']}  状态: {test['status']}")
    print(f"  观测统计量: {test['observed_stat']:.6g}")
    if result:
        print(
            f"  p = {result['p_value']:.6g} ({result['p_kind']})"
            + (f" ± {result['se']:.4g}" if result['se'] is not None else "")
        )
        if result['ci_low'] is not None:
            print(
                "  95% CI: "
                f"[{result['ci_low']:.6g}, {result['ci_high']:.6g}]"
            )
    else:
        print(f"  进度: {test['done_count']}")


def demo(db_path, n_draws, seed):
    """用可完全枚举的极小配对数据演示精确与抽样两条路径。"""
    with Platform(db_path) as pf:
        bundle = make_paired_example()
        batch = pf.create_batch("demo_batch", "配对置换演示：精确 + 蒙特卡洛对照")

        settings_exact = {
            "labels": ["treat", "ctrl"],
            "alternative": "two_sided",
            "allocation": "fixed",
            "run_mode": "exact",
        }
        t1 = pf.add_test(
            batch, bundle,
            null_hypothesis="H0: 处理与对照在配对内无差异（双侧，精确枚举）",
            settings=settings_exact,
        )
        pf.run(t1)

        settings_mc = {
            "labels": ["treat", "ctrl"],
            "alternative": "two_sided",
            "allocation": "fixed",
            "run_mode": "monte_carlo",
            "n_draws": n_draws,
            "chunk_size": max(1, n_draws // 4),
        }
        t2 = pf.add_test(
            batch, bundle,
            null_hypothesis=f"H0: 处理与对照在配对内无差异（双侧，{n_draws} 次抽样）",
            settings=settings_mc,
            seed=seed,
        )
        # 分块运行，模拟“暂停—恢复”：每次只跑一个 chunk。
        while pf.progress(t2)["status"] != "complete":
            pf.run(t2)

        print(f"批次: {batch}")
        for test in pf.list_tests(batch):
            _print_result(test, test["result"])


def run_file(args):
    with Platform(args.db) as pf:
        with open(args.data, encoding="utf-8") as fh:
            doc = json.load(fh)
        bundle = DataBundle.create(
            doc["name"], doc["version"], doc["rows"],
            group_col=doc.get("group_col", "group"),
            label_col=doc.get("label_col", "label"),
            value_col=doc.get("value_col", "value"),
        )
        batch = pf.create_batch(args.batch_name or f"batch-{bundle.name}")
        test_id = pf.add_test(
            batch, bundle,
            null_hypothesis=args.null,
            settings=json.loads(args.settings) if args.settings else {},
            seed=args.seed,
        )
        pf.run(test_id)
        test = pf.get_test(test_id)
        _print_result(test, pf.get_result(test_id))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="statplatform")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="用可枚举小数据运行演示")
    p_demo.add_argument("--db", default="platform.db")
    p_demo.add_argument("--draws", type=int, default=2000)
    p_demo.add_argument("--seed", type=int, default=20260929)

    p_run = sub.add_parser("run", help="从 JSON 文件运行一个批次")
    p_run.add_argument("data")
    p_run.add_argument("--db", default="platform.db")
    p_run.add_argument("--batch-name")
    p_run.add_argument("--null", required=True)
    p_run.add_argument("--settings")
    p_run.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    if args.command == "demo":
        demo(args.db, args.draws, args.seed)
    elif args.command == "run":
        run_file(args)


if __name__ == "__main__":
    main()
