from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .models import Dataset, TestSpec
from .permutation import GroupedPermutationTest
from .runner import PlatformRunner


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stat-platform")
    parser.add_argument("--database", default="experiments.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="create and run one batch")
    run.add_argument("csv", help="CSV containing group, condition, and value columns")
    run.add_argument("--name", default="dataset")
    run.add_argument("--version", default="working")
    run.add_argument("--batch-name")
    run.add_argument("--group-column", default="group")
    run.add_argument("--condition-column", default="condition")
    run.add_argument("--value-column", default="value")
    run.add_argument("--conditions", nargs=2, metavar=("CONTROL", "TREATMENT"))
    run.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
    )
    run.add_argument("--samples", type=int, default=10_000)
    run.add_argument("--max-exact-assignments", type=int, default=100_000)
    run.add_argument("--seed", type=int)
    run.add_argument("--checkpoint-every", type=int, default=None)
    run.add_argument(
        "--stopping",
        choices=("fixed", "ci-half-width", "sequential-decision"),
        default="fixed",
        help=(
            "stopping rule; sequential-decision requires pre-registered "
            "--looks and error budgets"
        ),
    )
    run.add_argument("--min-samples", type=int, default=0)
    run.add_argument("--max-samples", type=int, default=None)
    run.add_argument("--ci-half-width", type=float, default=None)
    run.add_argument(
        "--looks",
        type=int,
        nargs="+",
        default=None,
        help="strictly increasing cumulative sample counts at which decisions may be made",
    )
    run.add_argument("--p-high", type=float, default=0.05)
    run.add_argument("--p-low", type=float, default=None)
    run.add_argument("--epsilon-high", type=float, default=0.05)
    run.add_argument("--epsilon-low", type=float, default=0.05)
    run.add_argument(
        "--spending",
        choices=("pocock", "obrien-flanagan"),
        default="pocock",
    )

    resume = subparsers.add_parser("resume", help="resume an existing batch")
    resume.add_argument("batch_id", type=int)
    resume.add_argument("--checkpoint-every", type=int, default=None)

    status = subparsers.add_parser(
        "status", help="show current evidence and unmet stopping conditions"
    )
    status.add_argument("test_id", type=int)

    compare = subparsers.add_parser(
        "compare", help="unadjusted and adjusted conclusions for one batch"
    )
    compare.add_argument("batch_id", type=int)
    compare.add_argument(
        "--rule", choices=("none", "bonferroni", "holm"), default="holm"
    )
    compare.add_argument("--alpha", type=float, default=0.05)

    result = subparsers.add_parser("result", help="show a completed test result")
    result.add_argument("test_id", type=int)
    return parser


def build_stopping(args: argparse.Namespace) -> dict[str, Any]:
    stopping: dict[str, Any] = {"type": args.stopping}
    if args.stopping == "fixed":
        return stopping
    stopping["min_samples"] = args.min_samples
    if args.max_samples is not None:
        stopping["max_samples"] = args.max_samples
    if args.stopping == "ci-half-width":
        if args.ci_half_width is None:
            raise SystemExit("--ci-half-width is required for ci-half-width stopping")
        stopping["max_half_width"] = args.ci_half_width
        return stopping
    if not args.looks:
        raise SystemExit(
            "sequential-decision requires --looks with the pre-selected "
            "cumulative sample counts (e.g. --looks 200 500 1000)"
        )
    stopping.update(
        {
            "looks": list(args.looks),
            "p_high": args.p_high,
            "epsilon_high": args.epsilon_high,
            "epsilon_low": args.epsilon_low,
            "spending": args.spending,
        }
    )
    if args.p_low is not None:
        stopping["p_low"] = args.p_low
    return stopping


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with PlatformRunner(args.database) as runner:
        if args.command == "run":
            records = read_csv(args.csv)
            dataset = Dataset.from_records(
                records,
                name=args.name,
                version=args.version,
                group_column=args.group_column,
                condition_column=args.condition_column,
                value_column=args.value_column,
            )
            settings: dict[str, Any] = {"alternative": args.alternative}
            if args.conditions:
                settings["conditions"] = tuple(args.conditions)
            settings["stopping"] = build_stopping(args)
            spec = TestSpec(
                test_type=GroupedPermutationTest.test_type,
                null_hypothesis="Exchangeable condition labels within groups; mean(condition 0) = mean(condition 1).",
                settings=settings,
            )
            batch_id = runner.create_batch(
                dataset,
                name=args.batch_name,
                specs=[spec],
                samples=args.samples,
                max_exact_assignments=args.max_exact_assignments,
                seed=args.seed,
            )
            try:
                results = runner.run_batch(
                    batch_id,
                    max_new_samples=args.checkpoint_every,
                )
            except RuntimeError:
                print(
                    json.dumps(
                        {
                            "batch_id": batch_id,
                            "status": "checkpointed",
                            "message": "Persisted the requested sample chunk; resume the same batch.",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(json.dumps([serialize_result(item) for item in results], ensure_ascii=False, indent=2))
        elif args.command == "resume":
            try:
                results = runner.run_batch(
                    args.batch_id,
                    max_new_samples=args.checkpoint_every,
                )
            except RuntimeError:
                print(
                    json.dumps(
                        {
                            "batch_id": args.batch_id,
                            "status": "checkpointed",
                            "message": "Persisted the requested sample chunk; resume the same batch.",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(json.dumps([serialize_result(item) for item in results], ensure_ascii=False, indent=2))
        elif args.command == "compare":
            print(
                json.dumps(
                    runner.compare_batch(
                        args.batch_id, rule=args.rule, alpha=args.alpha
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "status":
            print(
                json.dumps(
                    runner.test_status(args.test_id),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(serialize_result(runner.get_result(args.test_id)), ensure_ascii=False, indent=2))


def serialize_result(item: object) -> dict[str, Any]:
    from .models import TestResult

    result = item if isinstance(item, TestResult) else item
    data = {
        "test_id": result.test_id,
        "batch_id": result.batch_id,
        "null_hypothesis": result.null_hypothesis,
        **result.result.to_dict(),
    }
    return data


if __name__ == "__main__":
    main()
