from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .models import Dataset, TestSpec
from .permutation import GroupedPermutationTest
from .runner import PlatformRunner
from .storage import dumps


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
    run.add_argument("--adaptive-min-samples", type=int, default=None)
    run.add_argument("--adaptive-ci-half-width", type=float, default=None)

    add_test = subparsers.add_parser("add-test", help="attach another hypothesis to a batch")
    add_test.add_argument("batch_id", type=int)
    add_test.add_argument("csv", help="CSV containing group, condition, and value columns")
    add_test.add_argument("--null", required=True, help="null hypothesis statement")
    add_test.add_argument("--conditions", nargs=2, metavar=("CONTROL", "TREATMENT"))
    add_test.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
    )
    add_test.add_argument("--samples", type=int, default=10_000)
    add_test.add_argument("--max-exact-assignments", type=int, default=100_000)
    add_test.add_argument("--adaptive-min-samples", type=int, default=None)
    add_test.add_argument("--adaptive-ci-half-width", type=float, default=None)

    resume = subparsers.add_parser("resume", help="resume an existing batch")
    resume.add_argument("batch_id", type=int)
    resume.add_argument("--checkpoint-every", type=int, default=None)

    result = subparsers.add_parser("result", help="show a completed test result")
    result.add_argument("test_id", type=int)

    compare = subparsers.add_parser(
        "compare", help="unadjusted and adjusted conclusions for a batch"
    )
    compare.add_argument("batch_id", type=int)
    compare.add_argument(
        "--method",
        choices=("bonferroni", "holm", "benjamini-hochberg"),
        default="holm",
    )
    compare.add_argument("--alpha", type=float, default=0.05)
    return parser


def _settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {"alternative": args.alternative}
    if args.conditions:
        settings["conditions"] = tuple(args.conditions)
    adaptive: dict[str, Any] = {}
    if args.adaptive_min_samples is not None:
        adaptive["min_samples"] = args.adaptive_min_samples
    if args.adaptive_ci_half_width is not None:
        adaptive["ci_half_width"] = args.adaptive_ci_half_width
    if adaptive:
        settings["adaptive"] = adaptive
    return settings


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
            settings = _settings_from_args(args)
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
        elif args.command == "add-test":
            records = read_csv(args.csv)
            dataset = Dataset.from_records(records)
            spec = TestSpec(
                test_type=GroupedPermutationTest.test_type,
                null_hypothesis=args.null,
                settings=_settings_from_args(args),
            )
            batch = runner.store.find_one("batches", id=args.batch_id)
            if batch is None:
                raise SystemExit(f"unknown batch {args.batch_id}")
            stored = runner.store.find_one("datasets", id=int(batch["dataset_id"]))
            if dumps(dataset.records) != stored["records_json"]:
                raise SystemExit("CSV records do not match the batch dataset")
            test_id = runner.add_test(
                args.batch_id,
                spec,
                samples=args.samples,
                max_exact_assignments=args.max_exact_assignments,
                seed=batch.get("seed"),
            )
            print(json.dumps({"test_id": test_id, "batch_id": args.batch_id}, ensure_ascii=False))
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
            report = runner.compare_batch(
                args.batch_id, method=args.method, alpha=args.alpha
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, default=list))
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
