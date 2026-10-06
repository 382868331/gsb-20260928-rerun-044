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
    run.add_argument(
        "--hypothesis-plan",
        type=Path,
        default=None,
        help="JSON list/dict with pre-planned hierarchy, gates, risk shares, and allocations",
    )
    run.add_argument("--family-alpha", type=float, default=0.05)
    run.add_argument("--checkpoint-every", type=int, default=None)
    run.add_argument(
        "--stopping",
        choices=("fixed", "ci-half-width", "decision"),
        default="fixed",
        help="adaptive stopping rule; non-fixed rules record repeated looks",
    )
    run.add_argument("--min-samples", type=int, default=0)
    run.add_argument("--max-samples", type=int, default=None)
    run.add_argument("--ci-half-width", type=float, default=None)
    run.add_argument("--decision-alpha", type=float, default=None)
    run.add_argument(
        "--looks",
        type=int,
        nargs="+",
        default=None,
        help="pre-planned cumulative sample counts at which decisions may occur",
    )
    run.add_argument(
        "--num-looks",
        type=int,
        default=5,
        help="number of evenly spaced looks when --looks is not supplied",
    )
    run.add_argument(
        "--risk-significant",
        type=float,
        default=0.05,
        help="allowed exact probability of declaring significant at q0",
    )
    run.add_argument(
        "--risk-not-significant",
        type=float,
        default=0.05,
        help="allowed exact probability of declaring not-significant at q0",
    )
    run.add_argument(
        "--q-sig",
        type=float,
        default=0.01,
        help="true-p level at which significant power is reported (below alpha)",
    )
    run.add_argument(
        "--q-nonsig",
        type=float,
        default=0.2,
        help="true-p level at which not-significant power is reported (above alpha)",
    )

    resume = subparsers.add_parser("resume", help="resume an existing batch")
    resume.add_argument("batch_id", type=int)
    resume.add_argument("--checkpoint-every", type=int, default=None)

    peek = subparsers.add_parser(
        "peek", help="show current evidence and unmet stopping conditions"
    )
    peek.add_argument("test_id", type=int)

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
    allocate = subparsers.add_parser(
        "allocate",
        help="read-only study comparing legal risk-quota allocation schemes",
    )
    allocate.add_argument("batch_id", type=int)
    allocate.add_argument(
        "--priority",
        action="append",
        nargs="+",
        default=None,
        metavar="KEY",
        help="candidate priority order listing every hypothesis key; repeatable",
    )
    allocate.add_argument(
        "--route",
        action="append",
        nargs=2,
        default=None,
        metavar=("FROM", "TO"),
        help="extra allowed transfer route; repeatable",
    )
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
    if args.stopping == "decision":
        stopping["alpha"] = (
            args.decision_alpha if args.decision_alpha is not None else 0.05
        )
        stopping["risk_significant"] = args.risk_significant
        stopping["risk_not_significant"] = args.risk_not_significant
        stopping["q_sig"] = args.q_sig
        stopping["q_nonsig"] = args.q_nonsig
        stopping["number_of_looks"] = args.num_looks
        if args.looks:
            stopping["looks"] = list(args.looks)
    if args.stopping == "ci-half-width":
        stopping["number_of_looks"] = args.num_looks
        if args.looks:
            stopping["looks"] = list(args.looks)
    return stopping


def build_specs(args: argparse.Namespace) -> list[TestSpec]:
    if args.hypothesis_plan is None:
        settings: dict[str, Any] = {"alternative": args.alternative}
        if args.conditions:
            settings["conditions"] = tuple(args.conditions)
        settings["stopping"] = build_stopping(args)
        return [
            TestSpec(
                test_type=GroupedPermutationTest.test_type,
                null_hypothesis="Exchangeable condition labels within groups; mean(condition 0) = mean(condition 1).",
                settings=settings,
            )
        ]
    with args.hypothesis_plan.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    hypotheses = payload.get("hypotheses", payload) if isinstance(payload, dict) else payload
    specs = []
    for item in hypotheses:
        settings = dict(item.get("settings") or {})
        settings.setdefault("alternative", args.alternative)
        settings.setdefault("conditions", tuple(args.conditions)) if args.conditions else None
        hierarchy = item.get("hierarchy")
        if hierarchy:
            settings["hierarchy"] = hierarchy
        specs.append(
            TestSpec(
                test_type=GroupedPermutationTest.test_type,
                null_hypothesis=item.get(
                    "null_hypothesis",
                    "Exchangeable condition labels within groups; equal means.",
                ),
                settings=settings,
            )
        )
    return specs


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
            batch_id = runner.create_batch(
                dataset,
                name=args.batch_name,
                specs=build_specs(args),
                samples=args.samples,
                max_exact_assignments=args.max_exact_assignments,
                seed=args.seed,
                family_alpha=args.family_alpha,
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
        elif args.command == "allocate":
            print(
                json.dumps(
                    runner.study_allocation(
                        args.batch_id,
                        priorities=args.priority,
                        extra_routes=args.route,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            if args.command == "peek":
                print(json.dumps(runner.peek_test(args.test_id), ensure_ascii=False, indent=2))
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
