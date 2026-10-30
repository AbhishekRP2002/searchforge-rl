"""`searchforge-report`: one command from finished runs to a provider verdict.

Exit codes: 0 report written; 1 gate failed (`--gate` only); 2 comparison
refused because the arms differ in more than the provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from searchforge.analysis.report import STATUS_REFUSED, build_report, render_markdown
from searchforge.analysis.runs import load_run

EXIT_GATE_FAILED = 1
EXIT_REFUSED = 2


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="searchforge-report",
        description="Compare search providers across one or more finished vf-eval runs.",
    )
    parser.add_argument(
        "runs", nargs="+", type=Path, help="run directories (or traces.jsonl paths)"
    )
    parser.add_argument(
        "--baseline", help="compare every arm against this one instead of all pairs"
    )
    parser.add_argument(
        "--allow-diff",
        type=_csv,
        default=[],
        metavar="KEY,...",
        help="accept these setting differences between arms; the report is stamped "
        "NOT A PROVIDER-ONLY COMPARISON",
    )
    parser.add_argument(
        "--expect",
        type=_csv,
        default=[],
        metavar="PROVIDER,...",
        help="providers that must be present",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="fail closed (exit 1) on any validity error or warning",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.0,
        help="share of failed rollouts per arm before a warning (default 0)",
    )
    parser.add_argument(
        "--min-slice-n",
        type=int,
        default=30,
        help="slices with fewer questions are marked as too small (default 30)",
    )
    parser.add_argument("--score", help="reward to treat as correctness (default: auto)")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--out", type=Path, help="write the report here instead of stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        [load_run(path) for path in args.runs],
        baseline=args.baseline,
        allow_diff=args.allow_diff,
        expected=args.expect,
        score=args.score,
        max_failure_rate=args.max_failure_rate,
        min_slice_n=args.min_slice_n,
    )
    text = (
        json.dumps(report, indent=2, default=str)
        if args.format == "json"
        else render_markdown(report)
    )
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)

    code = 0
    if report["status"] == STATUS_REFUSED:
        code = EXIT_REFUSED
    elif args.gate and report["validity"]:
        code = EXIT_GATE_FAILED
    if code:
        sys.exit(code)
    return code
