"""Command line entry point: `oppscan run`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oppscan import db
from oppscan.fake_etsy import fake_transport
from oppscan.fake_llm import FakeLLM
from oppscan.pipeline import Paths, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oppscan", description="Etsy template opportunity scanner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="collect, score and write a report")
    run.add_argument("--fixtures", action="store_true",
                     help="use the fake Etsy API and fake LLM (no network, no keys)")
    run.add_argument("--resume", metavar="RUN_ID", help="resume a paused run")
    run.add_argument("--config-dir", type=Path, default=Path("config"))
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    paths = Paths(
        db=args.data_dir / ("fixtures.duckdb" if args.fixtures else "oppscan.duckdb"),
        config_dir=args.config_dir,
        reports_dir=args.reports_dir / "fixtures" if args.fixtures else args.reports_dir,
        archive_dir=args.data_dir / "archive",
    )
    extra = {}
    if args.fixtures:
        extra = {"transport": fake_transport(db.utcnow()), "llm": FakeLLM(),
                 "api_key": "fixture", "qps": 1e6}
    try:
        result = run_pipeline(paths, resume=args.resume, **extra)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"run {result.run_id}: {result.status}")
    for reason in result.reasons:
        print(f"  - {reason}")
    if result.html:
        print(f"report: {result.html}")
    return 2 if result.status == "paused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
