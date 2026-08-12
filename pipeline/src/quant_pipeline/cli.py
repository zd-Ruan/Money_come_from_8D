from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .audit import audit_and_snapshot
from .config import load_config
from .report import generate_report
from .runner import run_pipeline
from .web import create_app


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "baseline.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliable ETF research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="validate and snapshot current data")
    audit_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    run_parser = subparsers.add_parser("run", help="run rolling training, backtests, gates, and report")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--run-id")

    report_parser = subparsers.add_parser("report", help="regenerate a frozen run report")
    report_parser.add_argument("run_dir", type=Path)

    serve_parser = subparsers.add_parser("serve", help="serve the local experiment dashboard")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    if args.command == "audit":
        result = audit_and_snapshot(load_config(args.config))
        print(json.dumps(result.report, ensure_ascii=False, indent=2))
    elif args.command == "run":
        config = load_config(args.config)
        run_dir = run_pipeline(config, run_id=args.run_id)
        print(run_dir / "report.html")
    elif args.command == "report":
        print(generate_report(args.run_dir))
    elif args.command == "serve":
        uvicorn.run(create_app(PIPELINE_ROOT), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
