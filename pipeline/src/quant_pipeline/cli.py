from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .audit import audit_and_snapshot
from .comparison import generate_comparison_json
from .config import load_config
from .integrity import resolve_run_directory
from .io import read_json
from .report import generate_report
from .runner import run_pipeline
from .web import create_app


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "baseline.yaml"


def resolve_comparison_output(output: Path | None, baseline_run_id: str, candidate_run_id: str) -> Path:
    comparisons_root = (PIPELINE_ROOT / "comparisons").resolve()
    if output is None:
        target = comparisons_root / f"{baseline_run_id}__vs__{candidate_run_id}.json"
    else:
        raw = Path(output)
        target = raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()
    if target.parent != comparisons_root:
        raise ValueError(f"comparison output must be a direct child of {comparisons_root}")
    if target.suffix.lower() != ".json":
        raise ValueError("comparison output must use a .json filename")
    return target


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

    compare_parser = subparsers.add_parser("compare", help="strictly compare paired completed runs")
    compare_parser.add_argument("--baseline-run", required=True)
    compare_parser.add_argument("--candidate-run", required=True)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.add_argument("--overwrite", action="store_true")

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
    elif args.command == "compare":
        runs_root = PIPELINE_ROOT / "runs"
        baseline_run = resolve_run_directory(runs_root, args.baseline_run)
        candidate_run = resolve_run_directory(runs_root, args.candidate_run)
        if not baseline_run.is_dir() or not candidate_run.is_dir():
            parser.error("baseline and candidate run directories must exist")
        try:
            output = resolve_comparison_output(args.output, args.baseline_run, args.candidate_run)
            comparison_path = generate_comparison_json(
                baseline_run,
                candidate_run,
                output,
                overwrite=args.overwrite,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        result = read_json(comparison_path, {})
        print(json.dumps({"path": str(comparison_path), **result}, ensure_ascii=False, indent=2))
    elif args.command == "serve":
        uvicorn.run(create_app(PIPELINE_ROOT), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
