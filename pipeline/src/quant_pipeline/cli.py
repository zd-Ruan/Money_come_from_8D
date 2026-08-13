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
from .research_cli import (
    execute_research_workspace,
    initialize_research_workspace,
    resolve_research_workspace,
    validate_research_workspace,
    write_research_experiment_manifest,
)
from .runner import run_pipeline
from .web import create_app


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "baseline.yaml"
DEFAULT_RESEARCH_ROOT = PIPELINE_ROOT / "research"


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

    research_parser = subparsers.add_parser(
        "research", help="pre-register and inspect factor research controls"
    )
    research_subparsers = research_parser.add_subparsers(dest="research_command", required=True)
    research_init = research_subparsers.add_parser(
        "init", help="freeze a plan, config, experiment battery, and unopened state ledger"
    )
    research_init.add_argument("--study-id", required=True)
    research_init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    research_init.add_argument("--calendar", type=Path, required=True)
    research_init.add_argument("--discovery-end", required=True)
    research_init.add_argument("--confirmation-end", required=True)
    research_init.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)

    research_status = research_subparsers.add_parser(
        "status", help="validate control artifacts without reading result metrics"
    )
    research_status.add_argument("--study-id", required=True)
    research_status.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)

    research_manifest = research_subparsers.add_parser(
        "manifest", help="export the current validated experiment specifications"
    )
    research_manifest.add_argument("--study-id", required=True)
    research_manifest.add_argument("--output", type=Path, required=True)
    research_manifest.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)

    research_execute = research_subparsers.add_parser(
        "execute", help="run or resume the frozen one-shot research protocol"
    )
    research_execute.add_argument("--study-id", required=True)
    research_execute.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    research_execute.add_argument(
        "--repository-root",
        type=Path,
        default=PIPELINE_ROOT.parent,
        help="repository root used to rehydrate frozen relative paths",
    )

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
    elif args.command == "research":
        try:
            workspace = resolve_research_workspace(args.research_root, args.study_id)
            if args.research_command == "init":
                workspace = initialize_research_workspace(
                    research_root=args.research_root,
                    study_id=args.study_id,
                    config_path=args.config,
                    calendar_path=args.calendar,
                    discovery_end=args.discovery_end,
                    confirmation_end=args.confirmation_end,
                )
                result = validate_research_workspace(workspace)
            elif args.research_command == "status":
                result = validate_research_workspace(workspace)
            elif args.research_command == "manifest":
                output = write_research_experiment_manifest(workspace, args.output)
                result = {"valid": True, "workspace": str(workspace), "output": str(output)}
            else:
                result = execute_research_workspace(
                    workspace, repository_root=args.repository_root
                )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
