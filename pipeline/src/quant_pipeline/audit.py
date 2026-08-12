from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io import now_shanghai, read_json, sha256_file, sha256_lines, write_json_atomic


AUDIT_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class AuditResult:
    snapshot_id: str
    snapshot_dir: Path
    report: dict[str, Any]


def ensure_future_calendar_boundary(calendar_path: Path) -> Path:
    calendar = pd.to_datetime(pd.read_csv(calendar_path, header=None).iloc[:, 0])
    future_path = calendar_path.with_name(f"{calendar_path.stem}_future{calendar_path.suffix}")
    future = pd.Series(dtype="datetime64[ns]")
    if future_path.exists():
        future = pd.to_datetime(pd.read_csv(future_path, header=None).iloc[:, 0])
    valid_prefix = len(future) > len(calendar) and future.iloc[: len(calendar)].reset_index(drop=True).equals(
        calendar.reset_index(drop=True)
    )
    if not valid_prefix:
        boundary = calendar.iloc[-1] + pd.Timedelta(days=1)
        values = [*calendar.dt.strftime("%Y-%m-%d"), boundary.strftime("%Y-%m-%d")]
        future_path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return future_path


def read_instruments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", names=["symbol", "start_date", "end_date"], dtype=str)
    frame["symbol"] = frame["symbol"].str.upper()
    return frame.sort_values("symbol").reset_index(drop=True)


def _file_inventory(paths: list[Path], root: Path) -> list[str]:
    return [
        f"{path.relative_to(root).as_posix()}|{path.stat().st_size}|{sha256_file(path)}"
        for path in sorted(paths)
    ]


def evaluate_upstream_validation(
    report: dict[str, Any],
    universe_count: int,
    configured_end: pd.Timestamp,
    raw_file_count: int,
    normalized_file_count: int,
    external_raw_count: int,
    external_normalized_count: int,
) -> tuple[list[str], list[str]]:
    blocking: list[str] = []
    warnings: list[str] = []
    if int(report.get("universe_count", -1)) != universe_count:
        blocking.append("upstream validation universe count does not match the active whitelist")

    expected_end = configured_end.date().isoformat()
    if report.get("min_latest_date") != expected_end or report.get("max_latest_date") != expected_end:
        blocking.append("upstream validation latest dates do not match the configured data end")

    expected_cache_issues = {
        f"raw file count {raw_file_count} != universe {universe_count}",
        f"normalized file count {normalized_file_count} != universe {universe_count}",
    }
    issue_messages = {str(issue.get("error", "")) for issue in report.get("issues", [])}
    extras_explain_counts = (
        raw_file_count == universe_count + external_raw_count
        and normalized_file_count == universe_count + external_normalized_count
    )
    unexpected_issues = issue_messages - expected_cache_issues
    if unexpected_issues:
        blocking.extend(f"upstream validation: {issue}" for issue in sorted(unexpected_issues))
    elif issue_messages and extras_explain_counts:
        warnings.append(
            "upstream total-file count warnings are fully explained by retained pool-external cache files; "
            "the active whitelist is complete"
        )
    elif issue_messages:
        blocking.append("upstream validation file-count issues are not explained by pool-external cache files")
    elif not bool(report.get("training_ready", False)):
        blocking.append("upstream validation is not training-ready for an unexplained reason")
    return blocking, warnings


def audit_and_snapshot(config: dict[str, Any]) -> AuditResult:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    universe_path = paths["universe"]
    instruments_path = paths["instruments"]
    provider = paths["qlib_provider"]
    validation_report_path = paths["validation_report"]
    calendar_path = provider / "calendars" / "day.txt"
    features_dir = provider / "features"
    raw_dir = universe_path.parent / "raw"
    normalized_dir = universe_path.parent / "normalized"

    required = [universe_path, instruments_path, calendar_path, features_dir, validation_report_path]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise FileNotFoundError(f"required data paths are missing: {missing_required}")
    future_calendar_path = ensure_future_calendar_boundary(calendar_path)

    universe = pd.read_csv(universe_path, dtype={"code": str})
    universe["symbol"] = universe["symbol"].str.upper()
    instruments = read_instruments(instruments_path)
    universe_symbols = set(universe["symbol"])
    instrument_symbols = set(instruments["symbol"])

    missing_in_instruments = sorted(universe_symbols - instrument_symbols)
    extra_in_instruments = sorted(instrument_symbols - universe_symbols)
    missing_feature_dirs = sorted(
        symbol for symbol in instrument_symbols if not (features_dir / symbol.lower()).is_dir()
    )
    missing_raw = sorted(symbol for symbol in universe_symbols if not (raw_dir / f"{symbol.lower()}.csv").exists())
    missing_normalized = sorted(
        symbol for symbol in universe_symbols if not (normalized_dir / f"{symbol.lower()}.csv").exists()
    )
    raw_symbols = {path.stem.upper() for path in raw_dir.glob("*.csv")}
    normalized_symbols = {path.stem.upper() for path in normalized_dir.glob("*.csv")}
    external_raw_symbols = raw_symbols - universe_symbols
    external_normalized_symbols = normalized_symbols - universe_symbols

    calendar = pd.to_datetime(pd.read_csv(calendar_path, header=None).iloc[:, 0])
    configured_end = pd.Timestamp(config["data"]["end_date"])
    calendar_end = calendar.max()
    future_calendar = pd.to_datetime(pd.read_csv(future_calendar_path, header=None).iloc[:, 0])
    stale_days = int((configured_end - calendar_end).days)
    upstream_validation = read_json(validation_report_path, {})

    feature_files = [path for symbol in instrument_symbols for path in (features_dir / symbol.lower()).glob("*.bin")]
    inventory = _file_inventory(feature_files, provider)
    source_fingerprint = sha256_lines(
        [
            AUDIT_SCHEMA_VERSION,
            sha256_file(universe_path),
            sha256_file(instruments_path),
            sha256_file(calendar_path),
            sha256_file(future_calendar_path),
            sha256_file(validation_report_path),
            *inventory,
        ]
    )
    snapshot_id = f"{calendar_end:%Y%m%d}-{source_fingerprint[:12]}"
    snapshot_dir = paths["snapshots"] / snapshot_id

    blocking_issues = []
    warning_issues = []
    if universe["symbol"].duplicated().any():
        blocking_issues.append("universe contains duplicate symbols")
    if instruments["symbol"].duplicated().any():
        blocking_issues.append("instrument file contains duplicate symbols")
    if missing_in_instruments or extra_in_instruments:
        blocking_issues.append("universe and instrument whitelist differ")
    if missing_feature_dirs or missing_raw or missing_normalized:
        blocking_issues.append("one or more whitelisted ETFs are missing data")
    if stale_days != 0:
        blocking_issues.append(f"calendar end {calendar_end.date()} does not match configured end {configured_end.date()}")
    upstream_blocking, upstream_warnings = evaluate_upstream_validation(
        upstream_validation,
        universe_count=len(universe),
        configured_end=configured_end,
        raw_file_count=len(raw_symbols),
        normalized_file_count=len(normalized_symbols),
        external_raw_count=len(external_raw_symbols),
        external_normalized_count=len(external_normalized_symbols),
    )
    blocking_issues.extend(upstream_blocking)
    warning_issues.extend(upstream_warnings)
    if config["data"]["universe_mode"] != "point_in_time":
        warning_issues.append("current ETF snapshot is backfilled through history; survivor bias remains")

    report = {
        "generated_at": now_shanghai().isoformat(),
        "snapshot_id": snapshot_id,
        "source_fingerprint": source_fingerprint,
        "data_valid": not blocking_issues,
        "promotion_eligible": not blocking_issues and not warning_issues,
        "universe_mode": config["data"]["universe_mode"],
        "counts": {
            "universe": len(universe),
            "instruments": len(instruments),
            "feature_directories": sum((features_dir / symbol.lower()).is_dir() for symbol in instrument_symbols),
            "feature_files": len(feature_files),
            "raw_files_total": len(list(raw_dir.glob("*.csv"))),
            "normalized_files_total": len(list(normalized_dir.glob("*.csv"))),
            "pool_external_raw_symbols": len(external_raw_symbols),
            "pool_external_normalized_symbols": len(external_normalized_symbols),
        },
        "calendar": {
            "start": calendar.min().date().isoformat(),
            "end": calendar_end.date().isoformat(),
            "sessions": len(calendar),
            "future_boundary": future_calendar.iloc[len(calendar)].date().isoformat(),
        },
        "coverage": {
            "missing_in_instruments": missing_in_instruments,
            "extra_in_instruments": extra_in_instruments,
            "missing_feature_dirs": missing_feature_dirs,
            "missing_raw": missing_raw,
            "missing_normalized": missing_normalized,
        },
        "upstream_validation": {
            "generated_at": upstream_validation.get("generated_at"),
            "training_ready": upstream_validation.get("training_ready"),
            "issue_count": upstream_validation.get("issue_count"),
            "reported_amount_ratio": upstream_validation.get("reported_amount_ratio"),
            "whitelist_scoped_ready": not upstream_blocking,
        },
        "blocking_issues": blocking_issues,
        "warnings": warning_issues,
    }

    manifest_path = snapshot_dir / "manifest.json"
    if manifest_path.exists():
        frozen_report = read_json(manifest_path)
        if frozen_report.get("source_fingerprint") != source_fingerprint:
            raise RuntimeError(f"snapshot manifest fingerprint mismatch: {manifest_path}")
        report = frozen_report
    else:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        for source in [
            universe_path,
            instruments_path,
            calendar_path,
            future_calendar_path,
            validation_report_path,
        ]:
            shutil.copy2(source, snapshot_dir / source.name)
        write_json_atomic(manifest_path, report)
    return AuditResult(snapshot_id=snapshot_id, snapshot_dir=snapshot_dir, report=report)
