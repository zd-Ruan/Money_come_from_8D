from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import now_shanghai, read_json, sha256_file, sha256_lines, write_json_atomic


AUDIT_SCHEMA_VERSION = "6"
SOURCE_INVENTORY_SCHEMA_VERSION = 1
REQUIRED_QLIB_FIELDS = (
    "open",
    "close",
    "high",
    "low",
    "volume",
    "factor",
    "change",
    "amount",
    "vwap",
    "amount_estimated",
    "paused",
)


@dataclass(frozen=True)
class AuditResult:
    snapshot_id: str
    snapshot_dir: Path
    report: dict[str, Any]


def _future_calendar_contract(
    calendar_path: Path,
    calendar: pd.Series,
    *,
    persist: bool,
) -> tuple[Path, pd.Series, int, str]:
    """Resolve Qlib's future calendar without writing during a dry audit."""

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
        payload = ("\n".join(values) + "\n").encode("utf-8")
        future = pd.Series(pd.to_datetime(values))
        if persist:
            future_path.write_bytes(payload)
        return future_path, future, len(payload), hashlib.sha256(payload).hexdigest()
    return future_path, future, future_path.stat().st_size, sha256_file(future_path)


def ensure_future_calendar_boundary(calendar_path: Path) -> Path:
    calendar = pd.to_datetime(pd.read_csv(calendar_path, header=None).iloc[:, 0])
    future_path, _, _, _ = _future_calendar_contract(
        calendar_path, calendar, persist=True
    )
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


def _scoped_inventory(scope: str, paths: list[Path], root: Path) -> list[dict[str, Any]]:
    """Return a portable, deterministic inventory for independently checking a snapshot."""

    records = []
    for path in sorted(paths):
        records.append(
            {
                "scope": scope,
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_lines([payload])


def _seal_snapshot_manifest(report: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(report)
    sealed.pop("manifest_sha256", None)
    sealed["manifest_sha256"] = _json_sha256(sealed)
    return sealed


def _verify_snapshot_manifest(report: Any, manifest_path: Path) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise RuntimeError(f"snapshot manifest is not a JSON object: {manifest_path}")
    expected = report.get("manifest_sha256")
    unsigned = copy.deepcopy(report)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(expected, str) or len(expected) != 64 or _json_sha256(unsigned) != expected:
        raise RuntimeError(f"snapshot manifest checksum mismatch: {manifest_path}")
    return report


def evaluate_provider_features(
    features_dir: Path,
    instruments: pd.DataFrame,
    calendar: pd.Series,
) -> tuple[list[str], dict[str, Any]]:
    """Validate the daily Qlib float32 contract for every whitelisted symbol."""

    issues: list[str] = []
    missing_fields: dict[str, list[str]] = {}
    malformed_files: list[str] = []
    malformed_symbols: set[str] = set()
    inconsistent_symbols: list[str] = []
    interval_mismatch_symbols: list[str] = []
    calendar_values = pd.DatetimeIndex(calendar).normalize()
    calendar_positions = {timestamp: index for index, timestamp in enumerate(calendar_values)}

    for row in instruments.itertuples(index=False):
        symbol = str(row.symbol).upper()
        symbol_dir = features_dir / symbol.lower()
        missing = [
            field
            for field in REQUIRED_QLIB_FIELDS
            if not (symbol_dir / f"{field}.day.bin").is_file()
        ]
        if missing:
            missing_fields[symbol] = missing

        signatures: dict[str, tuple[int, int]] = {}
        for field in REQUIRED_QLIB_FIELDS:
            path = symbol_dir / f"{field}.day.bin"
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size < 8 or size % 4:
                malformed_files.append(f"{symbol}/{path.name}")
                malformed_symbols.add(symbol)
                continue
            with path.open("rb") as handle:
                raw_header = handle.read(4)
            if len(raw_header) != 4:
                malformed_files.append(f"{symbol}/{path.name}")
                malformed_symbols.add(symbol)
                continue
            start_value = struct.unpack("<f", raw_header)[0]
            if (
                not math.isfinite(start_value)
                or start_value < 0
                or start_value != int(start_value)
            ):
                malformed_files.append(f"{symbol}/{path.name}")
                malformed_symbols.add(symbol)
                continue
            start_index = int(start_value)
            value_count = size // 4 - 1
            if start_index >= len(calendar_values) or start_index + value_count > len(calendar_values):
                malformed_files.append(f"{symbol}/{path.name}")
                malformed_symbols.add(symbol)
                continue
            values = np.fromfile(path, dtype="<f4", count=value_count, offset=4)
            if len(values) != value_count or np.isinf(values).any() or not np.isfinite(values).any():
                malformed_files.append(f"{symbol}/{path.name}")
                malformed_symbols.add(symbol)
                continue
            signatures[field] = (start_index, value_count)

        if signatures and len(set(signatures.values())) != 1:
            inconsistent_symbols.append(symbol)
            continue
        if len(signatures) != len(REQUIRED_QLIB_FIELDS):
            continue

        start_date = pd.Timestamp(row.start_date).normalize()
        end_date = pd.Timestamp(row.end_date).normalize()
        expected_start = calendar_positions.get(start_date)
        expected_end = calendar_positions.get(end_date)
        actual_start, actual_count = next(iter(signatures.values()))
        if (
            expected_start is None
            or expected_end is None
            or expected_end < expected_start
            or actual_start != expected_start
            or actual_count != expected_end - expected_start + 1
        ):
            interval_mismatch_symbols.append(symbol)

    if missing_fields:
        issues.append(
            f"Qlib provider is missing required fields for {len(missing_fields)} ETFs"
        )
    if malformed_files:
        issues.append(f"Qlib provider contains {len(malformed_files)} malformed feature files")
    if inconsistent_symbols:
        issues.append(
            f"Qlib provider has inconsistent field lengths for {len(inconsistent_symbols)} ETFs"
        )
    if interval_mismatch_symbols:
        issues.append(
            "Qlib provider feature ranges do not match instrument/calendar ranges for "
            f"{len(interval_mismatch_symbols)} ETFs"
        )
    return issues, {
        "required_fields": list(REQUIRED_QLIB_FIELDS),
        "missing_fields": missing_fields,
        "malformed_files": malformed_files,
        "inconsistent_length_symbols": inconsistent_symbols,
        "interval_mismatch_symbols": interval_mismatch_symbols,
        "validated_symbols": len(instruments)
        - len(
            set(missing_fields)
            | malformed_symbols
            | set(inconsistent_symbols)
            | set(interval_mismatch_symbols)
        ),
    }


def _frozen_control_records(
    data_files: list[Path],
    data_root: Path,
    provider_files: list[Path],
    provider_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    destinations: set[str] = set()
    for scope, files, root in (
        ("data", data_files, data_root),
        ("qlib_provider", provider_files, provider_root),
    ):
        for path in files:
            destination = path.name
            if destination in destinations:
                raise RuntimeError(f"snapshot control filename collision: {destination}")
            destinations.add(destination)
            records.append(
                {
                    "scope": scope,
                    "source_path": path.relative_to(root).as_posix(),
                    "snapshot_path": destination,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sorted(records, key=lambda record: (record["scope"], record["source_path"]))


def _verify_frozen_controls(snapshot_dir: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        relative = Path(str(record.get("snapshot_path", "")))
        if not relative.name or relative.is_absolute() or len(relative.parts) != 1:
            raise RuntimeError(f"snapshot control path is invalid: {relative}")
        path = snapshot_dir / relative
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"snapshot control file mismatch: {path}")


def _normalise_report_bool(series: pd.Series, *, name: str) -> pd.Series:
    """Parse collector booleans without treating the string 'False' as true."""

    if series.dtype == bool:
        return series
    values = series.fillna("").astype(str).str.strip().str.lower()
    invalid = ~values.isin({"true", "false"})
    if invalid.any():
        raise ValueError(f"{name} contains invalid boolean values")
    return values.eq("true")


def evaluate_corporate_action_collection(
    report: pd.DataFrame,
    actions: pd.DataFrame,
    universe_symbols: set[str],
    cache_dir: Path,
) -> tuple[list[str], dict[str, Any], list[Path]]:
    """Validate complete-universe publication and every referenced source hash."""

    issues: list[str] = []
    required_report = {
        "symbol",
        "error",
        "full_universe_scope",
        "published",
        "cache_sha256",
    }
    required_actions = {
        "symbol",
        "record_date",
        "ex_date",
        "cash_payment_date",
        "cash_dividend_per_old_share",
        "share_ratio",
        "fractional_share_treatment",
        "source_url",
        "source_sha256",
    }
    if missing := required_report - set(report.columns):
        issues.append(f"corporate-action report lacks columns: {sorted(missing)}")
    if missing := required_actions - set(actions.columns):
        issues.append(f"corporate-action table lacks columns: {sorted(missing)}")
    if issues:
        return issues, {
            "report_rows": len(report),
            "events": len(actions),
            "source_cache_files": 0,
            "complete_universe": False,
        }, []

    symbols = report["symbol"].fillna("").astype(str).str.strip().str.upper()
    if symbols.duplicated().any() or set(symbols) != universe_symbols:
        issues.append("corporate-action report does not cover the frozen universe exactly once")
    errors = report["error"].fillna("").astype(str).str.strip()
    if not errors.eq("").all():
        issues.append("corporate-action report contains collection failures")
    try:
        full_scope = _normalise_report_bool(report["full_universe_scope"], name="full_universe_scope")
        published = _normalise_report_bool(report["published"], name="published")
        if not full_scope.all() or not published.all():
            issues.append("corporate-action collection was not published for the full universe")
    except ValueError as exc:
        issues.append(str(exc))

    action_symbols = actions["symbol"].fillna("").astype(str).str.strip().str.upper()
    if not set(action_symbols).issubset(universe_symbols):
        issues.append("corporate-action table contains symbols outside the frozen universe")
    if actions.duplicated(["symbol", "ex_date"]).any():
        issues.append("corporate-action table contains duplicate symbol/ex-date events")

    report_hashes = dict(zip(symbols, report["cache_sha256"].fillna("").astype(str)))
    cache_files: list[Path] = []
    bad_cache: list[str] = []
    for symbol in sorted(universe_symbols):
        cache_path = cache_dir / f"{symbol}.html"
        cache_files.append(cache_path)
        expected = report_hashes.get(symbol, "")
        if not cache_path.is_file() or len(expected) != 64 or sha256_file(cache_path) != expected:
            bad_cache.append(symbol)
    if bad_cache:
        issues.append(
            f"corporate-action source cache is missing or hash-mismatched for {len(bad_cache)} ETFs"
        )

    if not actions.empty:
        declared = actions["source_sha256"].fillna("").astype(str)
        actual = action_symbols.map(report_hashes).fillna("")
        if not declared.eq(actual).all():
            issues.append("corporate-action event provenance does not match the frozen source cache")

    summary = {
        "report_rows": len(report),
        "events": len(actions),
        "source_cache_files": sum(path.is_file() for path in cache_files),
        "complete_universe": not issues,
    }
    return issues, summary, cache_files


def evaluate_upstream_validation(
    report: dict[str, Any],
    universe_count: int,
    configured_end: pd.Timestamp,
    raw_file_count: int,
    normalized_file_count: int,
    external_raw_count: int,
    external_normalized_count: int,
    max_stale_days: int = 0,
) -> tuple[list[str], list[str]]:
    blocking: list[str] = []
    warnings: list[str] = []
    if int(report.get("universe_count", -1)) != universe_count:
        blocking.append("upstream validation universe count does not match the active whitelist")

    expected_end = configured_end.date().isoformat()
    minimum_latest_date = pd.Timestamp(configured_end) - pd.Timedelta(days=int(max_stale_days))
    actual_min = pd.to_datetime(report.get("min_latest_date", ""), errors="coerce")
    if actual_min < minimum_latest_date or report.get("max_latest_date") != expected_end:
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
            "upstream total-file count warnings are fully explained by retained pool-external "
            "cache files; per-symbol whitelist coverage was verified separately"
        )
    elif issue_messages:
        blocking.append("upstream validation file-count issues are not explained by pool-external cache files")
    elif not bool(report.get("training_ready", False)):
        blocking.append("upstream validation is not training-ready for an unexplained reason")
    return blocking, warnings


def audit_and_snapshot(config: dict[str, Any], *, persist: bool = True) -> AuditResult:
    """Audit the provider and optionally persist its content-addressed snapshot.

    ``persist=False`` is the pre-claim path: it computes the same source
    fingerprint and report while creating neither ``day_future.txt`` nor a
    snapshot directory.
    """

    paths = {key: Path(value) for key, value in config["paths"].items()}
    universe_path = paths["universe"]
    instruments_path = paths["instruments"]
    provider = paths["qlib_provider"]
    validation_report_path = paths["validation_report"]
    calendar_path = provider / "calendars" / "day.txt"
    features_dir = provider / "features"
    raw_dir = universe_path.parent / "raw"
    normalized_dir = universe_path.parent / "normalized"
    corporate_actions_path = universe_path.parent / "corporate_actions.csv"
    corporate_action_report_path = universe_path.parent / "corporate_action_report.csv"
    corporate_action_cache_dir = universe_path.parent / "corporate_action_cache"

    required = [
        universe_path,
        instruments_path,
        calendar_path,
        features_dir,
        validation_report_path,
        corporate_actions_path,
        corporate_action_report_path,
        corporate_action_cache_dir,
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise FileNotFoundError(f"required data paths are missing: {missing_required}")
    calendar = pd.to_datetime(pd.read_csv(calendar_path, header=None).iloc[:, 0])
    (
        future_calendar_path,
        future_calendar,
        future_calendar_size,
        future_calendar_sha256,
    ) = _future_calendar_contract(calendar_path, calendar, persist=persist)

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

    corporate_action_report = pd.read_csv(corporate_action_report_path)
    corporate_actions = pd.read_csv(corporate_actions_path)
    action_issues, action_summary, action_cache_files = evaluate_corporate_action_collection(
        corporate_action_report,
        corporate_actions,
        universe_symbols,
        corporate_action_cache_dir,
    )

    configured_end = pd.Timestamp(config["data"]["end_date"])
    calendar_end = calendar.max()
    upstream_validation = read_json(validation_report_path, {})

    provider_issues, provider_summary = evaluate_provider_features(
        features_dir, instruments, calendar
    )

    feature_files = [path for symbol in instrument_symbols for path in (features_dir / symbol.lower()).glob("*.bin")]
    feature_inventory = _file_inventory(feature_files, provider)
    raw_files = [raw_dir / f"{symbol.lower()}.csv" for symbol in sorted(universe_symbols)]
    raw_inventory = _file_inventory([path for path in raw_files if path.is_file()], universe_path.parent)
    normalized_files = [
        normalized_dir / f"{symbol.lower()}.csv" for symbol in sorted(universe_symbols)
    ]
    normalized_inventory = _file_inventory(
        [path for path in normalized_files if path.is_file()], universe_path.parent
    )
    action_inventory = _file_inventory(
        [path for path in action_cache_files if path.is_file()], universe_path.parent
    )
    control_files = [
        universe_path,
        validation_report_path,
        corporate_actions_path,
        corporate_action_report_path,
    ]
    provider_control_files = [instruments_path, calendar_path]
    frozen_controls = _frozen_control_records(
        control_files,
        universe_path.parent,
        provider_control_files,
        provider,
    )
    frozen_controls.append(
        {
            "scope": "qlib_provider",
            "source_path": future_calendar_path.relative_to(provider).as_posix(),
            "snapshot_path": future_calendar_path.name,
            "size": future_calendar_size,
            "sha256": future_calendar_sha256,
        }
    )
    frozen_controls.sort(key=lambda record: (record["scope"], record["source_path"]))
    future_calendar_inventory = {
        "scope": "qlib_provider",
        "path": future_calendar_path.relative_to(provider).as_posix(),
        "size": future_calendar_size,
        "sha256": future_calendar_sha256,
    }
    source_inventory = {
        "schema_version": SOURCE_INVENTORY_SCHEMA_VERSION,
        "files": [
            *_scoped_inventory("data", control_files, universe_path.parent),
            *_scoped_inventory("qlib_provider", provider_control_files, provider),
            future_calendar_inventory,
            *_scoped_inventory("qlib_provider", feature_files, provider),
            *_scoped_inventory(
                "data", [path for path in raw_files if path.is_file()], universe_path.parent
            ),
            *_scoped_inventory(
                "data",
                [path for path in normalized_files if path.is_file()],
                universe_path.parent,
            ),
            *_scoped_inventory(
                "data", [path for path in action_cache_files if path.is_file()], universe_path.parent
            ),
        ],
    }
    source_fingerprint = sha256_lines(
        [
            AUDIT_SCHEMA_VERSION,
            _json_sha256(
                {
                    "configured_end": configured_end.date().isoformat(),
                    "universe_mode": config["data"]["universe_mode"],
                }
            ),
            sha256_file(universe_path),
            sha256_file(instruments_path),
            sha256_file(calendar_path),
            future_calendar_sha256,
            sha256_file(validation_report_path),
            sha256_file(corporate_actions_path),
            sha256_file(corporate_action_report_path),
            *feature_inventory,
            *raw_inventory,
            *normalized_inventory,
            *action_inventory,
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
    blocking_issues.extend(provider_issues)
    if calendar_end < configured_end:
        blocking_issues.append(
            f"provider calendar end {calendar_end.date()} is before configured end {configured_end.date()}"
        )
    upstream_blocking, upstream_warnings = evaluate_upstream_validation(
        upstream_validation,
        universe_count=len(universe),
        configured_end=calendar_end,
        raw_file_count=len(raw_symbols),
        normalized_file_count=len(normalized_symbols),
        external_raw_count=len(external_raw_symbols),
        external_normalized_count=len(external_normalized_symbols),
        max_stale_days=int(config["data"].get("max_stale_days", 0)),
    )
    blocking_issues.extend(upstream_blocking)
    blocking_issues.extend(action_issues)
    warning_issues.extend(upstream_warnings)
    if config["data"]["universe_mode"] != "point_in_time":
        warning_issues.append("current ETF snapshot is backfilled through history; survivor bias remains")

    report = {
        "generated_at": now_shanghai().isoformat(),
        "snapshot_id": snapshot_id,
        "source_fingerprint": source_fingerprint,
        "audit_identity": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "configured_end": configured_end.date().isoformat(),
            "universe_mode": config["data"]["universe_mode"],
        },
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
            "corporate_action_events": len(corporate_actions),
            "corporate_action_source_files": action_summary["source_cache_files"],
        },
        "calendar": {
            "start": calendar.min().date().isoformat(),
            "end": calendar_end.date().isoformat(),
            "requested_end": configured_end.date().isoformat(),
            "sessions": len(calendar),
            "future_boundary": future_calendar.iloc[len(calendar)].date().isoformat(),
        },
        "coverage": {
            "missing_in_instruments": missing_in_instruments,
            "extra_in_instruments": extra_in_instruments,
            "missing_feature_dirs": missing_feature_dirs,
            "missing_raw": missing_raw,
            "missing_normalized": missing_normalized,
            "provider_features": provider_summary,
        },
        "upstream_validation": {
            "generated_at": upstream_validation.get("generated_at"),
            "training_ready": upstream_validation.get("training_ready"),
            "issue_count": upstream_validation.get("issue_count"),
            "reported_amount_ratio": upstream_validation.get("reported_amount_ratio"),
            "whitelist_scoped_ready": not upstream_blocking,
        },
        "corporate_actions": action_summary,
        "source_inventory": {
            "path": "source_inventory.json",
            "file_count": len(source_inventory["files"]),
        },
        "frozen_controls": {"files": frozen_controls},
        "blocking_issues": blocking_issues,
        "warnings": warning_issues,
    }

    inventory_payload = json.dumps(
        source_inventory,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    report["source_inventory"]["sha256"] = hashlib.sha256(inventory_payload).hexdigest()
    manifest_path = snapshot_dir / "manifest.json"
    if not persist:
        report = _seal_snapshot_manifest(report)
    elif manifest_path.exists():
        frozen_report = _verify_snapshot_manifest(read_json(manifest_path), manifest_path)
        inventory_path = snapshot_dir / "source_inventory.json"
        if read_json(inventory_path) != source_inventory:
            raise RuntimeError(f"snapshot source inventory mismatch: {inventory_path}")
        inventory_sha256 = sha256_file(inventory_path)
        if frozen_report.get("source_inventory", {}).get("sha256") != inventory_sha256:
            raise RuntimeError(f"snapshot source inventory digest mismatch: {inventory_path}")
        if frozen_report.get("frozen_controls", {}).get("files") != frozen_controls:
            raise RuntimeError(f"snapshot frozen-control inventory mismatch: {manifest_path}")
        _verify_frozen_controls(snapshot_dir, frozen_controls)
        expected_report = copy.deepcopy(report)
        expected_report["generated_at"] = frozen_report.get("generated_at")
        expected_report["source_inventory"]["sha256"] = inventory_sha256
        expected_report = _seal_snapshot_manifest(expected_report)
        if frozen_report != expected_report:
            raise RuntimeError(f"snapshot manifest differs from the current audit: {manifest_path}")
        report = frozen_report
    else:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        for source in [*control_files, *provider_control_files, future_calendar_path]:
            shutil.copy2(source, snapshot_dir / source.name)
        _verify_frozen_controls(snapshot_dir, frozen_controls)
        inventory_path = snapshot_dir / "source_inventory.json"
        write_json_atomic(inventory_path, source_inventory)
        report["source_inventory"]["sha256"] = sha256_file(inventory_path)
        report = _seal_snapshot_manifest(report)
        write_json_atomic(manifest_path, report)
    return AuditResult(snapshot_id=snapshot_id, snapshot_dir=snapshot_dir, report=report)
