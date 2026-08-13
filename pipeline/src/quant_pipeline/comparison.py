"""Paired, fail-closed comparison of completed pipeline runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .integrity import combine_runtime_code_sha256, verify_artifact_checksums
from .io import now_shanghai, read_json, sha256_file, write_json_atomic
from .metrics import evaluation_frame, hac_t_stat


COMPARISON_SCHEMA_VERSION = 1
DEFAULT_HAC_THRESHOLD = 1.96
DEFAULT_HAC_MAX_LAG = 5
DEFAULT_COMPARISONS_ROOT = Path(__file__).resolve().parents[2] / "comparisons"
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def _incomparable(
    baseline_run: Path,
    candidate_run: Path,
    reasons: list[str],
    *,
    baseline_id: str | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_status": "completed",
        "status": "incomparable",
        "baseline_run_id": baseline_id or baseline_run.name,
        "candidate_run_id": candidate_id or candidate_run.name,
        "comparable": False,
        "reasons": list(dict.fromkeys(reasons)) or ["run comparability could not be established"],
        "deltas": None,
        "decision": {
            "claim": "No improvement conclusion is permitted because the runs are not comparable.",
            "criteria": {},
        },
    }


def _required_json(run_dir: Path, filename: str) -> dict[str, Any] | list[Any]:
    path = run_dir / filename
    value = read_json(path)
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{filename} must contain a JSON object or array")
    return value


def _source_identity(manifest: dict[str, Any]) -> tuple[str, str]:
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("manifest is missing data identity metadata")
    nested_snapshot_id = data.get("snapshot_id")
    top_snapshot_id = manifest.get("snapshot_id")
    if not isinstance(nested_snapshot_id, str) or not nested_snapshot_id:
        raise ValueError("manifest is missing data.snapshot_id")
    if not isinstance(top_snapshot_id, str) or not top_snapshot_id:
        raise ValueError("manifest is missing top-level snapshot_id")
    if nested_snapshot_id != top_snapshot_id:
        raise ValueError("manifest data.snapshot_id conflicts with top-level snapshot_id")

    fingerprint = data.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(fingerprint):
        raise ValueError("source_fingerprint must be a 64-character hexadecimal digest")
    return fingerprint.lower(), nested_snapshot_id


def _code_identity(manifest: dict[str, Any], role: str) -> str:
    code = manifest.get("code")
    if not isinstance(code, dict):
        raise ValueError(f"{role} manifest is missing code identity metadata")
    components = {}
    for field in ("pipeline_source_sha256", "qlib_package_sha256", "runtime_code_sha256"):
        digest = code.get(field)
        if not isinstance(digest, str) or not _FINGERPRINT_PATTERN.fullmatch(digest):
            raise ValueError(
                f"{role} manifest code.{field} must be a 64-character hexadecimal digest"
            )
        components[field] = digest.lower()
    expected = combine_runtime_code_sha256(
        components["pipeline_source_sha256"], components["qlib_package_sha256"]
    )
    if components["runtime_code_sha256"] != expected:
        raise ValueError(f"{role} manifest runtime code identity does not match its components")
    return expected


def _verify_run_integrity(run_dir: Path, manifest: dict[str, Any], role: str) -> None:
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError(f"{role} manifest is missing integrity metadata")
    checksum_manifest = integrity.get("checksum_manifest")
    if not isinstance(checksum_manifest, str) or not checksum_manifest or "\\" in checksum_manifest:
        raise ValueError(f"{role} manifest integrity.checksum_manifest is invalid")
    raw_checksum_path = Path(checksum_manifest)
    if raw_checksum_path.is_absolute():
        raise ValueError(f"{role} checksum manifest path must be relative to the run directory")
    checksum_path = (run_dir / raw_checksum_path).resolve()
    try:
        checksum_path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"{role} checksum manifest resolves outside the run directory") from exc
    if checksum_path.is_symlink():
        raise ValueError(f"{role} checksum manifest must not be a symbolic link")
    if not checksum_path.is_file():
        raise ValueError(f"{role} checksum manifest is missing: {checksum_manifest}")

    declared_checksum = integrity.get("checksum_sha256")
    if not isinstance(declared_checksum, str) or not _FINGERPRINT_PATTERN.fullmatch(
        declared_checksum
    ):
        raise ValueError(f"{role} manifest integrity.checksum_sha256 is invalid")
    actual_checksum = sha256_file(checksum_path)
    if actual_checksum != declared_checksum.lower():
        raise ValueError(f"{role} checksum manifest SHA-256 does not match manifest metadata")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("artifact_checksums") != checksum_manifest:
        raise ValueError(f"{role} manifest artifact checksum path declarations do not match")
    if integrity.get("verified") is not True:
        raise ValueError(f"{role} manifest does not record successful artifact verification")

    verification = verify_artifact_checksums(
        run_dir,
        raw_checksum_path,
        check_unexpected=True,
    )
    artifact_count = integrity.get("artifact_count")
    if (
        isinstance(artifact_count, bool)
        or not isinstance(artifact_count, int)
        or artifact_count != verification["expected_count"]
    ):
        raise ValueError(f"{role} manifest integrity.artifact_count is invalid")
    if not verification["valid"]:
        details = []
        for category in ("missing", "modified", "unexpected"):
            paths = verification[category]
            if paths:
                rendered = ", ".join(paths[:10])
                if len(paths) > 10:
                    rendered += f", ... ({len(paths)} total)"
                details.append(f"{category}: {rendered}")
        raise ValueError(
            f"{role} artifact checksum verification failed"
            + (f" ({'; '.join(details)})" if details else "")
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_candidate_catalog(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    features = config.get("features")
    if not isinstance(features, dict) or features.get("mode") != "alpha158_plus_original":
        raise ValueError("candidate features.mode must be alpha158_plus_original")
    families = features.get("families")
    if not isinstance(families, list) or not families or not all(isinstance(item, str) and item for item in families):
        raise ValueError("candidate features.families must be a non-empty list of names")
    if families != list(dict.fromkeys(families)):
        raise ValueError("candidate features.families must not contain duplicates")

    catalog = manifest.get("factor_catalog")
    if not isinstance(catalog, dict):
        raise ValueError("candidate manifest is missing frozen factor_catalog")
    digest = catalog.get("sha256")
    if not isinstance(digest, str) or not _FINGERPRINT_PATTERN.fullmatch(digest):
        raise ValueError("candidate factor_catalog has an invalid sha256")
    unsigned = {key: value for key, value in catalog.items() if key != "sha256"}
    expected = hashlib.sha256(_canonical_json(unsigned).encode("ascii")).hexdigest()
    if digest.lower() != expected:
        raise ValueError("candidate factor_catalog sha256 does not match its content")
    catalog_families = catalog.get("families")
    if catalog_families != list(dict.fromkeys(families)):
        raise ValueError("candidate feature families do not match frozen factor_catalog")
    factors = catalog.get("factors")
    if not isinstance(factors, list) or not factors:
        raise ValueError("candidate factor_catalog has no frozen factors")
    factor_names: set[str] = set()
    for position, factor in enumerate(factors):
        if not isinstance(factor, dict):
            raise ValueError(f"candidate factor_catalog factor {position} is not an object")
        required = {"name", "family", "expression", "direction", "hypothesis", "lookback"}
        if required - set(factor):
            raise ValueError(f"candidate factor_catalog factor {position} is incomplete")
        name = factor["name"]
        if not isinstance(name, str) or not name or name in factor_names:
            raise ValueError("candidate factor_catalog factor names must be unique non-empty strings")
        factor_names.add(name)
        if factor["family"] not in catalog_families:
            raise ValueError(f"candidate factor_catalog factor {name} is outside selected families")


def _validate_feature_roles(
    baseline_config: dict[str, Any],
    baseline_manifest: dict[str, Any],
    candidate_config: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> None:
    baseline_features = baseline_config.get("features")
    if baseline_features is None:
        baseline_mode = "alpha158"
    elif isinstance(baseline_features, dict):
        baseline_mode = baseline_features.get("mode")
    else:
        raise ValueError("baseline features must be an object")
    if baseline_mode != "alpha158":
        raise ValueError("baseline features.mode must be alpha158")
    if isinstance(baseline_features, dict) and baseline_features.get("families") not in (None, []):
        raise ValueError("Alpha158 baseline feature families must be empty")
    if baseline_manifest.get("factor_catalog") not in (None, {}):
        raise ValueError("Alpha158 baseline must not contain a factor_catalog")
    _validate_candidate_catalog(candidate_config, candidate_manifest)


def _normalized_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.pop("_meta", None)
    normalized.pop("features", None)

    project = normalized.get("project")
    if isinstance(project, dict):
        project.pop("name", None)
        project.pop("description", None)
    report = normalized.get("report")
    if isinstance(report, dict):
        report.pop("title", None)
    paths = normalized.get("paths")
    if isinstance(paths, dict):
        paths.pop("runs", None)
        paths.pop("registry", None)
    return normalized


def _value_differences(left: Any, right: Any, path: str = "config") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_value_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_value_differences(left_item, right_item, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]


def _validate_datetime_index(frame: pd.DataFrame, artifact: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame = frame.copy()
            frame.index = pd.DatetimeIndex(frame.index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{artifact} index must be datetime-like") from exc
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{artifact} index must be unique and increasing")
    return frame


def _read_parquet(path: Path, artifact: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"could not read {artifact}: {type(exc).__name__}") from exc
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{artifact} must contain a dataframe")
    return frame


def _load_paired_artifacts(
    baseline_run: Path,
    candidate_run: Path,
    baseline_config: dict[str, Any],
    candidate_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    base_slippage = baseline_config.get("execution", {}).get("base_slippage_bps_per_side")
    candidate_slippage = candidate_config.get("execution", {}).get("base_slippage_bps_per_side")
    if isinstance(base_slippage, bool) or not isinstance(base_slippage, (int, float)):
        raise ValueError("baseline base slippage is missing or invalid")
    if base_slippage != candidate_slippage or float(base_slippage) < 0 or not float(base_slippage).is_integer():
        raise ValueError("base slippage differs or is invalid")
    scenario = f"slippage_{int(base_slippage):02d}bps"

    baseline_raw = _read_parquet(
        baseline_run / "backtests" / scenario / "report.parquet", "baseline report"
    )
    candidate_raw = _read_parquet(
        candidate_run / "backtests" / scenario / "report.parquet", "candidate report"
    )
    baseline_raw = _validate_datetime_index(baseline_raw, "baseline report")
    candidate_raw = _validate_datetime_index(candidate_raw, "candidate report")
    if not baseline_raw.index.equals(candidate_raw.index):
        raise ValueError("base report test dates differ")
    for frame, label in ((baseline_raw, "baseline report"), (candidate_raw, "candidate report")):
        missing = {"return", "cost", "bench"} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns {sorted(missing)}")
        values = frame[["return", "cost", "bench"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite return data")
    if not np.allclose(
        baseline_raw["bench"].to_numpy(dtype=float),
        candidate_raw["bench"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("base report benchmark returns differ")
    baseline_aligned = evaluation_frame(baseline_raw)
    candidate_aligned = evaluation_frame(candidate_raw)
    for frame, label in ((baseline_aligned, "baseline report"), (candidate_aligned, "candidate report")):
        if ((1.0 + frame[["strategy_net", "benchmark"]]) <= 0).any().any():
            raise ValueError(f"{label} contains a return that makes wealth non-positive")

    baseline_signal = _validate_datetime_index(
        _read_parquet(baseline_run / "signal_metrics.parquet", "baseline signal_metrics"),
        "baseline signal_metrics",
    )
    candidate_signal = _validate_datetime_index(
        _read_parquet(candidate_run / "signal_metrics.parquet", "candidate signal_metrics"),
        "candidate signal_metrics",
    )
    if not baseline_signal.index.equals(candidate_signal.index):
        raise ValueError("signal metric test dates differ")
    for metric in ("ic", "rank_ic"):
        if metric not in baseline_signal or metric not in candidate_signal:
            raise ValueError(f"signal_metrics is missing {metric}")
        if not baseline_signal[metric].isna().equals(candidate_signal[metric].isna()):
            raise ValueError(f"paired {metric} missing-value masks differ")
        paired = pd.concat([baseline_signal[metric], candidate_signal[metric]], axis=1).dropna()
        if paired.empty or not np.isfinite(paired.to_numpy(dtype=float)).all():
            raise ValueError(f"paired {metric} values are empty or non-finite")

    baseline_folds = _required_json(baseline_run, "folds.json")
    candidate_folds = _required_json(candidate_run, "folds.json")
    if not isinstance(baseline_folds, list) or not all(isinstance(item, dict) for item in baseline_folds):
        raise ValueError("baseline folds.json must be an array of objects")
    if baseline_folds != candidate_folds:
        raise ValueError("rolling fold dates or definitions differ")
    if not baseline_folds:
        raise ValueError("rolling folds are empty")
    return baseline_raw, baseline_aligned, candidate_aligned, baseline_signal, candidate_signal, baseline_folds


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _paired_stat(candidate: pd.Series, baseline: pd.Series, max_lag: int) -> dict[str, Any]:
    paired = pd.concat([candidate.rename("candidate"), baseline.rename("baseline")], axis=1).dropna()
    difference = paired["candidate"] - paired["baseline"]
    return {
        "observations": len(difference),
        "baseline_mean": float(paired["baseline"].mean()),
        "candidate_mean": float(paired["candidate"].mean()),
        "mean_difference": float(difference.mean()),
        "hac_t_stat": _finite_or_none(hac_t_stat(difference, max_lag=max_lag)),
    }


def _terminal_relative_wealth(frame: pd.DataFrame) -> float:
    strategy_wealth = float((1.0 + frame["strategy_net"]).prod())
    benchmark_wealth = float((1.0 + frame["benchmark"]).prod())
    if not math.isfinite(strategy_wealth) or not math.isfinite(benchmark_wealth):
        raise ValueError("terminal strategy and benchmark wealth must be finite")
    if strategy_wealth <= 0 or benchmark_wealth <= 0:
        raise ValueError("terminal strategy and benchmark wealth must be positive")
    return strategy_wealth / benchmark_wealth


def _portfolio_summary_value(
    portfolio: dict[str, Any],
    field: str,
    expected: float,
    artifact: str,
) -> None:
    value = portfolio.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{artifact} portfolio.{field} is missing or non-finite")
    if not math.isclose(float(value), expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{artifact} portfolio.{field} does not match its independent report")


def _validate_fold_summary(
    summary: dict[str, Any],
    fold: dict[str, Any],
    raw: pd.DataFrame,
    aligned: pd.DataFrame,
    artifact: str,
) -> bool:
    for field in (
        "fold",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
        "purge_bars",
    ):
        if summary.get(field) != fold.get(field):
            raise ValueError(f"{artifact} {field} does not match folds.json")

    portfolio = summary.get("portfolio")
    if not isinstance(portfolio, dict):
        raise ValueError(f"{artifact} is missing portfolio metadata")
    if portfolio.get("reset_cash") is not True:
        raise ValueError(f"{artifact} portfolio is not an independent cash-reset backtest")
    complete = portfolio.get("complete_for_gate")
    if not isinstance(complete, bool):
        raise ValueError(f"{artifact} portfolio.complete_for_gate must be boolean")

    raw_start = raw.index.min().date().isoformat()
    raw_end = raw.index.max().date().isoformat()
    evaluation_start = aligned.index.min().date().isoformat()
    evaluation_end = aligned.index.max().date().isoformat()
    expected_boundaries = {
        "start": raw_start,
        "initial_execution_date": raw_start,
        "evaluation_start_date": evaluation_start,
        "end": raw_end,
        "evaluation_end_date": evaluation_end,
    }
    for field, expected in expected_boundaries.items():
        if portfolio.get(field) != expected:
            raise ValueError(f"{artifact} portfolio.{field} does not match its independent report")
    days = portfolio.get("days")
    if isinstance(days, bool) or not isinstance(days, int) or days != len(aligned):
        raise ValueError(f"{artifact} portfolio.days does not match its independent report")

    strategy = float((1.0 + aligned["strategy_net"]).prod() - 1.0)
    benchmark = float((1.0 + aligned["benchmark"]).prod() - 1.0)
    relative_excess = (1.0 + strategy) / (1.0 + benchmark) - 1.0
    _portfolio_summary_value(portfolio, "net_cumulative_return", strategy, artifact)
    _portfolio_summary_value(portfolio, "benchmark_cumulative_return", benchmark, artifact)
    _portfolio_summary_value(portfolio, "excess_cumulative_return", relative_excess, artifact)
    return complete


def _load_independent_fold(
    run_dir: Path,
    fold: dict[str, Any],
    fold_id: int,
    role: str,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    fold_root = run_dir / "folds" / f"fold_{fold_id:02d}"
    artifact = f"{role} fold {fold_id}"
    summary = _required_json(fold_root, "summary.json")
    if not isinstance(summary, dict):
        raise ValueError(f"{artifact} summary.json must contain an object")
    raw = _validate_datetime_index(
        _read_parquet(fold_root / "backtest" / "report.parquet", f"{artifact} report"),
        f"{artifact} report",
    )
    missing = {"return", "cost", "bench"} - set(raw.columns)
    if missing:
        raise ValueError(f"{artifact} report is missing columns {sorted(missing)}")
    if not np.isfinite(raw[["return", "cost", "bench"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{artifact} report contains non-finite return data")
    aligned = evaluation_frame(raw)
    if ((1.0 + aligned[["strategy_net", "benchmark"]]) <= 0).any().any():
        raise ValueError(f"{artifact} report contains a return that makes wealth non-positive")
    complete = _validate_fold_summary(summary, fold, raw, aligned, artifact)
    return raw, aligned, complete


def _fold_comparison(
    baseline_run: Path,
    candidate_run: Path,
    folds: list[dict[str, Any]],
    continuous_report_dates: pd.DatetimeIndex,
    signal_dates: pd.DatetimeIndex,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    records = []
    calendar = pd.DatetimeIndex(continuous_report_dates.union(signal_dates)).sort_values()
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("paired report and signal dates do not form a valid trading calendar")
    wins = losses = ties = 0
    complete_folds = residual_folds = 0
    previous_signal_end: pd.Timestamp | None = None
    previous_evaluation_end: pd.Timestamp | None = None
    for position, fold in enumerate(folds, start=1):
        fold_id = fold.get("fold", position)
        if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 1:
            raise ValueError(f"fold {position} has an invalid fold identifier")
        try:
            start = pd.Timestamp(fold["test_start"])
            end = pd.Timestamp(fold["test_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"fold {fold_id} has invalid test dates") from exc
        if start > end:
            raise ValueError(f"fold {fold_id} test_start is after test_end")
        if previous_signal_end is not None and start <= previous_signal_end:
            raise ValueError(f"fold {fold_id} overlaps or is out of chronological order")
        previous_signal_end = end

        fold_signal_dates = signal_dates[(signal_dates >= start) & (signal_dates <= end)]
        if fold_signal_dates.empty:
            raise ValueError(f"fold {fold_id} has no paired signal dates")
        first_signal_position = calendar.get_loc(fold_signal_dates.min())
        last_signal_position = calendar.get_loc(fold_signal_dates.max())
        if not isinstance(first_signal_position, int) or not isinstance(last_signal_position, int):
            raise ValueError(f"fold {fold_id} signal dates are ambiguous")
        if first_signal_position + 1 >= len(calendar) or last_signal_position + 2 >= len(calendar):
            raise ValueError(f"fold {fold_id} lacks t+1 execution or t+2 realization dates")
        expected_raw_start = calendar[first_signal_position + 1]
        expected_raw_end = calendar[last_signal_position + 2]

        baseline_raw, baseline, baseline_complete = _load_independent_fold(
            baseline_run, fold, fold_id, "baseline"
        )
        candidate_raw, candidate, candidate_complete = _load_independent_fold(
            candidate_run, fold, fold_id, "candidate"
        )
        if not baseline_raw.index.equals(candidate_raw.index):
            raise ValueError(f"fold {fold_id} independent report dates differ")
        if baseline_raw.index.min() != expected_raw_start or baseline_raw.index.max() != expected_raw_end:
            raise ValueError(
                f"fold {fold_id} independent report boundaries do not match signal t+1/t+2 timing"
            )
        if not baseline.index.equals(candidate.index):
            raise ValueError(f"fold {fold_id} independent evaluation dates differ")
        if not np.allclose(
            baseline_raw["bench"].to_numpy(dtype=float),
            candidate_raw["bench"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"fold {fold_id} independent benchmark returns differ")
        if baseline_complete != candidate_complete:
            raise ValueError(f"fold {fold_id} complete_for_gate metadata differs")
        if previous_evaluation_end is not None and baseline.index.min() <= previous_evaluation_end:
            raise ValueError(f"fold {fold_id} independent evaluation dates overlap")
        previous_evaluation_end = baseline.index.max()

        baseline_wealth = float((1.0 + baseline["strategy_net"]).prod())
        candidate_wealth = float((1.0 + candidate["strategy_net"]).prod())
        benchmark_wealth = float((1.0 + baseline["benchmark"]).prod())
        if not all(math.isfinite(value) and value > 0 for value in (baseline_wealth, candidate_wealth, benchmark_wealth)):
            raise ValueError(f"fold {fold_id} terminal wealth is not finite")
        baseline_relative = baseline_wealth / benchmark_wealth
        candidate_relative = candidate_wealth / benchmark_wealth
        difference = candidate_relative - baseline_relative
        if difference > tie_tolerance:
            outcome = "win"
        elif difference < -tie_tolerance:
            outcome = "loss"
        else:
            outcome = "tie"
        if baseline_complete:
            complete_folds += 1
            wins += outcome == "win"
            losses += outcome == "loss"
            ties += outcome == "tie"
        else:
            residual_folds += 1
        records.append(
            {
                "fold": fold_id,
                "signal_start": fold_signal_dates.min().date().isoformat(),
                "signal_end": fold_signal_dates.max().date().isoformat(),
                "start": baseline.index.min().date().isoformat(),
                "end": baseline.index.max().date().isoformat(),
                "observations": len(baseline),
                "complete_for_gate": baseline_complete,
                "included_in_win_rate": baseline_complete,
                "baseline_terminal_wealth": baseline_wealth,
                "candidate_terminal_wealth": candidate_wealth,
                "benchmark_terminal_wealth": benchmark_wealth,
                "baseline_terminal_relative_wealth": baseline_relative,
                "candidate_terminal_relative_wealth": candidate_relative,
                "terminal_relative_wealth_difference": difference,
                "outcome": outcome,
            }
        )
    return {
        "folds": len(records),
        "complete_folds": complete_folds,
        "residual_folds": residual_folds,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / complete_folds if complete_folds else None,
        "records": records,
    }


def compare_completed_runs(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    *,
    hac_threshold: float = DEFAULT_HAC_THRESHOLD,
    hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> dict[str, Any]:
    """Compare a completed Alpha158 baseline with one frozen-candidate run.

    Comparability is fail-closed.  An ``improved`` verdict requires statistically
    significant paired performance, majority fold wins, non-negative IC and
    RankIC deltas, and significance in at least one of the two signal metrics.
    """

    baseline_run = Path(baseline_run_dir).resolve()
    candidate_run = Path(candidate_run_dir).resolve()
    if isinstance(hac_threshold, bool) or not isinstance(hac_threshold, (int, float)):
        raise ValueError("hac_threshold must be finite and positive")
    if not math.isfinite(float(hac_threshold)) or hac_threshold <= 0:
        raise ValueError("hac_threshold must be finite and positive")
    if isinstance(hac_max_lag, bool) or not isinstance(hac_max_lag, int) or hac_max_lag < 0:
        raise ValueError("hac_max_lag must be a non-negative integer")
    if baseline_run == candidate_run:
        return _incomparable(baseline_run, candidate_run, ["baseline and candidate directories are identical"])

    reasons: list[str] = []
    baseline_manifest: dict[str, Any] = {}
    candidate_manifest: dict[str, Any] = {}
    baseline_config: dict[str, Any] = {}
    candidate_config: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    candidate_metrics: dict[str, Any] = {}
    try:
        if not baseline_run.is_dir() or not candidate_run.is_dir():
            raise ValueError("both run directories must exist")
        baseline_manifest_value = _required_json(baseline_run, "manifest.json")
        candidate_manifest_value = _required_json(candidate_run, "manifest.json")
        baseline_config_value = _required_json(baseline_run, "config.json")
        candidate_config_value = _required_json(candidate_run, "config.json")
        baseline_metrics_value = _required_json(baseline_run, "metrics.json")
        candidate_metrics_value = _required_json(candidate_run, "metrics.json")
        if not all(
            isinstance(value, dict)
            for value in (
                baseline_manifest_value,
                candidate_manifest_value,
                baseline_config_value,
                candidate_config_value,
                baseline_metrics_value,
                candidate_metrics_value,
            )
        ):
            raise ValueError("manifest, config, and metrics artifacts must contain JSON objects")
        baseline_manifest = baseline_manifest_value
        candidate_manifest = candidate_manifest_value
        baseline_config = baseline_config_value
        candidate_config = candidate_config_value
        baseline_metrics = baseline_metrics_value
        candidate_metrics = candidate_metrics_value
        _verify_run_integrity(baseline_run, baseline_manifest, "baseline")
        _verify_run_integrity(candidate_run, candidate_manifest, "candidate")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _incomparable(baseline_run, candidate_run, [str(exc)])

    baseline_id = baseline_manifest.get("run_id")
    candidate_id = candidate_manifest.get("run_id")
    if baseline_manifest.get("status") != "completed":
        reasons.append("baseline manifest status is not completed")
    if candidate_manifest.get("status") != "completed":
        reasons.append("candidate manifest status is not completed")
    if not isinstance(baseline_id, str) or not baseline_id:
        reasons.append("baseline manifest is missing run_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        reasons.append("candidate manifest is missing run_id")
    if isinstance(baseline_id, str) and baseline_id == candidate_id:
        reasons.append("baseline and candidate run_id values are identical")

    try:
        baseline_fingerprint, baseline_snapshot = _source_identity(baseline_manifest)
        candidate_fingerprint, candidate_snapshot = _source_identity(candidate_manifest)
        if baseline_fingerprint != candidate_fingerprint:
            reasons.append("data source_fingerprint differs")
        if baseline_snapshot != candidate_snapshot:
            reasons.append("snapshot_id differs")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        reasons.append(str(exc))
        baseline_fingerprint = candidate_fingerprint = None
        baseline_snapshot = candidate_snapshot = None

    try:
        baseline_source_tree = _code_identity(baseline_manifest, "baseline")
        candidate_source_tree = _code_identity(candidate_manifest, "candidate")
        if baseline_source_tree != candidate_source_tree:
            reasons.append("code.runtime_code_sha256 differs")
    except ValueError as exc:
        reasons.append(str(exc))
        baseline_source_tree = candidate_source_tree = None

    try:
        _validate_feature_roles(baseline_config, baseline_manifest, candidate_config, candidate_manifest)
    except ValueError as exc:
        reasons.append(str(exc))

    config_differences = _value_differences(
        _normalized_experiment_config(baseline_config),
        _normalized_experiment_config(candidate_config),
    )
    if config_differences:
        reasons.append("non-feature experiment configuration differs: " + ", ".join(config_differences[:20]))
    for field in ("last_realized_signal_date", "backtest_end_date"):
        left_value = baseline_metrics.get(field)
        right_value = candidate_metrics.get(field)
        if not isinstance(left_value, str) or not isinstance(right_value, str):
            reasons.append(f"both metrics artifacts must record {field}")
        elif left_value != right_value:
            reasons.append(f"metrics.{field} differs")

    if reasons:
        return _incomparable(
            baseline_run,
            candidate_run,
            reasons,
            baseline_id=baseline_id if isinstance(baseline_id, str) else None,
            candidate_id=candidate_id if isinstance(candidate_id, str) else None,
        )

    try:
        baseline_raw, baseline_returns, candidate_returns, baseline_signal, candidate_signal, folds = _load_paired_artifacts(
            baseline_run, candidate_run, baseline_config, candidate_config
        )
        actual_backtest_end = baseline_returns.index.max().date().isoformat()
        if baseline_metrics["backtest_end_date"] != actual_backtest_end:
            raise ValueError("metrics.backtest_end_date does not match the base report")
        actual_signal_end = baseline_signal.index.max().date().isoformat()
        if baseline_metrics["last_realized_signal_date"] != actual_signal_end:
            raise ValueError("metrics.last_realized_signal_date does not match signal_metrics")
        daily = _paired_stat(candidate_returns["strategy_net"], baseline_returns["strategy_net"], hac_max_lag)
        ic = _paired_stat(candidate_signal["ic"], baseline_signal["ic"], hac_max_lag)
        rank_ic = _paired_stat(candidate_signal["rank_ic"], baseline_signal["rank_ic"], hac_max_lag)
        baseline_relative = _terminal_relative_wealth(baseline_returns)
        candidate_relative = _terminal_relative_wealth(candidate_returns)
        fold_results = _fold_comparison(
            baseline_run,
            candidate_run,
            folds,
            baseline_raw.index,
            baseline_signal.index,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _incomparable(
            baseline_run,
            candidate_run,
            [str(exc)],
            baseline_id=baseline_id,
            candidate_id=candidate_id,
        )

    terminal_difference = candidate_relative - baseline_relative
    daily_t = daily["hac_t_stat"]
    ic_t = ic["hac_t_stat"]
    rank_ic_t = rank_ic["hac_t_stat"]
    criteria = {
        "terminal_relative_wealth_positive": terminal_difference > 0,
        "daily_return_difference_positive": daily["mean_difference"] > 0,
        "daily_return_hac_significant": daily_t is not None and daily_t >= hac_threshold,
        "fold_win_rate_majority": fold_results["win_rate"] is not None
        and fold_results["win_rate"] > 0.5,
        "ic_difference_non_negative": ic["mean_difference"] >= 0,
        "rank_ic_difference_non_negative": rank_ic["mean_difference"] >= 0,
        "at_least_one_signal_hac_significant": (ic_t is not None and ic_t >= hac_threshold)
        or (rank_ic_t is not None and rank_ic_t >= hac_threshold),
    }
    improved = all(criteria.values())
    status = "improved" if improved else "not_improved"
    failed = [name for name, passed in criteria.items() if not passed]
    claim = (
        "Candidate improvement is supported under the predeclared paired criteria."
        if improved
        else "Improvement is not established; no positive claim is permitted."
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_status": "completed",
        "status": status,
        "baseline_run_id": baseline_id,
        "candidate_run_id": candidate_id,
        "comparable": True,
        "reasons": [] if improved else [f"improvement criterion not met: {name}" for name in failed],
        "conditions": {
            "source_fingerprint": baseline_fingerprint,
            "snapshot_id": baseline_snapshot,
            "runtime_code_sha256": baseline_source_tree,
            "base_slippage_bps_per_side": baseline_config["execution"]["base_slippage_bps_per_side"],
            "evaluation_start_date": baseline_returns.index.min().date().isoformat(),
            "evaluation_end_date": baseline_returns.index.max().date().isoformat(),
            "return_observations": len(baseline_returns),
            "signal_start_date": baseline_signal.index.min().date().isoformat(),
            "signal_end_date": baseline_signal.index.max().date().isoformat(),
            "signal_observations": len(baseline_signal),
        },
        "thresholds": {
            "hac_t_stat": hac_threshold,
            "hac_max_lag": hac_max_lag,
            "fold_win_rate": 0.5,
        },
        "deltas": {
            "terminal_relative_wealth": {
                "baseline": baseline_relative,
                "candidate": candidate_relative,
                "difference": terminal_difference,
            },
            "daily_strategy_return": daily,
            "ic": ic,
            "rank_ic": rank_ic,
            "folds": fold_results,
        },
        "decision": {"claim": claim, "criteria": criteria},
        "scope": "paired incremental research evidence only; not live-performance or promotion evidence",
    }


def generate_comparison_json(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    hac_threshold: float = DEFAULT_HAC_THRESHOLD,
    hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> Path:
    """Atomically write a paired comparison without replacing one by default."""

    baseline_run = Path(baseline_run_dir).resolve()
    candidate_run = Path(candidate_run_dir).resolve()
    target = (
        Path(output_path).resolve()
        if output_path is not None
        else (DEFAULT_COMPARISONS_ROOT / f"{baseline_run.name}__vs__{candidate_run.name}.json").resolve()
    )
    for run_dir in (baseline_run, candidate_run):
        try:
            target.relative_to(run_dir)
        except ValueError:
            continue
        raise ValueError("comparison output must not be written inside an immutable run directory")
    if target.exists() and not overwrite:
        existing = read_json(target)
        if isinstance(existing, dict) and existing.get("comparison_status") == "completed":
            raise FileExistsError(f"completed comparison already exists: {target}")
        raise FileExistsError(f"comparison output already exists: {target}")
    if target.exists() and overwrite:
        existing = read_json(target)
        if (
            not isinstance(existing, dict)
            or existing.get("schema_version") != COMPARISON_SCHEMA_VERSION
            or existing.get("comparison_status") != "completed"
            or existing.get("status") not in {"improved", "not_improved", "incomparable"}
        ):
            raise ValueError(f"refusing to overwrite a non-comparison artifact: {target}")
    if target.suffix.lower() != ".json":
        raise ValueError("comparison output must use a .json filename")
    result = compare_completed_runs(
        baseline_run_dir,
        candidate_run_dir,
        hac_threshold=hac_threshold,
        hac_max_lag=hac_max_lag,
    )
    result["generated_at"] = now_shanghai().isoformat()
    write_json_atomic(target, result)
    return target


# Compact public alias for callers that do not need to distinguish calculation
# from JSON generation.
compare_runs = compare_completed_runs
