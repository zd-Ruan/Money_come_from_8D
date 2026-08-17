from __future__ import annotations

import gc
import json
import math
import shutil
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .action_audit import (
    CorporateActionAuditError,
    CorporateActionAuditResult,
    audit_corporate_actions,
    detect_material_factor_changes,
)
from .audit import AuditResult, audit_and_snapshot
from .config import json_ready_config
from .coverage import calculate_prediction_coverage, load_qlib_coverage_inputs
from .environment import DEFAULT_ENVIRONMENT_LOCK, validate_locked_environment
from .factors import (
    ORIGINAL_RESEARCH_CANDIDATES,
    build_alpha158_factor_handler,
    factor_catalog_manifest,
)
from .integrity import (
    generate_artifact_checksums,
    generate_integrity_seal,
    resolve_run_directory,
    runtime_code_identity,
    validate_run_id,
    verify_artifact_checksums,
)
from .io import git_state, now_shanghai, sha256_file, write_json_atomic
from .metrics import (
    annualized_return,
    beta_alpha,
    compounded_return,
    daily_ic,
    evaluation_frame,
    finite,
    hac_t_stat,
    independent_portfolio_performance,
    information_ratio,
    max_drawdown,
    relative_wealth_drawdown,
)
from .registry import update_registry
from .raw_backtest import RawBacktestConfig, run_raw_backtest
from .windows import RollingFold, build_rolling_folds, load_calendar, shift_session, validate_fold_boundaries


def make_run_id(project_name: str) -> str:
    stamp = now_shanghai().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{project_name}"


def _workspace_relative(path: Path | str, workspace: Path) -> str:
    """Return a portable workspace-relative path without leaking local paths."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(workspace.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside the workspace: {resolved.name}") from exc


def _sanitize_workspace_text(value: str, workspace: Path) -> str:
    return value.replace(str(workspace), "<workspace>").replace(workspace.as_posix(), "<workspace>")


def _qlib_manifest_version(environment: dict[str, Any], qlib_module: Any) -> str:
    installed = environment.get("packages", {}).get("pyqlib")
    if isinstance(installed, str) and installed:
        return installed
    source_version = getattr(qlib_module, "__version__", None)
    return str(source_version) if source_version else "source-checkout"


def backtest_bounds(calendar: pd.DatetimeIndex, first_signal_date: str, last_signal_date: str) -> tuple[str, str]:
    """Return initial close-execution date and final realization date."""
    start = shift_session(calendar, first_signal_date, 1)
    end = shift_session(calendar, last_signal_date, 2)
    return start, end


def fold_is_complete(
    fold: RollingFold,
    calendar: pd.DatetimeIndex,
    required_days: int,
    last_realized_signal_date: str | None = None,
) -> bool:
    positions = {timestamp.date().isoformat(): index for index, timestamp in enumerate(calendar)}
    effective_end = fold.test_end
    if last_realized_signal_date is not None:
        effective_end = min(
            (pd.Timestamp(fold.test_end), pd.Timestamp(last_realized_signal_date))
        ).date().isoformat()
    actual = max(0, positions[effective_end] - positions[fold.test_start] + 1)
    return actual >= int(required_days)


def select_backtest_predictions(predictions: pd.DataFrame, last_signal_date: str) -> pd.DataFrame:
    dates = predictions.index.get_level_values("datetime")
    selected = predictions.loc[dates <= pd.Timestamp(last_signal_date)]
    if selected.empty:
        raise RuntimeError("no fully realized out-of-sample signal dates are available for backtesting")
    if selected["score"].isna().any():
        raise RuntimeError("backtest predictions contain missing scores")
    return selected


def resolve_pipeline_data_bounds(
    config: dict[str, Any], research_request: dict[str, Any] | None = None
) -> tuple[str, str | None]:
    """Return the source-data and authorized final-signal bounds.

    The internal stage record is accepted only alongside a validated research
    request.  Its last source session must be exactly the final label-maturity
    session, which prevents the handler and folds from reading a later stage.
    """

    internal = config.get("_research_stage")
    if research_request is None:
        if internal is not None:
            raise ValueError("_research_stage requires a validated stage-bound request")
        return str(config["data"]["end_date"]), None
    if not isinstance(internal, dict):
        raise ValueError("validated research run is missing its derived stage bounds")
    partition = research_request["partition"]
    expected = {
        "stage": research_request["stage"],
        "request_sha256": research_request["request_sha256"],
        "prediction_end": partition["end"],
        "source_data_end": partition["source_data_end"],
        "exposure_registry_sha256": research_request["exposure_registry_sha256"],
        "evidence_class": research_request["evidence_class"],
        "claim_classification": research_request["claim_classification"],
    }
    if internal != expected:
        raise ValueError("derived research stage bounds differ from the validated request")
    maturity = partition["label_maturity_sessions"]
    if (
        len(maturity) != int(config["data"]["label_horizon_bars"])
        or maturity[-1] != internal["source_data_end"]
        or pd.Timestamp(internal["prediction_end"]) >= pd.Timestamp(internal["source_data_end"])
    ):
        raise ValueError("research source-data bound does not exactly mature the final stage label")
    return internal["source_data_end"], internal["prediction_end"]


def _flatten_feature_names(columns) -> list[str]:
    names = []
    for column in columns:
        if isinstance(column, tuple):
            names.append("__".join(str(part) for part in column if part != ""))
        else:
            names.append(str(column))
    return names


def _prepare_arrays(dataset, fold: RollingFold):
    from qlib.data.dataset.handler import DataHandlerLP

    train = dataset.prepare(
        slice(fold.train_start, fold.train_end), col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    ).dropna(subset=[("label", "LABEL0")])
    valid = dataset.prepare(
        slice(fold.valid_start, fold.valid_end), col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    ).dropna(subset=[("label", "LABEL0")])
    test_features = dataset.prepare(
        slice(fold.test_start, fold.test_end), col_set="feature", data_key=DataHandlerLP.DK_I
    )
    test_labels = dataset.prepare(
        slice(fold.test_start, fold.test_end), col_set="label", data_key=DataHandlerLP.DK_L
    )
    return train, valid, test_features, test_labels


def _model_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
    model = config["model"]
    params = {
        "objective": model["objective"],
        "metric": "l2",
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "max_depth": int(model["max_depth"]),
        "min_data_in_leaf": int(model["min_data_in_leaf"]),
        "feature_fraction": float(model["feature_fraction"]),
        "bagging_fraction": float(model["bagging_fraction"]),
        "bagging_freq": int(model["bagging_freq"]),
        "lambda_l1": float(model["lambda_l1"]),
        "lambda_l2": float(model["lambda_l2"]),
        "num_threads": int(model["num_threads"]),
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(config["project"]["random_seeds"][0]),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": int(model.get("verbosity", -1)),
    }
    device_type = str(model.get("device_type", "cpu"))
    params["device_type"] = device_type
    if device_type == "gpu":
        params.update(
            {
                "gpu_platform_id": int(model.get("gpu_platform_id", 0)),
                "gpu_device_id": int(model.get("gpu_device_id", 0)),
                "gpu_use_dp": bool(model.get("gpu_use_dp", True)),
                "max_bin": int(model.get("max_bin", 63)),
            }
        )
        params.pop("deterministic", None)
    return params


def validate_lightgbm_device(config: dict[str, Any]) -> None:
    """Fail before data preparation when the configured learner is unavailable."""
    model = config["model"]
    if str(model.get("device_type", "cpu")) != "gpu":
        return
    rng = np.random.default_rng(20260812)
    probe = lgb.Dataset(
        rng.normal(size=(4096, 8)),
        label=rng.normal(size=4096),
    )
    try:
        lgb.train(
            {
                "objective": "regression",
                "device_type": "gpu",
                "gpu_platform_id": int(model.get("gpu_platform_id", 0)),
                "gpu_device_id": int(model.get("gpu_device_id", 0)),
                "gpu_use_dp": bool(model.get("gpu_use_dp", True)),
                "max_bin": int(model.get("max_bin", 63)),
                "num_threads": int(model.get("num_threads", 1)),
                "verbosity": int(model.get("verbosity", 1)),
            },
            probe,
            num_boost_round=1,
        )
    except lgb.basic.LightGBMError as exc:
        raise RuntimeError("configured LightGBM GPU learner is unavailable") from exc


def validate_frozen_factor_provider(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all frozen candidates through Qlib's real raw-data path.

    The smoke check uses one active whitelist instrument and a recent bounded
    window.  It initializes no model and persists neither a handler nor output.
    """

    import qlib
    from qlib.data.dataset.handler import DataHandlerLP

    names = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
    if len(names) != 18 or len(set(names)) != 18:
        raise RuntimeError("the frozen research catalog must contain exactly 18 factors")

    provider = Path(config["paths"]["qlib_provider"]).resolve()
    calendar_path = provider / "calendars" / "day.txt"
    instruments_path = Path(config["paths"]["instruments"]).resolve()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(pd.read_csv(calendar_path, header=None).iloc[:, 0], errors="raise")
    ).normalize()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("provider calendar must be non-empty, unique, and increasing")
    configured_end = pd.Timestamp(config["data"]["end_date"]).normalize()
    end_position = int(calendar.searchsorted(configured_end, side="right")) - 1
    maximum_lookback = max(int(factor.lookback) for factor in ORIGINAL_RESEARCH_CANDIDATES)
    if end_position < maximum_lookback + 4:
        raise RuntimeError("provider has too few sessions for the frozen-factor smoke check")

    handler_end_position = end_position
    smoke_end_position = handler_end_position - int(config["data"]["label_horizon_bars"])
    smoke_start_position = smoke_end_position - 4
    load_start_position = smoke_start_position - maximum_lookback - 2
    if load_start_position < 0:
        raise RuntimeError("provider has too little warm-up history for the frozen-factor smoke check")
    load_start = calendar[load_start_position]
    smoke_start = calendar[smoke_start_position]
    smoke_end = calendar[smoke_end_position]
    handler_end = calendar[handler_end_position]

    instruments = pd.read_csv(
        instruments_path,
        sep="\t",
        names=["symbol", "start_date", "end_date"],
        dtype=str,
    )
    instruments["symbol"] = instruments["symbol"].str.upper()
    instruments["start_date"] = pd.to_datetime(instruments["start_date"], errors="raise")
    instruments["end_date"] = pd.to_datetime(instruments["end_date"], errors="raise")
    eligible = instruments.loc[
        (instruments["start_date"] <= load_start)
        & (instruments["end_date"] >= handler_end)
    ]
    if eligible.empty:
        raise RuntimeError("no whitelist ETF spans the frozen-factor smoke window")
    benchmark = str(config["data"]["benchmark"]).upper()
    symbol = benchmark if benchmark in set(eligible["symbol"]) else str(eligible.iloc[0]["symbol"])

    qlib.init(
        provider_uri=str(provider),
        region=config["data"]["region"],
        kernels=1,
    )
    handler = build_alpha158_factor_handler(
        instruments=[symbol],
        start_time=load_start.date().isoformat(),
        end_time=handler_end.date().isoformat(),
        fit_start_time=load_start.date().isoformat(),
        fit_end_time=smoke_end.date().isoformat(),
        label=([config["data"]["label"]], ["LABEL0"]),
        factor_names=names,
        infer_processors=[],
        learn_processors=[],
    )
    raw = handler.fetch(
        slice(smoke_start.date().isoformat(), smoke_end.date().isoformat()),
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_R,
    )
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise RuntimeError("frozen-factor Qlib DK_R smoke returned no observations")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("frozen-factor Qlib DK_R smoke returned an invalid column contract")
    feature = raw["feature"]
    label = raw["label"]
    occurrences = [list(feature.columns).count(name) for name in names]
    if len(names) != 18 or any(count != 1 for count in occurrences):
        raise RuntimeError("Qlib DK_R did not expose exactly the 18 frozen factor columns")
    if list(label.columns) != ["LABEL0"]:
        raise RuntimeError("Qlib DK_R did not expose the frozen label column")

    selected = pd.concat([feature.loc[:, names], label.loc[:, ["LABEL0"]]], axis=1)
    dates = pd.DatetimeIndex(selected.index.get_level_values("datetime")).normalize()
    symbols = set(selected.index.get_level_values("instrument").astype(str).str.upper())
    if dates.min() < smoke_start or dates.max() > smoke_end or symbols != {symbol}:
        raise RuntimeError("Qlib DK_R smoke escaped its instrument or date bounds")
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any() or np.isnan(values).any():
        raise RuntimeError("Qlib DK_R frozen factors or label contain non-finite smoke values")
    return {
        "instrument": symbol,
        "start": smoke_start.date().isoformat(),
        "end": smoke_end.date().isoformat(),
        "observations": len(numeric),
        "factor_count": len(names),
        "data_key": DataHandlerLP.DK_R,
    }


def train_fold(dataset, fold: RollingFold, config: dict[str, Any], fold_dir: Path) -> tuple[pd.DataFrame, dict]:
    train, valid, test_features, test_labels = _prepare_arrays(dataset, fold)
    feature_names = _flatten_feature_names(train["feature"].columns)
    train_set = lgb.Dataset(
        train["feature"].to_numpy(),
        label=train["label"].iloc[:, 0].to_numpy(),
        feature_name=feature_names,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid["feature"].to_numpy(),
        label=valid["label"].iloc[:, 0].to_numpy(),
        feature_name=feature_names,
        reference=train_set,
        free_raw_data=False,
    )

    fold_dir.mkdir(parents=True, exist_ok=False)
    seed_predictions: list[np.ndarray] = []
    best_iterations = []
    importance_frames = []
    seeds = [int(seed) for seed in config["project"]["random_seeds"]]
    for seed in seeds:
        evaluations: dict[str, Any] = {}
        booster = lgb.train(
            _model_params(config, seed),
            train_set,
            num_boost_round=int(config["model"]["num_boost_round"]),
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(int(config["model"]["early_stopping_rounds"]), verbose=False),
                lgb.record_evaluation(evaluations),
            ],
        )
        seed_predictions.append(booster.predict(test_features.to_numpy(), num_iteration=booster.best_iteration))
        best_iterations.append(int(booster.best_iteration))
        booster.save_model(str(fold_dir / f"model_seed_{seed}.txt"), num_iteration=booster.best_iteration)
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "gain": booster.feature_importance(importance_type="gain"),
                    "split": booster.feature_importance(importance_type="split"),
                    "seed": seed,
                }
            )
        )
        write_json_atomic(fold_dir / f"eval_seed_{seed}.json", evaluations)

    prediction = pd.DataFrame(
        {"score": np.mean(seed_predictions, axis=0), "score_std": np.std(seed_predictions, axis=0)},
        index=test_features.index,
    )
    for seed, values in zip(seeds, seed_predictions):
        prediction[f"score_seed_{seed}"] = values
    label = test_labels.iloc[:, 0].rename("label")
    prediction = prediction.join(label, how="left")
    prediction["fold"] = fold.fold

    importance = pd.concat(importance_frames, ignore_index=True)
    importance.groupby("feature", as_index=False)[["gain", "split"]].mean().sort_values(
        "gain", ascending=False
    ).to_parquet(fold_dir / "feature_importance.parquet", index=False)

    fold_ic, fold_rank_ic = daily_ic(prediction)
    summary = {
        **fold.to_dict(),
        "rows": {
            "train": len(train),
            "valid": len(valid),
            "test_features": len(test_features),
            "test_labels": int(prediction["label"].notna().sum()),
        },
        "instruments": {
            "train": train.index.get_level_values("instrument").nunique(),
            "valid": valid.index.get_level_values("instrument").nunique(),
            "test": test_features.index.get_level_values("instrument").nunique(),
        },
        "best_iterations": best_iterations,
        "device_type": str(config["model"].get("device_type", "cpu")),
        "ic": finite(float(fold_ic.mean())),
        "rank_ic": finite(float(fold_rank_ic.mean())),
        "prediction_seed_std_mean": finite(float(prediction["score_std"].mean())),
        "effective_test_end": prediction.loc[prediction["label"].notna()]
        .index.get_level_values("datetime")
        .max()
        .date()
        .isoformat(),
    }
    write_json_atomic(fold_dir / "summary.json", summary)
    del train, valid, test_features, test_labels, train_set, valid_set
    gc.collect()
    return prediction, summary


def raw_factor_daily_rank_ic(
    dataset,
    factor_names: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Compute raw, unmodelled factor RankIC against the frozen forward label."""

    if not factor_names:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))
    from qlib.data.dataset.handler import DataHandlerLP

    raw = dataset.prepare(
        slice(start_date, end_date),
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_R,
    )
    missing = sorted(set(factor_names) - set(raw["feature"].columns))
    if missing:
        raise RuntimeError(f"raw factor evidence is missing columns: {missing}")
    label = raw[("label", "LABEL0")].rename("label")
    records: dict[str, pd.Series] = {}
    for name in factor_names:
        values = pd.concat(
            [raw[("feature", name)].rename("factor"), label], axis=1
        ).dropna()
        metric = values.groupby(level="datetime").apply(
            lambda frame: frame["factor"].corr(frame["label"], method="spearman")
        )
        metric.name = f"{name}__rank_ic"
        records[metric.name] = metric
    result = pd.DataFrame(records).sort_index()
    result.index = pd.DatetimeIndex(result.index, name="datetime")
    return result


def _data_root(config: dict[str, Any]) -> Path:
    universe = Path(config["paths"]["universe"]).resolve()
    root = universe.parent
    if not root.is_dir():
        raise FileNotFoundError(f"ETF data directory does not exist: {root}")
    return root


def _frame_record(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) != 1:
        raise RuntimeError("audit summary must contain exactly one row")
    result: dict[str, Any] = {}
    for key, value in frame.iloc[0].items():
        if pd.isna(value):
            result[str(key)] = None
        elif isinstance(value, pd.Timestamp):
            result[str(key)] = value.date().isoformat()
        elif hasattr(value, "item"):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _read_complete_corporate_actions(
    data_root: Path,
    expected_symbols: set[str] | None = None,
) -> pd.DataFrame:
    actions_path = data_root / "corporate_actions.csv"
    report_path = data_root / "corporate_action_report.csv"
    if not actions_path.is_file():
        raise FileNotFoundError(
            "audited corporate-action table is missing; a partial collection report cannot be backtested"
        )
    if not report_path.is_file():
        raise FileNotFoundError("corporate-action collection report is missing")
    report = pd.read_csv(report_path)
    required_report = {"symbol", "error", "full_universe_scope", "published"}
    missing_report = required_report - set(report.columns)
    if missing_report:
        raise ValueError(f"corporate-action report lacks audit columns: {sorted(missing_report)}")
    errors = report["error"].fillna("").astype(str).str.strip()
    def audited_boolean(column: str) -> pd.Series:
        values = report[column]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False)
        normalized = values.fillna("").astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError(f"corporate-action report {column} must contain explicit booleans")
        return normalized.eq("true")

    full_scope = audited_boolean("full_universe_scope")
    published = audited_boolean("published")
    if report.empty or not errors.eq("").all() or not full_scope.all() or not published.all():
        raise RuntimeError("corporate-action collection is incomplete or was not published for the full universe")
    report_symbols = set(report["symbol"].astype(str).str.upper())
    if expected_symbols is not None and report_symbols != expected_symbols:
        missing = sorted(expected_symbols - report_symbols)
        extra = sorted(report_symbols - expected_symbols)
        raise RuntimeError(
            "corporate-action collection symbol coverage differs from the active universe: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    actions = pd.read_csv(actions_path)
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
    missing_actions = required_actions - set(actions.columns)
    if missing_actions:
        raise ValueError(f"corporate_actions.csv lacks columns: {sorted(missing_actions)}")
    if actions["source_url"].isna().any() or actions["source_sha256"].isna().any():
        raise ValueError("corporate_actions.csv contains unaudited source provenance")
    return actions


def _load_raw_tables(data_root: Path, symbols: set[str], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in sorted(symbols):
        path = data_root / "raw" / f"{symbol.lower()}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"raw ETF history is missing for {symbol}")
        frame = pd.read_csv(
            path,
            usecols=["date", "symbol", "raw_open", "raw_close", "raw_high", "raw_low", "volume"],
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[frame["date"].isin(calendar)]
        frame = frame.loc[pd.to_numeric(frame["volume"], errors="coerce") > 0]
        frames.append(frame)
    if not frames:
        raise ValueError("no raw ETF histories were requested")
    return pd.concat(frames, ignore_index=True)


def _load_factor_tables(data_root: Path, symbols: set[str], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in sorted(symbols):
        path = data_root / "normalized" / f"{symbol.lower()}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"normalized ETF history is missing for {symbol}")
        frame = pd.read_csv(path, usecols=["date", "symbol", "factor"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[frame["date"].isin(calendar)]
        frames.append(frame)
    if not frames:
        raise ValueError("no normalized factor histories were requested")
    return pd.concat(frames, ignore_index=True)


def _pretraining_audit_calendar(config: dict[str, Any]) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    """Include one prior session so an action on data.start_date is auditable."""

    path = Path(config["paths"]["qlib_provider"]) / "calendars" / "day.txt"
    values = pd.to_datetime(pd.read_csv(path, header=None).iloc[:, 0], errors="raise")
    calendar = pd.DatetimeIndex(values).normalize()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("provider calendar must be non-empty, unique, and increasing")
    start = pd.Timestamp(config["data"]["start_date"])
    end = pd.Timestamp(
        config.get("_research_stage", {}).get(
            "source_data_end", config["data"]["end_date"]
        )
    )
    start_position = int(calendar.searchsorted(start, side="left"))
    end_position = int(calendar.searchsorted(end, side="right")) - 1
    if start_position >= len(calendar) or end_position < start_position:
        raise ValueError("configured data range is outside the provider calendar")
    first_data_session = calendar[start_position]
    audit_start_position = max(0, start_position - 1)
    return calendar[audit_start_position : end_position + 1], first_data_session


def run_pretraining_corporate_action_audit(
    config: dict[str, Any],
    output_dir: Path | None,
) -> CorporateActionAuditResult:
    """Run the economic factor/action gate, optionally persisting its evidence."""

    data_root = _data_root(config)
    universe = pd.read_csv(config["paths"]["universe"], usecols=["symbol"])
    universe_symbols = set(universe["symbol"].astype(str).str.upper())
    benchmark_symbol = str(config["data"]["benchmark"]).upper()
    audit_symbols = universe_symbols | {benchmark_symbol}
    actions = _read_complete_corporate_actions(data_root, universe_symbols)
    action_symbols = set(actions["symbol"].astype(str).str.upper())
    unknown_actions = action_symbols - audit_symbols
    if unknown_actions:
        raise ValueError(
            "corporate_actions.csv contains symbols outside the audited universe: "
            f"{sorted(unknown_actions)}"
        )

    audit_calendar, first_data_session = _pretraining_audit_calendar(config)
    factors = _load_factor_tables(data_root, audit_symbols, audit_calendar)
    factor_changes = detect_material_factor_changes(factors)
    changed_symbols = set(factor_changes["symbol"].astype(str))
    if changed_symbols:
        raw_prices = _load_raw_tables(data_root, changed_symbols, audit_calendar)[
            ["date", "symbol", "raw_close"]
        ]
    else:
        raw_prices = pd.DataFrame(columns=["date", "symbol", "raw_close"])
    ex_dates = pd.to_datetime(actions["ex_date"], errors="coerce")
    scoped_actions = actions.loc[
        ex_dates.between(first_data_session, audit_calendar[-1])
    ].copy()
    result = audit_corporate_actions(
        factors,
        scoped_actions,
        audit_calendar,
        raw_prices=raw_prices,
        raise_on_failure=False,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        result.summary.to_parquet(output_dir / "summary.parquet", index=False)
        result.details.to_parquet(output_dir / "details.parquet", index=False)
        result.factor_changes.to_parquet(output_dir / "factor_changes.parquet", index=False)
        write_json_atomic(output_dir / "summary.json", _frame_record(result.summary))
    if not result.passed:
        raise CorporateActionAuditError(result)
    return result


def _raw_backtest_config(config: dict[str, Any], slippage_bps: int) -> RawBacktestConfig:
    strategy = config["strategy"]
    execution = config["execution"]
    return RawBacktestConfig(
        initial_cash=float(execution["account"]),
        topk=int(strategy["topk"]),
        n_drop=int(strategy["n_drop"]),
        hold_thresh=int(strategy["hold_thresh"]),
        risk_degree=float(strategy["risk_degree"]),
        commission_bps_per_side=float(execution["commission_bps_per_side"]),
        min_commission=float(execution["min_cost"]),
        slippage_bps_per_side=float(slippage_bps),
        lot_size=int(execution["trade_unit"]),
        max_daily_volume_participation=float(execution["max_daily_volume_participation"]),
        standard_limit_ratio=float(execution["standard_limit_ratio"]),
        wide_limit_ratio=float(execution["wide_limit_ratio"]),
        price_tick=float(execution["price_tick"]),
    )


def prepare_raw_backtest_inputs(
    config: dict[str, Any],
    predictions: pd.DataFrame,
    backtest_start: str,
    backtest_end: str,
    *,
    pretraining_action_audit: CorporateActionAuditResult | None = None,
) -> dict[str, Any]:
    data_root = _data_root(config)
    calendar_path = Path(config["paths"]["qlib_provider"]) / "calendars" / "day.txt"
    full_calendar = load_calendar(
        calendar_path,
        config["data"]["start_date"],
        config.get("_research_stage", {}).get(
            "source_data_end", config["data"]["end_date"]
        ),
    )
    first_signal = pd.Timestamp(
        predictions.index.get_level_values("datetime").min()
    )
    end = pd.Timestamp(backtest_end)
    audit_start = full_calendar[max(0, full_calendar.get_loc(first_signal) - 1)]
    audit_calendar = full_calendar[(full_calendar >= audit_start) & (full_calendar <= end)]
    execution_calendar = full_calendar[
        (full_calendar >= first_signal) & (full_calendar <= end)
    ]
    if execution_calendar.empty or pd.Timestamp(backtest_start) not in execution_calendar:
        raise ValueError("backtest bounds do not contain the first signal and execution sessions")
    universe = pd.read_csv(config["paths"]["universe"], usecols=["symbol"])
    universe_symbols = set(universe["symbol"].astype(str).str.upper())
    benchmark_symbol = str(config["data"]["benchmark"]).upper()
    audit_symbols = set(universe_symbols) | {benchmark_symbol}
    actions = _read_complete_corporate_actions(data_root, universe_symbols)
    action_symbols = set(actions["symbol"].astype(str).str.upper())
    unknown_actions = action_symbols - audit_symbols
    if unknown_actions:
        raise ValueError(f"corporate_actions.csv contains symbols outside the audited universe: {sorted(unknown_actions)}")
    prediction_symbols = set(predictions.index.get_level_values("instrument").map(str))
    backtest_symbols = prediction_symbols | {benchmark_symbol}
    if pretraining_action_audit is None:
        raw_for_audit = _load_raw_tables(data_root, audit_symbols, audit_calendar)
        factors = _load_factor_tables(data_root, audit_symbols, audit_calendar)
        scoped_actions = actions.loc[
            pd.to_datetime(actions["ex_date"], errors="coerce").between(
                audit_calendar[0], audit_calendar[-1]
            )
        ].copy()
        action_audit = audit_corporate_actions(
            factors,
            scoped_actions,
            audit_calendar,
            raw_prices=raw_for_audit[["date", "symbol", "raw_close"]],
        )
        raw_bars = raw_for_audit.loc[
            raw_for_audit["symbol"].isin(backtest_symbols)
            & raw_for_audit["date"].isin(execution_calendar)
        ].copy()
    else:
        if not pretraining_action_audit.passed:
            raise CorporateActionAuditError(pretraining_action_audit)
        action_audit = pretraining_action_audit
        raw_bars = _load_raw_tables(data_root, backtest_symbols, execution_calendar)
    active_actions = actions.loc[
        actions["symbol"].isin(backtest_symbols)
        & (
            pd.to_datetime(actions["record_date"], errors="coerce").between(
                execution_calendar[0], execution_calendar[-1]
            )
            | pd.to_datetime(actions["ex_date"], errors="coerce").between(
                execution_calendar[0], execution_calendar[-1]
            )
            | pd.to_datetime(actions["cash_payment_date"], errors="coerce").between(
                execution_calendar[0], execution_calendar[-1]
            )
        )
    ].copy()
    benchmark_close = raw_bars.loc[
        raw_bars["symbol"] == benchmark_symbol, ["date", "symbol", "raw_close"]
    ]
    return {
        "calendar": execution_calendar,
        "raw_bars": raw_bars.loc[raw_bars["symbol"] != benchmark_symbol].copy()
        if benchmark_symbol not in prediction_symbols
        else raw_bars.copy(),
        "benchmark_close": benchmark_close,
        "corporate_actions": active_actions,
        "corporate_action_audit": action_audit,
    }


def slice_prepared_raw_backtest_inputs(
    prepared: dict[str, Any],
    predictions: pd.DataFrame,
    backtest_end: str,
) -> dict[str, Any]:
    """Derive a fold view from immutable run-level data without repeating I/O or audit."""

    first_signal = pd.Timestamp(predictions.index.get_level_values("datetime").min())
    end = pd.Timestamp(backtest_end)
    calendar = prepared["calendar"]
    selected_calendar = calendar[(calendar >= first_signal) & (calendar <= end)]
    symbols = set(predictions.index.get_level_values("instrument").map(str))
    raw = prepared["raw_bars"]
    actions = prepared["corporate_actions"]
    return {
        "calendar": selected_calendar,
        "raw_bars": raw.loc[
            raw["date"].isin(selected_calendar) & raw["symbol"].isin(symbols)
        ].copy(),
        "benchmark_close": prepared["benchmark_close"].loc[
            prepared["benchmark_close"]["date"].isin(selected_calendar)
        ].copy(),
        "corporate_actions": actions.loc[
            actions["symbol"].isin(symbols | set(prepared["benchmark_close"]["symbol"].unique()))
            & (
                pd.to_datetime(actions["record_date"], errors="coerce").between(
                    selected_calendar[0], selected_calendar[-1]
                )
                | pd.to_datetime(actions["ex_date"], errors="coerce").between(
                    selected_calendar[0], selected_calendar[-1]
                )
                | pd.to_datetime(actions["cash_payment_date"], errors="coerce").between(
                    selected_calendar[0], selected_calendar[-1]
                )
            )
        ].copy(),
        "corporate_action_audit": prepared["corporate_action_audit"],
    }


def _execution_indicators(index: pd.DatetimeIndex, executions: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        0.0,
        index=index,
        columns=["intent_count", "filled_intent_count", "target_notional", "fill_notional", "fill_rate"],
    )
    if executions.empty:
        return frame
    grouped = executions.groupby("execution_date", sort=True)
    frame["intent_count"] = grouped.size().reindex(index, fill_value=0).astype(float)
    frame["filled_intent_count"] = grouped["fill_shares"].apply(
        lambda value: float((value > 0).sum())
    ).reindex(index, fill_value=0.0)
    frame["target_notional"] = grouped["target_notional"].sum(min_count=1).reindex(index).fillna(0.0)
    frame["fill_notional"] = grouped["fill_notional"].sum().reindex(index, fill_value=0.0)
    frame["fill_rate"] = np.where(
        frame["target_notional"] > 0,
        frame["fill_notional"] / frame["target_notional"],
        0.0,
    )
    return frame


def _cost_only_stress_report(
    base_report: pd.DataFrame,
    base_executions: pd.DataFrame,
    base_execution_summary: dict[str, Any],
    stress_slippage_bps: int,
    base_slippage_bps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild a report applying stress slippage only to the cost leg.

    The base scenario's execution path (positions, fills, return path) is
    reused unchanged; only the slippage rate on filled notionals rises, so the
    net-return series is monotonically non-increasing in the stress level.
    """

    filled = base_executions.loc[base_executions["fill_shares"] > 0]
    dates = pd.DatetimeIndex(base_report.index, name="date")
    if filled.empty:
        daily_commission = pd.Series(0.0, index=dates)
        daily_notional = pd.Series(0.0, index=dates)
    else:
        daily_commission = filled.groupby("execution_date")["commission"].sum()
        daily_notional = filled.groupby("execution_date")["fill_notional"].sum()
        daily_commission = daily_commission.reindex(dates, fill_value=0.0)
        daily_notional = daily_notional.reindex(dates, fill_value=0.0)
    slippage = daily_notional.astype(float) * stress_slippage_bps / 10_000.0
    daily_cost = daily_commission.astype(float) + slippage
    initial = float(base_execution_summary["initial_account"])
    report = base_report.copy()
    previous = initial
    cost_ratios: list[float] = []
    account_path: list[float] = []
    for date in dates:
        gross = float(report.at[date, "return"])
        cost_ratio = float(daily_cost.at[date]) / previous
        previous = previous * (1.0 + gross - cost_ratio)
        cost_ratios.append(cost_ratio)
        account_path.append(previous)
    report["cost"] = cost_ratios
    report["account"] = account_path
    report["cash"] = report["account"] - report["value"] - report["receivable"]
    execution_summary = dict(base_execution_summary)
    execution_summary["slippage_bps_per_side"] = float(stress_slippage_bps)
    execution_summary["slippage_total"] = float(slippage.sum())
    execution_summary["total_cost"] = float(daily_cost.sum())
    execution_summary["cost_only_stress"] = True
    execution_summary["base_execution_path_slippage_bps"] = float(base_slippage_bps)
    base_config = dict(execution_summary.get("config", {}))
    execution_summary["config"] = {
        **base_config,
        "slippage_bps_per_side": float(stress_slippage_bps),
        "cost_only_stress": True,
    }
    return report, execution_summary


def run_backtest(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    slippage_bps: int,
    backtest_start: str,
    backtest_end: str,
    *,
    prepared_inputs: dict[str, Any] | None = None,
):
    prepared = prepared_inputs or prepare_raw_backtest_inputs(
        config, predictions, backtest_start, backtest_end
    )
    result = run_raw_backtest(
        predictions,
        prepared["raw_bars"],
        prepared["corporate_actions"],
        prepared["calendar"],
        _raw_backtest_config(config, slippage_bps),
        benchmark_close=prepared["benchmark_close"],
        benchmark_symbol=str(config["data"]["benchmark"]),
        factor_jumps_pre_audited=True,
    )
    report = result.report.loc[pd.Timestamp(backtest_start) : pd.Timestamp(backtest_end)].copy()
    if report.empty or report.index[0] != pd.Timestamp(backtest_start):
        raise RuntimeError("raw-share report does not start on the requested first execution date")
    positions = result.positions.loc[
        result.positions["date"].between(pd.Timestamp(backtest_start), pd.Timestamp(backtest_end))
    ].reset_index(drop=True)
    executions = result.executions.loc[
        result.executions["execution_date"].between(
            pd.Timestamp(backtest_start), pd.Timestamp(backtest_end)
        )
    ].reset_index(drop=True)
    action_ledger = result.corporate_action_ledger.loc[
        result.corporate_action_ledger["date"].between(
            pd.Timestamp(backtest_start), pd.Timestamp(backtest_end)
        )
    ].reset_index(drop=True)
    symbol_attribution = result.symbol_attribution.loc[
        result.symbol_attribution["date"].between(
            pd.Timestamp(backtest_start), pd.Timestamp(backtest_end)
        )
    ].reset_index(drop=True)
    symbol_attribution["net_pnl_cny"] = symbol_attribution["net_pnl"]
    indicator_frame = _execution_indicators(report.index, executions)
    audit = prepared["corporate_action_audit"]
    indicator_object = {
        "engine": "raw_share_daily_v1",
        "corporate_action_ledger": action_ledger,
        "symbol_attribution": symbol_attribution,
        "corporate_action_audit_summary": audit.summary,
        "corporate_action_audit_details": audit.details,
    }
    execution_summary = dict(result.summary)
    execution_summary["total_cost"] = float(execution_summary.pop("cost_total"))
    execution_summary["price_limit_audit"] = {
        "mode": "ohlc_proven_tier_conservative",
        "standard_limit_ratio": float(config["execution"]["standard_limit_ratio"]),
        "wide_limit_ratio": float(config["execution"]["wide_limit_ratio"]),
        "research_only": True,
    }
    execution_summary["corporate_action_audit"] = _frame_record(audit.summary)
    execution_summary["zero_fill_order_rate"] = execution_summary["zero_fill_intent_rate"]
    execution_summary["zero_fill_order_count"] = execution_summary["zero_fill_intent_count"]
    execution_summary["notional_fill_rate"] = execution_summary["fill_rate"]
    return report, positions, indicator_frame, indicator_object, executions, execution_summary


def summarize_backtest(
    report: pd.DataFrame,
    indicators: pd.DataFrame,
    slippage_bps: int,
    execution_summary: dict[str, Any],
) -> dict[str, Any]:
    aligned = evaluation_frame(report)
    net = aligned["strategy_net"]
    benchmark = aligned["benchmark"]
    excess = net - benchmark
    beta, alpha = beta_alpha(net, benchmark)
    ledger_total_cost = float(execution_summary.get("total_cost") or 0.0)
    report_total_cost = float((report["cost"] * report["account"].shift(1)).iloc[1:].sum())
    report_total_cost += float(report["cost"].iloc[0]) * float(execution_summary["initial_account"])
    qlib_total_cost = report_total_cost
    if not math.isclose(qlib_total_cost, ledger_total_cost, rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError(
            f"execution ledger cost {ledger_total_cost} does not match portfolio cost {qlib_total_cost}"
        )
    relative_terminal = (1.0 + compounded_return(net)) / (1.0 + compounded_return(benchmark)) - 1.0
    correlation = (
        None
        if float(net.std(ddof=1)) <= 0.0 or float(benchmark.std(ddof=1)) <= 0.0
        else finite(float(net.corr(benchmark)))
    )
    return {
        "slippage_bps_per_side": slippage_bps,
        "raw_execution_days": len(report),
        "days": len(aligned),
        "initial_execution_date": pd.Timestamp(report.index[0]).date().isoformat(),
        "evaluation_start_date": pd.Timestamp(aligned.index[0]).date().isoformat(),
        "evaluation_end_date": pd.Timestamp(aligned.index[-1]).date().isoformat(),
        "alignment_method": "initial_cost_compounded_into_first_realized_return",
        "net_cumulative_return": finite(compounded_return(net)),
        "net_annualized_return": finite(annualized_return(net)),
        "benchmark_cumulative_return": finite(compounded_return(benchmark)),
        "excess_cumulative_return": finite(relative_terminal),
        "excess_annualized_return": finite(float(excess.mean() * 252)),
        "information_ratio": finite(information_ratio(excess)),
        "excess_hac_t_stat": finite(hac_t_stat(excess)),
        "strategy_max_drawdown": finite(max_drawdown(net)),
        "relative_wealth_max_drawdown": finite(relative_wealth_drawdown(net, benchmark)),
        "beta": finite(beta),
        "beta_adjusted_alpha_annualized": finite(alpha),
        "strategy_benchmark_correlation": correlation,
        "average_daily_turnover": finite(float(report["turnover"].mean())),
        "total_cost": finite(qlib_total_cost),
        "fill_rate": finite(execution_summary.get("fill_rate")),
        "notional_fill_rate": finite(execution_summary.get("notional_fill_rate")),
        "execution": execution_summary,
        "average_cash_utilization": finite(float((1.0 - report["cash"] / report["account"]).mean())),
        "terminal_account": finite(float(report["account"].iloc[-1])),
        "max_drawdown": finite(max_drawdown(net)),
        "single_etf_abs_contribution_share": execution_summary.get(
            "max_single_etf_gross_abs_contribution_share"
        ),
        "symbol_attribution_concentration": execution_summary.get(
            "symbol_attribution_concentration"
        ),
    }


def _run_research_backtest_folds(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    research_request: dict[str, Any],
    calendar: pd.DatetimeIndex,
    prepared_inputs: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Run each frozen research fold from a fresh CNY cash account."""

    partition = research_request.get("partition")
    metric_contract = research_request.get("metric_contract", {}).get("portfolio")
    if not isinstance(partition, dict) or not isinstance(metric_contract, dict):
        raise ValueError("research request is missing its portfolio fold contract")
    research_folds = partition.get("research_folds")
    if not isinstance(research_folds, list) or not research_folds:
        raise ValueError("research request must contain at least one research fold")
    if metric_contract.get("research_folds") != research_folds:
        raise ValueError("research portfolio and partition fold contracts differ")

    account = float(config["execution"]["account"])
    contracted_account = float(metric_contract.get("initial_account", float("nan")))
    stress_slippage = metric_contract.get("stress_slippage_bps_per_side")
    if not math.isclose(account, 20_000.0, rel_tol=0.0, abs_tol=1e-9) or not math.isclose(
        contracted_account, account, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("research folds require an independently reset CNY 20,000 account")
    if isinstance(stress_slippage, bool) or stress_slippage != 10:
        raise ValueError("research folds require the frozen 10 bps per-side stress slippage")
    configured_stress = config["execution"].get("stress_slippage_bps_per_side", [])
    if 10 not in [int(value) for value in configured_stress]:
        raise ValueError("research configuration does not produce the required 10 bps scenario")
    if (
        int(config["data"].get("label_horizon_bars", 0)) != 2
        or config["data"].get("label") != "Ref($close, -2) / Ref($close, -1) - 1"
        or partition.get("portfolio_execution_lag_bars") != 1
        or partition.get("portfolio_realization_lag_bars") != 2
    ):
        raise ValueError("research folds require T close signal, T+1 close execution, and T+2 label realization")
    configured_fold_days = int(config.get("gates", {}).get("research_fold_days", 0))
    if partition.get("research_fold_signal_sessions") != 21 or configured_fold_days != 21:
        raise ValueError("research folds require exactly 21 signal sessions")

    calendar = pd.DatetimeIndex(calendar)
    if calendar.tz is not None or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("research fold calendar must be unique, timezone-naive, and increasing")
    prediction_dates = pd.DatetimeIndex(
        predictions.index.get_level_values("datetime").unique()
    ).sort_values()
    expected_stage_dates = pd.DatetimeIndex(partition.get("sessions", []))
    if not prediction_dates.equals(expected_stage_dates):
        raise ValueError("research backtest predictions do not exactly equal authorized signal sessions")
    if len(research_folds) != math.ceil(len(expected_stage_dates) / 21):
        raise ValueError("research fold count does not match 21-session partitioning")
    covered_sessions: list[Any] = []
    for position, fold_contract in enumerate(research_folds, start=1):
        if not isinstance(fold_contract, dict) or fold_contract.get("fold") != position:
            raise ValueError("research folds must be mappings numbered consecutively from one")
        fold_sessions = fold_contract.get("signal_sessions")
        if not isinstance(fold_sessions, list):
            raise ValueError(f"research fold {position} signal sessions must be a list")
        covered_sessions.extend(fold_sessions)
    if not pd.DatetimeIndex(covered_sessions).equals(expected_stage_dates):
        raise ValueError("research folds do not exactly cover the authorized signal sessions")

    summaries: list[dict[str, Any]] = []
    prior_signal_end: pd.Timestamp | None = None
    for position, fold_contract in enumerate(research_folds, start=1):
        signal_sessions = pd.DatetimeIndex(fold_contract.get("signal_sessions", []))
        raw_sessions = pd.DatetimeIndex(fold_contract.get("raw_report_sessions", []))
        evaluation_sessions = pd.DatetimeIndex(fold_contract.get("evaluation_sessions", []))
        signal_count = int(fold_contract.get("signal_observations", 0))
        if (
            signal_count != len(signal_sessions)
            or not 1 <= signal_count <= 21
            or (position < len(research_folds) and signal_count != 21)
            or bool(fold_contract.get("complete_for_gate")) != (signal_count == 21)
        ):
            raise ValueError(f"research fold {position} has an invalid 21-session completeness contract")
        if (
            signal_sessions.empty
            or signal_sessions.has_duplicates
            or not signal_sessions.is_monotonic_increasing
            or fold_contract.get("signal_start") != signal_sessions[0].date().isoformat()
            or fold_contract.get("signal_end") != signal_sessions[-1].date().isoformat()
        ):
            raise ValueError(f"research fold {position} signal boundaries are invalid")
        if prior_signal_end is not None and signal_sessions[0] <= prior_signal_end:
            raise ValueError("research fold signals overlap or are not chronological")
        prior_signal_end = signal_sessions[-1]

        fold_start, fold_end = backtest_bounds(
            calendar,
            signal_sessions[0].date().isoformat(),
            signal_sessions[-1].date().isoformat(),
        )
        expected_raw = calendar[
            (calendar >= pd.Timestamp(fold_start)) & (calendar <= pd.Timestamp(fold_end))
        ]
        if (
            not raw_sessions.equals(expected_raw)
            or fold_contract.get("raw_report_start") != fold_start
            or fold_contract.get("raw_report_end") != fold_end
            or not evaluation_sessions.equals(raw_sessions[1:])
            or fold_contract.get("evaluation_start") != evaluation_sessions[0].date().isoformat()
            or fold_contract.get("evaluation_end") != evaluation_sessions[-1].date().isoformat()
        ):
            raise ValueError(f"research fold {position} T+1/T+2 date contract is invalid")

        prediction_index_dates = predictions.index.get_level_values("datetime")
        fold_predictions = predictions.loc[prediction_index_dates.isin(signal_sessions)]
        actual_fold_dates = pd.DatetimeIndex(
            fold_predictions.index.get_level_values("datetime").unique()
        ).sort_values()
        if fold_predictions.empty or not actual_fold_dates.equals(signal_sessions):
            raise RuntimeError(f"research fold {position} predictions are incomplete")
        fold_inputs = slice_prepared_raw_backtest_inputs(
            prepared_inputs, fold_predictions, fold_end
        )
        (
            report,
            positions,
            indicators,
            indicator_object,
            executions,
            execution_summary,
        ) = run_backtest(
            fold_predictions,
            config,
            10,
            fold_start,
            fold_end,
            prepared_inputs=fold_inputs,
        )
        actual_report_dates = pd.DatetimeIndex(report.index)
        actual_evaluation_dates = pd.DatetimeIndex(evaluation_frame(report).index)
        if not actual_report_dates.equals(raw_sessions) or not actual_evaluation_dates.equals(
            evaluation_sessions
        ):
            raise RuntimeError(f"research fold {position} output dates differ from the frozen contract")
        if (
            not math.isclose(
                float(execution_summary["initial_account"]), account, rel_tol=0.0, abs_tol=1e-9
            )
            or not math.isclose(
                float(execution_summary["config"]["initial_cash"]),
                account,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or float(execution_summary["config"]["slippage_bps_per_side"]) != 10.0
        ):
            raise RuntimeError(f"research fold {position} did not reset the contracted account")

        summary = summarize_backtest(report, indicators, 10, execution_summary)
        benchmark_terminal = account * (1.0 + float(summary["benchmark_cumulative_return"]))
        summary.update(
            {
                "fold": position,
                "signal_start": fold_contract["signal_start"],
                "signal_end": fold_contract["signal_end"],
                "signal_observations": signal_count,
                "signal_sessions": list(fold_contract["signal_sessions"]),
                "raw_report_start": fold_contract["raw_report_start"],
                "raw_report_end": fold_contract["raw_report_end"],
                "raw_report_sessions": list(fold_contract["raw_report_sessions"]),
                "evaluation_start": fold_contract["evaluation_start"],
                "evaluation_end": fold_contract["evaluation_end"],
                "evaluation_sessions": list(fold_contract["evaluation_sessions"]),
                "complete_for_gate": bool(fold_contract["complete_for_gate"]),
                "initial_account_value": account,
                "terminal_account_value": float(summary["terminal_account"]),
                "benchmark_terminal_account": benchmark_terminal,
                "benchmark_terminal_account_value": benchmark_terminal,
                "single_etf_abs_contribution_share": execution_summary.get(
                    "max_single_etf_gross_abs_contribution_share"
                ),
                "symbol_attribution_concentration": execution_summary.get(
                    "symbol_attribution_concentration"
                ),
            }
        )
        fold_dir = run_dir / "folds" / f"research_fold_{position:02d}" / "backtest"
        fold_dir.mkdir(parents=True, exist_ok=False)
        report.to_parquet(fold_dir / "report.parquet")
        indicators.to_parquet(fold_dir / "indicators.parquet")
        executions.to_parquet(fold_dir / "executions.parquet", index=False)
        positions.to_parquet(fold_dir / "positions.parquet", index=False)
        indicator_object["symbol_attribution"].to_parquet(
            fold_dir / "symbol_attribution.parquet", index=False
        )
        indicator_object["corporate_action_ledger"].to_parquet(
            fold_dir / "corporate_actions.parquet", index=False
        )
        write_json_atomic(fold_dir / "summary.json", summary)
        summaries.append(summary)

    summary_path = run_dir / "folds" / "research_folds.json"
    write_json_atomic(summary_path, summaries)
    return summaries


def evaluate_gates(
    config: dict[str, Any], audit: AuditResult, fold_summaries: list[dict], metrics: dict[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    base = metrics["base"]
    required_stress = str(int(gates["required_stress_slippage_bps"]))
    stress = metrics["stress"][required_stress]
    complete_folds = [
        summary for summary in fold_summaries if summary.get("portfolio", {}).get("complete_for_gate") is True
    ]
    positive_folds = sum(
        (summary["portfolio"].get("excess_cumulative_return") or 0) > 0 for summary in complete_folds
    )
    fold_ratio = positive_folds / len(complete_folds) if complete_folds else 0.0

    checks = [
        {"name": "data_valid", "passed": audit.report["data_valid"], "value": audit.report["data_valid"]},
        {
            "name": "point_in_time_universe",
            "passed": config["data"]["universe_mode"] == "point_in_time",
            "value": config["data"]["universe_mode"],
            "blocking_for_promotion": bool(gates["require_historical_point_in_time_universe_for_promotion"]),
        },
        {
            "name": "minimum_test_days",
            "passed": int(base["days"]) >= int(gates["min_test_days"]),
            "value": base["days"],
            "threshold": gates["min_test_days"],
        },
        {
            "name": "prediction_coverage",
            "passed": metrics["prediction_coverage"] >= float(gates["min_prediction_coverage"]),
            "value": metrics["prediction_coverage"],
            "threshold": gates["min_prediction_coverage"],
        },
        {
            "name": "fill_rate",
            "passed": base["fill_rate"] is not None
            and base["fill_rate"] >= float(gates["min_fill_rate"]),
            "value": base["fill_rate"],
            "threshold": gates["min_fill_rate"],
        },
        {
            "name": "zero_fill_order_rate",
            "passed": base["execution"]["zero_fill_order_rate"] is not None
            and base["execution"]["zero_fill_order_rate"] <= float(gates["max_zero_fill_order_rate"]),
            "value": base["execution"]["zero_fill_order_rate"],
            "threshold": gates["max_zero_fill_order_rate"],
        },
        {
            "name": "minimum_complete_folds",
            "passed": len(complete_folds) >= int(gates["min_complete_folds"]),
            "value": len(complete_folds),
            "threshold": gates["min_complete_folds"],
        },
        {
            "name": "fold_positive_excess_ratio",
            "passed": fold_ratio >= float(gates["min_positive_fold_excess_ratio"]),
            "value": fold_ratio,
            "threshold": gates["min_positive_fold_excess_ratio"],
        },
        {
            "name": "excess_hac_t_stat",
            "passed": (base["excess_hac_t_stat"] or -math.inf) >= float(gates["min_hac_t_stat"]),
            "value": base["excess_hac_t_stat"],
            "threshold": gates["min_hac_t_stat"],
        },
        {
            "name": "rank_ic_hac_t_stat",
            "passed": (metrics["rank_ic_hac_t_stat"] or -math.inf)
            >= float(gates["min_rank_ic_hac_t_stat"]),
            "value": metrics["rank_ic_hac_t_stat"],
            "threshold": gates["min_rank_ic_hac_t_stat"],
        },
        {
            "name": "max_drawdown",
            "passed": abs(base["strategy_max_drawdown"] or 1) <= float(gates["max_strategy_drawdown"]),
            "value": base["strategy_max_drawdown"],
            "threshold": gates["max_strategy_drawdown"],
        },
        {
            "name": "stress_positive_excess",
            "passed": (stress["excess_cumulative_return"] or -1) > 0,
            "value": stress["excess_cumulative_return"],
            "threshold": 0,
        },
    ]
    promotion_eligible = all(check["passed"] for check in checks)
    return {
        "status": "candidate" if promotion_eligible else "research_only",
        "promotion_eligible": promotion_eligible,
        "passed": sum(check["passed"] for check in checks),
        "total": len(checks),
        "checks": checks,
    }


def enforce_research_exposure_gate(
    gates: dict[str, Any], research_control: dict[str, Any] | None
) -> dict[str, Any]:
    """Prevent exposed research from inheriting a promotion-like run status."""

    if research_control is None:
        return gates
    result = deepcopy(gates)
    evidence_class = research_control.get("evidence_class")
    claim_classification = research_control.get("claim_classification")
    passed = evidence_class == "prospective_unseen" and claim_classification == "research_only"
    result["checks"].append(
        {
            "name": "research_exposure_not_historically_exposed",
            "passed": passed,
            "value": evidence_class,
            "threshold": "prospective_unseen",
            "blocking_for_promotion": True,
        }
    )
    result["total"] = len(result["checks"])
    result["passed"] = sum(check["passed"] for check in result["checks"])
    result["promotion_eligible"] = bool(result["promotion_eligible"] and passed)
    result["status"] = "candidate" if result["promotion_eligible"] else "research_only"
    result["research_claim_classification"] = claim_classification
    return result


def _seal_completed_run(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Finalize the manifest once, then create and verify its detached outer seal."""
    manifest_path = run_dir / "manifest.json"
    checksum_path = generate_artifact_checksums(run_dir)
    artifact_integrity = verify_artifact_checksums(run_dir, require_seal=False)
    if not artifact_integrity["valid"]:
        raise RuntimeError(f"artifact checksum verification failed: {artifact_integrity}")

    manifest["integrity"] = {
        "checksum_manifest": checksum_path.name,
        "checksum_sha256": sha256_file(checksum_path),
        "artifact_count": artifact_integrity["expected_count"],
        "seal_manifest": "integrity_seal.json",
        "verified": True,
    }
    manifest.setdefault("completed_at", now_shanghai().isoformat())
    manifest["status"] = "completed"
    write_json_atomic(manifest_path, manifest)

    seal_path = generate_integrity_seal(run_dir, checksum_path)
    sealed_integrity = verify_artifact_checksums(run_dir, seal_path=seal_path)
    if not sealed_integrity["valid"]:
        raise RuntimeError(f"sealed artifact verification failed: {sealed_integrity}")
    return sealed_integrity


def run_pipeline(
    config: dict[str, Any],
    run_id: str | None = None,
    *,
    research_plan: dict[str, Any] | None = None,
    research_request: dict[str, Any] | None = None,
    research_state_path: Path | None = None,
) -> Path:
    import qlib
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    research_control: dict[str, Any] | None = None
    supplied_research = (
        research_plan is not None,
        research_request is not None,
        research_state_path is not None,
    )
    if any(supplied_research) and not all(supplied_research):
        raise ValueError(
            "research_plan, research_request, and research_state_path must be supplied together"
        )
    if "_research_stage" in config:
        raise ValueError("_research_stage is reserved for a validated stage-bound request")
    if research_plan is not None and research_request is not None:
        from .research_runner import prepare_stage_pipeline_config, validate_claimed_stage_run

        validate_claimed_stage_run(research_state_path, research_plan, research_request)
        config, validated_request = prepare_stage_pipeline_config(
            config, research_plan, research_request
        )
        research_control = {
            "protocol_version": validated_request["protocol_version"],
            "plan_id": validated_request["plan_id"],
            "plan_sha256": validated_request["plan_sha256"],
            "request_sha256": validated_request["request_sha256"],
            "stage": validated_request["stage"],
            "experiment_id": validated_request["experiment"]["experiment_id"],
            "experiment_spec_sha256": validated_request["experiment"]["spec_sha256"],
            "partition_sha256": validated_request["partition"]["sessions_sha256"],
            "portfolio_evaluation_sessions_sha256": validated_request["partition"][
                "portfolio_evaluation_sessions_sha256"
            ],
            "label_maturity_sessions_sha256": validated_request["partition"][
                "label_maturity_sessions_sha256"
            ],
            "source_data_end": validated_request["partition"]["source_data_end"],
            "exposure_registry_sha256": validated_request[
                "exposure_registry_sha256"
            ],
            "evidence_class": validated_request["evidence_class"],
            "claim_classification": validated_request["claim_classification"],
        }

    research_source_data_end, research_prediction_end = resolve_pipeline_data_bounds(
        config, validated_request if research_control is not None else None
    )

    workspace = Path(config["_meta"]["workspace_root"]).resolve()
    source_root = Path(__file__).resolve().parent
    qlib_file = getattr(qlib, "__file__", None)
    if not isinstance(qlib_file, str) or not qlib_file:
        raise RuntimeError("the imported Qlib package path is unavailable")
    qlib_package_root = Path(qlib_file).resolve().parent
    initial_code_identity = runtime_code_identity(source_root, qlib_package_root)
    initial_git_state = git_state(workspace)
    feature_mode = config["features"]["mode"]
    selected_families = config["features"].get("families") or None
    selected_factor_names = config["features"].get("factor_names") or None
    if selected_families is not None and selected_factor_names is not None:
        raise ValueError("features.families and features.factor_names are mutually exclusive")
    frozen_factor_catalog = (
        factor_catalog_manifest(selected_families, selected_factor_names)
        if feature_mode == "alpha158_plus_original"
        else None
    )
    run_id = validate_run_id(run_id or make_run_id(config["project"]["name"]))
    run_dir = resolve_run_directory(Path(config["paths"]["runs"]), run_id)
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    created_at = now_shanghai().isoformat()
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "status": "running",
        "code": initial_code_identity,
        "git": initial_git_state,
    }
    if frozen_factor_catalog is not None:
        manifest["factor_catalog"] = frozen_factor_catalog
    if research_control is not None:
        manifest["research"] = research_control
        write_json_atomic(run_dir / "research_request.json", validated_request)
    write_json_atomic(manifest_path, manifest)

    try:
        configured_environment_lock = Path(
            config["paths"].get("environment_lock", DEFAULT_ENVIRONMENT_LOCK)
        ).resolve()
        environment = validate_locked_environment(configured_environment_lock)
        environment_lock_path = run_dir / configured_environment_lock.name
        shutil.copy2(configured_environment_lock, environment_lock_path)
        if sha256_file(environment_lock_path) != environment["lock"]["sha256"]:
            raise RuntimeError("environment lock changed while the run was starting")
        validate_lightgbm_device(config)
        audit = audit_and_snapshot(config)
        if not audit.report["data_valid"]:
            raise RuntimeError(f"data quality gate failed: {audit.report['blocking_issues']}")
        pretraining_action_audit = run_pretraining_corporate_action_audit(
            config,
            run_dir / "audits" / "corporate_actions_pretraining",
        )

        provider = Path(config["paths"]["qlib_provider"])
        calendar_path = provider / "calendars" / "day.txt"
        calendar = load_calendar(
            calendar_path, config["data"]["start_date"], research_source_data_end
        )
        rolling_calendar = (
            calendar[calendar <= pd.Timestamp(research_prediction_end)]
            if research_control is not None
            else calendar
        )
        folds = build_rolling_folds(
            rolling_calendar,
            train_start_date=config["data"]["start_date"],
            test_start_date=config["data"]["test_start_date"],
            validation_days=int(config["rolling"]["validation_days"]),
            test_days=int(config["rolling"]["test_days"]),
            purge_bars=int(config["rolling"]["purge_bars"]),
        )
        validate_fold_boundaries(folds, rolling_calendar)
        write_json_atomic(run_dir / "folds.json", [fold.to_dict() for fold in folds])

        qlib.init(provider_uri=str(provider), region=config["data"]["region"], kernels=4)
        handler_kwargs = {
            "instruments": config["data"]["market"],
            "start_time": config["data"]["start_date"],
            "end_time": research_source_data_end,
            "fit_start_time": config["data"]["start_date"],
            "fit_end_time": folds[0].train_end,
            "label": ([config["data"]["label"]], ["LABEL0"]),
            "filter_pipe": [
                {
                    "filter_type": "ExpressionDFilter",
                    "rule_expression": config["data"]["liquidity_expression"],
                    "filter_start_time": None,
                    "filter_end_time": None,
                    "keep": False,
                }
            ],
        }
        if feature_mode == "alpha158_plus_original":
            handler = build_alpha158_factor_handler(
                **handler_kwargs,
                families=selected_families,
                factor_names=selected_factor_names,
            )
        elif feature_mode == "alpha360":
            from qlib.contrib.data.handler import Alpha360

            handler = Alpha360(**handler_kwargs)
        elif feature_mode == "alpha191":
            from quant_pipeline.alpha191_handler import Alpha191

            handler = Alpha191(**handler_kwargs)
        else:
            handler = Alpha158(**handler_kwargs)
        dataset = DatasetH(handler=handler, segments={})

        prediction_frames = []
        fold_summaries = []
        for fold in folds:
            prediction, summary = train_fold(dataset, fold, config, run_dir / "folds" / f"fold_{fold.fold:02d}")
            prediction_frames.append(prediction)
            fold_summaries.append(summary)

        predictions = pd.concat(prediction_frames).sort_index()
        if predictions.index.duplicated().any():
            raise RuntimeError("rolling predictions contain duplicate index rows")
        predictions.to_parquet(run_dir / "predictions.parquet")
        raw_factor_metrics = raw_factor_daily_rank_ic(
            dataset,
            list(selected_factor_names or []),
            str(config["data"]["test_start_date"]),
            str(research_prediction_end or config["data"]["end_date"]),
        )
        del dataset, handler, prediction_frames
        gc.collect()

        labeled = predictions.dropna(subset=["label"])
        if labeled.empty:
            raise RuntimeError("no realized out-of-sample labels are available for backtesting")
        ic, rank_ic = daily_ic(labeled)
        signal_metrics = pd.concat([ic, rank_ic], axis=1)
        if research_control is not None:
            expected_metric_sessions = pd.DatetimeIndex(
                validated_request["metric_contract"]["signal"]["sessions"], name="datetime"
            )
            if not signal_metrics.index.equals(expected_metric_sessions):
                raise RuntimeError(
                    "stage signal metric dates do not exactly equal the authorized sessions"
                )
        signal_metrics.to_parquet(run_dir / "signal_metrics.parquet")
        if research_control is not None and selected_factor_names:
            expected_metric_sessions = pd.DatetimeIndex(
                validated_request["metric_contract"]["signal"]["sessions"],
                name="datetime",
            )
            if not raw_factor_metrics.index.equals(expected_metric_sessions):
                raise RuntimeError(
                    "raw factor metric dates do not exactly equal the authorized sessions"
                )
        raw_factor_metrics.to_parquet(run_dir / "raw_factor_metrics.parquet")

        backtest_summaries: dict[str, dict[str, Any]] = {}
        base_slippage = int(config["execution"]["base_slippage_bps_per_side"])
        scenarios = sorted(
            set(int(value) for value in config["execution"]["stress_slippage_bps_per_side"]) | {base_slippage}
        )
        base_report = None
        base_indicators = None
        last_signal_date = (
            research_prediction_end
            if research_control is not None
            else shift_session(
                calendar,
                research_source_data_end,
                -int(config["data"]["label_horizon_bars"]),
            )
        )
        backtest_predictions = select_backtest_predictions(predictions, last_signal_date)
        first_signal_date = pd.Timestamp(
            backtest_predictions.index.get_level_values("datetime").min()
        ).date().isoformat()
        backtest_start, backtest_end = backtest_bounds(calendar, first_signal_date, last_signal_date)
        prepared_base_inputs = prepare_raw_backtest_inputs(
            config,
            backtest_predictions,
            backtest_start,
            backtest_end,
            pretraining_action_audit=pretraining_action_audit,
        )
        for slippage in scenarios:
            scenario_dir = run_dir / "backtests" / f"slippage_{slippage:02d}bps"
            scenario_dir.mkdir(parents=True)
            report, positions, indicator_frame, indicator_object, execution_frame, execution_summary = run_backtest(
                backtest_predictions,
                config,
                slippage,
                backtest_start,
                backtest_end,
                prepared_inputs=prepared_base_inputs,
            )
            report.to_parquet(scenario_dir / "report.parquet")
            indicator_frame.to_parquet(scenario_dir / "indicators.parquet")
            execution_frame.to_parquet(scenario_dir / "executions.parquet", index=False)
            positions.to_parquet(scenario_dir / "positions.parquet", index=False)
            indicator_object["symbol_attribution"].to_parquet(
                scenario_dir / "symbol_attribution.parquet", index=False
            )
            indicator_object["corporate_action_ledger"].to_parquet(
                scenario_dir / "corporate_actions.parquet", index=False
            )
            indicator_object["corporate_action_audit_summary"].to_parquet(
                scenario_dir / "corporate_action_audit_summary.parquet", index=False
            )
            indicator_object["corporate_action_audit_details"].to_parquet(
                scenario_dir / "corporate_action_audit_details.parquet", index=False
            )
            prepared_base_inputs["corporate_action_audit"].factor_changes.to_parquet(
                scenario_dir / "corporate_action_factor_changes.parquet", index=False
            )
            summary = summarize_backtest(report, indicator_frame, slippage, execution_summary)
            write_json_atomic(scenario_dir / "summary.json", summary)
            backtest_summaries[str(slippage)] = summary
            if slippage == base_slippage:
                base_report = report
                base_indicators = indicator_frame
            del positions, indicator_object
            gc.collect()

        if base_report is None:
            raise RuntimeError("base slippage backtest did not produce a report")
        base_executions = pd.read_parquet(
            run_dir / "backtests" / f"slippage_{base_slippage:02d}bps" / "executions.parquet"
        )
        base_execution_summary = backtest_summaries[str(base_slippage)]["execution"]
        for slippage in scenarios:
            if slippage <= base_slippage:
                continue
            cost_only_dir = run_dir / "backtests" / f"slippage_costonly_{slippage:02d}bps"
            cost_only_dir.mkdir(parents=True)
            cost_only_report, cost_only_execution_summary = _cost_only_stress_report(
                base_report,
                base_executions,
                base_execution_summary,
                slippage,
                base_slippage,
            )
            cost_only_report.to_parquet(cost_only_dir / "report.parquet")
            cost_only_summary = summarize_backtest(
                cost_only_report, base_indicators, slippage, cost_only_execution_summary
            )
            cost_only_summary["stress_mode"] = "cost_only"
            write_json_atomic(cost_only_dir / "summary.json", cost_only_summary)
            backtest_summaries[f"costonly_{slippage}"] = cost_only_summary
        research_fold_summaries = None
        if research_control is not None:
            research_fold_summaries = _run_research_backtest_folds(
                backtest_predictions,
                config,
                validated_request,
                calendar,
                prepared_base_inputs,
                run_dir,
            )
        for summary, fold in zip(fold_summaries, folds):
            fold_dates = backtest_predictions.index.get_level_values("datetime")
            fold_predictions = backtest_predictions.loc[
                (fold_dates >= pd.Timestamp(fold.test_start))
                & (fold_dates <= pd.Timestamp(fold.test_end))
            ]
            if fold_predictions.empty:
                raise RuntimeError(f"fold {fold.fold} has no realizable backtest predictions")
            fold_first = pd.Timestamp(fold_predictions.index.get_level_values("datetime").min()).date().isoformat()
            fold_last = pd.Timestamp(fold_predictions.index.get_level_values("datetime").max()).date().isoformat()
            fold_start, fold_end = backtest_bounds(calendar, fold_first, fold_last)
            prepared_fold_inputs = slice_prepared_raw_backtest_inputs(
                prepared_base_inputs, fold_predictions, fold_end
            )
            fold_dir = run_dir / "folds" / f"fold_{fold.fold:02d}" / "backtest"
            fold_dir.mkdir(parents=True, exist_ok=False)
            (
                fold_report,
                fold_positions,
                fold_indicators,
                fold_indicator_object,
                fold_executions,
                fold_execution_summary,
            ) = run_backtest(
                fold_predictions,
                config,
                base_slippage,
                fold_start,
                fold_end,
                prepared_inputs=prepared_fold_inputs,
            )
            fold_report.to_parquet(fold_dir / "report.parquet")
            fold_indicators.to_parquet(fold_dir / "indicators.parquet")
            fold_executions.to_parquet(fold_dir / "executions.parquet", index=False)
            fold_positions.to_parquet(fold_dir / "positions.parquet", index=False)
            fold_indicator_object["symbol_attribution"].to_parquet(
                fold_dir / "symbol_attribution.parquet", index=False
            )
            fold_indicator_object["corporate_action_ledger"].to_parquet(
                fold_dir / "corporate_actions.parquet", index=False
            )
            fold_indicator_object["corporate_action_audit_summary"].to_parquet(
                fold_dir / "corporate_action_audit_summary.parquet", index=False
            )
            fold_indicator_object["corporate_action_audit_details"].to_parquet(
                fold_dir / "corporate_action_audit_details.parquet", index=False
            )
            prepared_fold_inputs["corporate_action_audit"].factor_changes.to_parquet(
                fold_dir / "corporate_action_factor_changes.parquet", index=False
            )
            summary["portfolio"] = independent_portfolio_performance(fold_report)
            summary["portfolio"].update(
                {
                    "start": fold_start,
                    "end": fold_end,
                    "complete_for_gate": fold_is_complete(
                        fold,
                        calendar,
                        int(config["rolling"]["test_days"]),
                        last_signal_date,
                    ),
                    "execution": fold_execution_summary,
                }
            )
            write_json_atomic(
                run_dir / "folds" / f"fold_{int(summary['fold']):02d}" / "summary.json", summary
            )
            del fold_positions, fold_indicator_object
            gc.collect()

        coverage_inputs = load_qlib_coverage_inputs(
            config["data"]["market"],
            config["data"]["test_start_date"],
            last_signal_date,
            config["data"]["liquidity_expression"],
        )
        coverage = calculate_prediction_coverage(
            coverage_inputs.active_spans,
            coverage_inputs.calendar,
            coverage_inputs.calendar,
            coverage_inputs.eligibility,
            backtest_predictions,
        )
        metrics = {
            "base_slippage_bps_per_side": base_slippage,
            "base": backtest_summaries[str(base_slippage)],
            "stress": backtest_summaries,
            "prediction_rows": len(predictions),
            "labeled_prediction_rows": len(labeled),
            "backtest_prediction_rows": len(backtest_predictions),
            "last_realized_signal_date": last_signal_date,
            "backtest_start_date": backtest_start,
            "backtest_end_date": backtest_end,
            "prediction_coverage": coverage["coverage"],
            "prediction_coverage_audit": coverage,
            "prediction_days": predictions.index.get_level_values("datetime").nunique(),
            "prediction_instruments": predictions.index.get_level_values("instrument").nunique(),
            "ic": finite(float(ic.mean())),
            "ic_hac_t_stat": finite(hac_t_stat(ic)),
            "rank_ic": finite(float(rank_ic.mean())),
            "rank_ic_hac_t_stat": finite(hac_t_stat(rank_ic)),
            "folds": fold_summaries,
        }
        if research_fold_summaries is not None:
            metrics["research_folds"] = research_fold_summaries
        gates = enforce_research_exposure_gate(
            evaluate_gates(config, audit, fold_summaries, metrics), research_control
        )
        write_json_atomic(run_dir / "metrics.json", metrics)
        write_json_atomic(run_dir / "gates.json", gates)
        write_json_atomic(run_dir / "config.json", json_ready_config(config))

        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "created_at": created_at,
            "status": "reporting",
            "classification": gates["status"],
            "snapshot_id": audit.snapshot_id,
            "snapshot_manifest": _workspace_relative(audit.snapshot_dir / "manifest.json", workspace),
            "config": _workspace_relative(Path(config["_meta"]["config_path"]), workspace),
            "data": {
                "snapshot_id": audit.snapshot_id,
                "source_fingerprint": audit.report["source_fingerprint"],
                "snapshot_manifest": _workspace_relative(
                    audit.snapshot_dir / "manifest.json", workspace
                ),
                "universe_mode": audit.report["universe_mode"],
            },
            "code": initial_code_identity,
            "environment": {
                "python": (
                    f"{environment['python']['implementation']} "
                    f"{environment['python']['version']}"
                ),
                "platform": environment["platform"],
                # A source checkout has no installed ``pyqlib`` distribution
                # metadata. Its runtime tree is still content-hashed above.
                "qlib": _qlib_manifest_version(environment, qlib),
                "lightgbm": environment["packages"]["lightgbm"],
                "packages": environment["packages"],
                "lock": environment["lock"],
                "lightgbm_build": environment["lightgbm_build"],
                "opencl_loader": environment["opencl_loader"],
                "model_device_type": str(config["model"].get("device_type", "cpu")),
                "gpu_probe_passed": True,
            },
            "git": initial_git_state,
            "artifacts": {
                "predictions": "predictions.parquet",
                "signal_metrics": "signal_metrics.parquet",
                "raw_factor_metrics": "raw_factor_metrics.parquet",
                "metrics": "metrics.json",
                "gates": "gates.json",
                "report": "report.html",
                "pretraining_corporate_action_audit": (
                    "audits/corporate_actions_pretraining/summary.json"
                ),
                "artifact_checksums": "artifact_checksums.json",
                "integrity_seal": "integrity_seal.json",
                "environment_lock": environment_lock_path.name,
            },
        }
        if frozen_factor_catalog is not None:
            manifest["factor_catalog"] = frozen_factor_catalog
        if research_control is not None:
            manifest["research"] = research_control
            manifest["artifacts"]["research_request"] = "research_request.json"
        write_json_atomic(manifest_path, manifest)
        from .report import generate_report

        generate_report(run_dir)

        final_audit = audit_and_snapshot(config)
        if (
            final_audit.snapshot_id != audit.snapshot_id
            or final_audit.report.get("source_fingerprint") != audit.report.get("source_fingerprint")
        ):
            raise RuntimeError("data source changed after the initial audit")
        if runtime_code_identity(source_root, qlib_package_root) != initial_code_identity:
            raise RuntimeError("pipeline or imported Qlib code changed while the run was executing")

        manifest["completed_at"] = now_shanghai().isoformat()
        registry_record = {
                "run_id": run_id,
                "created_at": created_at,
                "completed_at": manifest["completed_at"],
                "status": "completed",
                "classification": gates["status"],
                "run_dir": _workspace_relative(run_dir, workspace),
                "snapshot_id": audit.snapshot_id,
                "metrics": {
                    "net_cumulative_return": metrics["base"]["net_cumulative_return"],
                    "benchmark_cumulative_return": metrics["base"]["benchmark_cumulative_return"],
                    "ic": metrics["ic"],
                    "rank_ic": metrics["rank_ic"],
                    "max_drawdown": metrics["base"]["strategy_max_drawdown"],
                    "relative_wealth_max_drawdown": metrics["base"]["relative_wealth_max_drawdown"],
                },
            }
        try:
            update_registry(Path(config["paths"]["registry"]), registry_record)
            manifest["registry"] = {"updated": True}
        except Exception as registry_exc:
            manifest["registry"] = {
                "updated": False,
                "error": _sanitize_workspace_text(
                    f"{type(registry_exc).__name__}: {registry_exc}", workspace
                ),
            }
        _seal_completed_run(run_dir, manifest)
        return run_dir
    except Exception as exc:
        sanitized_traceback = _sanitize_workspace_text(traceback.format_exc(), workspace)
        failure = {
            **manifest,
            "completed_at": now_shanghai().isoformat(),
            "status": "failed",
            "error": _sanitize_workspace_text(f"{type(exc).__name__}: {exc}", workspace),
            "traceback": sanitized_traceback,
        }
        if isinstance(failure.get("integrity"), dict):
            failure["integrity"] = {
                **failure["integrity"],
                "verified": False,
                "seal_invalidated_by_failure": True,
            }
        write_json_atomic(manifest_path, failure)
        try:
            update_registry(
                Path(config["paths"]["registry"]),
                {
                "run_id": run_id,
                "created_at": created_at,
                "completed_at": failure["completed_at"],
                "status": "failed",
                "classification": "invalid",
                "run_dir": _workspace_relative(run_dir, workspace),
                "error": failure["error"],
                },
            )
        except Exception:
            pass
        raise
