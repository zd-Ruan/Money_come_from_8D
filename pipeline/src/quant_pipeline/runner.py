from __future__ import annotations

import gc
import json
import math
import pickle
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .audit import AuditResult, audit_and_snapshot
from .config import json_ready_config
from .coverage import calculate_prediction_coverage, load_qlib_coverage_inputs
from .factors import build_alpha158_factor_handler, factor_catalog_manifest
from .integrity import generate_artifact_checksums, resolve_run_directory, source_tree_sha256, validate_run_id
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
from .small_account import SmallAccountExchange, summarize_execution_records
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


def run_backtest(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    slippage_bps: int,
    backtest_start: str,
    backtest_end: str,
):
    from qlib.backtest import backtest

    strategy_config = config["strategy"]
    execution = config["execution"]
    commission_rate = float(execution["commission_bps_per_side"]) / 10000.0
    strategy = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": predictions["score"],
            "topk": int(strategy_config["topk"]),
            "n_drop": int(strategy_config["n_drop"]),
            "hold_thresh": int(strategy_config["hold_thresh"]),
            "risk_degree": float(strategy_config["risk_degree"]),
            "only_tradable": True,
        },
    }
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "indicator_config": {"ffr_config": {"weight_method": "value_weighted"}},
        },
    }
    participation = float(execution["max_daily_volume_participation"])
    exchange = SmallAccountExchange(
        freq="day",
        start_time=backtest_start,
        end_time=backtest_end,
        codes=config["data"]["market"],
        deal_price=execution["deal_price"],
        limit_threshold=float(execution["limit_threshold"]),
        volume_threshold=("current", f"{participation} * $volume"),
        commission_rate=commission_rate,
        min_commission=float(execution["min_cost"]),
        slippage_bps=float(slippage_bps),
        trade_unit=int(execution["trade_unit"]),
    )
    portfolio, indicators = backtest(
        start_time=backtest_start,
        end_time=backtest_end,
        strategy=strategy,
        executor=executor,
        benchmark=config["data"]["benchmark"],
        account=float(execution["account"]),
        exchange_kwargs={"exchange": exchange},
    )
    report, positions = portfolio["1day"]
    indicator_frame, indicator_object = indicators["1day"]
    execution_records = exchange.execution_records
    execution_summary = summarize_execution_records(execution_records)
    execution_frame = pd.DataFrame(record.to_dict() for record in execution_records)
    return report, positions, indicator_frame, indicator_object, execution_frame, execution_summary


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
    qlib_total_cost = float(report["total_cost"].iloc[-1])
    ledger_total_cost = float(execution_summary.get("total_cost") or 0.0)
    if not math.isclose(qlib_total_cost, ledger_total_cost, rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError(
            f"execution ledger cost {ledger_total_cost} does not match portfolio cost {qlib_total_cost}"
        )
    relative_terminal = (1.0 + compounded_return(net)) / (1.0 + compounded_return(benchmark)) - 1.0
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
        "strategy_benchmark_correlation": finite(float(net.corr(benchmark))),
        "average_daily_turnover": finite(float(report["turnover"].mean())),
        "total_cost": finite(qlib_total_cost),
        "fill_rate": finite(execution_summary.get("fill_rate")),
        "execution": execution_summary,
        "average_cash_utilization": finite(float((1.0 - report["cash"] / report["account"]).mean())),
        "terminal_account": finite(float(report["account"].iloc[-1])),
    }


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


def run_pipeline(config: dict[str, Any], run_id: str | None = None) -> Path:
    import qlib
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    workspace = Path(config["_meta"]["workspace_root"]).resolve()
    source_root = Path(__file__).resolve().parent
    initial_source_sha256 = source_tree_sha256(source_root)
    initial_git_state = git_state(workspace)
    feature_mode = config["features"]["mode"]
    frozen_factor_catalog = (
        factor_catalog_manifest(config["features"].get("families") or None)
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
        "code": {"source_tree_sha256": initial_source_sha256},
        "git": initial_git_state,
    }
    if frozen_factor_catalog is not None:
        manifest["factor_catalog"] = frozen_factor_catalog
    write_json_atomic(manifest_path, manifest)

    try:
        validate_lightgbm_device(config)
        audit = audit_and_snapshot(config)
        if not audit.report["data_valid"]:
            raise RuntimeError(f"data quality gate failed: {audit.report['blocking_issues']}")

        provider = Path(config["paths"]["qlib_provider"])
        calendar_path = provider / "calendars" / "day.txt"
        calendar = load_calendar(calendar_path, config["data"]["start_date"], config["data"]["end_date"])
        folds = build_rolling_folds(
            calendar,
            train_start_date=config["data"]["start_date"],
            test_start_date=config["data"]["test_start_date"],
            validation_days=int(config["rolling"]["validation_days"]),
            test_days=int(config["rolling"]["test_days"]),
            purge_bars=int(config["rolling"]["purge_bars"]),
        )
        validate_fold_boundaries(folds, calendar)
        write_json_atomic(run_dir / "folds.json", [fold.to_dict() for fold in folds])

        qlib.init(provider_uri=str(provider), region=config["data"]["region"], kernels=4)
        handler_kwargs = {
            "instruments": config["data"]["market"],
            "start_time": config["data"]["start_date"],
            "end_time": config["data"]["end_date"],
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
                families=config["features"].get("families") or None,
            )
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
        del dataset, handler, prediction_frames
        gc.collect()

        labeled = predictions.dropna(subset=["label"])
        if labeled.empty:
            raise RuntimeError("no realized out-of-sample labels are available for backtesting")
        ic, rank_ic = daily_ic(labeled)
        signal_metrics = pd.concat([ic, rank_ic], axis=1)
        signal_metrics.to_parquet(run_dir / "signal_metrics.parquet")

        backtest_summaries: dict[str, dict[str, Any]] = {}
        base_slippage = int(config["execution"]["base_slippage_bps_per_side"])
        scenarios = sorted(
            set(int(value) for value in config["execution"]["stress_slippage_bps_per_side"]) | {base_slippage}
        )
        base_report = None
        base_indicators = None
        last_signal_date = shift_session(
            calendar,
            config["data"]["end_date"],
            -int(config["data"]["label_horizon_bars"]),
        )
        backtest_predictions = select_backtest_predictions(predictions, last_signal_date)
        first_signal_date = pd.Timestamp(
            backtest_predictions.index.get_level_values("datetime").min()
        ).date().isoformat()
        backtest_start, backtest_end = backtest_bounds(calendar, first_signal_date, last_signal_date)
        for slippage in scenarios:
            scenario_dir = run_dir / "backtests" / f"slippage_{slippage:02d}bps"
            scenario_dir.mkdir(parents=True)
            report, positions, indicator_frame, indicator_object, execution_frame, execution_summary = run_backtest(
                backtest_predictions, config, slippage, backtest_start, backtest_end
            )
            report.to_parquet(scenario_dir / "report.parquet")
            indicator_frame.to_parquet(scenario_dir / "indicators.parquet")
            execution_frame.to_parquet(scenario_dir / "executions.parquet", index=False)
            with (scenario_dir / "positions.pkl").open("wb") as handle:
                pickle.dump(positions, handle)
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
            fold_dir = run_dir / "folds" / f"fold_{fold.fold:02d}" / "backtest"
            fold_dir.mkdir(parents=True, exist_ok=False)
            (
                fold_report,
                fold_positions,
                fold_indicators,
                fold_indicator_object,
                fold_executions,
                fold_execution_summary,
            ) = run_backtest(fold_predictions, config, base_slippage, fold_start, fold_end)
            fold_report.to_parquet(fold_dir / "report.parquet")
            fold_indicators.to_parquet(fold_dir / "indicators.parquet")
            fold_executions.to_parquet(fold_dir / "executions.parquet", index=False)
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
        gates = evaluate_gates(config, audit, fold_summaries, metrics)
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
            "code": {"source_tree_sha256": initial_source_sha256},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "qlib": getattr(qlib, "__version__", "unknown"),
                "lightgbm": lgb.__version__,
                "model_device_type": str(config["model"].get("device_type", "cpu")),
            },
            "git": initial_git_state,
            "artifacts": {
                "predictions": "predictions.parquet",
                "signal_metrics": "signal_metrics.parquet",
                "metrics": "metrics.json",
                "gates": "gates.json",
                "report": "report.html",
                "artifact_checksums": "artifact_checksums.json",
            },
        }
        if frozen_factor_catalog is not None:
            manifest["factor_catalog"] = frozen_factor_catalog
        write_json_atomic(manifest_path, manifest)
        from .report import generate_report

        generate_report(run_dir)

        final_audit = audit_and_snapshot(config)
        if (
            final_audit.snapshot_id != audit.snapshot_id
            or final_audit.report.get("source_fingerprint") != audit.report.get("source_fingerprint")
        ):
            raise RuntimeError("data source changed after the initial audit")
        if source_tree_sha256(source_root) != initial_source_sha256:
            raise RuntimeError("pipeline source changed while the run was executing")

        checksum_path = generate_artifact_checksums(run_dir)
        from .integrity import verify_artifact_checksums

        integrity = verify_artifact_checksums(run_dir)
        if not integrity["valid"]:
            raise RuntimeError(f"artifact checksum verification failed: {integrity}")
        manifest["integrity"] = {
            "checksum_manifest": checksum_path.name,
            "checksum_sha256": sha256_file(checksum_path),
            "artifact_count": integrity["expected_count"],
            "verified": True,
        }
        manifest["completed_at"] = now_shanghai().isoformat()
        manifest["status"] = "completed"
        write_json_atomic(manifest_path, manifest)
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
        write_json_atomic(manifest_path, manifest)
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
