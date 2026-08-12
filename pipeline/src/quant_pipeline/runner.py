from __future__ import annotations

import gc
import json
import math
import pickle
import platform
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .audit import AuditResult, audit_and_snapshot
from .config import json_ready_config
from .io import git_state, now_shanghai, write_json_atomic
from .metrics import (
    annualized_return,
    beta_alpha,
    compounded_return,
    correlation_t_stat,
    daily_ic,
    finite,
    hac_t_stat,
    information_ratio,
    max_drawdown,
    period_portfolio_performance,
)
from .registry import update_registry
from .windows import RollingFold, build_rolling_folds, load_calendar, shift_session, validate_fold_boundaries


def make_run_id(project_name: str) -> str:
    stamp = now_shanghai().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{project_name}"


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
    return {
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
        "verbosity": -1,
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
    predictions: pd.DataFrame, config: dict[str, Any], slippage_bps: int, backtest_end: str
):
    from qlib.backtest import backtest

    strategy_config = config["strategy"]
    execution = config["execution"]
    commission = int(execution["commission_bps_per_side"])
    total_cost = (commission + slippage_bps) / 10000.0
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
    portfolio, indicators = backtest(
        start_time=config["data"]["test_start_date"],
        end_time=backtest_end,
        strategy=strategy,
        executor=executor,
        benchmark=config["data"]["benchmark"],
        account=float(execution["account"]),
        exchange_kwargs={
            "codes": config["data"]["market"],
            "limit_threshold": float(execution["limit_threshold"]),
            "deal_price": execution["deal_price"],
            "open_cost": total_cost,
            "close_cost": total_cost,
            "min_cost": float(execution["min_cost"]),
            "volume_threshold": ("current", f"{participation} * $volume"),
        },
    )
    report, positions = portfolio["1day"]
    indicator_frame, indicator_object = indicators["1day"]
    return report, positions, indicator_frame, indicator_object


def summarize_backtest(report: pd.DataFrame, indicators: pd.DataFrame, slippage_bps: int) -> dict[str, Any]:
    net = report["return"].fillna(0.0) - report["cost"].fillna(0.0)
    benchmark = report["bench"].fillna(0.0)
    excess = net - benchmark
    beta, alpha = beta_alpha(net, benchmark)
    fill_rate = float(indicators["ffr"].dropna().mean()) if "ffr" in indicators else float("nan")
    return {
        "slippage_bps_per_side": slippage_bps,
        "days": len(report),
        "net_cumulative_return": finite(compounded_return(net)),
        "net_annualized_return": finite(annualized_return(net)),
        "benchmark_cumulative_return": finite(compounded_return(benchmark)),
        "excess_annualized_return": finite(float(excess.mean() * 252)),
        "information_ratio": finite(information_ratio(excess)),
        "excess_hac_t_stat": finite(hac_t_stat(excess)),
        "strategy_max_drawdown": finite(max_drawdown(net)),
        "excess_max_drawdown": finite(max_drawdown(excess)),
        "beta": finite(beta),
        "beta_adjusted_alpha_annualized": finite(alpha),
        "strategy_benchmark_correlation": finite(float(net.corr(benchmark))),
        "average_daily_turnover": finite(float(report["turnover"].mean())),
        "total_cost": finite(float(report["total_cost"].iloc[-1])),
        "fill_rate": finite(fill_rate),
        "terminal_account": finite(float(report["account"].iloc[-1])),
    }


def evaluate_gates(
    config: dict[str, Any], audit: AuditResult, fold_summaries: list[dict], metrics: dict[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    base = metrics["base"]
    required_stress = str(int(gates["required_stress_slippage_bps"]))
    stress = metrics["stress"][required_stress]
    positive_folds = sum(
        (summary.get("portfolio", {}).get("excess_cumulative_return") or 0) > 0
        for summary in fold_summaries
    )
    fold_ratio = positive_folds / len(fold_summaries)

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
            "passed": (base["fill_rate"] or 0) >= float(gates["min_fill_rate"]),
            "value": base["fill_rate"],
            "threshold": gates["min_fill_rate"],
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
            "name": "ic_t_stat",
            "passed": (metrics["ic_t_stat"] or -math.inf) >= float(gates["min_ic_t_stat"]),
            "value": metrics["ic_t_stat"],
            "threshold": gates["min_ic_t_stat"],
        },
        {
            "name": "max_drawdown",
            "passed": abs(base["strategy_max_drawdown"] or 1) <= float(gates["max_strategy_drawdown"]),
            "value": base["strategy_max_drawdown"],
            "threshold": gates["max_strategy_drawdown"],
        },
        {
            "name": "stress_positive_excess",
            "passed": (stress["net_cumulative_return"] or -1) > (stress["benchmark_cumulative_return"] or 0),
            "value": (stress["net_cumulative_return"] or -1) - (stress["benchmark_cumulative_return"] or 0),
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

    run_id = run_id or make_run_id(config["project"]["name"])
    run_dir = Path(config["paths"]["runs"]) / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    created_at = now_shanghai().isoformat()
    manifest_path = run_dir / "manifest.json"
    manifest = {"run_id": run_id, "created_at": created_at, "status": "running"}
    write_json_atomic(manifest_path, manifest)

    try:
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
        handler = Alpha158(
            instruments=config["data"]["market"],
            start_time=config["data"]["start_date"],
            end_time=config["data"]["end_date"],
            fit_start_time=config["data"]["start_date"],
            fit_end_time=folds[0].train_end,
            label=([config["data"]["label"]], ["LABEL0"]),
            filter_pipe=[
                {
                    "filter_type": "ExpressionDFilter",
                    "rule_expression": config["data"]["liquidity_expression"],
                    "filter_start_time": None,
                    "filter_end_time": None,
                    "keep": False,
                }
            ],
        )
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
        backtest_end = shift_session(calendar, last_signal_date, int(config["data"]["label_horizon_bars"]))
        for slippage in scenarios:
            scenario_dir = run_dir / "backtests" / f"slippage_{slippage:02d}bps"
            scenario_dir.mkdir(parents=True)
            report, positions, indicator_frame, indicator_object = run_backtest(
                backtest_predictions, config, slippage, backtest_end
            )
            report.to_parquet(scenario_dir / "report.parquet")
            indicator_frame.to_parquet(scenario_dir / "indicators.parquet")
            with (scenario_dir / "positions.pkl").open("wb") as handle:
                pickle.dump(positions, handle)
            summary = summarize_backtest(report, indicator_frame, slippage)
            write_json_atomic(scenario_dir / "summary.json", summary)
            backtest_summaries[str(slippage)] = summary
            if slippage == base_slippage:
                base_report = report
                base_indicators = indicator_frame
            del positions, indicator_object
            gc.collect()

        if base_report is None:
            raise RuntimeError("base slippage backtest did not produce a report")
        for summary in fold_summaries:
            portfolio_end = min(pd.Timestamp(summary["test_end"]), pd.Timestamp(backtest_end)).date().isoformat()
            summary["portfolio"] = period_portfolio_performance(
                base_report, summary["test_start"], portfolio_end
            )
            summary["portfolio"]["start"] = summary["test_start"]
            summary["portfolio"]["end"] = portfolio_end
            write_json_atomic(
                run_dir / "folds" / f"fold_{int(summary['fold']):02d}" / "summary.json", summary
            )

        expected_rows = len(predictions)
        prediction_coverage = float(predictions["score"].notna().sum() / expected_rows) if expected_rows else 0.0
        metrics = {
            "base_slippage_bps_per_side": base_slippage,
            "base": backtest_summaries[str(base_slippage)],
            "stress": backtest_summaries,
            "prediction_rows": len(predictions),
            "labeled_prediction_rows": len(labeled),
            "backtest_prediction_rows": len(backtest_predictions),
            "last_realized_signal_date": last_signal_date,
            "backtest_end_date": backtest_end,
            "prediction_coverage": prediction_coverage,
            "prediction_days": predictions.index.get_level_values("datetime").nunique(),
            "prediction_instruments": predictions.index.get_level_values("instrument").nunique(),
            "ic": finite(float(ic.mean())),
            "ic_t_stat": finite(correlation_t_stat(ic)),
            "rank_ic": finite(float(rank_ic.mean())),
            "rank_ic_t_stat": finite(correlation_t_stat(rank_ic)),
            "folds": fold_summaries,
        }
        gates = evaluate_gates(config, audit, fold_summaries, metrics)
        write_json_atomic(run_dir / "metrics.json", metrics)
        write_json_atomic(run_dir / "gates.json", gates)
        write_json_atomic(run_dir / "config.json", json_ready_config(config))

        workspace = Path(config["_meta"]["workspace_root"])
        manifest = {
            "run_id": run_id,
            "created_at": created_at,
            "status": "reporting",
            "classification": gates["status"],
            "snapshot_id": audit.snapshot_id,
            "snapshot_manifest": str(audit.snapshot_dir / "manifest.json"),
            "config": str(config["_meta"]["config_path"]),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "qlib": getattr(qlib, "__version__", "unknown"),
                "lightgbm": lgb.__version__,
            },
            "git": git_state(workspace / "qlib"),
            "artifacts": {
                "predictions": "predictions.parquet",
                "signal_metrics": "signal_metrics.parquet",
                "metrics": "metrics.json",
                "gates": "gates.json",
                "report": "report.html",
            },
        }
        write_json_atomic(manifest_path, manifest)
        from .report import generate_report

        generate_report(run_dir)
        manifest["completed_at"] = now_shanghai().isoformat()
        manifest["status"] = "completed"
        write_json_atomic(manifest_path, manifest)
        update_registry(
            Path(config["paths"]["registry"]),
            {
                "run_id": run_id,
                "created_at": created_at,
                "completed_at": manifest["completed_at"],
                "status": "completed",
                "classification": gates["status"],
                "run_dir": str(run_dir.resolve()),
                "snapshot_id": audit.snapshot_id,
                "metrics": {
                    "net_cumulative_return": metrics["base"]["net_cumulative_return"],
                    "benchmark_cumulative_return": metrics["base"]["benchmark_cumulative_return"],
                    "ic": metrics["ic"],
                    "rank_ic": metrics["rank_ic"],
                    "max_drawdown": metrics["base"]["strategy_max_drawdown"],
                },
            },
        )
        return run_dir
    except Exception as exc:
        failure = {
            **manifest,
            "completed_at": now_shanghai().isoformat(),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(manifest_path, failure)
        update_registry(
            Path(config["paths"]["registry"]),
            {
                "run_id": run_id,
                "created_at": created_at,
                "completed_at": failure["completed_at"],
                "status": "failed",
                "classification": "invalid",
                "run_dir": str(run_dir.resolve()),
                "error": failure["error"],
            },
        )
        raise
