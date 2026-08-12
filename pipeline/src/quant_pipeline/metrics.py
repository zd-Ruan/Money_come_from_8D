from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def compounded_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    if returns.empty:
        return float("nan")
    total = 1.0 + compounded_return(returns)
    return float(total ** (periods / len(returns)) - 1.0) if total > 0 else -1.0


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def information_ratio(excess_returns: pd.Series, periods: int = 252) -> float:
    std = excess_returns.std(ddof=1)
    return float(excess_returns.mean() / std * math.sqrt(periods)) if std and np.isfinite(std) else float("nan")


def hac_t_stat(values: pd.Series, max_lag: int = 5) -> float:
    array = values.dropna().to_numpy(dtype=float)
    n_obs = len(array)
    if n_obs < max_lag + 3:
        return float("nan")
    centered = array - array.mean()
    long_run_variance = float(np.dot(centered, centered) / n_obs)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return float("nan")
    return float(array.mean() / math.sqrt(long_run_variance / n_obs))


def correlation_t_stat(values: pd.Series) -> float:
    clean = values.dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / (std / math.sqrt(len(clean)))) if len(clean) > 2 and std else float("nan")


def daily_ic(predictions: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    grouped = predictions.dropna(subset=["score", "label"]).groupby(level="datetime")
    ic = grouped.apply(lambda frame: frame["score"].corr(frame["label"]))
    rank_ic = grouped.apply(lambda frame: frame["score"].corr(frame["label"], method="spearman"))
    ic.name = "ic"
    rank_ic.name = "rank_ic"
    return ic, rank_ic


def beta_alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252) -> tuple[float, float]:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan"), float("nan")
    strategy = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]
    variance = benchmark.var(ddof=1)
    beta = float(strategy.cov(benchmark) / variance) if variance else float("nan")
    alpha = float((strategy.mean() - beta * benchmark.mean()) * periods)
    return beta, alpha


def period_portfolio_performance(report: pd.DataFrame, start: str, end: str) -> dict[str, float | int | None]:
    period = report.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if period.empty:
        return {
            "days": 0,
            "net_cumulative_return": None,
            "benchmark_cumulative_return": None,
            "excess_cumulative_return": None,
        }
    net_return = compounded_return(period["return"].fillna(0.0) - period["cost"].fillna(0.0))
    benchmark_return = compounded_return(period["bench"].fillna(0.0))
    return {
        "days": len(period),
        "net_cumulative_return": finite(net_return),
        "benchmark_cumulative_return": finite(benchmark_return),
        "excess_cumulative_return": finite(net_return - benchmark_return),
    }


def finite(value: float | int | None) -> float | int | None:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
