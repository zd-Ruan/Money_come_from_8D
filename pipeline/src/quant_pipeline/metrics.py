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


def relative_wealth_drawdown(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Maximum drawdown of strategy wealth relative to benchmark wealth."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    strategy_wealth = (1.0 + aligned.iloc[:, 0]).cumprod()
    benchmark_wealth = (1.0 + aligned.iloc[:, 1]).cumprod()
    relative = strategy_wealth / benchmark_wealth
    return float((relative / relative.cummax() - 1.0).min())


def evaluation_frame(report: pd.DataFrame, *, gross_tolerance: float = 1e-10) -> pd.DataFrame:
    """Align Qlib close-execution returns with the first realizable benchmark period.

    The first raw row is the initial close execution for the earliest signal. It has
    no holding-period return, but its transaction cost belongs to the first realized
    strategy return. Combining that cost with the next row keeps terminal wealth
    exact while removing benchmark performance that occurred before the strategy
    could be invested.
    """
    required = {"return", "cost", "bench"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"report is missing required columns: {sorted(missing)}")
    if len(report) < 2:
        raise ValueError("report needs an execution row and at least one realization row")
    if abs(float(report["return"].fillna(0.0).iloc[0])) > gross_tolerance:
        raise ValueError("initial execution row contains a non-zero gross return")

    raw_net = report["return"].fillna(0.0) - report["cost"].fillna(0.0)
    result = pd.DataFrame(index=report.index[1:])
    result["strategy_net"] = raw_net.iloc[1:].to_numpy()
    result.iloc[0, result.columns.get_loc("strategy_net")] = (
        (1.0 + float(raw_net.iloc[0])) * (1.0 + float(raw_net.iloc[1])) - 1.0
    )
    result["benchmark"] = report["bench"].fillna(0.0).iloc[1:].to_numpy()
    result["excess"] = result["strategy_net"] - result["benchmark"]

    raw_wealth = float((1.0 + raw_net).prod())
    aligned_wealth = float((1.0 + result["strategy_net"]).prod())
    if not math.isclose(raw_wealth, aligned_wealth, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("evaluation alignment changed terminal strategy wealth")
    return result


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
    """Legacy IID t-stat retained for backward-compatible report loading."""
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
    relative_excess = (1.0 + net_return) / (1.0 + benchmark_return) - 1.0
    return {
        "days": len(period),
        "net_cumulative_return": finite(net_return),
        "benchmark_cumulative_return": finite(benchmark_return),
        "excess_cumulative_return": finite(relative_excess),
    }


def independent_portfolio_performance(report: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize a reset-cash fold backtest on realizable holding periods."""
    aligned = evaluation_frame(report)
    strategy = compounded_return(aligned["strategy_net"])
    benchmark = compounded_return(aligned["benchmark"])
    relative_excess = (1.0 + strategy) / (1.0 + benchmark) - 1.0
    return {
        "days": len(aligned),
        "net_cumulative_return": finite(strategy),
        "benchmark_cumulative_return": finite(benchmark),
        "excess_cumulative_return": finite(relative_excess),
        "initial_execution_date": pd.Timestamp(report.index[0]).date().isoformat(),
        "evaluation_start_date": pd.Timestamp(aligned.index[0]).date().isoformat(),
        "evaluation_end_date": pd.Timestamp(aligned.index[-1]).date().isoformat(),
        "alignment_method": "initial_cost_compounded_into_first_realized_return",
        "reset_cash": True,
    }


def finite(value: float | int | None) -> float | int | None:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
