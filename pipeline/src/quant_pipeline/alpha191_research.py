"""Build Alpha191 factors on the ETF dataset and produce an IC report.

The runner loads per-symbol normalized CSVs (``data/cn_etf/normalized``),
pivots them into the wide layout required by :mod:`quant_pipeline.alpha191`,
computes the selected factors, persists a long parquet table, and writes a
per-factor IC / RankIC report for the configured forward-return horizons.

Example
-------
.. code-block:: powershell

    $env:PYTHONPATH = (Resolve-Path .\\qlib\\pipeline\\src).Path
    C:\\Exception\\quant\\python.exe -m quant_pipeline.alpha191_research ^
        --config .\\qlib\\pipeline\\configs\\alpha191_build.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quant_pipeline.alpha191 import (
    Alpha191Data,
    build_all_factors,
    factor_metadata_table,
)

_FIELD_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "vwap": "vwap",
}


@dataclass
class Alpha191BuildConfig:
    data_dir: str = "data/cn_etf"
    universe: str = "data/cn_etf/trainable_universe_2015.csv"
    benchmark: str = "SH510300"
    start_date: str = "2015-01-01"
    end_date: str = "2026-08-12"
    min_days: int = 60
    drop_paused: bool = True
    output_dir: str = "pipeline/runs/alpha191"
    factor_numbers: List[int] = field(default_factory=list)
    horizons: List[int] = field(default_factory=lambda: [1, 5])
    recent_years: List[int] = field(default_factory=lambda: [2])


def _resolve_repo_path(path: Path) -> Path:
    """Resolve a config path against the working directory or the ``qlib`` repo root.

    The pipeline is normally invoked from the repository root that contains
    ``qlib/`` (the My_Quant directory); config paths such as
    ``data/cn_etf/...`` live inside the ``qlib`` repo, so we fall back to
    ``qlib/<path>`` when the literal path does not exist.
    """

    if path.exists():
        return path
    for candidate in (Path.cwd() / "qlib" / path, Path(path).absolute()):
        if candidate.exists():
            return candidate
    return path


def _load_universe_symbols(universe_file: Path) -> List[str]:
    universe_file = _resolve_repo_path(universe_file)
    if not universe_file.exists():
        raise FileNotFoundError(f"universe file not found: {universe_file}")
    frame = pd.read_csv(universe_file)
    col = "symbol" if "symbol" in frame.columns else frame.columns[0]
    return [str(x) for x in frame[col].tolist()]


def load_wide_data(config: Alpha191BuildConfig) -> Tuple[Alpha191Data, pd.DatetimeIndex, List[str]]:
    """Load normalized CSVs, pivot to wide OHLCV, and align a common calendar."""

    data_dir = _resolve_repo_path(Path(config.data_dir))
    norm_dir = data_dir / "normalized"
    universe_symbols = _load_universe_symbols(Path(config.universe))
    wanted = set(universe_symbols)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    frames: Dict[str, Dict[str, pd.DataFrame]] = {field: {} for field in _FIELD_COLUMNS}
    calendars: List[pd.DatetimeIndex] = []
    used_symbols: List[str] = []
    for symbol in sorted(wanted):
        path = norm_dir / f"{symbol.lower()}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, parse_dates=["date"])
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
        if config.drop_paused and "paused" in raw.columns:
            raw = raw[raw["paused"] == 0]
        if len(raw) < config.min_days:
            continue
        raw = raw.set_index("date").sort_index()
        used_symbols.append(symbol)
        calendars.append(raw.index)
        for field, column in _FIELD_COLUMNS.items():
            if column not in raw.columns:
                continue
            frames[field][symbol] = raw[column].astype(float)

    if not used_symbols:
        raise ValueError("no symbols survived the filters; check universe and date range")

    calendar = pd.DatetimeIndex(sorted(set().union(*calendars)))
    wide: Dict[str, pd.DataFrame] = {}
    for field, series in frames.items():
        if not series:
            continue
        wide[field] = pd.DataFrame(series, index=calendar).sort_index()
        wide[field] = wide[field].reindex(columns=used_symbols)

    benchmark_close: Optional[pd.Series] = None
    benchmark_open: Optional[pd.Series] = None
    bench_path = norm_dir / f"{config.benchmark.lower()}.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path, parse_dates=["date"])
        bench = bench[(bench["date"] >= start) & (bench["date"] <= end)].set_index("date").sort_index()
        if config.drop_paused and "paused" in bench.columns:
            bench = bench[bench["paused"] == 0]
        benchmark_close = bench["close"].reindex(calendar).astype(float)
        benchmark_open = bench["open"].reindex(calendar).astype(float)
    else:
        print(f"[warn] benchmark {config.benchmark} not found; benchmark-relative alphas will be all-NaN")

    data = Alpha191Data(
        open=wide["open"],
        high=wide["high"],
        low=wide["low"],
        close=wide["close"],
        volume=wide["volume"],
        amount=wide.get("amount"),
        vwap=wide.get("vwap"),
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
    )
    return data, calendar, used_symbols


def _daily_corr(x: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Per-date cross-sectional Pearson correlation across symbol columns.

    Returns ``(ic, days)``: daily IC values and the count of valid symbol
    pairs per date. Dates with fewer than 5 valid pairs yield NaN IC.
    """

    xarr = x.to_numpy(dtype=float)
    yarr = y.to_numpy(dtype=float)
    valid = np.isfinite(xarr) & np.isfinite(yarr)
    xv = np.where(valid, xarr, np.nan)
    yv = np.where(valid, yarr, np.nan)
    n = valid.sum(axis=1)
    count = np.maximum(n, 1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = xv - np.nansum(xv, axis=1, keepdims=True) / count
        ym = yv - np.nansum(yv, axis=1, keepdims=True) / count
        cov = np.nansum(xm * ym, axis=1)
        var_x = np.nansum(xm * xm, axis=1)
        var_y = np.nansum(ym * ym, axis=1)
        ic = cov / np.sqrt(var_x * var_y)
    ic = np.where((n >= 5) & (var_x > 0) & (var_y > 0), ic, np.nan)
    return ic, n


def compute_ic_report(
    factors: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizons: Sequence[int],
    recent_years: Sequence[int],
) -> pd.DataFrame:
    """Per-factor daily cross-sectional IC / RankIC with t-stats.

    Forward return over ``h`` sessions is ``close(t+h)/close(t) - 1``. For each
    date the Pearson IC and Spearman RankIC (Pearson on cross-sectional ranks)
    of the factor against the forward return are computed across symbols; the
    t-statistic is the classic ``mean / std * sqrt(n_days)`` of the daily
    IC series.
    """

    rows: List[dict] = []
    end = close.index.max()
    fwd_cache: Dict[int, pd.DataFrame] = {h: close.shift(-h) / close - 1.0 for h in horizons}
    for name, factor in sorted(factors.items()):
        for h in horizons:
            fwd = fwd_cache[h]
            for years in recent_years:
                if years == 0:
                    f, fw = factor, fwd
                else:
                    cutoff = end - pd.DateOffset(years=years)
                    f, fw = factor[factor.index >= cutoff], fwd[fwd.index >= cutoff]
                ic, _ = _daily_corr(f, fw)
                f_rank = f.rank(axis=1)
                fw_rank = fw.rank(axis=1)
                ric, _ = _daily_corr(f_rank, fw_rank)
                ic = ic[np.isfinite(ic)]
                ric = ric[np.isfinite(ric)]
                n_days = len(ic)
                if n_days == 0:
                    continue
                rows.append(
                    {
                        "factor": name,
                        "horizon": h,
                        "window": f"last_{years}y" if years else "full",
                        "days": n_days,
                        "ic": float(np.mean(ic)),
                        "ic_std": float(np.std(ic)),
                        "ic_t": float(np.mean(ic) / (np.std(ic) + 1e-12) * np.sqrt(n_days)),
                        "rank_ic": float(np.mean(ric)),
                        "rank_ic_std": float(np.std(ric)),
                        "rank_ic_t": float(np.mean(ric) / (np.std(ric) + 1e-12) * np.sqrt(n_days)),
                        "ic_positive_ratio": float((ic > 0).mean()),
                        "coverage": float(f.notna().to_numpy().mean()),
                    }
                )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["factor", "horizon", "window"])
    return report


def run(config: Alpha191BuildConfig, progress: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Execute the build; returns a dict of produced artifact paths."""

    started = time.time()
    output_dir = _resolve_repo_path(Path(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    data, calendar, symbols = load_wide_data(config)
    print(f"[load] {len(symbols)} symbols, {len(calendar)} sessions, "
          f"{calendar.min().date()}..{calendar.max().date()}")

    def _progress(name: str) -> None:
        if progress is not None:
            print(f"  [{name}]", flush=True)

    numbers = config.factor_numbers or None
    factors, errors = build_all_factors(data, numbers=numbers, on_error="warn", progress=_progress)
    print(f"[compute] {len(factors)} factors, {len(errors)} errors")
    for error in errors:
        print(f"  [error] {error}")

    # Per-factor wide parquet files (index=date, columns=symbols).  A single
    # merged long table would be ~symbols*sessions*191 cells and is avoided by
    # default; stack_factors() remains available for small subsets.
    factors_dir = output_dir / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in sorted(factors.items()):
        frame.to_parquet(factors_dir / f"{name}.parquet")
    print(f"[save] {len(factors)} wide factor files -> {factors_dir}")

    metadata = factor_metadata_table()
    metadata_path = output_dir / "alpha191_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    report_path, markdown_path = write_report(factors, data.close, config, output_dir)

    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "symbols": len(symbols),
        "sessions": len(calendar),
        "calendar_start": str(calendar.min().date()),
        "calendar_end": str(calendar.max().date()),
        "factors_computed": len(factors),
        "factor_errors": errors,
        "artifacts": {
            "factors_dir": str(factors_dir),
            "metadata": str(metadata_path),
            "ic_report_csv": str(report_path),
            "ic_report_md": str(markdown_path),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] elapsed {summary['elapsed_seconds']}s; artifacts under {output_dir}")
    return summary["artifacts"]


def load_factors_dir(factors_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load per-factor wide parquet files written by :func:`run`."""

    factors: Dict[str, pd.DataFrame] = {}
    for path in sorted(factors_dir.glob("alpha*.parquet")):
        factors[path.stem] = pd.read_parquet(path)
    return factors


def write_report(
    factors: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    config: Alpha191BuildConfig,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Compute and persist the IC report; returns (csv_path, markdown_path)."""

    report = compute_ic_report(factors, close, config.horizons, config.recent_years)
    report_path = output_dir / "alpha191_ic_report.csv"
    report.to_csv(report_path, index=False)
    markdown_path = output_dir / "alpha191_ic_report.md"
    _write_markdown(report, markdown_path)
    print(f"[report] {report_path} ({len(report)} rows)")
    return report_path, markdown_path


def _write_markdown(report: pd.DataFrame, path: Path) -> None:
    lines = ["# Alpha191 因子 IC 报告", ""]
    if report.empty:
        lines.append("（无有效结果）")
    else:
        top = report.loc[report["horizon"] == 1].copy()
        top = top.sort_values("rank_ic", key=lambda s: s.abs(), ascending=False)
        lines.append("## 按 |RankIC|（1日前瞻）排序 Top 20")
        lines.append("")
        lines.append("| 因子 | 窗口 | IC | RankIC | RankIC_t | IC正比例 | 覆盖率 |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in top.head(20).iterrows():
            lines.append(
                f"| {row['factor']} | {row['window']} | {row['ic']:.4f} | {row['rank_ic']:.4f} | "
                f"{row['rank_ic_t']:.2f} | {row['ic_positive_ratio']:.2f} | {row['coverage']:.2f} |"
            )
        lines.append("")
        lines.append("## 说明")
        lines.append("")
        lines.append("- IC：每日截面 Pearson 相关；RankIC：Spearman 相关；t = mean/std*sqrt(days)。")
        lines.append("- 覆盖率为因子非空比例；因子为复权价计算，未做中性化。")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_config(path: Path) -> Alpha191BuildConfig:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    alpha = payload.get("alpha191", payload)
    kwargs = {k: v for k, v in alpha.items() if k in Alpha191BuildConfig.__dataclass_fields__}
    return Alpha191BuildConfig(**kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Alpha191 factors and IC report")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (alpha191 section)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-days", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--numbers", default=None, help="comma-separated alpha numbers, e.g. 1,2,101")
    parser.add_argument("--horizons", default=None, help="comma-separated forward horizons in sessions")
    parser.add_argument("--report-only", action="store_true", help="reuse saved factors and only rewrite the IC report")
    args = parser.parse_args(argv)

    config = _parse_config(args.config) if args.config else Alpha191BuildConfig()
    for attr, value in [
        ("data_dir", args.data_dir),
        ("universe", args.universe),
        ("benchmark", args.benchmark),
        ("start_date", args.start_date),
        ("end_date", args.end_date),
        ("min_days", args.min_days),
        ("output_dir", args.output_dir),
    ]:
        if value is not None:
            setattr(config, attr, value)
    if args.numbers is not None:
        config.factor_numbers = [int(x) for x in args.numbers.split(",") if x.strip()]
    if args.horizons is not None:
        config.horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    if args.report_only:
        output_dir = _resolve_repo_path(Path(config.output_dir))
        factors_dir = output_dir / "factors"
        if not factors_dir.exists():
            raise FileNotFoundError(f"no saved factors under {factors_dir}; run the build first")
        factors = load_factors_dir(factors_dir)
        data, calendar, symbols = load_wide_data(config)
        print(f"[load] {len(symbols)} symbols, {len(calendar)} sessions (report-only)")
        write_report(factors, data.close, config, output_dir)
        return 0

    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
