import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\zhendong ruan\Downloads\Code\My_Quant\qlib")
sys.path.insert(0, str(ROOT / "pipeline" / "src"))

from quant_pipeline.raw_backtest import RawBacktestConfig, run_raw_backtest
from quant_pipeline.metrics import compounded_return
from quant_pipeline.runner import _cost_only_stress_report

run_dir = ROOT / "pipeline" / "runs" / "baseline_cpu_20260813"
config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

calendar = pd.read_csv(
    ROOT / "data" / "cn_etf" / "qlib_data" / "calendars" / "day.txt", header=None
)[0]
calendar = pd.DatetimeIndex(pd.to_datetime(calendar))

predictions = pd.read_parquet(run_dir / "predictions.parquet")
last_signal = pd.Timestamp("2026-08-07")
scores = predictions.loc[
    predictions.index.get_level_values("datetime") <= last_signal, "score"
]
first_signal = scores.index.get_level_values("datetime").min()
sessions = calendar[(calendar >= first_signal) & (calendar <= pd.Timestamp("2026-08-11"))]

symbols = sorted(set(scores.index.get_level_values("instrument")))
raw_dir = ROOT / "data" / "cn_etf" / "raw"
frames = []
for symbol in symbols:
    path = raw_dir / f"{symbol.lower()}.csv"
    frames.append(
        pd.read_csv(
            path,
            parse_dates=["date"],
            usecols=["date", "symbol", "raw_open", "raw_close", "raw_high", "raw_low", "volume"],
        )
    )
bars = pd.concat(frames, ignore_index=True)
bars = bars[bars["date"].isin(sessions)]

benchmark = pd.read_csv(
    raw_dir / "sh510300.csv",
    parse_dates=["date"],
    usecols=["date", "symbol", "raw_close"],
)
benchmark = benchmark[benchmark["date"].isin(sessions)]

actions = pd.DataFrame(
    columns=[
        "symbol",
        "record_date",
        "ex_date",
        "cash_payment_date",
        "cash_dividend_per_old_share",
        "share_ratio",
        "fractional_share_treatment",
    ]
)

policy = RawBacktestConfig.from_mapping(config)
print("config:", {k: getattr(policy, k) for k in ("topk", "n_drop", "hold_thresh", "risk_degree")})

results = {}
for slippage in (0, 5, 10, 15, 20, 30):
    result = run_raw_backtest(
        scores,
        bars,
        actions,
        sessions,
        RawBacktestConfig.from_mapping(
            {**config, "execution": {**config["execution"], "slippage_bps_per_side": slippage}}
        ),
        benchmark_close=benchmark,
        benchmark_symbol="SH510300",
        factor_jumps_pre_audited=True,
    )
    report = result.report
    results[slippage] = result
    print(
        f"full rerun {slippage:>2}bps: net_cum={report['account'].iloc[-1]/20000-1:.6f} "
        f"terminal={report['account'].iloc[-1]:.2f} total_cost={result.summary['cost_total']:.2f} "
        f"commission_eff_bps={result.summary['commission_effective_bps']:.2f} "
        f"carry_forward={result.summary['missing_market_data_carry_forward_count']}"
    )

base = results[5]
base_exec = base.executions
base_summary = dict(base.summary)
for slippage in (10, 15, 20, 30):
    report, exec_summary = _cost_only_stress_report(
        base.report, base_exec, base_summary, slippage, 5
    )
    print(
        f"cost-only {slippage:>2}bps: net_cum={report['account'].iloc[-1]/20000-1:.6f} "
        f"terminal={report['account'].iloc[-1]:.2f} total_cost={exec_summary['total_cost']:.2f}"
    )
