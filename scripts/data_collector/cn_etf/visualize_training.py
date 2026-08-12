#!/usr/bin/env python
"""Generate an offline HTML report from a Qlib ETF training run."""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


CUR_DIR = Path(__file__).resolve().parent
QLIB_ROOT = CUR_DIR.parents[2]
DEFAULT_MLRUNS_DIR = QLIB_ROOT / "mlruns"
DEFAULT_OUTPUT = QLIB_ROOT / "data" / "cn_etf" / "training_report_latest.html"
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def read_simple_yaml(path: Path) -> dict[str, str]:
    """Read the scalar key/value subset used by MLflow meta.yaml files."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.startswith((" ", "-")):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def is_complete_run(run_dir: Path) -> bool:
    meta_path = run_dir / "meta.yaml"
    artifacts = run_dir / "artifacts"
    if not meta_path.exists() or not artifacts.is_dir():
        return False
    meta = read_simple_yaml(meta_path)
    required = [
        artifacts / "pred.pkl",
        artifacts / "label.pkl",
        artifacts / "sig_analysis" / "ic.pkl",
        artifacts / "sig_analysis" / "ric.pkl",
        artifacts / "portfolio_analysis" / "report_normal_1day.pkl",
    ]
    return meta.get("status") == "3" and all(path.exists() for path in required)


def find_latest_run(mlruns_dir: Path) -> Path:
    candidates = [path.parent for path in mlruns_dir.glob("*/*/meta.yaml") if is_complete_run(path.parent)]
    if not candidates:
        raise FileNotFoundError(f"no completed training run found under {mlruns_dir}")
    return max(candidates, key=lambda path: int(read_simple_yaml(path / "meta.yaml").get("end_time", "0")))


def read_metric(run_dir: Path, name: str) -> float:
    path = run_dir / "metrics" / name
    if not path.exists():
        return float("nan")
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    return float(line.split()[1])


def read_param(run_dir: Path, name: str, default: str = "-") -> str:
    path = run_dir / "params" / name
    return path.read_text(encoding="utf-8").strip() if path.exists() else default


def load_series(path: Path, name: str) -> pd.Series:
    value = pd.read_pickle(path)
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    series = pd.Series(value, name=name).dropna()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def cumulative_return(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def drawdown(cumulative: pd.Series) -> pd.Series:
    wealth = 1.0 + cumulative
    return wealth / wealth.cummax() - 1.0


def fmt_pct(value: float, digits: int = 2) -> str:
    return "-" if pd.isna(value) else f"{value * 100:.{digits}f}%"


def fmt_float(value: float, digits: int = 3) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}f}"


def chart_html(figure: go.Figure, height: int = 360, percent_y: bool = False) -> str:
    figure.update_layout(
        height=height,
        margin=dict(l=52, r=24, t=58, b=46),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12, color="#263238"),
        title_font=dict(size=16, color="#182026"),
        hoverlabel=dict(bgcolor="#182026", font_color="#ffffff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, linecolor="#dfe3e6", zeroline=False)
    figure.update_yaxes(gridcolor="#edf0f2", zeroline=False, tickformat=".1%" if percent_y else None)
    return figure.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)


def annual_summary(
    predictions: pd.DataFrame,
    report: pd.DataFrame,
    ic: pd.Series,
    rank_ic: pd.Series,
) -> pd.DataFrame:
    years = sorted(set(report.index.year) | set(ic.index.year))
    rows = []
    for year in years:
        year_report = report[report.index.year == year]
        year_pred = predictions[predictions.index.get_level_values("datetime").year == year]
        net_daily = year_report["return"].fillna(0.0) - year_report["cost"].fillna(0.0)
        net_curve = cumulative_return(net_daily)
        rows.append(
            {
                "year": year,
                "period": "全年" if year < report.index.max().year else "截至报告期",
                "days": len(year_report),
                "predictions": len(year_pred),
                "instruments": year_pred.index.get_level_values("instrument").nunique(),
                "net_return": cumulative_return(net_daily).iloc[-1],
                "gross_return": cumulative_return(year_report["return"]).iloc[-1],
                "benchmark_return": cumulative_return(year_report["bench"]).iloc[-1],
                "max_drawdown": drawdown(net_curve).min(),
                "ic": ic[ic.index.year == year].mean(),
                "rank_ic": rank_ic[rank_ic.index.year == year].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_report(run_dir: Path, output: Path) -> Path:
    run_dir = run_dir.resolve()
    if not is_complete_run(run_dir):
        raise ValueError(f"run is not complete or required artifacts are missing: {run_dir}")

    artifacts = run_dir / "artifacts"
    meta = read_simple_yaml(run_dir / "meta.yaml")
    pred = pd.read_pickle(artifacts / "pred.pkl")
    label = pd.read_pickle(artifacts / "label.pkl")
    predictions = pred.join(label).dropna()
    predictions.index = predictions.index.set_levels(
        pd.to_datetime(predictions.index.levels[0]), level="datetime"
    )

    report = pd.read_pickle(artifacts / "portfolio_analysis" / "report_normal_1day.pkl").copy()
    report.index = pd.to_datetime(report.index)
    report = report.sort_index()
    ic = load_series(artifacts / "sig_analysis" / "ic.pkl", "IC")
    rank_ic = load_series(artifacts / "sig_analysis" / "ric.pkl", "Rank IC")

    net_daily = report["return"].fillna(0.0) - report["cost"].fillna(0.0)
    gross_curve = cumulative_return(report["return"])
    net_curve = cumulative_return(net_daily)
    benchmark_curve = cumulative_return(report["bench"])
    excess_daily = net_daily - report["bench"].fillna(0.0)
    excess_curve = cumulative_return(excess_daily)
    net_drawdown = drawdown(net_curve)
    excess_drawdown = drawdown(excess_curve)

    metric_ic = read_metric(run_dir, "IC")
    metric_rank_ic = read_metric(run_dir, "Rank IC")
    annual_excess = read_metric(run_dir, "1day.excess_return_with_cost.annualized_return")
    information_ratio = read_metric(run_dir, "1day.excess_return_with_cost.information_ratio")
    excess_max_drawdown = read_metric(run_dir, "1day.excess_return_with_cost.max_drawdown")
    annual = annual_summary(predictions, report, ic, rank_ic)

    performance_fig = go.Figure()
    performance_fig.add_trace(
        go.Scatter(x=net_curve.index, y=net_curve, name="策略扣费后", line=dict(color="#13795b", width=3))
    )
    performance_fig.add_trace(
        go.Scatter(x=gross_curve.index, y=gross_curve, name="策略毛收益", line=dict(color="#3478a5", width=2))
    )
    performance_fig.add_trace(
        go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name="沪深300 ETF 基准", line=dict(color="#d28b28", width=2))
    )
    performance_fig.update_layout(title="累计收益曲线")
    performance_fig.update_yaxes(title="累计收益")
    performance_fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.07)

    drawdown_fig = go.Figure()
    drawdown_fig.add_trace(
        go.Scatter(
            x=net_drawdown.index,
            y=net_drawdown,
            name="策略净值回撤",
            fill="tozeroy",
            line=dict(color="#b44b4b", width=1.5),
            fillcolor="rgba(180,75,75,0.16)",
        )
    )
    drawdown_fig.add_trace(
        go.Scatter(
            x=excess_drawdown.index,
            y=excess_drawdown,
            name="扣费超额回撤",
            line=dict(color="#704c9a", width=1.8),
        )
    )
    drawdown_fig.update_layout(title="回撤")
    drawdown_fig.update_yaxes(title="回撤幅度")

    annual_fig = go.Figure()
    annual_fig.add_trace(
        go.Bar(x=annual["year"].astype(str), y=annual["net_return"], name="策略扣费后", marker_color="#13795b")
    )
    annual_fig.add_trace(
        go.Bar(x=annual["year"].astype(str), y=annual["benchmark_return"], name="基准", marker_color="#d28b28")
    )
    annual_fig.update_layout(title="分年度收益", barmode="group", hovermode="x")
    annual_fig.update_yaxes(title="区间收益")

    ic_fig = go.Figure()
    ic_fig.add_trace(
        go.Bar(x=ic.index, y=ic, name="日度 IC", marker_color="#78a6c8", opacity=0.42)
    )
    ic_fig.add_trace(
        go.Bar(x=rank_ic.index, y=rank_ic, name="日度 Rank IC", marker_color="#d9a65a", opacity=0.32)
    )
    ic_fig.add_trace(
        go.Scatter(x=ic.index, y=ic.rolling(20, min_periods=10).mean(), name="IC 20日均值", line=dict(color="#215e83", width=2.5))
    )
    ic_fig.add_trace(
        go.Scatter(
            x=rank_ic.index,
            y=rank_ic.rolling(20, min_periods=10).mean(),
            name="Rank IC 20日均值",
            line=dict(color="#b8751a", width=2.5),
        )
    )
    ic_fig.add_hline(y=0, line_color="#8b969c", line_width=1)
    ic_fig.update_layout(title="日度 IC 与 20 日滚动均值", barmode="overlay")
    ic_fig.update_yaxes(title="相关系数")

    ranked = predictions.copy()
    ranked["bucket"] = ranked.groupby(level="datetime")["score"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 10, labels=False) + 1
    )
    deciles = ranked.groupby("bucket").agg(score=("score", "mean"), label=("LABEL0", "mean"), count=("score", "size"))
    decile_fig = go.Figure()
    decile_fig.add_trace(
        go.Bar(
            x=[f"D{i}" for i in deciles.index],
            y=deciles["label"],
            marker_color=["#b44b4b" if value < 0 else "#13795b" for value in deciles["label"]],
            text=[f"{value:.3f}" for value in deciles["label"]],
            textposition="outside",
            name="标签均值",
        )
    )
    decile_fig.update_layout(title="预测分位与归一化标签均值", showlegend=False, hovermode="x")
    decile_fig.update_xaxes(title="预测得分分位（D10 最高）")
    decile_fig.update_yaxes(title="LABEL0 均值")

    coverage = pred.groupby(level="datetime")["score"].count()
    coverage_fig = go.Figure()
    coverage_fig.add_trace(
        go.Scatter(
            x=coverage.index,
            y=coverage,
            mode="lines",
            name="有效预测数",
            line=dict(color="#3478a5", width=2),
            fill="tozeroy",
            fillcolor="rgba(52,120,165,0.10)",
        )
    )
    coverage_fig.update_layout(title="每日有效预测覆盖", showlegend=False)
    coverage_fig.update_yaxes(title="ETF 数量", rangemode="tozero")

    execution_fig = go.Figure()
    execution_fig.add_trace(
        go.Scatter(
            x=report.index,
            y=report["turnover"],
            name="日换手率",
            line=dict(color="#d28b28", width=1.2),
            opacity=0.45,
        )
    )
    execution_fig.add_trace(
        go.Scatter(
            x=report.index,
            y=report["turnover"].rolling(20, min_periods=5).mean(),
            name="20日平均换手率",
            line=dict(color="#9b641d", width=2.5),
        )
    )
    execution_fig.update_layout(title="换手率")
    execution_fig.update_yaxes(title="换手率", tickformat=".0%")

    cost_fig = go.Figure()
    cost_fig.add_trace(
        go.Scatter(
            x=report.index,
            y=report["total_cost"],
            name="累计交易成本",
            line=dict(color="#704c9a", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(112,76,154,0.10)",
        )
    )
    cost_fig.update_layout(title="累计交易成本", showlegend=False)
    cost_fig.update_yaxes(title="人民币（元）", tickformat=",.0f")

    monthly = net_daily.groupby([net_daily.index.year, net_daily.index.month]).apply(
        lambda values: (1.0 + values).prod() - 1.0
    )
    heatmap = monthly.unstack(fill_value=np.nan).reindex(columns=range(1, 13))
    heat_limit = max(0.05, float(np.nanmax(np.abs(heatmap.to_numpy()))))
    month_names = [f"{month}月" for month in range(1, 13)]
    heatmap_fig = go.Figure(
        go.Heatmap(
            z=heatmap.to_numpy(),
            x=month_names,
            y=heatmap.index.astype(str),
            zmin=-heat_limit,
            zmax=heat_limit,
            zmid=0,
            colorscale=[[0, "#b44b4b"], [0.5, "#f3f4f4"], [1, "#13795b"]],
            text=np.where(np.isnan(heatmap.to_numpy()), "", np.vectorize(lambda value: f"{value:.1%}")(np.nan_to_num(heatmap.to_numpy()))),
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}<br>扣费后收益 %{z:.2%}<extra></extra>",
            colorbar=dict(title="月收益", tickformat=".0%"),
        )
    )
    heatmap_fig.update_layout(title="策略扣费后月度收益", hovermode="closest")
    heatmap_fig.update_yaxes(autorange="reversed")

    annual_rows = []
    for row in annual.itertuples(index=False):
        annual_rows.append(
            "<tr>"
            f"<td>{row.year}（{row.period}）</td>"
            f"<td class='num'>{row.days:,}</td>"
            f"<td class='num'>{row.predictions:,}</td>"
            f"<td class='num'>{row.instruments:,}</td>"
            f"<td class='num {'positive' if row.net_return >= 0 else 'negative'}'>{fmt_pct(row.net_return)}</td>"
            f"<td class='num'>{fmt_pct(row.benchmark_return)}</td>"
            f"<td class='num negative'>{fmt_pct(row.max_drawdown)}</td>"
            f"<td class='num'>{fmt_float(row.ic, 4)}</td>"
            f"<td class='num'>{fmt_float(row.rank_ic, 4)}</td>"
            "</tr>"
        )

    start_time = datetime.fromtimestamp(int(meta["start_time"]) / 1000).strftime("%Y-%m-%d %H:%M")
    end_time = datetime.fromtimestamp(int(meta["end_time"]) / 1000).strftime("%Y-%m-%d %H:%M")
    test_start = pred.index.get_level_values("datetime").min().strftime("%Y-%m-%d")
    test_end = pred.index.get_level_values("datetime").max().strftime("%Y-%m-%d")
    model_name = read_param(run_dir, "model.class")
    handler_name = read_param(run_dir, "dataset.kwargs.handler.class")
    market_name = read_param(run_dir, "dataset.kwargs.handler.kwargs.instruments")
    total_predictions = int(pred.notna().all(axis=1).sum())
    predicted_instruments = pred.index.get_level_values("instrument").nunique()
    avg_turnover = report["turnover"].mean()
    total_cost = report["total_cost"].iloc[-1]
    net_return = net_curve.iloc[-1]
    gross_return = gross_curve.iloc[-1]
    benchmark_return = benchmark_curve.iloc[-1]
    strategy_max_drawdown = net_drawdown.min()
    weak_signal = metric_ic < 0.02 or metric_rank_ic < 0.03
    signal_note = (
        "信号统计强度偏弱，当前收益对组合构建和测试期市场环境较敏感；建议继续做滚动训练、参数稳定性和样本外压力测试。"
        if weak_signal
        else "信号统计强度处于可继续验证区间，仍需通过滚动训练和样本外压力测试确认稳定性。"
    )

    css = """
    :root { --ink:#182026; --muted:#65727a; --line:#dfe3e6; --soft:#f5f7f7;
            --green:#13795b; --red:#b44b4b; --blue:#3478a5; --amber:#d28b28; --purple:#704c9a; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#fff; font-family:"Microsoft YaHei","Segoe UI",sans-serif; letter-spacing:0; }
    header { border-bottom:1px solid var(--line); background:#fff; }
    .header-inner, main { width:min(1440px, calc(100% - 40px)); min-width:0; margin:0 auto; }
    .header-inner { min-height:100px; display:flex; align-items:center; justify-content:space-between; gap:28px; }
    h1 { max-width:100%; font-size:26px; line-height:1.25; margin:0 0 8px; font-weight:700; white-space:normal; overflow-wrap:anywhere; }
    .subtitle, .run-meta { color:var(--muted); font-size:13px; line-height:1.65; }
    .run-meta { text-align:right; font-variant-numeric:tabular-nums; }
    main { padding:28px 0 48px; overflow-x:hidden; }
    .kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:26px; }
    .kpi { border:1px solid var(--line); border-radius:6px; padding:17px 18px; min-height:108px; }
    .kpi-label { color:var(--muted); font-size:12px; margin-bottom:10px; }
    .kpi-value { font-size:27px; line-height:1; font-weight:700; font-variant-numeric:tabular-nums; }
    .kpi-note { color:var(--muted); font-size:11px; margin-top:10px; white-space:normal; }
    .positive { color:var(--green); }
    .negative { color:var(--red); }
    .section { margin-top:36px; }
    .section-head { display:flex; align-items:end; justify-content:space-between; gap:20px; margin-bottom:12px; border-bottom:1px solid var(--line); padding-bottom:10px; }
    h2 { font-size:19px; margin:0; }
    .section-note { color:var(--muted); font-size:12px; text-align:right; }
    .grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }
    .chart { min-width:0; border-bottom:1px solid var(--line); }
    .wide-chart { min-width:0; border-bottom:1px solid var(--line); }
    .assessment { display:grid; grid-template-columns:180px minmax(0,1fr); gap:20px; align-items:center; border-left:4px solid var(--amber); background:#fff8eb; padding:15px 18px; margin:4px 0 24px; font-size:13px; line-height:1.7; }
    .assessment strong { font-size:14px; }
    .assessment span { min-width:0; white-space:normal; overflow-wrap:anywhere; }
    .details { display:flex; flex-wrap:wrap; gap:8px 24px; padding:12px 0 2px; color:var(--muted); font-size:12px; }
    .details b { color:var(--ink); font-weight:600; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); }
    table { width:100%; min-width:980px; border-collapse:collapse; font-size:12px; }
    th { background:#eef1f2; text-align:left; font-weight:600; padding:10px 12px; border-bottom:1px solid #cbd1d5; white-space:nowrap; }
    td { padding:10px 12px; border-bottom:1px solid #edf0f2; white-space:nowrap; }
    tbody tr:hover { background:#f5faf8; }
    th.num, td.num { text-align:right; font-variant-numeric:tabular-nums; }
    footer { color:var(--muted); font-size:11px; border-top:1px solid var(--line); margin-top:36px; padding-top:16px; line-height:1.7; }
    @media (max-width:900px) {
      .header-inner { align-items:flex-start; flex-direction:column; padding:18px 0; gap:8px; }
      .run-meta { text-align:left; }
      .kpis, .grid-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .assessment { grid-template-columns:1fr; gap:4px; }
    }
    @media (max-width:560px) {
      .header-inner, main { width:calc(100% - 24px); }
      h1 { font-size:22px; }
      .kpis, .grid-2 { grid-template-columns:1fr; }
      .section-head { align-items:flex-start; flex-direction:column; gap:6px; }
      .section-note { text-align:left; }
      .kpi { min-height:94px; }
    }
    """

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETF Alpha158 LightGBM 训练报告</title>
  <style>{css}</style>
  <script>{get_plotlyjs()}</script>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>ETF Alpha158 LightGBM 训练报告</h1>
        <div class="subtitle">A股 T+1 ETF · 收敛池 · 真实日线行情 · 严格样本外回测</div>
      </div>
      <div class="run-meta">Run {html.escape(meta.get('run_id', run_dir.name))}<br>{start_time} 至 {end_time}</div>
    </div>
  </header>
  <main>
    <section class="kpis" aria-label="核心指标">
      <div class="kpi"><div class="kpi-label">策略扣费后累计收益</div><div class="kpi-value positive">{fmt_pct(net_return)}</div><div class="kpi-note">毛收益 {fmt_pct(gross_return)} · 期末账户 {report['account'].iloc[-1]:,.0f} 元</div></div>
      <div class="kpi"><div class="kpi-label">基准累计收益</div><div class="kpi-value">{fmt_pct(benchmark_return)}</div><div class="kpi-note">基准 SH510300 · 同期区间</div></div>
      <div class="kpi"><div class="kpi-label">扣费超额年化</div><div class="kpi-value positive">{fmt_pct(annual_excess)}</div><div class="kpi-note">信息比率 {fmt_float(information_ratio, 2)}</div></div>
      <div class="kpi"><div class="kpi-label">扣费超额最大回撤</div><div class="kpi-value negative">{fmt_pct(excess_max_drawdown)}</div><div class="kpi-note">策略净值最大回撤 {fmt_pct(strategy_max_drawdown)}</div></div>
      <div class="kpi"><div class="kpi-label">IC / ICIR</div><div class="kpi-value">{fmt_float(metric_ic, 4)}</div><div class="kpi-note">ICIR {fmt_float(read_metric(run_dir, 'ICIR'), 3)}</div></div>
      <div class="kpi"><div class="kpi-label">Rank IC / ICIR</div><div class="kpi-value">{fmt_float(metric_rank_ic, 4)}</div><div class="kpi-note">Rank ICIR {fmt_float(read_metric(run_dir, 'Rank ICIR'), 3)}</div></div>
      <div class="kpi"><div class="kpi-label">有效预测</div><div class="kpi-value">{total_predictions:,}</div><div class="kpi-note">{predicted_instruments:,} 只 ETF · {coverage.size:,} 个交易日</div></div>
      <div class="kpi"><div class="kpi-label">交易成本</div><div class="kpi-value">{total_cost / 1e4:.1f} 万</div><div class="kpi-note">日均换手率 {fmt_pct(avg_turnover, 1)}</div></div>
    </section>

    <div class="assessment"><strong>模型判断</strong><span>{signal_note}</span></div>
    <div class="details">
      <span><b>模型</b> {html.escape(model_name)}</span><span><b>特征</b> {html.escape(handler_name)}</span>
      <span><b>市场</b> {html.escape(market_name)}</span><span><b>测试期</b> {test_start} 至 {test_end}</span>
      <span><b>数据截止</b> {read_param(run_dir, 'dataset.kwargs.handler.kwargs.end_time')}</span>
    </div>

    <section class="section">
      <div class="section-head"><h2>收益与风险</h2><div class="section-note">扣费后曲线由每日 return - cost 复利计算</div></div>
      <div class="wide-chart">{chart_html(performance_fig, 470, True)}</div>
      <div class="grid-2"><div class="chart">{chart_html(drawdown_fig, 370, True)}</div><div class="chart">{chart_html(annual_fig, 370, True)}</div></div>
      <div class="wide-chart">{chart_html(heatmap_fig, 330)}</div>
    </section>

    <section class="section">
      <div class="section-head"><h2>信号质量</h2><div class="section-note">IC 衡量预测得分与下一期归一化标签的横截面相关性</div></div>
      <div class="wide-chart">{chart_html(ic_fig, 430)}</div>
      <div class="grid-2"><div class="chart">{chart_html(decile_fig, 370)}</div><div class="chart">{chart_html(coverage_fig, 370)}</div></div>
    </section>

    <section class="section">
      <div class="section-head"><h2>执行与成本</h2><div class="section-note">买卖佣金均为 0.03%，包含最低佣金约束</div></div>
      <div class="grid-2"><div class="chart">{chart_html(execution_fig, 370)}</div><div class="chart">{chart_html(cost_fig, 370)}</div></div>
    </section>

    <section class="section">
      <div class="section-head"><h2>年度稳定性</h2><div class="section-note">2026 年为截至 {test_end} 的非完整年度</div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>期间</th><th class="num">回测日</th><th class="num">预测数</th><th class="num">覆盖 ETF</th><th class="num">扣费后收益</th><th class="num">基准收益</th><th class="num">策略最大回撤</th><th class="num">IC</th><th class="num">Rank IC</th></tr></thead>
          <tbody>{''.join(annual_rows)}</tbody>
        </table>
      </div>
    </section>

    <footer>本报告读取 Qlib/MLflow 原始训练产物生成。回测使用当前可投资 ETF 池，存在退市幸存者偏差；结果不构成投资建议。Run 目录：{html.escape(str(run_dir))}</footer>
  </main>
</body>
</html>"""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="MLflow run directory; defaults to the latest completed run")
    parser.add_argument("--mlruns-dir", type=Path, default=DEFAULT_MLRUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_dir = args.run_dir or find_latest_run(args.mlruns_dir)
    output = build_report(run_dir, args.output)
    print(output)


if __name__ == "__main__":
    main()
