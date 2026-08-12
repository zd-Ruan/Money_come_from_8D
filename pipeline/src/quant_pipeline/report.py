from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .io import read_json


PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def fmt_pct(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def fmt_num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def cumulative(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def chart_html(figure: go.Figure, height: int = 350, percent_y: bool = False) -> str:
    figure.update_layout(
        height=height,
        margin=dict(l=52, r=24, t=56, b=44),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12, color="#25313a"),
        title_font=dict(size=15, color="#17212a"),
        hoverlabel=dict(bgcolor="#17212a", font_color="#ffffff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, linecolor="#dce2e5", zeroline=False)
    figure.update_yaxes(gridcolor="#edf0f2", zeroline=False, tickformat=".1%" if percent_y else None)
    return figure.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)


def _performance_chart(report: pd.DataFrame) -> go.Figure:
    net = report["return"].fillna(0.0) - report["cost"].fillna(0.0)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=report.index, y=cumulative(net), name="策略扣费后", line=dict(color="#14765a", width=3)))
    figure.add_trace(
        go.Scatter(x=report.index, y=cumulative(report["return"]), name="策略毛收益", line=dict(color="#3478a5", width=2))
    )
    figure.add_trace(
        go.Scatter(x=report.index, y=cumulative(report["bench"]), name="基准", line=dict(color="#c98218", width=2))
    )
    figure.update_layout(title="滚动样本外累计收益")
    figure.update_yaxes(title="累计收益")
    figure.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.07)
    return figure


def _drawdown_chart(report: pd.DataFrame) -> go.Figure:
    net_curve = 1.0 + cumulative(report["return"].fillna(0.0) - report["cost"].fillna(0.0))
    drawdown = net_curve / net_curve.cummax() - 1.0
    figure = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill="tozeroy",
            line=dict(color="#b44b4b", width=1.5),
            fillcolor="rgba(180,75,75,0.15)",
            name="策略回撤",
        )
    )
    figure.update_layout(title="策略扣费后回撤", showlegend=False)
    figure.update_yaxes(title="回撤")
    return figure


def _stress_chart(stress: dict[str, dict[str, Any]]) -> go.Figure:
    rows = pd.DataFrame(stress.values()).sort_values("slippage_bps_per_side")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=rows["slippage_bps_per_side"].astype(str),
            y=rows["net_cumulative_return"],
            name="策略扣费后",
            marker_color="#14765a",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["slippage_bps_per_side"].astype(str),
            y=rows["benchmark_cumulative_return"],
            name="基准",
            line=dict(color="#c98218", width=2),
        )
    )
    figure.update_layout(title="单边滑点压力测试", barmode="group", hovermode="x")
    figure.update_xaxes(title="额外滑点（bp/边）")
    figure.update_yaxes(title="累计收益")
    return figure


def _signal_chart(signal: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=signal.index, y=signal["ic"], name="日度 IC", marker_color="#78a6c8", opacity=0.35))
    figure.add_trace(
        go.Scatter(
            x=signal.index,
            y=signal["ic"].rolling(20, min_periods=10).mean(),
            name="IC 20日均值",
            line=dict(color="#215e83", width=2.5),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=signal.index,
            y=signal["rank_ic"].rolling(20, min_periods=10).mean(),
            name="Rank IC 20日均值",
            line=dict(color="#b8751a", width=2.5),
        )
    )
    figure.add_hline(y=0, line_color="#8a959b", line_width=1)
    figure.update_layout(title="样本外信号稳定性")
    figure.update_yaxes(title="相关系数")
    return figure


def _fold_chart(folds: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(folds)
    figure = go.Figure()
    figure.add_trace(go.Bar(x=frame["fold"].astype(str), y=frame["ic"], name="IC", marker_color="#3478a5"))
    figure.add_trace(go.Bar(x=frame["fold"].astype(str), y=frame["rank_ic"], name="Rank IC", marker_color="#c98218"))
    figure.add_hline(y=0, line_color="#8a959b", line_width=1)
    figure.update_layout(title="各滚动窗口 IC", barmode="group", hovermode="x")
    figure.update_xaxes(title="Fold")
    figure.update_yaxes(title="相关系数")
    return figure


def _coverage_chart(predictions: pd.DataFrame) -> go.Figure:
    coverage = predictions["score"].groupby(level="datetime").count()
    uncertainty = predictions["score_std"].groupby(level="datetime").mean()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=coverage.index,
            y=coverage,
            name="有效 ETF 数",
            line=dict(color="#3478a5", width=2),
            fill="tozeroy",
            fillcolor="rgba(52,120,165,0.10)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=uncertainty.index,
            y=uncertainty,
            name="种子间预测标准差",
            line=dict(color="#704c9a", width=2),
            yaxis="y2",
        )
    )
    figure.update_layout(
        title="预测覆盖与随机种子不确定性",
        yaxis=dict(title="ETF 数量"),
        yaxis2=dict(title="预测标准差", overlaying="y", side="right", showgrid=False),
    )
    return figure


def _monthly_heatmap(report: pd.DataFrame) -> go.Figure:
    net = report["return"].fillna(0.0) - report["cost"].fillna(0.0)
    monthly = net.groupby([net.index.year, net.index.month]).apply(lambda values: (1.0 + values).prod() - 1.0)
    heatmap = monthly.unstack().reindex(columns=range(1, 13))
    values = heatmap.to_numpy()
    limit = max(0.05, float(np.nanmax(np.abs(values))))
    labels = np.where(np.isnan(values), "", np.vectorize(lambda value: f"{value:.1%}")(np.nan_to_num(values)))
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=[f"{month}月" for month in range(1, 13)],
            y=heatmap.index.astype(str),
            zmin=-limit,
            zmax=limit,
            zmid=0,
            colorscale=[[0, "#b44b4b"], [0.5, "#f3f4f4"], [1, "#14765a"]],
            text=labels,
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}<br>扣费后收益 %{z:.2%}<extra></extra>",
            colorbar=dict(title="月收益", tickformat=".0%"),
        )
    )
    figure.update_layout(title="扣费后月度收益", hovermode="closest")
    figure.update_yaxes(autorange="reversed")
    return figure


def generate_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "metrics.json")
    gates = read_json(run_dir / "gates.json")
    if not all([manifest, config, metrics, gates]):
        raise FileNotFoundError("run is missing manifest, config, metrics, or gates")

    base_slippage = int(metrics["base_slippage_bps_per_side"])
    base_dir = run_dir / "backtests" / f"slippage_{base_slippage:02d}bps"
    report = pd.read_parquet(base_dir / "report.parquet")
    report.index = pd.to_datetime(report.index)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    signal = pd.read_parquet(run_dir / "signal_metrics.parquet")
    signal.index = pd.to_datetime(signal.index)
    base = metrics["base"]

    check_rows = []
    for check in gates["checks"]:
        check_rows.append(
            "<tr>"
            f"<td>{html.escape(check['name'])}</td>"
            f"<td><span class='status {'pass' if check['passed'] else 'fail'}'>{'通过' if check['passed'] else '未通过'}</span></td>"
            f"<td class='num'>{html.escape(str(check.get('value', '-')))}</td>"
            f"<td class='num'>{html.escape(str(check.get('threshold', '-')))}</td>"
            "</tr>"
        )

    fold_rows = []
    for fold in metrics["folds"]:
        fold_rows.append(
            "<tr>"
            f"<td>{fold['fold']}</td><td>{fold['train_start']} 至 {fold['train_end']}</td>"
            f"<td>{fold['valid_start']} 至 {fold['valid_end']}</td><td>{fold['test_start']} 至 {fold['test_end']}</td>"
            f"<td class='num'>{fold['rows']['train']:,}</td><td class='num'>{fold['rows']['test_features']:,}</td>"
            f"<td class='num'>{fmt_num(fold['ic'], 4)}</td><td class='num'>{fmt_num(fold['rank_ic'], 4)}</td>"
            f"<td class='num'>{fmt_pct(fold['portfolio']['excess_cumulative_return'])}</td>"
            f"<td class='num'>{html.escape(str(fold['best_iterations']))}</td>"
            "</tr>"
        )

    classification = gates["status"]
    badge_text = "候选模型" if classification == "candidate" else "仅限研究"
    badge_class = "pass" if classification == "candidate" else "fail"
    point_in_time_warning = next(
        (check for check in gates["checks"] if check["name"] == "point_in_time_universe"), None
    )
    warning = (
        "历史时点 ETF 池尚未建立，当前结果存在幸存者偏差，系统不会允许晋级为实盘候选。"
        if point_in_time_warning and not point_in_time_warning["passed"]
        else "历史时点 ETF 池门禁已通过。"
    )

    css = """
    :root { --ink:#17212a; --muted:#64727b; --line:#dce2e5; --soft:#f5f7f7;
            --green:#14765a; --red:#b44b4b; --blue:#3478a5; --amber:#c98218; --purple:#704c9a; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#fff; font-family:"Microsoft YaHei","Segoe UI",sans-serif; letter-spacing:0; }
    header { border-bottom:1px solid var(--line); }
    .inner, main { width:min(1460px, calc(100% - 40px)); margin:0 auto; min-width:0; }
    .inner { min-height:96px; display:flex; justify-content:space-between; align-items:center; gap:24px; }
    h1 { margin:0 0 7px; font-size:24px; line-height:1.3; overflow-wrap:anywhere; }
    h2 { margin:0; font-size:18px; }
    .sub, .meta, .section-note { color:var(--muted); font-size:12px; line-height:1.65; }
    .meta { text-align:right; font-variant-numeric:tabular-nums; }
    main { padding:26px 0 48px; }
    .summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:6px; }
    .metric { padding:18px; min-height:104px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); }
    .metric:nth-child(4n) { border-right:0; }
    .label { color:var(--muted); font-size:12px; margin-bottom:9px; }
    .value { font-size:26px; line-height:1; font-weight:700; font-variant-numeric:tabular-nums; }
    .note { color:var(--muted); font-size:11px; margin-top:9px; }
    .positive { color:var(--green); } .negative { color:var(--red); }
    .alert { margin:20px 0; padding:14px 16px; border-left:4px solid var(--red); background:#fff4f2; font-size:13px; line-height:1.7; }
    .alert-head { display:flex; align-items:center; gap:10px; margin-bottom:3px; font-weight:700; }
    .status { display:inline-block; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:700; }
    .status.pass { color:#0d6248; background:#e8f5ef; } .status.fail { color:#9e3434; background:#fdebea; }
    .section { margin-top:34px; }
    .section-head { display:flex; justify-content:space-between; align-items:end; gap:18px; border-bottom:1px solid var(--line); padding-bottom:9px; margin-bottom:10px; }
    .grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .chart, .wide { min-width:0; border-bottom:1px solid var(--line); }
    .table-wrap { overflow:auto; border:1px solid var(--line); max-height:520px; }
    table { width:100%; border-collapse:collapse; min-width:780px; font-size:12px; }
    th { position:sticky; top:0; z-index:1; background:#eef1f2; text-align:left; padding:10px 12px; border-bottom:1px solid #cbd1d5; white-space:nowrap; }
    td { padding:9px 12px; border-bottom:1px solid #edf0f2; white-space:nowrap; }
    tbody tr:hover { background:#f5faf8; } th.num, td.num { text-align:right; font-variant-numeric:tabular-nums; }
    footer { border-top:1px solid var(--line); margin-top:34px; padding-top:15px; color:var(--muted); font-size:11px; line-height:1.7; }
    @media(max-width:900px) { .inner { align-items:flex-start; flex-direction:column; padding:17px 0; gap:8px; } .meta{text-align:left;} .summary,.grid-2{grid-template-columns:repeat(2,minmax(0,1fr));} .metric:nth-child(4n){border-right:1px solid var(--line);} .metric:nth-child(2n){border-right:0;} }
    @media(max-width:560px) { .inner,main{width:calc(100% - 24px);} h1{font-size:21px;} .summary,.grid-2{grid-template-columns:1fr;} .metric{border-right:0;} .section-head{align-items:flex-start;flex-direction:column;gap:5px;} }
    """

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(config['report']['title'])}</title><style>{css}</style><script>{get_plotlyjs()}</script></head>
<body><header><div class="inner"><div><h1>{html.escape(config['report']['title'])}</h1>
<div class="sub">Alpha158 · LightGBM 三种子集成 · Purged 滚动样本外 · 容量约束回测</div></div>
<div class="meta">Run {html.escape(manifest['run_id'])}<br>Snapshot {html.escape(manifest['snapshot_id'])}</div></div></header>
<main>
  <section class="summary">
    <div class="metric"><div class="label">运行分类</div><div class="value"><span class="status {badge_class}">{badge_text}</span></div><div class="note">质量门禁 {gates['passed']} / {gates['total']}</div></div>
    <div class="metric"><div class="label">基础场景净累计收益</div><div class="value {'positive' if base['net_cumulative_return'] >= 0 else 'negative'}">{fmt_pct(base['net_cumulative_return'])}</div><div class="note">单边滑点 {base_slippage}bp + 佣金 {config['execution']['commission_bps_per_side']}bp</div></div>
    <div class="metric"><div class="label">基准累计收益</div><div class="value">{fmt_pct(base['benchmark_cumulative_return'])}</div><div class="note">{html.escape(config['data']['benchmark'])}</div></div>
    <div class="metric"><div class="label">策略最大回撤</div><div class="value negative">{fmt_pct(base['strategy_max_drawdown'])}</div><div class="note">成交率 {fmt_pct(base['fill_rate'])}</div></div>
    <div class="metric"><div class="label">IC / t 值</div><div class="value">{fmt_num(metrics['ic'], 4)}</div><div class="note">t = {fmt_num(metrics['ic_t_stat'], 2)}</div></div>
    <div class="metric"><div class="label">Rank IC / t 值</div><div class="value">{fmt_num(metrics['rank_ic'], 4)}</div><div class="note">t = {fmt_num(metrics['rank_ic_t_stat'], 2)}</div></div>
    <div class="metric"><div class="label">扣费超额 HAC t</div><div class="value">{fmt_num(base['excess_hac_t_stat'], 2)}</div><div class="note">信息比率 {fmt_num(base['information_ratio'], 2)}</div></div>
    <div class="metric"><div class="label">市场暴露</div><div class="value">β {fmt_num(base['beta'], 2)}</div><div class="note">beta 调整年化 alpha {fmt_pct(base['beta_adjusted_alpha_annualized'])}</div></div>
  </section>
  <div class="alert"><div class="alert-head"><span class="status {badge_class}">{badge_text}</span>可信度结论</div>{html.escape(warning)}</div>

  <section class="section"><div class="section-head"><h2>收益与风险</h2><div class="section-note">所有日期均为严格样本外；基础成本含真实佣金、5bp 滑点和 5% 成交量上限</div></div>
    <div class="wide">{chart_html(_performance_chart(report), 470, True)}</div>
    <div class="grid-2"><div class="chart">{chart_html(_drawdown_chart(report), 350, True)}</div><div class="chart">{chart_html(_monthly_heatmap(report), 350)}</div></div>
  </section>
  <section class="section"><div class="section-head"><h2>稳健性</h2><div class="section-note">成本压力与滚动窗口必须同时通过，不能只看总净值</div></div>
    <div class="grid-2"><div class="chart">{chart_html(_stress_chart(metrics['stress']), 370, True)}</div><div class="chart">{chart_html(_fold_chart(metrics['folds']), 370)}</div></div>
    <div class="wide">{chart_html(_signal_chart(signal), 420)}</div><div class="wide">{chart_html(_coverage_chart(predictions), 380)}</div>
  </section>
  <section class="section"><div class="section-head"><h2>质量门禁</h2><div class="section-note">未通过任一强制门禁时只能保留为 research_only</div></div>
    <div class="table-wrap"><table><thead><tr><th>门禁</th><th>状态</th><th class="num">实际值</th><th class="num">阈值</th></tr></thead><tbody>{''.join(check_rows)}</tbody></table></div>
  </section>
  <section class="section"><div class="section-head"><h2>滚动训练审计</h2><div class="section-note">训练、验证、测试之间均隔离 {config['rolling']['purge_bars']} 个交易日</div></div>
    <div class="table-wrap"><table><thead><tr><th>Fold</th><th>训练期</th><th>验证期</th><th>测试期</th><th class="num">训练样本</th><th class="num">测试样本</th><th class="num">IC</th><th class="num">Rank IC</th><th class="num">组合超额</th><th class="num">最佳迭代</th></tr></thead><tbody>{''.join(fold_rows)}</tbody></table></div>
  </section>
  <footer>本报告只读取冻结运行产物，不在网页端重新计算关键指标。状态为“仅限研究”时不得将结果解释为实盘预期。数据、配置、模型、预测、持仓和成本场景均保存在 {html.escape(str(run_dir))}。</footer>
</main></body></html>"""

    output = run_dir / "report.html"
    output.write_text(document, encoding="utf-8")
    return output
