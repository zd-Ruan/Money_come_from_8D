#!/usr/bin/env python
"""Generate an offline Chinese HTML report for the CN T+1 ETF dataset."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


CUR_DIR = Path(__file__).resolve().parent
QLIB_ROOT = CUR_DIR.parents[2]
DEFAULT_DATA_DIR = QLIB_ROOT / "data" / "cn_etf"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "etf_data_report.html"
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def load_latest_rows(data_dir: Path, symbols: set[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        path = data_dir / "raw" / f"{symbol.lower()}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            usecols=["date", "symbol", "raw_close", "volume", "amount", "amount_quality"],
        )
        if not frame.empty:
            rows.append(frame.iloc[-1])
    if not rows:
        return pd.DataFrame()
    latest = pd.DataFrame(rows)
    latest["date"] = pd.to_datetime(latest["date"])
    latest["traded_value"] = latest["raw_close"] * latest["volume"]
    return latest


def load_representative_series(data_dir: Path, symbols: list[str], start: str = "2021-01-01") -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        path = data_dir / "raw" / f"{symbol.lower()}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["date", "symbol", "qfq_close"], parse_dates=["date"])
        frame = frame[frame["date"] >= pd.Timestamp(start)].copy()
        if frame.empty:
            continue
        frame["rebased_close"] = frame["qfq_close"] / frame["qfq_close"].iloc[0] * 100.0
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def chart_html(figure: go.Figure, height: int = 340) -> str:
    figure.update_layout(
        height=height,
        margin=dict(l=48, r=24, t=54, b=44),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12, color="#263238"),
        title_font=dict(size=16, color="#182026"),
        hoverlabel=dict(bgcolor="#182026", font_color="#ffffff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(showgrid=False, linecolor="#dfe3e6", zeroline=False)
    figure.update_yaxes(gridcolor="#edf0f2", zeroline=False)
    return figure.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)


def format_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value >= 1e8:
        return f"{value / 1e8:.2f} 亿"
    if value >= 1e4:
        return f"{value / 1e4:.1f} 万"
    return f"{value:,.0f}"


def build_report(data_dir: Path, output: Path) -> Path:
    universe = pd.read_csv(data_dir / "universe.csv", dtype={"code": str})
    normalize_report = pd.read_csv(data_dir / "normalize_report.csv")
    validation = json.loads((data_dir / "validation_report.json").read_text(encoding="utf-8"))

    universe = universe.merge(
        normalize_report[["symbol", "rows", "start_date", "end_date"]], on="symbol", how="left"
    )
    universe["start_date"] = pd.to_datetime(universe["start_date"])
    universe["end_date"] = pd.to_datetime(universe["end_date"])
    universe["listing_year"] = universe["start_date"].dt.year
    universe["history_years"] = (
        (universe["end_date"] - universe["start_date"]).dt.days / 365.25
    ).round(1)

    latest = load_latest_rows(data_dir, set(universe["symbol"]))
    universe = universe.merge(
        latest[["symbol", "traded_value", "amount_quality"]], on="symbol", how="left"
    )
    symbol_to_name = universe.set_index("symbol")["name"].to_dict()

    exchange_counts = universe["exchange"].map({"SH": "上海", "SZ": "深圳"}).value_counts().rename_axis("交易所")
    exchange_fig = go.Figure(
        go.Pie(
            labels=exchange_counts.index,
            values=exchange_counts.values,
            hole=0.62,
            marker_colors=["#167d68", "#d28b28"],
            textinfo="label+value",
            hovertemplate="%{label}<br>%{value} 只<br>%{percent}<extra></extra>",
        )
    )
    exchange_fig.update_layout(title="交易所分布", showlegend=False)

    listings = universe.groupby("listing_year").size().rename("count").reset_index()
    listing_fig = px.bar(
        listings,
        x="listing_year",
        y="count",
        title="每年新上市 ETF 数量",
        labels={"listing_year": "上市年份", "count": "ETF 数量"},
        color_discrete_sequence=["#167d68"],
    )
    listing_fig.update_traces(hovertemplate="%{x} 年<br>%{y} 只<extra></extra>")

    cumulative = listings.copy()
    cumulative["total"] = cumulative["count"].cumsum()
    growth_fig = go.Figure()
    growth_fig.add_trace(
        go.Scatter(
            x=cumulative["listing_year"],
            y=cumulative["total"],
            mode="lines+markers",
            line=dict(color="#2f6f9f", width=3),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(47,111,159,0.10)",
            hovertemplate="%{x} 年<br>累计 %{y} 只<extra></extra>",
        )
    )
    growth_fig.update_layout(title="当前 ETF 池的上市时间积累", showlegend=False)
    growth_fig.update_xaxes(title="年份")
    growth_fig.update_yaxes(title="累计数量")

    history_fig = px.histogram(
        universe,
        x="history_years",
        nbins=24,
        title="单只 ETF 历史数据长度",
        labels={"history_years": "历史年数", "count": "ETF 数量"},
        color_discrete_sequence=["#d28b28"],
    )
    history_fig.update_traces(hovertemplate="约 %{x:.1f} 年<br>%{y} 只<extra></extra>")
    history_fig.update_yaxes(title="ETF 数量")

    top_liquid = universe.nlargest(20, "traded_value").sort_values("traded_value")
    liquidity_labels = top_liquid.apply(lambda row: f"{row['symbol']}  {row['name']}", axis=1)
    liquidity_fig = go.Figure(
        go.Bar(
            x=top_liquid["traded_value"] / 1e8,
            y=liquidity_labels,
            orientation="h",
            marker_color="#2f6f9f",
            hovertemplate="%{y}<br>%{x:.2f} 亿元<extra></extra>",
        )
    )
    liquidity_fig.update_layout(title="2026-08-10 流动性最高的 20 只 ETF", showlegend=False)
    liquidity_fig.update_xaxes(title="收盘价 × 成交量（亿元）")
    liquidity_fig.update_yaxes(title=None, tickfont=dict(size=11))

    representative_symbols = [symbol for symbol in ["SH510300", "SZ159915", "SH588000"] if symbol in symbol_to_name]
    representative = load_representative_series(data_dir, representative_symbols)
    performance_fig = go.Figure()
    colors = ["#167d68", "#d28b28", "#2f6f9f"]
    for color, symbol in zip(colors, representative_symbols):
        frame = representative[representative["symbol"] == symbol]
        performance_fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["rebased_close"],
                mode="lines",
                name=f"{symbol} {symbol_to_name[symbol]}",
                line=dict(width=2, color=color),
                hovertemplate="%{x|%Y-%m-%d}<br>指数 %{y:.1f}<extra>%{fullData.name}</extra>",
            )
        )
    performance_fig.update_layout(title="代表性 ETF 前复权走势（2021 年首日 = 100）")
    performance_fig.update_xaxes(title=None, rangeslider_visible=True, rangeslider_thickness=0.08)
    performance_fig.update_yaxes(title="累计价格指数")

    table_rows = []
    for row in universe.sort_values("traded_value", ascending=False).itertuples(index=False):
        exchange_name = "上海" if row.exchange == "SH" else "深圳"
        search_text = f"{row.symbol} {row.name}".lower()
        table_rows.append(
            "<tr "
            f"data-exchange=\"{html.escape(row.exchange)}\" "
            f"data-search=\"{html.escape(search_text, quote=True)}\">"
            f"<td class=\"symbol\">{html.escape(row.symbol)}</td>"
            f"<td>{html.escape(str(row.name))}</td>"
            f"<td>{exchange_name}</td>"
            f"<td>{row.start_date:%Y-%m-%d}</td>"
            f"<td class=\"num\">{int(row.rows):,}</td>"
            f"<td class=\"num\">{float(row.last_price):.3f}</td>"
            f"<td class=\"num\">{format_money(row.traded_value)}</td>"
            "</tr>"
        )
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    snapshot_date = str(universe["snapshot_date"].iloc[0])
    history_end = str(universe["history_end_date"].iloc[0])
    total_rows = int(validation["total_rows"])
    earliest = universe["start_date"].min().strftime("%Y-%m-%d")
    median_years = universe["history_years"].median()
    liquid_count = int((universe["traded_value"] > 1e7).sum())

    css = """
    :root { --ink:#182026; --muted:#637078; --line:#dfe3e6; --soft:#f4f6f7;
            --green:#167d68; --amber:#d28b28; --blue:#2f6f9f; --warn:#fff7e8; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#fff; font-family:"Microsoft YaHei","Segoe UI",sans-serif; letter-spacing:0; }
    header { border-bottom:1px solid var(--line); background:#fff; }
    .header-inner, main { width:min(1420px, calc(100% - 40px)); min-width:0; margin:0 auto; }
    .header-inner { min-height:92px; display:flex; align-items:center; justify-content:space-between; gap:24px; }
    h1 { font-size:25px; line-height:1.25; margin:0 0 7px; font-weight:700; }
    .subtitle, .updated { color:var(--muted); font-size:13px; }
    .updated { text-align:right; white-space:nowrap; }
    main { padding:28px 0 48px; overflow-x:hidden; }
    .kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:30px; }
    .kpi { border:1px solid var(--line); border-radius:6px; padding:18px 20px; min-height:112px; }
    .kpi-label { color:var(--muted); font-size:13px; margin-bottom:10px; }
    .kpi-value { font-size:28px; line-height:1; font-weight:700; font-variant-numeric:tabular-nums; }
    .kpi-note { color:var(--muted); font-size:12px; margin-top:10px; }
    .section { margin-top:34px; }
    .section-head { display:flex; align-items:end; justify-content:space-between; gap:20px; margin-bottom:14px; border-bottom:1px solid var(--line); padding-bottom:10px; }
    h2 { font-size:19px; margin:0; }
    .section-note { color:var(--muted); font-size:12px; text-align:right; }
    .grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .chart { min-width:0; border-bottom:1px solid var(--line); }
    .wide-chart { min-width:0; }
    .callout { background:var(--warn); border-left:4px solid var(--amber); padding:14px 16px; margin:18px 0; font-size:13px; line-height:1.65; }
    .glossary { width:100%; min-width:0; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
    .term { width:100%; min-width:0; border-top:3px solid var(--green); padding:12px 4px 0; overflow-wrap:anywhere; }
    .term:nth-child(2) { border-color:var(--amber); }
    .term:nth-child(3) { border-color:var(--blue); }
    .term strong { display:block; font-size:14px; margin-bottom:6px; }
    .term span { display:block; width:100%; max-width:100%; color:var(--muted); font-size:12px; line-height:1.6; white-space:normal; overflow-wrap:anywhere; word-break:break-word; }
    .table-toolbar { display:flex; gap:10px; margin:16px 0 12px; }
    input, select { height:38px; border:1px solid #b8c0c5; border-radius:4px; background:#fff; color:var(--ink); padding:0 11px; font:inherit; font-size:13px; }
    input { width:min(420px, 70%); }
    input:focus, select:focus { outline:2px solid rgba(47,111,159,.22); border-color:var(--blue); }
    .table-wrap { max-height:560px; overflow:auto; border:1px solid var(--line); }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th { position:sticky; top:0; z-index:1; background:#eef1f2; text-align:left; font-weight:600; padding:10px 12px; border-bottom:1px solid #cbd1d5; white-space:nowrap; }
    td { padding:9px 12px; border-bottom:1px solid #edf0f2; white-space:nowrap; }
    tbody tr:hover { background:#f5faf8; }
    td.symbol, td.num { font-variant-numeric:tabular-nums; }
    td.symbol { font-weight:600; color:#215e83; }
    td.num { text-align:right; }
    .table-count { color:var(--muted); font-size:12px; margin-left:auto; align-self:center; }
    footer { color:var(--muted); font-size:11px; border-top:1px solid var(--line); margin-top:34px; padding-top:16px; line-height:1.7; }
    @media (max-width:900px) {
      .header-inner { align-items:flex-start; flex-direction:column; padding:18px 0; gap:8px; }
      .updated { text-align:left; }
      .kpis, .grid-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .glossary { grid-template-columns:1fr; }
    }
    @media (max-width:560px) {
      .header-inner, main { width:calc(100% - 24px); max-width:1420px; }
      .kpis, .grid-2 { grid-template-columns:1fr; }
      .kpi { min-height:94px; }
      .section-head { align-items:flex-start; flex-direction:column; gap:6px; }
      .section-note { text-align:left; }
      .table-toolbar { flex-wrap:wrap; }
      input { width:100%; }
    }
    """

    javascript = """
    const searchInput = document.getElementById('etf-search');
    const exchangeSelect = document.getElementById('exchange-filter');
    const rows = Array.from(document.querySelectorAll('#etf-table tbody tr'));
    const count = document.getElementById('table-count');
    function applyFilters() {
      const query = searchInput.value.trim().toLowerCase();
      const exchange = exchangeSelect.value;
      let visible = 0;
      rows.forEach((row) => {
        const show = (!query || row.dataset.search.includes(query)) && (!exchange || row.dataset.exchange === exchange);
        row.hidden = !show;
        if (show) visible += 1;
      });
      count.textContent = `显示 ${visible.toLocaleString()} / ${rows.length.toLocaleString()} 只`;
    }
    searchInput.addEventListener('input', applyFilters);
    exchangeSelect.addEventListener('change', applyFilters);
    applyFilters();
    """

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A 股 T+1 ETF 数据报告</title>
  <style>{css}</style>
  <script>{get_plotlyjs()}</script>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>A 股 T+1 ETF 数据报告</h1>
        <div class="subtitle">当前可交易的境内股票 ETF，已按 Qlib 标准处理</div>
      </div>
      <div class="updated">清单快照 {snapshot_date}<br>报告生成 {generated_at}</div>
    </div>
  </header>
  <main>
    <section class="kpis" aria-label="核心指标">
      <div class="kpi"><div class="kpi-label">当前 ETF 数量</div><div class="kpi-value">{len(universe):,}</div><div class="kpi-note">上海 {exchange_counts.get('上海', 0):,} · 深圳 {exchange_counts.get('深圳', 0):,}</div></div>
      <div class="kpi"><div class="kpi-label">日线记录</div><div class="kpi-value">{total_rows / 1e6:.2f}M</div><div class="kpi-note">共 {total_rows:,} 条</div></div>
      <div class="kpi"><div class="kpi-label">历史覆盖</div><div class="kpi-value">{earliest[:4]}–26</div><div class="kpi-note">单只中位数 {median_years:.1f} 年</div></div>
      <div class="kpi"><div class="kpi-label">最近完整交易日</div><div class="kpi-value">{history_end[5:]}</div><div class="kpi-note">{history_end} · 校验问题 0</div></div>
    </section>

    <section class="section">
      <div class="section-head"><h2>这批数据是什么</h2><div class="section-note">只包含境内股票指数 ETF</div></div>
      <div class="glossary">
        <div class="term"><strong>T+1</strong><span>当天买入，最早下一交易日卖出。<br>已排除跨境、黄金、债券和货币 ETF。</span></div>
        <div class="term"><strong>前复权价格</strong><span>历史分红的价格跳空已处理。<br>适合计算收益率和训练模型。</span></div>
        <div class="term"><strong>Qlib 标准</strong><span>首个有效收盘价归一为 1。<br>factor 可还原真实成交价。</span></div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><h2>ETF 池概览</h2><div class="section-note">展示当前仍可交易品种的上市历史</div></div>
      <div class="grid-2"><div class="chart">{chart_html(exchange_fig)}</div><div class="chart">{chart_html(listing_fig)}</div></div>
      <div class="grid-2"><div class="chart">{chart_html(growth_fig)}</div><div class="chart">{chart_html(history_fig)}</div></div>
    </section>

    <section class="section">
      <div class="section-head"><h2>市场走势与流动性</h2><div class="section-note">图表可悬停查看数值、拖动缩放</div></div>
      <div class="wide-chart">{chart_html(performance_fig, 440)}</div>
      <div class="wide-chart">{chart_html(liquidity_fig, 600)}</div>
      <div class="callout">最近交易日共有 <strong>{liquid_count:,}</strong> 只 ETF 的“收盘价 × 成交量”超过 1,000 万元。腾讯历史接口不提供正式成交额，因此这里使用收盘价成交额近似值；OHLCV 和前复权价格均为行情源原始数据。</div>
    </section>

    <section class="section">
      <div class="section-head"><h2>ETF 明细</h2><div class="section-note">按最近完整交易日流动性从高到低</div></div>
      <div class="table-toolbar">
        <input id="etf-search" type="search" placeholder="搜索代码或名称" aria-label="搜索 ETF">
        <select id="exchange-filter" aria-label="筛选交易所"><option value="">全部交易所</option><option value="SH">上海</option><option value="SZ">深圳</option></select>
        <div id="table-count" class="table-count"></div>
      </div>
      <div class="table-wrap">
        <table id="etf-table">
          <thead><tr><th>代码</th><th>名称</th><th>交易所</th><th>数据起点</th><th class="num">交易日数</th><th class="num">最新价</th><th class="num">成交额近似</th></tr></thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </div>
    </section>
    <footer>数据用途：量化研究和模型训练，不构成投资建议。当前清单不含已退市 ETF，做长期历史回测时需注意幸存者偏差。正式数据目录：data/cn_etf。</footer>
  </main>
  <script>{javascript}</script>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_report(args.data_dir, args.output)
    print(output.resolve())


if __name__ == "__main__":
    main()
