"""数据全流程可视化：从原始行情到预测结果的可解释浏览页面。

只读取已冻结的运行产物与数据目录，不在网页端重算任何核心指标；
所有图表均基于冻结产物或原始数据文件做展示性聚合。
"""
from __future__ import annotations

import html
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

# Alpha360 六个原始字段与其经济含义（用于给非量化读者解释）
ALPHA360_FIELDS = [
    ("CLOSE", "收盘价", "每天最后一笔成交的价格，代表市场对该 ETF 当天价值的最终共识"),
    ("OPEN", "开盘价", "每天第一笔成交的价格，反映隔夜信息在开盘瞬间的冲击"),
    ("HIGH", "最高价", "当天达到的最高成交价，代表日内多头最强势的位置"),
    ("LOW", "最低价", "当天达到的最低成交价，代表日内空头最强势的位置"),
    ("VWAP", "成交均价", "按成交量加权的平均价，代表当天资金的真实平均成本"),
    ("VOLUME", "成交量", "当天成交的份额数，代表这只 ETF 的活跃程度与关注度"),
]

RAW_FIELDS = [
    ("open / close / high / low", "开、收、高、低四个价格"),
    ("volume / amount", "成交量与成交额"),
    ("vwap", "按量加权的成交均价"),
    ("factor", "复权因子（用于对齐分红、拆分造成的价格跳空）"),
    ("change / paused", "涨跌幅与是否停牌"),
]


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _chart_html(figure: go.Figure, height: int = 360, percent_y: bool = False) -> str:
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


def _resolve(workspace_root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = Path(relative)
    if path.is_absolute():
        return path
    return (workspace_root / path).resolve()


# ---------------------------------------------------------------------------
# 数据加载（只读）
# ---------------------------------------------------------------------------

def _load_instruments(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, sep="\t", header=None, names=["symbol", "list_date", "delist_date"])
        frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
        return frame
    except Exception:
        return None


def _load_calendar(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    try:
        days = pd.read_csv(path, header=None, names=["date"])
        days["date"] = pd.to_datetime(days["date"], errors="coerce")
        return days.dropna()
    except Exception:
        return None


def _load_universe_meta(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
        for col in ("total_market_value", "float_market_value", "spot_amount"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame
    except Exception:
        return None


def _load_benchmark_csv(path: Path | None, symbol: str) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame.dropna(subset=["date"])
    except Exception:
        return None


def _load_corporate_actions(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
        frame["record_date"] = pd.to_datetime(frame["record_date"], errors="coerce")
        frame["cash_dividend_per_old_share"] = pd.to_numeric(
            frame.get("cash_dividend_per_old_share"), errors="coerce"
        )
        frame["share_ratio"] = pd.to_numeric(frame.get("share_ratio"), errors="coerce")
        return frame
    except Exception:
        return None


def _load_predictions(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_parquet(path)
        return frame
    except Exception:
        return None


def _load_signal(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        return frame
    except Exception:
        return None


def _load_importance(fold_dir: Path) -> pd.DataFrame | None:
    path = fold_dir / "feature_importance.parquet"
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_fold_summaries(run_dir: Path, n_folds: int) -> list[dict[str, Any]]:
    summaries = []
    for index in range(1, n_folds + 1):
        path = run_dir / "folds" / f"fold_{index:02d}" / "summary.json"
        value = read_json(path)
        summaries.append(value if isinstance(value, dict) else {})
    return summaries


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def _raw_price_chart(raw: pd.DataFrame, symbol: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=raw["date"], y=raw["raw_close"], name="收盘价(未复权)", line=dict(color="#14765a", width=2))
    )
    figure.add_trace(
        go.Scatter(
            x=raw["date"], y=raw["volume"], name="成交量",
            yaxis="y2", fill="tozeroy", line=dict(color="rgba(200,130,24,0.45)", width=1),
        )
    )
    figure.update_layout(
        title=f"{symbol}（沪深300 ETF 基准）原始日线",
        yaxis=dict(title="价格（元）"),
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)),
    )
    return figure


def _normalized_chart(norm: pd.DataFrame, symbol: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=norm["date"], y=norm["close"], name="归一化收盘价", line=dict(color="#14765a", width=2))
    )
    figure.add_trace(
        go.Scatter(
            x=norm["date"], y=norm["factor"], name="复权因子", yaxis="y2",
            line=dict(color="#3478a5", width=2),
        )
    )
    figure.update_layout(
        title=f"{symbol} 归一化价格与复权因子",
        yaxis=dict(title="归一化价格"),
        yaxis2=dict(title="复权因子", overlaying="y", side="right", showgrid=False),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)),
    )
    return figure


def _listing_chart(instruments: pd.DataFrame) -> go.Figure:
    counts = (
        instruments.dropna(subset=["list_date"])
        .sort_values("list_date")
        .assign(cumcount=lambda frame: range(1, len(frame) + 1))
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=counts["list_date"], y=counts["cumcount"],
            name="累计可交易 ETF 数量", line=dict(color="#14765a", width=2.5), fill="tozeroy",
        )
    )
    figure.update_layout(title="ETF 上市时间线与累计可交易数量", yaxis=dict(title="累计数量"))
    return figure


def _marketcap_chart(meta: pd.DataFrame) -> go.Figure:
    values = meta["total_market_value"].dropna() / 1e8  # 亿元
    if values.empty:
        return go.Figure()
    figure = go.Figure(
        go.Histogram(x=values, nbinsx=60, name="ETF 总市值", marker_color="#3478a5", opacity=0.85)
    )
    figure.update_layout(title="ETF 总市值分布（亿元，对数横轴）", xaxis=dict(type="log"), yaxis=dict(title="数量"))
    return figure


def _corporate_actions_chart(ca: pd.DataFrame) -> go.Figure:
    frame = ca.dropna(subset=["record_date"]).copy()
    frame["year"] = frame["record_date"].dt.year
    has_cash = frame["cash_dividend_per_old_share"].fillna(0) > 0
    has_split = frame["share_ratio"].fillna(1) != 1
    cash_yearly = frame[has_cash].groupby("year").size()
    split_yearly = frame[has_split].groupby("year").size()
    years = sorted(set(cash_yearly.index) | set(split_yearly.index))
    figure = go.Figure()
    figure.add_trace(
        go.Bar(x=years, y=[int(cash_yearly.get(year, 0)) for year in years], name="现金分红", marker_color="#14765a")
    )
    figure.add_trace(
        go.Bar(x=years, y=[int(split_yearly.get(year, 0)) for year in years], name="份额折算", marker_color="#c98218")
    )
    figure.update_layout(title="公司行动事件年度分布（分红 vs 份额折算）", barmode="stack", yaxis=dict(title="事件数"))
    return figure


def _feature_matrix_heatmap(norm: pd.DataFrame, symbol: str) -> go.Figure:
    """用归一化数据复现 Alpha360 的 6 字段 × 60 天特征矩阵（仅作特征定义展示）。"""
    frame = norm.sort_values("date").reset_index(drop=True)
    if frame.empty:
        return go.Figure()
    t = len(frame) - 1
    sample_date = frame.iloc[t]["date"]

    def window60(column: str) -> np.ndarray:
        series = frame[column].iloc[max(0, t - 59): t + 1].reset_index(drop=True)
        values = [np.nan] * (60 - len(series)) + list(series.astype(float))
        return np.array(values)

    close = window60("close")
    close_t = close[-1]
    price_matrix = np.vstack(
        [
            close / close_t,
            window60("open") / close_t,
            window60("high") / close_t,
            window60("low") / close_t,
            window60("vwap") / close_t,
        ]
    )
    figure = go.Figure(
        go.Heatmap(
            z=price_matrix,
            x=[str(i) for i in range(60)],
            y=["CLOSE", "OPEN", "HIGH", "LOW", "VWAP"],
            zmid=1.0,
            colorscale=[[0, "#2c7fb8"], [0.5, "#f7f7f7"], [1, "#c8402f"]],
            colorbar=dict(title="相对今日收盘"),
            hovertemplate="字段 %{y}<br>往前 %{x} 天<br>值 %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{symbol} 某交易日的特征矩阵示例（价格类 300 维，另有成交量 60 维）",
        xaxis=dict(title="往前第 N 天（0=今天）"),
        yaxis=dict(title="字段"),
        height=340,
    )
    return figure


def _fold_timeline(folds: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    colors = {"train": "#3478a5", "valid": "#c98218", "test": "#14765a"}
    labels = {"train": "训练期", "valid": "验证期", "test": "测试期"}
    for fold in folds:
        y = int(fold.get("fold", 0))
        for stage in ("train", "valid", "test"):
            start = pd.to_datetime(fold.get(f"{stage}_start"), errors="coerce")
            end = pd.to_datetime(fold.get(f"{stage}_end"), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            # 训练期是扩展窗口（回溯到 2005），图表上把它截到可视区间起点
            if stage == "train":
                start = max(start, pd.to_datetime("2024-06-01"))
            figure.add_trace(
                go.Scatter(
                    x=[start, end], y=[y, y], mode="lines",
                    line=dict(color=colors[stage], width=11),
                    name=labels[stage], legendgroup=stage,
                    showlegend=(y == folds[0].get("fold")),
                    hovertemplate=f"Fold {y} · {labels[stage]}<br>%{{x|%Y-%m-%d}}<extra></extra>",
                )
            )
    figure.update_layout(
        title="滚动训练：7 个 fold 的训练 / 验证 / 测试时间窗",
        yaxis=dict(title="Fold", tickmode="linear", dtick=1),
        xaxis=dict(title="日期"),
        height=420,
    )
    return figure


def _fold_samples_chart(folds: list[dict[str, Any]]) -> go.Figure:
    rows = []
    for fold in folds:
        row = fold.get("rows", {})
        rows.append(
            {
                "fold": str(fold.get("fold", "-")),
                "训练样本": int(row.get("train", 0) or 0),
                "验证样本": int(row.get("valid", 0) or 0),
                "测试样本": int(row.get("test_features", 0) or 0),
            }
        )
    frame = pd.DataFrame(rows)
    figure = go.Figure()
    for name, color in (("训练样本", "#3478a5"), ("验证样本", "#c98218"), ("测试样本", "#14765a")):
        figure.add_trace(go.Bar(x=frame["fold"], y=frame[name], name=name, marker_color=color))
    figure.update_layout(title="每个 fold 实际喂给模型 / 用于评测的样本行数", barmode="group", yaxis=dict(title="样本行数"))
    return figure


def _importance_chart(importance: pd.DataFrame, fold: int) -> go.Figure:
    top = importance.sort_values("gain", ascending=False).head(20).iloc[::-1]
    figure = go.Figure(
        go.Bar(x=top["gain"], y=top["feature"], orientation="h", marker_color="#14765a")
    )
    figure.update_layout(title=f"Fold {fold} 特征重要性 Top 20（按增益）", xaxis=dict(title="增益"))
    return figure


def _score_label_chart(predictions: pd.DataFrame, fold: int) -> go.Figure:
    sample = predictions[predictions["fold"] == fold]
    if sample.empty:
        return go.Figure()
    sample = sample.sample(min(len(sample), 4000), random_state=0)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sample["score"], y=sample["label"], mode="markers",
            marker=dict(color="#3478a5", opacity=0.25, size=5), name="单只 ETF 单日",
        )
    )
    figure.update_layout(
        title=f"Fold {fold} 预测打分 vs 真实未来收益（相关性越强越好）",
        xaxis=dict(title="模型预测 score"), yaxis=dict(title="真实未来收益 label（已标准化）"),
    )
    return figure


def _coverage_chart(predictions: pd.DataFrame) -> go.Figure:
    coverage = predictions["score"].groupby(level="datetime").count()
    figure = go.Figure(
        go.Scatter(
            x=coverage.index, y=coverage, name="每日有预测的 ETF 数量",
            line=dict(color="#3478a5", width=2), fill="tozeroy", fillcolor="rgba(52,120,165,0.10)",
        )
    )
    figure.update_layout(title="每日预测覆盖的 ETF 数量", yaxis=dict(title="ETF 数量"))
    return figure


def _ic_chart(signal: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=signal.index, y=signal["ic"], name="日度 IC", marker_color="#78a6c8", opacity=0.35))
    figure.add_trace(
        go.Scatter(
            x=signal.index, y=signal["ic"].rolling(20, min_periods=10).mean(),
            name="IC 20日均值", line=dict(color="#215e83", width=2.5),
        )
    )
    figure.add_hline(y=0, line_color="#8a959b", line_width=1)
    figure.update_layout(title="样本外信号质量：IC（预测与真实收益的相关性）", yaxis=dict(title="IC"))
    return figure


# ---------------------------------------------------------------------------
# 页面拼装
# ---------------------------------------------------------------------------

_CSS = """
:root { --ink:#17212a; --muted:#64727b; --line:#dce2e5; --soft:#f5f7f7;
        --green:#14765a; --red:#b44b4b; --blue:#3478a5; --amber:#c98218; --purple:#704c9a; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#fff; font-family:"Microsoft YaHei","Segoe UI",sans-serif; }
header { border-bottom:1px solid var(--line); }
.inner, main { width:min(1460px, calc(100% - 40px)); margin:0 auto; min-width:0; }
.inner { min-height:96px; display:flex; justify-content:space-between; align-items:center; gap:24px; }
h1 { margin:0 0 7px; font-size:24px; }
h2 { margin:0; font-size:18px; }
h3 { margin:0; font-size:14px; }
.sub, .meta, .note, .explain { color:var(--muted); font-size:12px; line-height:1.7; }
.meta { text-align:right; font-variant-numeric:tabular-nums; }
main { padding:26px 0 48px; }
.pipeline { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0; }
.step { flex:1 1 150px; border:1px solid var(--line); border-top:3px solid var(--blue); border-radius:6px; padding:12px 13px; background:var(--soft); }
.step .k { font-weight:700; font-size:13px; margin-bottom:5px; }
.step .v { color:var(--muted); font-size:11px; line-height:1.5; }
.explain { background:#f2f6f8; border-left:4px solid var(--blue); padding:13px 15px; margin:12px 0; }
.stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:14px 0; }
.stat { border:1px solid var(--line); border-radius:6px; padding:14px 15px; }
.stat .k { color:var(--muted); font-size:11px; margin-bottom:7px; }
.stat .v { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }
.section { margin-top:38px; }
.section-head { border-bottom:1px solid var(--line); padding-bottom:9px; margin-bottom:10px; }
.section-num { display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; border-radius:50%; background:var(--blue); color:#fff; font-weight:700; font-size:13px; margin-right:9px; }
.grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.chart, .wide { min-width:0; }
.table-wrap { overflow:auto; border:1px solid var(--line); max-height:520px; }
table { width:100%; border-collapse:collapse; min-width:720px; font-size:12px; }
th { position:sticky; top:0; background:#eef1f2; text-align:left; padding:10px 12px; border-bottom:1px solid #cbd1d5; white-space:nowrap; }
td { padding:9px 12px; border-bottom:1px solid #edf0f2; white-space:nowrap; }
td.wrap { white-space:normal; min-width:320px; line-height:1.55; }
tbody tr:hover { background:#f5faf8; }
th.num, td.num { text-align:right; font-variant-numeric:tabular-nums; }
.field-table { max-width:960px; }
footer { border-top:1px solid var(--line); margin-top:38px; padding-top:15px; color:var(--muted); font-size:11px; line-height:1.7; }
@media(max-width:900px) { .inner { flex-direction:column; align-items:flex-start; gap:8px; } .meta{text-align:left;} .stats{grid-template-columns:repeat(2,1fr);} .grid-2{grid-template-columns:1fr;} }
"""


def render_walkthrough(run_id: str, run_dir: Path, pipeline_root: Path) -> str:
    workspace_root = pipeline_root.parent
    manifest = read_json(run_dir / "manifest.json") or {}
    config = read_json(run_dir / "config.json") or {}
    metrics = read_json(run_dir / "metrics.json") or {}

    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    benchmark = str(data_cfg.get("benchmark", "SH510300"))
    benchmark_file = benchmark.lower()

    qlib_provider = _resolve(workspace_root, paths.get("qlib_provider"))
    instruments_path = _resolve(workspace_root, paths.get("instruments"))
    universe_path = _resolve(workspace_root, paths.get("universe"))
    validation_path = _resolve(workspace_root, paths.get("validation_report"))
    raw_dir = workspace_root / "data" / "cn_etf" / "raw"
    normalized_dir = workspace_root / "data" / "cn_etf" / "normalized"

    instruments = _load_instruments(instruments_path)
    calendar = _load_calendar(qlib_provider / "calendars" / "day.txt" if qlib_provider else None)
    meta = _load_universe_meta(qlib_provider / "metadata" / "t1_etf_universe.csv" if qlib_provider else None)
    validation = read_json(validation_path) if validation_path else None
    raw = _load_benchmark_csv(raw_dir / f"{benchmark_file}.csv", benchmark)
    norm = _load_benchmark_csv(normalized_dir / f"{benchmark_file}.csv", benchmark)
    corporate_actions = _load_corporate_actions(workspace_root / "data" / "cn_etf" / "corporate_actions.csv")

    predictions = _load_predictions(run_dir / "predictions.parquet")
    signal = _load_signal(run_dir / "signal_metrics.parquet")

    folds = metrics.get("folds") or []
    n_folds = len(folds)
    fold_summaries = _load_fold_summaries(run_dir, n_folds)

    base = metrics.get("base", {})
    feature_mode = (config.get("features") or {}).get("mode", "alpha360")

    # 组装统计卡片
    def stat(k: str, v: str) -> str:
        return f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>'

    # 数据准备统计
    raw_count = int((validation or {}).get("raw_file_count", 0) or 0)
    total_rows = int((validation or {}).get("total_rows", 0) or 0)
    universe_count = int((validation or {}).get("universe_count", 0) or 0)
    min_date = (validation or {}).get("min_latest_date", "-")
    max_date = (validation or {}).get("max_latest_date", "-")
    data_stats = "".join(
        [
            stat("ETF 数量", f"{universe_count:,}" if universe_count else "-"),
            stat("原始数据总行数", f"{total_rows:,}" if total_rows else "-"),
            stat("原始文件数", f"{raw_count:,}" if raw_count else "-"),
            stat("数据最新日期", f"{min_date} ~ {max_date}" if min_date != "-" else "-"),
        ]
    )

    # 日历统计
    cal_days = len(calendar) if calendar is not None else 0
    cal_start = str(calendar.iloc[0]["date"].date()) if calendar is not None and len(calendar) else "-"
    cal_end = str(calendar.iloc[-1]["date"].date()) if calendar is not None and len(calendar) else "-"

    # 上市日期统计
    if instruments is not None and "list_date" in instruments:
        listed = instruments.dropna(subset=["list_date"])
        first_list = str(listed["list_date"].min().date()) if len(listed) else "-"
        last_list = str(listed["list_date"].max().date()) if len(listed) else "-"
    else:
        listed = None
        first_list = last_list = "-"

    # fold 样本汇总表
    fold_rows = []
    for fold in folds:
        row = fold.get("rows", {})
        instruments_map = fold.get("instruments", {})
        fold_rows.append(
            "<tr>"
            f"<td>{fold.get('fold', '-')}</td>"
            f"<td>{fold.get('train_start', '-')} ~ {fold.get('train_end', '-')}</td>"
            f"<td>{fold.get('valid_start', '-')} ~ {fold.get('valid_end', '-')}</td>"
            f"<td>{fold.get('test_start', '-')} ~ {fold.get('test_end', '-')}</td>"
            f"<td class='num'>{_fmt_int(row.get('train'))}</td>"
            f"<td class='num'>{_fmt_int(row.get('valid'))}</td>"
            f"<td class='num'>{_fmt_int(row.get('test_features'))}</td>"
            f"<td class='num'>{_fmt_int(instruments_map.get('train'))}</td>"
            f"<td class='num'>{_fmt_int(instruments_map.get('valid'))}</td>"
            f"<td class='num'>{_fmt_int(instruments_map.get('test'))}</td>"
            f"<td class='num'>{_fmt_num(fold.get('ic'), 4)}</td>"
            "</tr>"
        )
    fold_table = "".join(fold_rows) or "<tr><td colspan='11'>暂无 fold 数据</td></tr>"

    # 特征重要性（合并所有 fold 的均值，取 top20）
    importance_parts = []
    for index, summary in enumerate(fold_summaries, start=1):
        importance = _load_importance(run_dir / "folds" / f"fold_{index:02d}")
        if importance is not None and not importance.empty:
            importance_parts.append(importance)
    importance_chart = ""
    importance_table = ""
    if importance_parts:
        combined = pd.concat(importance_parts)
        mean_gain = combined.groupby("feature")["gain"].mean().sort_values(ascending=False)
        top_features = mean_gain.head(20)
        importance_chart = _chart_html(
            go.Figure(
                go.Bar(x=top_features.values[::-1], y=top_features.index[::-1], orientation="h", marker_color="#14765a")
            ).update_layout(title="全样本特征重要性 Top 20（多 fold 平均增益）", xaxis=dict(title="平均增益")),
            400,
        )
        importance_rows = []
        for feature, gain in mean_gain.head(30).items():
            importance_rows.append(
                f"<tr><td>{html.escape(str(feature))}</td><td class='num'>{_fmt_num(gain, 1)}</td></tr>"
            )
        importance_table = (
            '<div class="table-wrap field-table"><table><thead><tr><th>特征</th><th class="num">平均增益</th></tr></thead>'
            f"<tbody>{''.join(importance_rows)}</tbody></table></div>"
        )

    # 第一个 fold 的重要性（最直观展示单个模型的注意力）
    first_importance = _load_importance(run_dir / "folds" / "fold_01") if n_folds else None
    first_importance_chart = ""
    if first_importance is not None and not first_importance.empty:
        top_first = first_importance.sort_values("gain", ascending=False).head(20).iloc[::-1]
        first_importance_chart = _chart_html(
            go.Figure(
                go.Bar(x=top_first["gain"], y=top_first["feature"], orientation="h", marker_color="#3478a5")
            ).update_layout(title="Fold 1 特征重要性 Top 20（单个模型视角）", xaxis=dict(title="增益")),
            400,
        )

    # 预测 vs 真实
    score_label_chart = ""
    if predictions is not None and "fold" in predictions.columns and not predictions.empty:
        score_label_chart = _chart_html(_score_label_chart(predictions, 1), 380)

    coverage_chart = _chart_html(_coverage_chart(predictions), 380) if predictions is not None else ""
    ic_chart = _chart_html(_ic_chart(signal), 380) if signal is not None else ""
    fold_timeline = _chart_html(_fold_timeline(folds), 430) if folds else ""
    fold_samples = _chart_html(_fold_samples_chart(folds), 380) if folds else ""
    raw_chart = _chart_html(_raw_price_chart(raw, benchmark), 400) if raw is not None else ""
    norm_chart = _chart_html(_normalized_chart(norm, benchmark), 400) if norm is not None else ""
    listing_chart = _chart_html(_listing_chart(instruments), 380) if instruments is not None else ""
    marketcap_chart = _chart_html(_marketcap_chart(meta), 380) if meta is not None else ""

    # 公司行动
    ca_chart = ""
    ca_stats = ""
    if corporate_actions is not None and not corporate_actions.empty:
        ca_chart = _chart_html(_corporate_actions_chart(corporate_actions), 380)
        n_total = len(corporate_actions)
        n_cash = int((corporate_actions["cash_dividend_per_old_share"].fillna(0) > 0).sum())
        n_split = int((corporate_actions["share_ratio"].fillna(1) != 1).sum())
        n_symbols = corporate_actions["symbol"].nunique()
        ca_min = str(corporate_actions["record_date"].min().date()) if corporate_actions["record_date"].notna().any() else "-"
        ca_max = str(corporate_actions["record_date"].max().date()) if corporate_actions["record_date"].notna().any() else "-"
        ca_stats = "".join(
            [
                stat("公司行动事件总数", f"{n_total:,}"),
                stat("现金分红事件", f"{n_cash:,}"),
                stat("份额折算事件", f"{n_split:,}"),
                stat("涉及 ETF 数", f"{n_symbols:,}"),
            ]
        )
        # 示例事件表
        sample_events = corporate_actions.head(4)
        event_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('symbol', '-'))) }</td>"
            f"<td>{html.escape(str(row.get('record_date', '-'))) }</td>"
            f"<td class='num'>{_fmt_num(row.get('cash_dividend_per_old_share'), 4)}</td>"
            f"<td class='num'>{_fmt_num(row.get('share_ratio'), 4)}</td>"
            "</tr>"
            for _, row in sample_events.iterrows()
        )
        ca_example_table = (
            '<div class="table-wrap field-table"><table><thead><tr><th>ETF</th><th>登记日</th>'
            '<th class="num">每股现金分红(元)</th><th class="num">份额折算比例</th></tr></thead>'
            f"<tbody>{event_rows}</tbody></table></div>"
        )
    else:
        ca_example_table = ""

    # 特征矩阵示例
    feature_matrix_chart = _chart_html(_feature_matrix_heatmap(norm, benchmark), 380) if norm is not None else ""

    # 字段表
    field_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(label)}</td><td class='wrap'>{html.escape(meaning)}</td></tr>"
        for name, label, meaning in ALPHA360_FIELDS
    )

    description = str(config.get("project", {}).get("description", ""))
    label_expr = str(data_cfg.get("label", "-"))
    label_horizon = str(data_cfg.get("label_horizon_bars", "-"))

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>数据全流程 · {html.escape(run_id)}</title><style>{_CSS}</style><script>{get_plotlyjs()}</script></head>
<body><header><div class="inner"><div><h1>数据全流程 · 从原始行情到预测</h1>
<div class="sub">{html.escape(description)}</div></div>
<div class="meta">Run {html.escape(run_id)}<br>特征 {html.escape(feature_mode)} · 快照 {html.escape(manifest.get('snapshot_id', '-'))}</div></div></header>
<main>

  <div class="pipeline">
    <div class="step"><div class="k">① 原始行情</div><div class="v">新浪日线，每只 ETF 一个 CSV</div></div>
    <div class="step"><div class="k">② 公司行动+复权</div><div class="v">分红/折算 → 复权因子 → 归一化</div></div>
    <div class="step"><div class="k">③ 特征工程</div><div class="v">Alpha360：6字段×60天=360维</div></div>
    <div class="step"><div class="k">④ 滚动训练</div><div class="v">7个 fold，训练/验证/测试隔离</div></div>
    <div class="step"><div class="k">⑤ 每日打分</div><div class="v">给每只 ETF 预测未来收益</div></div>
    <div class="step"><div class="k">⑥ 回测账本</div><div class="v">真实份额+现金+公司行动</div></div>
  </div>

  <section class="section"><div class="section-head"><h2><span class="section-num">1</span>原始数据从哪来</h2>
  <div class="note">数据来源：新浪财经。每只 ETF 单独一个 CSV，记录每天的开高低收、成交量、成交额和复权因子。</div></div>
  <div class="explain"><b>一句话解释：</b>这一步只是把交易所每天的交易记录按 ETF 存成表格，还没做任何加工。字段有 {len(RAW_FIELDS)} 类：{("、".join(name for name, _ in RAW_FIELDS))}。下面是基准 {html.escape(benchmark)} 的原始价格走势。</div>
  {data_stats}
  {raw_chart}
  </section>

  <section class="section"><div class="section-head"><h2><span class="section-num">2</span>公司行动与复权</h2>
  <div class="note">ETF 会分红、会份额折算，这些"公司行动"会改变价格和份额，是复权因子的来源。</div></div>
  <div class="explain"><b>一句话解释：</b>ETF 分红后价格会“看起来”跌一截，其实是把钱分给了你；份额折算会改变你手里的份额数。复权因子把这些“假跳空”拉平，让历史价格变成可比的连续序列，避免模型把分红误当成暴跌信号。每个事件有 3 个关键日期：<b>登记日</b>（冻结权利）→ <b>除息日</b>（计入应收）→ <b>发放日</b>（变成可用现金）。</div>
  {ca_stats}
  {ca_chart}
  {ca_example_table}
  <div class="note" style="margin-top:16px">下面是基准 {html.escape(benchmark)} 的归一化价格与复权因子走势（复权因子由上面的公司行动反推而来）：</div>
  {norm_chart}
  </section>

  <section class="section"><div class="section-head"><h2><span class="section-num">3</span>股票池（universe）：哪些 ETF 可以被交易</h2>
  <div class="note">模型只能从“当时已经上市”的 ETF 里选。下面展示 ETF 的上市节奏与规模分布。</div></div>
  <div class="explain"><b>关键认知：</b>第一只 ETF 2005 年才上市，绝大多数 ETF 是 2015 年后才有的。当前研究用的是“现时快照”股票池，历史回测存在幸存者偏差——所以结果只能标注为 research_only。第一只上市 {first_list}，最新上市 {last_list}。</div>
  <div class="grid-2"><div class="chart">{listing_chart}</div><div class="chart">{marketcap_chart}</div></div>
  </section>

  <section class="section"><div class="section-head"><h2><span class="section-num">4</span>特征工程：Alpha360 是怎么算出来的</h2>
  <div class="note">360 个特征 = 6 个原始字段 × 60 个时间窗口（今天往前 0~59 天）。</div></div>
  <div class="explain"><b>一句话解释：</b>Alpha360 不做复杂的财务指标，而是把最近 60 天的“开/收/高/低/均价/成交量”都除以<b>今天的收盘价</b>，得到 360 个“相对位置”特征。这样模型就能看到“价格这 60 天是怎么走过来的、现在处在什么相对水平”，从而预测未来 2 天的收益。成交量则除以今天的成交量。</div>
  <div class="table-wrap field-table"><table><thead><tr><th>字段</th><th>名称</th><th>含义</th></tr></thead><tbody>{field_rows}</tbody></table></div>
  <div class="explain" style="margin-top:14px"><b>实际长什么样：</b>下面这张热力图是基准 {html.escape(benchmark)} 在某一个交易日的真实特征矩阵——每一行是一个字段，每一列是“往前第 N 天”，颜色代表“那天价格相对今天收盘价的高低”（红=比今天高，蓝=比今天低，白=持平）。模型每次就是看这样一张 <b>6 字段 × 60 天 = 360 个数字</b>的“图”来做预测（成交量另除以今日成交量，共 60 维）。</div>
  {feature_matrix_chart}
  <div class="grid-2"><div class="chart">{importance_chart}</div><div class="chart" style="margin-top:0">{first_importance_chart if first_importance is not None else ""}</div></div>
  {importance_table}
  </section>

  <section class="section"><div class="section-head"><h2><span class="section-num">5</span>滚动训练：训练 / 验证 / 测试怎么切</h2>
  <div class="note">训练、验证、测试之间用 purge_bars 隔离，防止未来信息泄漏到过去。</div></div>
  <div class="explain"><b>一句话解释：</b>不能拿同一段数据既训练又考试。所以把时间切成 7 个 fold：每个 fold 用<b>过去所有历史</b>训练、用<b>最近 126 天</b>验证选最优迭代、再用<b>随后 63 天</b>做样本外考试。标签是“未来 2 天收益”({html.escape(label_expr)})。</div>
  {fold_timeline}
  {fold_samples}
  <div class="table-wrap"><table><thead><tr><th>Fold</th><th>训练期</th><th>验证期</th><th>测试期</th><th class="num">训练样本</th><th class="num">验证样本</th><th class="num">测试样本</th><th class="num">训练ETF</th><th class="num">验证ETF</th><th class="num">测试ETF</th><th class="num">IC</th></tr></thead><tbody>{fold_table}</tbody></table></div>
  </section>

  <section class="section"><div class="section-head"><h2><span class="section-num">6</span>模型看到什么、预测什么</h2>
  <div class="note">模型输入是 360 维特征矩阵，输出是每只 ETF 每天的预测得分（score），目标是把 score 排序后挑最强的几只买入。</div></div>
  <div class="explain"><b>一句话解释：</b>模型对每天每只 ETF 打一个分（score）。这个分不是价格，而是“模型认为这只 ETF 未来 2 天会涨多少”。score 越高越该买。下面左图看“预测分 vs 真实涨跌”是否对得上（点越沿对角线越好），右图看每天覆盖多少只 ETF。</div>
  <div class="grid-2"><div class="chart">{score_label_chart}</div><div class="chart">{coverage_chart}</div></div>
  {ic_chart}
  </section>

  <section class="section"><div class="section-head"><h2><span class="section-num">7</span>回测结果</h2>
  <div class="note">预测打分经过选股规则（top 5、5日最小持有、整手、佣金滑点）后，进入原始份额+现金账本回测。</div></div>
  <div class="explain"><b>一句话解释：</b>把模型的每日打分翻译成“明天买什么、卖什么”，再用真实价格、真实基金份额、人民币现金逐笔记账，扣除佣金滑点后得到最终净值。完整图表见 <a href="/runs/{html.escape(run_id)}">运行报告</a>。</div>
  <div class="stats">
    {stat("净累计收益", _fmt_pct(base.get("net_cumulative_return")))}
    {stat("基准累计收益", _fmt_pct(base.get("benchmark_cumulative_return")))}
    {stat("策略最大回撤", _fmt_pct(base.get("strategy_max_drawdown")))}
    {stat("信息比率", _fmt_num(base.get("information_ratio"), 2))}
  </div>
  </section>

  <footer>本页面只读取已冻结的运行产物与数据目录做展示性可视化，不在网页端重算核心指标；它不改变任何冻结产物的完整性。所有数字请以 <a href="/runs/{html.escape(run_id)}">运行报告</a> 与运行目录内的冻结文件为准。</footer>
</main></body></html>"""

    return document
