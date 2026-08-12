from __future__ import annotations

import html
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .io import read_json
from .registry import list_runs


RUN_FIELDS = (
    "run_id",
    "created_at",
    "completed_at",
    "status",
    "classification",
    "snapshot_id",
    "metrics",
)
RUN_METRIC_FIELDS = (
    "net_cumulative_return",
    "benchmark_cumulative_return",
    "ic",
    "rank_ic",
    "max_drawdown",
)
MANIFEST_FIELDS = ("run_id", "created_at", "completed_at", "status", "classification", "snapshot_id")
ENVIRONMENT_FIELDS = ("python", "platform", "qlib", "lightgbm")
METRIC_FIELDS = (
    "base_slippage_bps_per_side",
    "prediction_rows",
    "labeled_prediction_rows",
    "last_realized_signal_date",
    "backtest_end_date",
    "prediction_coverage",
    "prediction_days",
    "prediction_instruments",
    "ic",
    "ic_t_stat",
    "rank_ic",
    "rank_ic_t_stat",
)
PERFORMANCE_FIELDS = (
    "slippage_bps_per_side",
    "days",
    "net_cumulative_return",
    "net_annualized_return",
    "benchmark_cumulative_return",
    "excess_annualized_return",
    "information_ratio",
    "excess_hac_t_stat",
    "strategy_max_drawdown",
    "excess_max_drawdown",
    "beta",
    "beta_adjusted_alpha_annualized",
    "strategy_benchmark_correlation",
    "average_daily_turnover",
    "total_cost",
    "fill_rate",
    "terminal_account",
)
FOLD_FIELDS = (
    "fold",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
    "purge_bars",
    "instruments",
    "best_iterations",
    "ic",
    "rank_ic",
    "prediction_seed_std_mean",
    "effective_test_end",
)
FOLD_ROW_FIELDS = ("train", "valid", "test_features", "test_labels")
FOLD_PORTFOLIO_FIELDS = (
    "days",
    "net_cumulative_return",
    "benchmark_cumulative_return",
    "excess_cumulative_return",
    "start",
    "end",
)
GATE_FIELDS = ("status", "promotion_eligible", "passed", "total")
GATE_CHECK_FIELDS = ("name", "passed", "value", "threshold", "blocking_for_promotion")

CLASSIFICATION_LABELS = {
    "candidate": "候选模型",
    "research_only": "仅限研究",
    "invalid": "无效",
}
RUN_STATUS_LABELS = {
    "completed": "已完成",
    "failed": "失败",
    "running": "运行中",
    "reporting": "生成报告中",
}


def _select(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in fields if field in value}


def _public_run(run: Any) -> dict[str, Any]:
    public = _select(run, RUN_FIELDS)
    if "metrics" in public:
        public["metrics"] = _select(public["metrics"], RUN_METRIC_FIELDS)
    return public


def _public_manifest(manifest: Any) -> dict[str, Any]:
    public = _select(manifest, MANIFEST_FIELDS)
    if isinstance(manifest, dict) and "environment" in manifest:
        public["environment"] = _select(manifest["environment"], ENVIRONMENT_FIELDS)
    return public


def _public_metrics(metrics: Any) -> dict[str, Any]:
    public = _select(metrics, METRIC_FIELDS)
    if not isinstance(metrics, dict):
        return public
    if "base" in metrics:
        public["base"] = _select(metrics["base"], PERFORMANCE_FIELDS)
    if isinstance(metrics.get("stress"), dict):
        public["stress"] = {
            str(name): _select(performance, PERFORMANCE_FIELDS)
            for name, performance in metrics["stress"].items()
        }
    if isinstance(metrics.get("folds"), list):
        public["folds"] = []
        for fold in metrics["folds"]:
            item = _select(fold, FOLD_FIELDS)
            if isinstance(fold, dict) and "rows" in fold:
                item["rows"] = _select(fold["rows"], FOLD_ROW_FIELDS)
            if isinstance(fold, dict) and "portfolio" in fold:
                item["portfolio"] = _select(fold["portfolio"], FOLD_PORTFOLIO_FIELDS)
            public["folds"].append(item)
    return public


def _public_gates(gates: Any) -> dict[str, Any]:
    public = _select(gates, GATE_FIELDS)
    if isinstance(gates, dict) and isinstance(gates.get("checks"), list):
        public["checks"] = [_select(check, GATE_CHECK_FIELDS) for check in gates["checks"]]
    return public


def _display(value: Any, labels: dict[str, str], fallback: str) -> str:
    return labels.get(str(value), fallback)


def _text(value: Any, fallback: str = "-") -> str:
    return html.escape(str(value if value is not None else fallback))


def create_app(pipeline_root: Path) -> FastAPI:
    pipeline_root = pipeline_root.resolve()
    registry_path = pipeline_root / "registry.json"
    app = FastAPI(title="My Quant Pipeline", docs_url="/api/docs")

    @app.get("/api/runs")
    def runs_api():
        return {"runs": [_public_run(run) for run in list_runs(registry_path)]}

    @app.get("/api/runs/{run_id}")
    def run_api(run_id: str):
        run_dir = (pipeline_root / "runs" / run_id).resolve()
        if run_dir.parent != (pipeline_root / "runs").resolve() or not run_dir.is_dir():
            raise HTTPException(404, "run not found")
        return {
            "manifest": _public_manifest(read_json(run_dir / "manifest.json")),
            "metrics": _public_metrics(read_json(run_dir / "metrics.json")),
            "gates": _public_gates(read_json(run_dir / "gates.json")),
        }

    @app.get("/runs/{run_id}")
    def run_report(run_id: str):
        report = (pipeline_root / "runs" / run_id / "report.html").resolve()
        if report.parent.parent != (pipeline_root / "runs").resolve() or not report.exists():
            raise HTTPException(404, "report not found")
        return FileResponse(report, media_type="text/html; charset=utf-8")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        runs = list_runs(registry_path)
        rows = []
        for run in runs:
            metrics = run.get("metrics", {})
            classification = str(run.get("classification", "invalid"))
            badge_class = classification if classification in CLASSIFICATION_LABELS else "invalid"
            classification_label = _display(classification, CLASSIFICATION_LABELS, "未知")
            run_status = str(run.get("status", ""))
            status_label = _display(run_status, RUN_STATUS_LABELS, "未知")
            encoded_run_id = html.escape(quote(str(run.get("run_id", "")), safe=""), quote=True)
            report_link = f"/runs/{encoded_run_id}" if run_status == "completed" else "#"
            rows.append(
                "<tr>"
                f"<td><a href='{report_link}'>{_text(run.get('run_id'))}</a></td>"
                f"<td><span class='badge {badge_class}'>{classification_label}</span></td>"
                f"<td>{status_label}</td>"
                f"<td>{_text(run.get('snapshot_id'))}</td>"
                f"<td class='num'>{_pct(metrics.get('net_cumulative_return'))}</td>"
                f"<td class='num'>{_pct(metrics.get('benchmark_cumulative_return'))}</td>"
                f"<td class='num'>{_num(metrics.get('ic'), 4)}</td>"
                f"<td class='num'>{_pct(metrics.get('max_drawdown'))}</td>"
                f"<td>{_text(run.get('completed_at'))}</td>"
                "</tr>"
            )
        content = "".join(rows) or "<tr><td colspan='9'>暂无运行记录</td></tr>"
        return HTMLResponse(
            f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>My Quant Pipeline</title><style>
body{{margin:0;color:#17212a;font-family:"Microsoft YaHei","Segoe UI",sans-serif;letter-spacing:0;background:#fff}}header{{border-bottom:1px solid #dce2e5}}.inner,main{{width:min(1380px,calc(100% - 40px));margin:0 auto}}.inner{{min-height:86px;display:flex;align-items:center;justify-content:space-between;gap:20px}}h1{{font-size:23px;margin:0 0 6px}}.sub,.meta{{color:#64727b;font-size:12px}}main{{padding:28px 0}}h2{{font-size:17px;margin:0 0 12px}}.table{{overflow:auto;border:1px solid #dce2e5}}table{{width:100%;border-collapse:collapse;min-width:1050px;font-size:12px}}th{{text-align:left;background:#eef1f2;padding:10px 12px;border-bottom:1px solid #cbd1d5;white-space:nowrap}}td{{padding:10px 12px;border-bottom:1px solid #edf0f2;white-space:nowrap}}tr:hover{{background:#f5faf8}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}a{{color:#215e83;text-decoration:none;font-weight:600}}a:hover{{text-decoration:underline}}.badge{{padding:3px 7px;border-radius:4px;font-size:11px;font-weight:700}}.candidate{{color:#0d6248;background:#e8f5ef}}.research_only{{color:#9e3434;background:#fdebea}}.invalid{{color:#64727b;background:#edf0f2}}@media(max-width:560px){{.inner,main{{width:calc(100% - 24px)}}.inner{{align-items:flex-start;flex-direction:column;padding:16px 0;gap:5px}}}}
</style></head><body><header><div class='inner'><div><h1>My Quant Pipeline</h1><div class='sub'>ETF 研究、审计、滚动训练与回测</div></div><div class='meta'>API <a href='/api/docs'>/api/docs</a></div></div></header><main><h2>实验运行</h2><div class='table'><table><thead><tr><th>Run</th><th>分类</th><th>状态</th><th>数据快照</th><th>净收益</th><th>基准</th><th>IC</th><th>回撤</th><th>完成时间</th></tr></thead><tbody>{content}</tbody></table></div></main></body></html>"""
        )

    return app


def _pct(value):
    return "-" if value is None else f"{value * 100:.2f}%"


def _num(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"
