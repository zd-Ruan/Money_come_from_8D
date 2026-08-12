from __future__ import annotations

import html
import json
import math
import re
from numbers import Real
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
    "relative_wealth_max_drawdown",
)
MANIFEST_FIELDS = ("run_id", "created_at", "completed_at", "status", "classification", "snapshot_id")
ENVIRONMENT_FIELDS = ("python", "platform", "qlib", "lightgbm")
METRIC_FIELDS = (
    "base_slippage_bps_per_side",
    "backtest_prediction_rows",
    "prediction_rows",
    "labeled_prediction_rows",
    "last_realized_signal_date",
    "backtest_end_date",
    "prediction_coverage",
    "prediction_days",
    "prediction_instruments",
    "ic",
    "ic_hac_t_stat",
    "ic_t_stat",
    "rank_ic",
    "rank_ic_hac_t_stat",
    "rank_ic_t_stat",
)
PERFORMANCE_FIELDS = (
    "slippage_bps_per_side",
    "raw_execution_days",
    "days",
    "evaluation_start_date",
    "evaluation_end_date",
    "initial_execution_date",
    "alignment_method",
    "net_cumulative_return",
    "net_annualized_return",
    "benchmark_cumulative_return",
    "excess_annualized_return",
    "information_ratio",
    "excess_hac_t_stat",
    "strategy_max_drawdown",
    "relative_wealth_max_drawdown",
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
    "complete_for_gate",
    "reset_cash",
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
COMPARISON_STATUS_LABELS = {
    "improved": "改进成立",
    "not_improved": "未证明改进",
    "incomparable": "不可比较",
}
COMPARISON_SCALAR_FIELDS = (
    "schema_version",
    "comparison_status",
    "status",
    "baseline_run_id",
    "candidate_run_id",
    "comparable",
    "scope",
    "generated_at",
)
COMPARISON_CONDITION_FIELDS = (
    "source_fingerprint",
    "snapshot_id",
    "base_slippage_bps_per_side",
    "evaluation_start_date",
    "evaluation_end_date",
    "return_observations",
    "signal_start_date",
    "signal_end_date",
    "signal_observations",
)
COMPARISON_THRESHOLD_FIELDS = ("hac_t_stat", "hac_max_lag", "fold_win_rate")
TERMINAL_WEALTH_FIELDS = ("baseline", "candidate", "difference")
PAIRED_STAT_FIELDS = ("observations", "baseline_mean", "candidate_mean", "mean_difference", "hac_t_stat")
FOLD_DELTA_FIELDS = ("folds", "wins", "losses", "ties", "win_rate")
FOLD_RECORD_FIELDS = (
    "fold",
    "start",
    "end",
    "observations",
    "baseline_terminal_wealth",
    "candidate_terminal_wealth",
    "benchmark_terminal_wealth",
    "baseline_terminal_relative_wealth",
    "candidate_terminal_relative_wealth",
    "terminal_relative_wealth_difference",
    "outcome",
)
COMPARISON_CRITERIA_FIELDS = (
    "terminal_relative_wealth_positive",
    "daily_return_difference_positive",
    "daily_return_hac_significant",
    "fold_win_rate_majority",
    "ic_difference_non_negative",
    "rank_ic_difference_non_negative",
    "at_least_one_signal_hac_significant",
)
FACTOR_FIELDS = ("name", "family", "direction", "hypothesis", "lookback")

CRITERION_LABELS = {
    "terminal_relative_wealth_positive": "相对财富增量为正",
    "daily_return_difference_positive": "日策略收益差为正",
    "daily_return_hac_significant": "日策略收益差通过 HAC 显著性门槛",
    "fold_win_rate_majority": "滚动窗口胜率超过一半",
    "ic_difference_non_negative": "IC 均值增量非负",
    "rank_ic_difference_non_negative": "Rank IC 均值增量非负",
    "at_least_one_signal_hac_significant": "IC 或 Rank IC 至少一项通过 HAC 显著性门槛",
}
OUTCOME_LABELS = {"win": "候选胜", "loss": "基准胜", "tie": "持平"}
COMPARISON_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _select(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in fields if field in value}


def _select_scalars(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    public = {}
    for field in fields:
        item = value.get(field)
        if item is None or isinstance(item, (str, bool)):
            if field in value:
                public[field] = item
        elif isinstance(item, Real) and math.isfinite(float(item)):
            public[field] = item
    return public


def _public_run(run: Any) -> dict[str, Any]:
    public = _select(run, RUN_FIELDS)
    if "metrics" in public:
        public["metrics"] = _select(public["metrics"], RUN_METRIC_FIELDS)
    return public


def _public_manifest(manifest: Any) -> dict[str, Any]:
    public = _select(manifest, MANIFEST_FIELDS)
    if isinstance(manifest, dict) and "environment" in manifest:
        public["environment"] = _select(manifest["environment"], ENVIRONMENT_FIELDS)
    if isinstance(manifest, dict) and isinstance(manifest.get("factor_catalog"), dict):
        catalog = manifest["factor_catalog"]
        factors = catalog.get("factors") if isinstance(catalog.get("factors"), list) else []
        families = catalog.get("families") if isinstance(catalog.get("families"), list) else []
        public_catalog: dict[str, Any] = {
            "families": [family for family in families if isinstance(family, str)],
            "factors": [_select_scalars(factor, FACTOR_FIELDS) for factor in factors if isinstance(factor, dict)],
        }
        for field in ("catalog_version", "sha256"):
            if isinstance(catalog.get(field), str):
                public_catalog[field] = catalog[field]
        public["factor_catalog"] = public_catalog
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


def _public_comparison(comparison: Any) -> dict[str, Any]:
    public = _select_scalars(comparison, COMPARISON_SCALAR_FIELDS)
    if not isinstance(comparison, dict):
        return public
    if isinstance(comparison.get("reasons"), list):
        public["reasons"] = [str(reason) for reason in comparison["reasons"] if isinstance(reason, str)]
    if isinstance(comparison.get("conditions"), dict):
        public["conditions"] = _select_scalars(comparison["conditions"], COMPARISON_CONDITION_FIELDS)
    if isinstance(comparison.get("thresholds"), dict):
        public["thresholds"] = _select_scalars(comparison["thresholds"], COMPARISON_THRESHOLD_FIELDS)
    if isinstance(comparison.get("deltas"), dict):
        source = comparison["deltas"]
        deltas: dict[str, Any] = {}
        if isinstance(source.get("terminal_relative_wealth"), dict):
            deltas["terminal_relative_wealth"] = _select_scalars(
                source["terminal_relative_wealth"], TERMINAL_WEALTH_FIELDS
            )
        for name in ("daily_strategy_return", "ic", "rank_ic"):
            if isinstance(source.get(name), dict):
                deltas[name] = _select_scalars(source[name], PAIRED_STAT_FIELDS)
        if isinstance(source.get("folds"), dict):
            folds = _select_scalars(source["folds"], FOLD_DELTA_FIELDS)
            records = source["folds"].get("records")
            if isinstance(records, list):
                folds["records"] = [
                    _select_scalars(record, FOLD_RECORD_FIELDS) for record in records if isinstance(record, dict)
                ]
            deltas["folds"] = folds
        public["deltas"] = deltas
    if isinstance(comparison.get("decision"), dict):
        decision = comparison["decision"]
        public_decision = _select_scalars(decision, ("claim",))
        if isinstance(decision.get("criteria"), dict):
            public_decision["criteria"] = {
                name: decision["criteria"][name]
                for name in COMPARISON_CRITERIA_FIELDS
                if isinstance(decision["criteria"].get(name), bool)
            }
        public["decision"] = public_decision
    return public


def _load_comparison(path: Path) -> dict[str, Any] | None:
    try:
        comparison = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(comparison, dict):
        return None
    if comparison.get("comparison_status") != "completed":
        return None
    if comparison.get("status") not in COMPARISON_STATUS_LABELS:
        return None
    return comparison


def _list_comparisons(comparisons_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    if not comparisons_dir.is_dir():
        return []
    records = []
    for path in comparisons_dir.glob("*.json"):
        safe_path = _comparison_path(comparisons_dir, path.stem)
        comparison = _load_comparison(safe_path) if safe_path is not None else None
        if comparison is not None:
            records.append((path.stem, comparison))
    return sorted(records, key=lambda item: (str(item[1].get("generated_at", "")), item[0]), reverse=True)


def _comparison_reason(reason: str) -> str:
    prefix = "improvement criterion not met: "
    if reason.startswith(prefix):
        criterion = reason[len(prefix) :]
        return f"未满足改进条件：{CRITERION_LABELS.get(criterion, criterion)}"
    return reason


def _direct_child_dir(parent: Path, name: str) -> Path | None:
    parent = parent.resolve()
    child = (parent / name).resolve()
    if child.parent != parent or not child.is_dir():
        return None
    return child


def _comparison_path(comparisons_dir: Path, comparison_id: str) -> Path | None:
    if not COMPARISON_ID_PATTERN.fullmatch(comparison_id):
        return None
    comparisons_dir = comparisons_dir.resolve()
    path = (comparisons_dir / f"{comparison_id}.json").resolve()
    if path.parent != comparisons_dir or path.stem != comparison_id or not path.is_file():
        return None
    return path


def _factor_catalog_for_comparison(runs_dir: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    candidate_id = comparison.get("candidate_run_id")
    if not isinstance(candidate_id, str):
        return {}
    candidate_dir = _direct_child_dir(runs_dir, candidate_id)
    if candidate_dir is None:
        return {}
    try:
        manifest = read_json(candidate_dir / "manifest.json")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return _public_manifest(manifest).get("factor_catalog", {})


def _comparison_payload(
    comparison_id: str,
    comparison: dict[str, Any],
    runs_dir: Path,
) -> dict[str, Any]:
    payload = {"comparison_id": comparison_id, **_public_comparison(comparison)}
    catalog = _factor_catalog_for_comparison(runs_dir, comparison)
    if catalog:
        payload["factor_catalog"] = catalog
    return payload


def _find_candidate_comparison(
    comparisons: list[tuple[str, dict[str, Any]]], candidate_run_id: str
) -> tuple[str, dict[str, Any]] | None:
    return next(
        (
            (comparison_id, comparison)
            for comparison_id, comparison in comparisons
            if comparison.get("candidate_run_id") == candidate_run_id
        ),
        None,
    )


def _render_comparison(
    comparison_id: str,
    comparison: dict[str, Any],
    factor_catalog: dict[str, Any],
) -> HTMLResponse:
    public = _public_comparison(comparison)
    status = str(public.get("status", "incomparable"))
    status_label = _display(status, COMPARISON_STATUS_LABELS, "未知")
    deltas = public.get("deltas") if isinstance(public.get("deltas"), dict) else {}
    terminal = deltas.get("terminal_relative_wealth", {})
    daily = deltas.get("daily_strategy_return", {})
    ic_delta = deltas.get("ic", {})
    rank_delta = deltas.get("rank_ic", {})
    folds = deltas.get("folds", {})
    conditions = public.get("conditions") if isinstance(public.get("conditions"), dict) else {}
    thresholds = public.get("thresholds") if isinstance(public.get("thresholds"), dict) else {}
    decision = public.get("decision") if isinstance(public.get("decision"), dict) else {}
    criteria = decision.get("criteria") if isinstance(decision.get("criteria"), dict) else {}

    conclusion = {
        "improved": "在预先声明的配对标准下，候选方案的改进得到支持。",
        "not_improved": "尚未证明候选方案改进，不应作出正向结论。",
        "incomparable": "两次运行不可比较，不允许得出改进结论。",
    }.get(status, "无法形成审计结论。")
    scope = (
        "仅作为配对增量研究证据，不代表实盘表现或模型晋级证据。"
        if public.get("scope")
        else "审计范围未记录。"
    )

    metric_rows = (
        ("终端相对财富", terminal.get("baseline"), terminal.get("candidate"), terminal.get("difference"), None),
        (
            "日策略收益",
            daily.get("baseline_mean"),
            daily.get("candidate_mean"),
            daily.get("mean_difference"),
            daily.get("hac_t_stat"),
        ),
        ("IC", ic_delta.get("baseline_mean"), ic_delta.get("candidate_mean"), ic_delta.get("mean_difference"), ic_delta.get("hac_t_stat")),
        (
            "Rank IC",
            rank_delta.get("baseline_mean"),
            rank_delta.get("candidate_mean"),
            rank_delta.get("mean_difference"),
            rank_delta.get("hac_t_stat"),
        ),
    )
    metric_content = "".join(
        "<tr>"
        f"<td>{label}</td><td class='num'>{_num(baseline, 6)}</td>"
        f"<td class='num'>{_num(candidate, 6)}</td><td class='num'>{_num(difference, 6)}</td>"
        f"<td class='num'>{_num(hac_t, 2)}</td></tr>"
        for label, baseline, candidate, difference, hac_t in metric_rows
    )
    criterion_content = "".join(
        "<tr>"
        f"<td>{_text(CRITERION_LABELS.get(name, name))}</td>"
        f"<td><span class='check {'pass' if passed is True else 'fail'}'>{'通过' if passed is True else '未通过'}</span></td>"
        "</tr>"
        for name, passed in criteria.items()
    ) or "<tr><td colspan='2'>无可执行判定条件</td></tr>"
    reason_content = "".join(
        f"<li>{_text(_comparison_reason(reason))}</li>" for reason in public.get("reasons", [])
    ) or "<li>无附加原因</li>"
    fold_content = "".join(
        "<tr>"
        f"<td>{_text(record.get('fold'))}</td><td>{_text(record.get('start'))}</td><td>{_text(record.get('end'))}</td>"
        f"<td class='num'>{_text(record.get('observations'))}</td>"
        f"<td class='num'>{_num(record.get('baseline_terminal_relative_wealth'), 6)}</td>"
        f"<td class='num'>{_num(record.get('candidate_terminal_relative_wealth'), 6)}</td>"
        f"<td class='num'>{_num(record.get('terminal_relative_wealth_difference'), 6)}</td>"
        f"<td>{_text(OUTCOME_LABELS.get(str(record.get('outcome')), '未知'))}</td></tr>"
        for record in folds.get("records", [])
        if isinstance(record, dict)
    ) or "<tr><td colspan='8'>无逐折结果</td></tr>"
    condition_rows = (
        ("数据源指纹", conditions.get("source_fingerprint")),
        ("数据快照", conditions.get("snapshot_id")),
        ("单边滑点", f"{_num(conditions.get('base_slippage_bps_per_side'), 0)} bps"),
        (
            "收益评估区间",
            f"{_text(conditions.get('evaluation_start_date'))} 至 {_text(conditions.get('evaluation_end_date'))}",
        ),
        ("配对收益样本", conditions.get("return_observations")),
        (
            "信号评估区间",
            f"{_text(conditions.get('signal_start_date'))} 至 {_text(conditions.get('signal_end_date'))}",
        ),
        ("配对信号样本", conditions.get("signal_observations")),
        ("HAC t 门槛", _num(thresholds.get("hac_t_stat"), 2)),
        ("HAC 最大滞后阶数", thresholds.get("hac_max_lag")),
        ("Fold 胜率门槛", _pct(thresholds.get("fold_win_rate"))),
    )
    condition_content = "".join(
        f"<tr><td>{label}</td><td class='wrap'>{_text(value)}</td></tr>" for label, value in condition_rows
    )
    factors = factor_catalog.get("factors") if isinstance(factor_catalog.get("factors"), list) else []
    factor_content = "".join(
        "<tr>"
        f"<td>{_text(factor.get('name'))}</td><td>{_text(factor.get('family'))}</td>"
        f"<td>{'正向' if factor.get('direction') == 1 else '反向' if factor.get('direction') == -1 else _text(factor.get('direction'))}</td>"
        f"<td class='num'>{_text(factor.get('lookback'))}</td><td class='wrap'>{_text(factor.get('hypothesis'))}</td></tr>"
        for factor in factors
        if isinstance(factor, dict)
    ) or "<tr><td colspan='5'>候选运行没有可展示的冻结因子目录</td></tr>"

    return HTMLResponse(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>比较审计 · {_text(comparison_id)}</title><style>
body{{margin:0;color:#17212a;font-family:"Microsoft YaHei","Segoe UI",sans-serif;letter-spacing:0;background:#fff}}header{{border-bottom:1px solid #dce2e5}}.inner,main{{width:min(1260px,calc(100% - 40px));margin:0 auto}}.inner{{padding:20px 0}}h1{{font-size:22px;margin:5px 0}}h2{{font-size:16px;margin:28px 0 10px}}p{{margin:5px 0;line-height:1.6}}a{{color:#215e83;text-decoration:none;font-weight:600}}.meta{{color:#64727b;font-size:12px}}.summary{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border:1px solid #dce2e5}}.summary>div{{padding:13px;border-right:1px solid #e4e8ea}}.summary>div:last-child{{border:0}}.label{{color:#64727b;font-size:11px;margin-bottom:5px}}.value{{font-size:15px;font-weight:700;overflow-wrap:anywhere}}.badge,.check{{display:inline-block;padding:3px 7px;border-radius:4px;font-size:11px;font-weight:700}}.improved,.pass{{color:#0d6248;background:#e8f5ef}}.not_improved,.fail{{color:#9e3434;background:#fdebea}}.incomparable{{color:#72520b;background:#fff4d8}}.table{{overflow:auto;border:1px solid #dce2e5}}table{{width:100%;border-collapse:collapse;min-width:760px;font-size:12px}}th{{text-align:left;background:#eef1f2;padding:9px 11px;border-bottom:1px solid #cbd1d5;white-space:nowrap}}td{{padding:9px 11px;border-bottom:1px solid #edf0f2;vertical-align:top;white-space:nowrap}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}td.wrap{{white-space:normal;min-width:300px;line-height:1.5}}ul{{margin:0;padding-left:20px;line-height:1.7}}code{{font-size:11px;overflow-wrap:anywhere}}@media(max-width:760px){{.inner,main{{width:calc(100% - 24px)}}.summary{{grid-template-columns:1fr 1fr}}.summary>div{{border-bottom:1px solid #e4e8ea}}}}
</style></head><body><header><div class='inner'><div class='meta'><a href='/'>返回运行列表</a> · 冻结比较审计</div><h1>基准与候选配对比较</h1><p>{_text(conclusion)}</p><p class='meta'>{_text(scope)}</p></div></header><main>
<section class='summary'><div><div class='label'>审计结论</div><div class='value'><span class='badge {status}'>{_text(status_label)}</span></div></div><div><div class='label'>可比性</div><div class='value'>{'可比较' if public.get('comparable') is True else '不可比较'}</div></div><div><div class='label'>基准运行</div><div class='value'>{_text(public.get('baseline_run_id'))}</div></div><div><div class='label'>候选运行</div><div class='value'>{_text(public.get('candidate_run_id'))}</div></div><div><div class='label'>Fold 胜率</div><div class='value'>{_pct(folds.get('win_rate'))}</div></div><div><div class='label'>胜 / 负 / 平</div><div class='value'>{_text(folds.get('wins'))} / {_text(folds.get('losses'))} / {_text(folds.get('ties'))}</div></div></section>
<h2>关键增量与配对 HAC t</h2><div class='table'><table><thead><tr><th>指标</th><th>基准</th><th>候选</th><th>候选 - 基准</th><th>配对 HAC t</th></tr></thead><tbody>{metric_content}</tbody></table></div>
<h2>可比性条件与阈值</h2><div class='table'><table><thead><tr><th>项目</th><th>冻结值</th></tr></thead><tbody>{condition_content}</tbody></table></div>
<h2>预声明判定条件</h2><div class='table'><table><thead><tr><th>条件</th><th>结果</th></tr></thead><tbody>{criterion_content}</tbody></table></div>
<h2>审计原因</h2><ul>{reason_content}</ul>
<h2>逐折相对财富</h2><div class='table'><table><thead><tr><th>Fold</th><th>开始</th><th>结束</th><th>样本</th><th>基准相对财富</th><th>候选相对财富</th><th>差值</th><th>结果</th></tr></thead><tbody>{fold_content}</tbody></table></div>
<h2>冻结原创因子目录</h2><p class='meta'>目录版本 {_text(factor_catalog.get('catalog_version'))} · SHA-256 <code>{_text(factor_catalog.get('sha256'))}</code></p><div class='table'><table><thead><tr><th>因子</th><th>族群</th><th>方向</th><th>Lookback</th><th>预注册假设</th></tr></thead><tbody>{factor_content}</tbody></table></div>
<p class='meta' style='margin:24px 0'>比较编号 {_text(comparison_id)} · 生成时间 {_text(public.get('generated_at'))}</p></main></body></html>"""
    )


def _display(value: Any, labels: dict[str, str], fallback: str) -> str:
    return labels.get(str(value), fallback)


def _text(value: Any, fallback: str = "-") -> str:
    return html.escape(str(value if value is not None else fallback))


def create_app(pipeline_root: Path) -> FastAPI:
    pipeline_root = pipeline_root.resolve()
    registry_path = pipeline_root / "registry.json"
    runs_dir = (pipeline_root / "runs").resolve()
    comparisons_dir = (pipeline_root / "comparisons").resolve()
    app = FastAPI(title="My Quant Pipeline", docs_url="/api/docs")

    @app.get("/api/runs")
    def runs_api():
        return {"runs": [_public_run(run) for run in list_runs(registry_path)]}

    @app.get("/api/runs/{run_id}")
    def run_api(run_id: str):
        run_dir = _direct_child_dir(runs_dir, run_id)
        if run_dir is None:
            raise HTTPException(404, "run not found")
        return {
            "manifest": _public_manifest(read_json(run_dir / "manifest.json")),
            "metrics": _public_metrics(read_json(run_dir / "metrics.json")),
            "gates": _public_gates(read_json(run_dir / "gates.json")),
        }

    @app.get("/api/comparisons")
    def comparisons_api():
        return {
            "comparisons": [
                _comparison_payload(comparison_id, comparison, runs_dir)
                for comparison_id, comparison in _list_comparisons(comparisons_dir)
            ]
        }

    @app.get("/api/comparisons/{comparison_id}")
    def comparison_api(comparison_id: str):
        path = _comparison_path(comparisons_dir, comparison_id)
        comparison = _load_comparison(path) if path is not None else None
        if comparison is None:
            raise HTTPException(404, "comparison not found")
        return _comparison_payload(comparison_id, comparison, runs_dir)

    @app.get("/api/runs/{run_id}/comparison")
    def run_comparison_api(run_id: str):
        if _direct_child_dir(runs_dir, run_id) is None:
            raise HTTPException(404, "run not found")
        match = _find_candidate_comparison(_list_comparisons(comparisons_dir), run_id)
        if match is None:
            raise HTTPException(404, "comparison not found")
        return _comparison_payload(match[0], match[1], runs_dir)

    @app.get("/comparisons/{comparison_id}", response_class=HTMLResponse)
    def comparison_page(comparison_id: str):
        path = _comparison_path(comparisons_dir, comparison_id)
        comparison = _load_comparison(path) if path is not None else None
        if comparison is None:
            raise HTTPException(404, "comparison not found")
        return _render_comparison(
            comparison_id,
            comparison,
            _factor_catalog_for_comparison(runs_dir, comparison),
        )

    @app.get("/runs/{run_id}/comparison", response_class=HTMLResponse)
    def run_comparison_page(run_id: str):
        if _direct_child_dir(runs_dir, run_id) is None:
            raise HTTPException(404, "run not found")
        match = _find_candidate_comparison(_list_comparisons(comparisons_dir), run_id)
        if match is None:
            raise HTTPException(404, "comparison not found")
        return _render_comparison(match[0], match[1], _factor_catalog_for_comparison(runs_dir, match[1]))

    @app.get("/runs/{run_id}")
    def run_report(run_id: str):
        report = (pipeline_root / "runs" / run_id / "report.html").resolve()
        if report.parent.parent != (pipeline_root / "runs").resolve() or not report.exists():
            raise HTTPException(404, "report not found")
        document = report.read_text(encoding="utf-8")
        sanitized = document
        local_paths = {str(pipeline_root), pipeline_root.as_posix(), str(report.parent), report.parent.as_posix()}
        manifest = read_json(report.parent / "manifest.json", {})
        if isinstance(manifest, dict) and manifest.get("config"):
            frozen_config = Path(str(manifest["config"]))
            if frozen_config.is_absolute():
                frozen_pipeline_root = frozen_config.parent.parent
                frozen_run = frozen_pipeline_root / "runs" / run_id
                local_paths.update(
                    {str(frozen_pipeline_root), frozen_pipeline_root.as_posix(), str(frozen_run), frozen_run.as_posix()}
                )
        for local_path in sorted(local_paths, key=len, reverse=True):
            sanitized = sanitized.replace(html.escape(local_path, quote=True), "对应运行目录")
            sanitized = sanitized.replace(local_path, "对应运行目录")
        if sanitized != document:
            return HTMLResponse(sanitized)
        return FileResponse(report, media_type="text/html; charset=utf-8")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        runs = list_runs(registry_path)
        comparisons_by_candidate = {}
        for comparison_id, comparison in _list_comparisons(comparisons_dir):
            candidate_id = comparison.get("candidate_run_id")
            if isinstance(candidate_id, str):
                comparisons_by_candidate.setdefault(candidate_id, (comparison_id, comparison))
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
            comparison_match = comparisons_by_candidate.get(run.get("run_id"))
            if comparison_match is None:
                comparison_link = "-"
            else:
                comparison_id, comparison = comparison_match
                comparison_status = str(comparison.get("status", "incomparable"))
                comparison_label = _display(comparison_status, COMPARISON_STATUS_LABELS, "未知")
                encoded_comparison_id = html.escape(quote(comparison_id, safe=""), quote=True)
                comparison_link = (
                    f"<a class='audit {comparison_status}' href='/comparisons/{encoded_comparison_id}'>"
                    f"比较审计 · {_text(comparison_label)}</a>"
                )
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
                f"<td>{comparison_link}</td>"
                f"<td>{_text(run.get('completed_at'))}</td>"
                "</tr>"
            )
        content = "".join(rows) or "<tr><td colspan='10'>暂无运行记录</td></tr>"
        return HTMLResponse(
            f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>My Quant Pipeline</title><style>
body{{margin:0;color:#17212a;font-family:"Microsoft YaHei","Segoe UI",sans-serif;letter-spacing:0;background:#fff}}header{{border-bottom:1px solid #dce2e5}}.inner,main{{width:min(1380px,calc(100% - 40px));margin:0 auto}}.inner{{min-height:86px;display:flex;align-items:center;justify-content:space-between;gap:20px}}h1{{font-size:23px;margin:0 0 6px}}.sub,.meta{{color:#64727b;font-size:12px}}main{{padding:28px 0}}h2{{font-size:17px;margin:0 0 12px}}.table{{overflow:auto;border:1px solid #dce2e5}}table{{width:100%;border-collapse:collapse;min-width:1160px;font-size:12px}}th{{text-align:left;background:#eef1f2;padding:10px 12px;border-bottom:1px solid #cbd1d5;white-space:nowrap}}td{{padding:10px 12px;border-bottom:1px solid #edf0f2;white-space:nowrap}}tr:hover{{background:#f5faf8}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}a{{color:#215e83;text-decoration:none;font-weight:600}}a:hover{{text-decoration:underline}}.badge,.audit{{padding:3px 7px;border-radius:4px;font-size:11px;font-weight:700}}.candidate,.audit.improved{{color:#0d6248;background:#e8f5ef}}.research_only,.audit.not_improved{{color:#9e3434;background:#fdebea}}.invalid{{color:#64727b;background:#edf0f2}}.audit.incomparable{{color:#72520b;background:#fff4d8}}@media(max-width:560px){{.inner,main{{width:calc(100% - 24px)}}.inner{{align-items:flex-start;flex-direction:column;padding:16px 0;gap:5px}}}}
</style></head><body><header><div class='inner'><div><h1>My Quant Pipeline</h1><div class='sub'>ETF 研究、审计、滚动训练与回测</div></div><div class='meta'>API <a href='/api/docs'>/api/docs</a></div></div></header><main><h2>实验运行</h2><div class='table'><table><thead><tr><th>Run</th><th>分类</th><th>状态</th><th>数据快照</th><th>净收益</th><th>基准</th><th>IC</th><th>回撤</th><th>增量审计</th><th>完成时间</th></tr></thead><tbody>{content}</tbody></table></div></main></body></html>"""
        )

    return app


def _pct(value):
    return "-" if not _finite_real(value) else f"{float(value) * 100:.2f}%"


def _num(value, digits=2):
    return "-" if not _finite_real(value) else f"{float(value):.{digits}f}"


def _finite_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))
