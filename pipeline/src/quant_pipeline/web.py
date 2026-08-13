from __future__ import annotations

import html
import hashlib
import json
import math
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .factor_research import RESEARCH_STAGES, read_research_state, validate_research_plan
from .integrity import verify_artifact_checksums
from .io import read_json
from .registry import list_runs
from .research_cli import (
    RESEARCH_EXECUTION_SCHEMA_VERSION,
    validate_research_workspace,
    validate_study_id,
)
from .research_runner import (
    build_research_experiment_manifest,
    build_stage_run_request,
    load_completed_stage_evidence,
)


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
TRUSTED_ACCOUNT_CNY = 20_000

RESEARCH_STAGE_LABELS = {
    "discovery": "发现期",
    "confirmation": "确认期",
    "locked_holdout": "锁定留出集",
}
RESEARCH_STAGE_STATUS_LABELS = {
    "unopened": "未打开",
    "claimed": "评估中",
    "completed": "已完成",
    "failed": "已失败",
}
RESEARCH_EXECUTION_STATUS_LABELS = {
    None: "尚未执行",
    "running": "运行中",
    "completed": "已完成",
    "failed": "已失败",
    "stopped_no_bh_candidate": "无 BH 候选，已停止",
    "stopped_confirmation_failed": "确认未通过，已停止",
}
RESEARCH_CRITERION_LABELS = {
    "joint_bh_rejected": "联合检验通过 BH q=0.10",
    "rank_ic_mean_difference_positive": "Rank IC 增量为正",
    "strategy_net_mean_difference_positive": "10 bps 压力净收益增量为正",
    "rank_ic_mean_difference_above_minimum": "Rank IC 增量为正",
    "rank_ic_one_sided_p_value_below_alpha": "Rank IC 单侧 p 值不高于 0.05",
    "strategy_net_mean_difference_above_minimum": "10 bps 压力净收益增量为正",
    "strategy_net_one_sided_p_value_below_alpha": "压力净收益单侧 p 值不高于 0.05",
    "terminal_account_improvement_positive": "候选终值高于基准",
    "terminal_relative_wealth_improvement_positive": "候选相对财富增量为正",
    "candidate_terminal_account_not_below_initial": "候选终值不低于 CNY 20,000",
    "candidate_execution_quality_passed": "成交质量通过",
    "baseline_execution_quality_passed": "基准成交质量通过",
    "candidate_beats_benchmark_at_10bps": "候选在 10 bps 压力成本后跑赢沪深 300 ETF",
    "candidate_max_drawdown_within_limit": "候选最大回撤不超过 25%",
    "paired_complete_fold_majority": "完整滚动折中至少 60% 跑赢基准",
    "single_etf_abs_contribution_share_within_limit": "单只 ETF 绝对收益贡献不超过 35%",
    "single_fold_abs_incremental_pnl_share_within_limit": "单折增量损益集中度不超过 50%",
    "signed_raw_factor_rank_ic_positive": "原始因子方向调整后 Rank IC 为正",
    "signed_raw_factor_fold_majority": "原始因子方向在至少 60% 完整折中为正",
    "all_signed_raw_factor_rank_ic_positive": "全部冻结因子方向调整后 Rank IC 为正",
    "all_signed_raw_factor_rank_ic_p_values_below_alpha": "全部冻结因子方向证据单侧 p 值不高于 0.05",
    "all_signed_raw_factor_fold_majorities": "全部冻结因子在至少 60% 完整折中方向为正",
}
RESEARCH_METRIC_FIELDS = (
    "observations",
    "baseline_mean",
    "candidate_mean",
    "mean_difference",
    "hac_max_lag",
    "hac_t_stat",
    "one_sided_p_value",
    "alternative",
)
RESEARCH_TERMINAL_FIELDS = (
    "account_currency",
    "initial_account",
    "stress_slippage_bps_per_side",
    "baseline_terminal_account",
    "candidate_terminal_account",
    "account_improvement",
    "relative_wealth_improvement",
    "account_improvement_positive",
    "relative_wealth_improvement_positive",
    "candidate_terminal_account_not_below_initial",
    "baseline_execution_quality_passed",
    "candidate_execution_quality_passed",
    "comparison",
)
RESEARCH_RESULT_FIELDS = (
    "analysis_status",
    "stage",
    "scope",
    "alpha",
    "minimum_mean_difference",
    "account_currency",
    "initial_account",
    "stress_slippage_bps_per_side",
    "confirmation_passed",
    "locked_holdout_passed",
)
DISCOVERY_RESEARCH_CRITERIA = {
    "joint_bh_rejected",
    "rank_ic_mean_difference_positive",
    "strategy_net_mean_difference_positive",
    "terminal_account_improvement_positive",
    "terminal_relative_wealth_improvement_positive",
    "candidate_terminal_account_not_below_initial",
    "candidate_execution_quality_passed",
    "baseline_execution_quality_passed",
    "candidate_beats_benchmark_at_10bps",
    "candidate_max_drawdown_within_limit",
    "paired_complete_fold_majority",
    "single_etf_abs_contribution_share_within_limit",
    "single_fold_abs_incremental_pnl_share_within_limit",
    "signed_raw_factor_rank_ic_positive",
    "signed_raw_factor_fold_majority",
}
CONFIRMATION_RESEARCH_CRITERIA = {
    "rank_ic_mean_difference_above_minimum",
    "rank_ic_one_sided_p_value_below_alpha",
    "strategy_net_mean_difference_above_minimum",
    "strategy_net_one_sided_p_value_below_alpha",
    "terminal_account_improvement_positive",
    "terminal_relative_wealth_improvement_positive",
    "candidate_terminal_account_not_below_initial",
    "candidate_execution_quality_passed",
    "baseline_execution_quality_passed",
    "candidate_beats_benchmark_at_10bps",
    "candidate_max_drawdown_within_limit",
    "paired_complete_fold_majority",
    "single_etf_abs_contribution_share_within_limit",
    "single_fold_abs_incremental_pnl_share_within_limit",
    "all_signed_raw_factor_rank_ic_positive",
    "all_signed_raw_factor_rank_ic_p_values_below_alpha",
    "all_signed_raw_factor_fold_majorities",
}

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
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        return None
    try:
        parent = parent.resolve()
        child = (parent / name).resolve()
    except (OSError, RuntimeError):
        return None
    return child if child.parent == parent and child.is_dir() else None


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _trusted_run(
    runs_dir: Path, run_id: Any
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Load the minimum immutable contract required for a public run."""

    run_dir = _direct_child_dir(runs_dir, run_id) if isinstance(run_id, str) else None
    if run_dir is None:
        return None
    manifest = _read_json_object(run_dir / "manifest.json")
    config = _read_json_object(run_dir / "config.json")
    if manifest is None or config is None:
        return None
    integrity = manifest.get("integrity")
    execution = config.get("execution")
    account = execution.get("account") if isinstance(execution, dict) else None
    if (
        manifest.get("run_id") != run_id
        or manifest.get("status") != "completed"
        or not isinstance(integrity, dict)
        or integrity.get("verified") is not True
        or not _finite_real(account)
        or float(account) != TRUSTED_ACCOUNT_CNY
    ):
        return None
    try:
        verification = verify_artifact_checksums(run_dir)
    except Exception:
        return None
    if verification.get("valid") is not True:
        return None
    return run_dir, manifest, config


def _trusted_registry_runs(registry_path: Path, runs_dir: Path) -> list[dict[str, Any]]:
    try:
        records = list_runs(registry_path)
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
        return []
    if not isinstance(records, list):
        return []

    trusted = []
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "completed":
            continue
        if _trusted_run(runs_dir, record.get("run_id")) is None:
            continue
        public_record = dict(record)
        if not isinstance(public_record.get("metrics"), dict):
            public_record["metrics"] = {}
        trusted.append(public_record)
    return trusted


def _trusted_comparisons(
    comparisons_dir: Path, runs_dir: Path
) -> list[tuple[str, dict[str, Any]]]:
    records = []
    for comparison_id, comparison in _list_comparisons(comparisons_dir):
        if _trusted_run(runs_dir, comparison.get("baseline_run_id")) is None:
            continue
        if _trusted_run(runs_dir, comparison.get("candidate_run_id")) is None:
            continue
        records.append((comparison_id, comparison))
    return records


class _ResearchWorkspaceInvalid(ValueError):
    """A study exists, but its sealed control artifacts cannot be trusted."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_object(path: Path, *, artifact: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise _ResearchWorkspaceInvalid(f"{artifact} is missing or unsafe")
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _ResearchWorkspaceInvalid(f"{artifact} is invalid") from exc
    if not isinstance(value, dict):
        raise _ResearchWorkspaceInvalid(f"{artifact} must contain an object")
    return value


def _research_workspace_path(research_dir: Path, study_id: str) -> Path | None:
    try:
        validated_id = validate_study_id(study_id)
    except (TypeError, ValueError):
        return None
    unresolved = research_dir / validated_id
    if not unresolved.exists() and not unresolved.is_symlink():
        return None
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise _ResearchWorkspaceInvalid("research workspace is unsafe")
    try:
        root = research_dir.resolve()
        workspace = unresolved.resolve()
    except (OSError, RuntimeError) as exc:
        raise _ResearchWorkspaceInvalid("research workspace cannot be resolved") from exc
    if workspace.parent != root:
        raise _ResearchWorkspaceInvalid("research workspace resolves outside its root")
    return workspace


def _public_research_metric(value: Any) -> dict[str, Any]:
    return _select_scalars(value, RESEARCH_METRIC_FIELDS)


def _public_research_terminal(value: Any) -> dict[str, Any]:
    return _select_scalars(value, RESEARCH_TERMINAL_FIELDS)


def _public_research_tests(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    public: dict[str, Any] = {}
    for name in ("rank_ic", "strategy_net"):
        if isinstance(value.get(name), dict):
            public[name] = _public_research_metric(value[name])
    if isinstance(value.get("joint_iut"), dict):
        public["joint_iut"] = _select_scalars(
            value["joint_iut"], ("method", "one_sided_p_value", "alternative")
        )
    if isinstance(value.get("terminal"), dict):
        public["terminal"] = _public_research_terminal(value["terminal"])
    if isinstance(value.get("benchmark"), dict):
        public["benchmark"] = _select_scalars(
            value["benchmark"],
            (
                "symbol",
                "baseline_terminal_account",
                "candidate_terminal_account",
                "baseline_beats_benchmark",
                "candidate_beats_benchmark",
            ),
        )
    if isinstance(value.get("drawdown"), dict):
        public["drawdown"] = _select_scalars(
            value["drawdown"],
            ("maximum_allowed", "baseline", "candidate", "candidate_within_limit"),
        )
    execution = value.get("execution_quality")
    if isinstance(execution, dict):
        public["execution_quality"] = {
            name: _select_scalars(
                execution.get(name),
                (
                    "minimum_intent_fill_rate",
                    "minimum_notional_fill_rate",
                    "maximum_zero_fill_intent_rate",
                    "intent_fill_rate",
                    "notional_fill_rate",
                    "zero_fill_intent_rate",
                    "execution_quality_passed",
                ),
            )
            for name in ("thresholds", "baseline", "candidate")
            if isinstance(execution.get(name), dict)
        }
    concentration = value.get("concentration")
    if isinstance(concentration, dict):
        public_concentration = _select_scalars(
            concentration,
            (
                "maximum_single_etf_abs_contribution_share",
                "candidate_single_etf_within_limit",
            ),
        )
        for source, target in (
            ("baseline_single_etf", "baseline_single_etf"),
            ("candidate_single_etf", "candidate_single_etf"),
            ("single_fold_abs_incremental_pnl", "single_fold_abs_incremental_pnl"),
        ):
            if isinstance(concentration.get(source), dict):
                public_concentration[target] = _select_scalars(
                    concentration[source],
                    (
                        "symbol",
                        "fold",
                        "numerator_cny",
                        "denominator_cny",
                        "share",
                        "passed",
                    ),
                )
        public["concentration"] = public_concentration
    folds = value.get("research_folds")
    if isinstance(folds, dict):
        public_folds = _select_scalars(
            folds,
            (
                "signal_sessions_per_fold",
                "minimum_complete_folds",
                "minimum_win_ratio",
                "complete_folds",
                "wins",
                "losses_or_ties",
                "win_ratio",
                "majority_positive_passed",
                "maximum_single_fold_abs_incremental_pnl_share",
            ),
        )
        records = folds.get("records")
        if isinstance(records, list):
            public_folds["records"] = [
                _select_scalars(
                    record,
                    (
                        "fold",
                        "signal_start",
                        "signal_end",
                        "complete_for_gate",
                        "baseline_terminal_account",
                        "candidate_terminal_account",
                        "candidate_benchmark_terminal_account",
                        "incremental_pnl_cny",
                        "candidate_minus_baseline_positive",
                    ),
                )
                for record in records
                if isinstance(record, dict)
            ]
        public["research_folds"] = public_folds
    raw = value.get("signed_raw_factor_rank_ic")
    if isinstance(raw, dict):
        public_raw: dict[str, Any] = {}
        for name, evidence in raw.items():
            if not isinstance(name, str) or not isinstance(evidence, dict):
                continue
            item = _select_scalars(
                evidence,
                (
                    "factor_name",
                    "expected_direction",
                    "observations",
                    "total_sessions",
                    "coverage",
                    "minimum_coverage",
                    "raw_rank_ic_mean",
                    "signed_rank_ic_mean",
                    "hac_max_lag",
                    "hac_t_stat",
                    "one_sided_p_value",
                    "alternative",
                ),
            )
            factor_folds = evidence.get("folds")
            if isinstance(factor_folds, dict):
                item["folds"] = _select_scalars(
                    factor_folds,
                    (
                        "minimum_complete_folds",
                        "minimum_positive_ratio",
                        "complete_folds",
                        "eligible_complete_folds",
                        "positive_folds",
                        "positive_ratio",
                        "all_complete_folds_have_required_coverage",
                        "majority_positive_passed",
                    ),
                )
            public_raw[name] = item
        public["signed_raw_factor_rank_ic"] = public_raw
    return public


def _public_research_criteria(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {
        name: passed
        for name, passed in value.items()
        if name in RESEARCH_CRITERION_LABELS and isinstance(passed, bool)
    }


def _require_research_tests(value: Any, *, artifact: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _ResearchWorkspaceInvalid(f"{artifact} paired evidence is missing")
    for name in ("rank_ic", "strategy_net"):
        metric = value.get(name)
        if not isinstance(metric, dict):
            raise _ResearchWorkspaceInvalid(f"{artifact} {name} evidence is missing")
        for field in (
            "observations",
            "baseline_mean",
            "candidate_mean",
            "mean_difference",
            "hac_t_stat",
            "one_sided_p_value",
        ):
            if field not in metric or (
                field != "observations" and not _finite_real(metric.get(field))
            ):
                raise _ResearchWorkspaceInvalid(f"{artifact} {name} evidence is invalid")
        if (
            isinstance(metric.get("observations"), bool)
            or not isinstance(metric.get("observations"), int)
            or metric["observations"] <= 0
        ):
            raise _ResearchWorkspaceInvalid(f"{artifact} {name} observations are invalid")
    joint = value.get("joint_iut")
    terminal = value.get("terminal")
    if not isinstance(joint, dict) or not _finite_real(joint.get("one_sided_p_value")):
        raise _ResearchWorkspaceInvalid(f"{artifact} joint evidence is invalid")
    if not isinstance(terminal, dict):
        raise _ResearchWorkspaceInvalid(f"{artifact} terminal evidence is missing")
    for field in (
        "initial_account",
        "stress_slippage_bps_per_side",
        "baseline_terminal_account",
        "candidate_terminal_account",
        "account_improvement",
        "relative_wealth_improvement",
    ):
        if not _finite_real(terminal.get(field)):
            raise _ResearchWorkspaceInvalid(f"{artifact} terminal evidence is invalid")
    if (
        float(terminal["initial_account"]) != TRUSTED_ACCOUNT_CNY
        or float(terminal["stress_slippage_bps_per_side"]) != 10.0
    ):
        raise _ResearchWorkspaceInvalid(f"{artifact} violates the account stress contract")
    for field in (
        "account_improvement_positive",
        "relative_wealth_improvement_positive",
        "candidate_terminal_account_not_below_initial",
        "baseline_execution_quality_passed",
        "candidate_execution_quality_passed",
    ):
        if not isinstance(terminal.get(field), bool):
            raise _ResearchWorkspaceInvalid(f"{artifact} execution decision is invalid")
    return value


def _public_discovery_result(
    result: dict[str, Any], state: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    factor_discovery = result.get("factor_discovery")
    if not isinstance(factor_discovery, dict):
        raise _ResearchWorkspaceInvalid("completed discovery result is incomplete")
    bh = factor_discovery.get("bh")
    if not isinstance(bh, dict):
        raise _ResearchWorkspaceInvalid("completed discovery result has no BH decision")
    records = factor_discovery.get("results")
    if not isinstance(records, list):
        raise _ResearchWorkspaceInvalid("completed discovery result has no factor records")
    expected_names = [item.get("hypothesis_id") for item in plan["factor_hypotheses"]]
    if len(records) != len(expected_names) or [
        item.get("hypothesis_id") if isinstance(item, dict) else None for item in records
    ] != expected_names:
        raise _ResearchWorkspaceInvalid("completed discovery factor battery is incomplete")
    public_records = []
    for record in records:
        if not isinstance(record, dict):
            raise _ResearchWorkspaceInvalid("completed discovery factor record is invalid")
        criteria = record.get("selection_criteria")
        if (
            not isinstance(criteria, dict)
            or set(criteria) != DISCOVERY_RESEARCH_CRITERIA
            or not all(isinstance(passed, bool) for passed in criteria.values())
        ):
            raise _ResearchWorkspaceInvalid("completed discovery selection rule is invalid")
        _require_research_tests(record, artifact=str(record.get("hypothesis_id")))
        item = _select_scalars(
            record,
            ("hypothesis_id", "family", "joint_p_value", "bh_q_value", "bh_rejected"),
        )
        item["selection_criteria"] = _public_research_criteria(criteria)
        item["tests"] = _public_research_tests(record)
        public_records.append(item)
    selected = result.get("selected_factor_names")
    if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
        raise _ResearchWorkspaceInvalid("completed discovery selection is invalid")
    selected_from_records = [
        record["hypothesis_id"]
        for record in records
        if all(record["selection_criteria"].values())
    ]
    if (
        selected != factor_discovery.get("selected_factor_names")
        or selected != selected_from_records
        or selected != state.get("discovery_eligible_factor_names")
    ):
        raise _ResearchWorkspaceInvalid("completed discovery selections do not reconcile")
    bh_results = bh.get("results")
    if (
        bh.get("method") != "Benjamini-Hochberg"
        or bh.get("fdr_q") != 0.10
        or bh.get("hypothesis_count") != len(expected_names)
        or not isinstance(bh_results, list)
        or len(bh_results) != len(expected_names)
    ):
        raise _ResearchWorkspaceInvalid("completed discovery BH family is invalid")
    families = result.get("family_ablations")
    family_records = families.get("results") if isinstance(families, dict) else None
    if not isinstance(family_records, list) or len(family_records) != len(plan["family_ablations"]):
        raise _ResearchWorkspaceInvalid("completed family ablation battery is incomplete")
    for record in family_records:
        if not isinstance(record, dict):
            raise _ResearchWorkspaceInvalid("completed family ablation record is invalid")
        _require_research_tests(record, artifact=str(record.get("family")))
    return {
        "analysis_status": result.get("analysis_status"),
        "selected_factor_names": list(selected),
        "family_count": (
            result.get("family_ablations", {}).get("family_count")
            if isinstance(result.get("family_ablations"), dict)
            else None
        ),
        "bh": _select_scalars(
            bh,
            (
                "method",
                "fdr_q",
                "hypothesis_count",
                "rejected_count",
                "cutoff_p_value",
            ),
        ),
        "factors": public_records,
    }


def _public_later_stage_result(
    result: dict[str, Any], stage: str, record: dict[str, Any]
) -> dict[str, Any]:
    public = _select_scalars(result, RESEARCH_RESULT_FIELDS)
    criteria = result.get("criteria")
    if (
        not isinstance(criteria, dict)
        or set(criteria) != CONFIRMATION_RESEARCH_CRITERIA
        or not all(isinstance(passed, bool) for passed in criteria.values())
    ):
        raise _ResearchWorkspaceInvalid(f"completed {stage} criteria are invalid")
    _require_research_tests(result.get("tests"), artifact=stage)
    public["criteria"] = _public_research_criteria(criteria)
    public["tests"] = _public_research_tests(result.get("tests"))
    passed_field = "confirmation_passed" if stage == "confirmation" else "locked_holdout_passed"
    if (
        not isinstance(result.get(passed_field), bool)
        or result[passed_field] is not all(criteria.values())
        or record.get(passed_field) is not result[passed_field]
    ):
        raise _ResearchWorkspaceInvalid(f"completed {stage} result has no decision")
    if (
        float(result.get("initial_account", -1)) != TRUSTED_ACCOUNT_CNY
        or float(result.get("stress_slippage_bps_per_side", -1)) != 10.0
    ):
        raise _ResearchWorkspaceInvalid(f"completed {stage} result violates the account contract")
    return public


def _validate_stage_result(
    stage: str,
    record: dict[str, Any],
    plan: dict[str, Any],
    frozen_spec_sha256: Any,
) -> dict[str, Any] | None:
    status = record.get("status")
    result = record.get("result")
    digest = record.get("result_sha256")
    if status != "completed":
        if result is not None or digest is not None:
            raise _ResearchWorkspaceInvalid(f"non-completed {stage} contains a result")
        return None
    if (
        not isinstance(result, dict)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or _json_sha256(result) != digest
    ):
        raise _ResearchWorkspaceInvalid(f"completed {stage} result seal is invalid")
    partition = plan["partitions"][stage]
    if (
        result.get("stage") != stage
        or result.get("plan_sha256") != plan["plan_sha256"]
        or result.get("partition_sha256") != partition["sessions_sha256"]
    ):
        raise _ResearchWorkspaceInvalid(f"completed {stage} result identity is invalid")
    if stage != "discovery" and result.get("frozen_spec_sha256") != frozen_spec_sha256:
        raise _ResearchWorkspaceInvalid(f"completed {stage} result uses another frozen candidate")
    expected_exposure = plan["exposure_provenance"]["stage_classification"][stage]
    if (
        result.get("exposure_registry_sha256")
        != plan["exposure_provenance"]["registry_sha256"]
        or result.get("evidence_class") != expected_exposure["evidence_class"]
        or result.get("claim_classification")
        != expected_exposure["claim_classification"]
        or record.get("exposure_registry_sha256")
        != result.get("exposure_registry_sha256")
        or record.get("evidence_class") != result.get("evidence_class")
        or record.get("claim_classification") != result.get("claim_classification")
    ):
        raise _ResearchWorkspaceInvalid(f"completed {stage} exposure provenance is invalid")
    return result


def _conservative_exposure(plan: dict[str, Any]) -> dict[str, Any]:
    provenance = plan.get("exposure_provenance")
    if not isinstance(provenance, dict):
        raise _ResearchWorkspaceInvalid("research plan has no verified exposure provenance")
    source = provenance.get("stage_classification")
    if (
        provenance.get("provenance_status") != "verified"
        or not isinstance(provenance.get("registry_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", provenance["registry_sha256"])
        or not isinstance(provenance.get("known_through_session"), str)
        or not isinstance(source, dict)
    ):
        raise _ResearchWorkspaceInvalid("research exposure provenance is invalid")
    stage_source = source
    registry_sha256 = provenance["registry_sha256"]
    known_through_session = provenance["known_through_session"]
    stages: dict[str, dict[str, str]] = {}
    for stage in RESEARCH_STAGES:
        classification = stage_source.get(stage)
        if not isinstance(classification, dict):
            raise _ResearchWorkspaceInvalid(f"research {stage} exposure provenance is missing")
        evidence_class = classification.get("evidence_class")
        claim_classification = classification.get("claim_classification")
        if (
            evidence_class != "retrospective_exposed"
            or claim_classification != "research_only"
            or classification.get("promotion_eligible") is not False
        ):
            raise _ResearchWorkspaceInvalid(
                f"research {stage} is not classified as historical research-only evidence"
            )
        stages[stage] = {
            "evidence_class": evidence_class,
            "evidence_label": "历史已暴露",
            "claim_classification": claim_classification,
        }
    return {
        "evidence_class": "retrospective_exposed",
        "evidence_label": "历史已暴露",
        "claim_classification": "research_only",
        "registry_sha256": registry_sha256,
        "known_through_session": known_through_session,
        "stages": stages,
    }


def _verify_execution_record(
    workspace: Path, plan: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    path = workspace / "execution.json"
    if not path.exists() and not path.is_symlink():
        return None, None
    value = _strict_json_object(path, artifact="research execution record")
    digest = value.get("execution_sha256")
    unsigned = {key: item for key, item in value.items() if key != "execution_sha256"}
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or _json_sha256(unsigned) != digest
        or unsigned.get("schema_version") != RESEARCH_EXECUTION_SCHEMA_VERSION
        or unsigned.get("plan_sha256") != plan["plan_sha256"]
        or unsigned.get("exposure_registry_sha256")
        != plan["exposure_provenance"]["registry_sha256"]
        or not isinstance(unsigned.get("stages"), dict)
    ):
        raise _ResearchWorkspaceInvalid("research execution record seal is invalid")
    for stage in RESEARCH_STAGES:
        state_record = state["stages"][stage]
        execution_stage = unsigned["stages"].get(stage)
        if execution_stage is not None and (
            not isinstance(execution_stage, dict)
            or any(
                execution_stage.get(field) != state_record.get(field)
                for field in (
                    "exposure_registry_sha256",
                    "evidence_class",
                    "claim_classification",
                )
            )
        ):
            raise _ResearchWorkspaceInvalid(
                f"research execution record and {stage} exposure differ"
            )
        if state_record["status"] == "completed":
            if (
                not isinstance(execution_stage, dict)
                or execution_stage.get("status") != "completed"
                or execution_stage.get("result_sha256") != state_record.get("result_sha256")
            ):
                raise _ResearchWorkspaceInvalid(
                    f"research execution record and {stage} state differ"
                )
    return unsigned, digest


def _runtime_research_config(
    workspace: Path, pipeline_root: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    record = _strict_json_object(
        workspace / "base_config.json", artifact="frozen base configuration"
    )
    config = record.get("config")
    digest = record.get("config_sha256")
    source = record.get("config_source")
    if (
        not isinstance(config, dict)
        or not isinstance(digest, str)
        or digest != plan["base_config_sha256"]
        or _json_sha256(config) != digest
        or not isinstance(source, str)
        or not source
    ):
        raise _ResearchWorkspaceInvalid("frozen base configuration is invalid")
    result = json.loads(json.dumps(config))
    for key, raw_value in result.get("paths", {}).items():
        path = Path(raw_value)
        result["paths"][key] = str(
            path if path.is_absolute() else (pipeline_root.parent / path).resolve()
        )
    config_path = (pipeline_root.parent / source).resolve()
    try:
        config_path.relative_to(pipeline_root.parent)
    except ValueError as exc:
        raise _ResearchWorkspaceInvalid("frozen config source resolves outside the repository") from exc
    result["_meta"] = {
        "config_path": str(config_path),
        "workspace_root": str(pipeline_root.parent),
    }
    return result


def _stage_experiment_specs(
    stage: str,
    experiments: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if stage == "discovery":
        discovery = experiments.get("discovery")
        if not isinstance(discovery, Mapping):
            raise _ResearchWorkspaceInvalid("discovery experiment manifest is invalid")
        specs = [
            discovery.get("baseline"),
            *(discovery.get("family_ablations") or []),
            *(discovery.get("single_factor_tests") or []),
        ]
    else:
        specs = [experiments.get("discovery", {}).get("baseline"), experiments.get("frozen_candidate")]
    if not specs or any(not isinstance(spec, dict) for spec in specs):
        raise _ResearchWorkspaceInvalid(f"{stage} experiment specifications are incomplete")
    by_id = {spec.get("experiment_id"): spec for spec in specs}
    if (
        len(by_id) != len(specs)
        or None in by_id
        or any(not isinstance(identifier, str) for identifier in by_id)
    ):
        raise _ResearchWorkspaceInvalid(f"{stage} experiment identities are invalid")
    return by_id


def _public_portfolio_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    portfolio = evidence.get("portfolio")
    if not isinstance(portfolio, Mapping):
        raise _ResearchWorkspaceInvalid("recomputed portfolio evidence is invalid")
    folds = portfolio.get("research_folds")
    if not isinstance(folds, list):
        raise _ResearchWorkspaceInvalid("recomputed research folds are invalid")
    complete = [fold for fold in folds if fold.get("complete_for_gate") is True]
    raw_factors = evidence.get("raw_factor_rank_ic")
    if not isinstance(raw_factors, Mapping):
        raise _ResearchWorkspaceInvalid("recomputed raw factor evidence is invalid")
    factor_records = []
    for name, values in raw_factors.items():
        finite = values.dropna()
        factor_records.append(
            {
                "factor_name": name,
                "observations": len(values),
                "finite_observations": len(finite),
                "coverage": len(finite) / len(values),
                "mean_rank_ic": float(finite.mean()),
            }
        )
    return {
        "benchmark": portfolio["benchmark"],
        "initial_account": portfolio["initial_account"],
        "terminal_account": portfolio["terminal_account"],
        "benchmark_terminal_account": portfolio["benchmark_terminal_account"],
        "strategy_max_drawdown": portfolio["strategy_max_drawdown"],
        "intent_fill_rate": portfolio["intent_fill_rate"],
        "notional_fill_rate": portfolio["notional_fill_rate"],
        "zero_fill_intent_rate": portfolio["zero_fill_intent_rate"],
        "single_etf_abs_contribution_share": portfolio[
            "single_etf_abs_contribution_share"
        ],
        "single_etf_abs_contribution_symbol": portfolio[
            "single_etf_abs_contribution_symbol"
        ],
        "single_etf_abs_contribution_numerator_cny": portfolio[
            "single_etf_abs_contribution_numerator_cny"
        ],
        "single_etf_abs_contribution_denominator_cny": portfolio[
            "single_etf_abs_contribution_denominator_cny"
        ],
        "complete_research_folds": len(complete),
        "research_folds": [
            {
                key: fold[key]
                for key in (
                    "fold",
                    "signal_start",
                    "signal_end",
                    "evaluation_start",
                    "evaluation_end",
                    "complete_for_gate",
                    "terminal_account",
                    "benchmark_terminal_account",
                    "single_etf_abs_contribution_share",
                )
            }
            for fold in folds
        ],
        "raw_factor_rank_ic": factor_records,
    }


def _paired_actual_folds(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_folds = baseline.get("research_folds")
    candidate_folds = candidate.get("research_folds")
    if (
        not isinstance(baseline_folds, list)
        or not isinstance(candidate_folds, list)
        or len(baseline_folds) != len(candidate_folds)
    ):
        raise _ResearchWorkspaceInvalid("recomputed paired research folds are incomplete")
    records = []
    wins = losses = ties = 0
    increments = []
    for base, proposed in zip(baseline_folds, candidate_folds):
        identity = (
            "fold",
            "signal_start",
            "signal_end",
            "evaluation_start",
            "evaluation_end",
            "complete_for_gate",
        )
        if any(base.get(key) != proposed.get(key) for key in identity):
            raise _ResearchWorkspaceInvalid("recomputed paired research fold identities differ")
        incremental = float(proposed["terminal_account"]) - float(base["terminal_account"])
        complete = proposed["complete_for_gate"] is True
        if complete:
            increments.append((int(proposed["fold"]), incremental))
            if math.isclose(incremental, 0.0, rel_tol=0.0, abs_tol=1e-8):
                outcome = "tie"
                ties += 1
            elif incremental > 0.0:
                outcome = "win"
                wins += 1
            else:
                outcome = "loss"
                losses += 1
        else:
            outcome = "incomplete"
        records.append(
            {
                "fold": proposed["fold"],
                "complete_for_gate": complete,
                "baseline_terminal_account": base["terminal_account"],
                "candidate_terminal_account": proposed["terminal_account"],
                "benchmark_terminal_account": proposed["benchmark_terminal_account"],
                "incremental_pnl_cny": incremental,
                "outcome": outcome,
            }
        )
    complete_count = wins + losses + ties
    denominator = float(sum(abs(value) for _, value in increments))
    if denominator > 1e-12:
        dominant_fold, dominant_value = max(
            increments, key=lambda item: (abs(item[1]), -item[0])
        )
        concentration = {
            "fold": dominant_fold,
            "numerator_cny": abs(dominant_value),
            "denominator_cny": denominator,
            "share": abs(dominant_value) / denominator,
        }
    else:
        concentration = {
            "fold": None,
            "numerator_cny": 0.0,
            "denominator_cny": 0.0,
            "share": None,
        }
    return {
        "complete_folds": complete_count,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_ratio": wins / complete_count if complete_count else 0.0,
        "single_fold_abs_incremental_pnl_concentration": concentration,
        "records": records,
    }


def _public_actual_stage_evidence(
    actual: Mapping[str, Mapping[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_id = "alpha158_baseline"
    baseline = actual.get(baseline_id)
    if not isinstance(baseline, Mapping):
        raise _ResearchWorkspaceInvalid("recomputed actual baseline is missing")
    directions = {
        item["hypothesis_id"]: item["expected_direction"]
        for item in plan["factor_hypotheses"]
    }
    experiments = json.loads(json.dumps(actual))
    for evidence in experiments.values():
        for factor in evidence["raw_factor_rank_ic"]:
            direction = directions.get(factor["factor_name"])
            if direction not in (-1, 1):
                raise _ResearchWorkspaceInvalid("raw factor direction is invalid")
            factor["expected_direction"] = direction
            factor["signed_mean_rank_ic"] = factor["mean_rank_ic"] * direction
    comparisons = {
        identifier: _paired_actual_folds(baseline, candidate)
        for identifier, candidate in actual.items()
        if identifier != baseline_id
    }
    return {"experiments": experiments, "fold_comparisons": comparisons}


def _load_actual_stage_evidence(
    *,
    stage: str,
    result: Mapping[str, Any],
    execution: Mapping[str, Any] | None,
    experiments: Mapping[str, Any],
    plan: Mapping[str, Any],
    base_config: Mapping[str, Any],
    runs_dir: Path,
) -> dict[str, dict[str, Any]] | None:
    run_ids = result.get("run_ids")
    if run_ids is None:
        return None
    if not isinstance(run_ids, dict):
        raise _ResearchWorkspaceInvalid(f"completed {stage} run_ids are invalid")
    specs = _stage_experiment_specs(stage, experiments)
    if set(run_ids) != set(specs) or any(
        not isinstance(run_id, str) or not run_id for run_id in run_ids.values()
    ):
        raise _ResearchWorkspaceInvalid(f"completed {stage} run_ids are incomplete")
    execution_stage = execution.get("stages", {}).get(stage) if execution is not None else None
    experiment_records = execution_stage.get("experiments") if isinstance(execution_stage, Mapping) else None
    if not isinstance(experiment_records, list):
        raise _ResearchWorkspaceInvalid(f"completed {stage} execution runs are missing")
    execution_by_id = {
        item.get("experiment_id"): item
        for item in experiment_records
        if isinstance(item, Mapping)
    }
    if set(execution_by_id) != set(specs):
        raise _ResearchWorkspaceInvalid(f"completed {stage} execution battery is incomplete")

    actual: dict[str, dict[str, Any]] = {}
    for experiment_id, spec in specs.items():
        run_id = run_ids[experiment_id]
        execution_record = execution_by_id[experiment_id]
        request = build_stage_run_request(plan, spec, plan["partitions"][stage])
        if (
            execution_record.get("status") != "completed"
            or execution_record.get("run_id") != run_id
            or execution_record.get("request_sha256") != request["request_sha256"]
        ):
            raise _ResearchWorkspaceInvalid(
                f"completed {stage} execution identity differs from the sealed result"
            )
        try:
            run_dir = (runs_dir / run_id).resolve()
            if run_dir.parent != runs_dir:
                raise ValueError("run resolves outside the runs root")
            evidence, _ = load_completed_stage_evidence(
                run_dir, base_config, plan, request
            )
        except Exception as exc:
            raise _ResearchWorkspaceInvalid(
                f"completed {stage} run evidence failed verification"
            ) from exc
        actual[experiment_id] = _public_portfolio_evidence(evidence)
    return actual


def _cross_check_stage_actual(
    stage: str, result: Mapping[str, Any], actual: Mapping[str, Mapping[str, Any]]
) -> None:
    baseline_id = "alpha158_baseline"
    baseline = actual.get(baseline_id)
    if not isinstance(baseline, Mapping):
        raise _ResearchWorkspaceInvalid(f"completed {stage} actual baseline is missing")
    if stage == "discovery":
        factor_results = result.get("factor_discovery", {}).get("results")
        if not isinstance(factor_results, list):
            raise _ResearchWorkspaceInvalid("completed discovery factor results are invalid")
        for record in factor_results:
            experiment_id = f"single_factor__{record.get('hypothesis_id')}"
            candidate = actual.get(experiment_id)
            terminal = record.get("terminal")
            if not isinstance(candidate, Mapping) or not isinstance(terminal, Mapping):
                raise _ResearchWorkspaceInvalid("completed discovery actual evidence is incomplete")
            for declared, computed in (
                (terminal.get("baseline_terminal_account"), baseline["terminal_account"]),
                (terminal.get("candidate_terminal_account"), candidate["terminal_account"]),
            ):
                if not _finite_real(declared) or not math.isclose(
                    float(declared), float(computed), rel_tol=1e-10, abs_tol=1e-6
                ):
                    raise _ResearchWorkspaceInvalid(
                        "completed discovery terminal values differ from sealed runs"
                    )
    else:
        candidate_ids = [identifier for identifier in actual if identifier != baseline_id]
        terminal = result.get("tests", {}).get("terminal")
        if len(candidate_ids) != 1 or not isinstance(terminal, Mapping):
            raise _ResearchWorkspaceInvalid(f"completed {stage} actual evidence is incomplete")
        candidate = actual[candidate_ids[0]]
        for declared, computed in (
            (terminal.get("baseline_terminal_account"), baseline["terminal_account"]),
            (terminal.get("candidate_terminal_account"), candidate["terminal_account"]),
        ):
            if not _finite_real(declared) or not math.isclose(
                float(declared), float(computed), rel_tol=1e-10, abs_tol=1e-6
            ):
                raise _ResearchWorkspaceInvalid(
                    f"completed {stage} terminal values differ from sealed runs"
                )


def _research_payload(
    research_dir: Path, study_id: str, runs_dir: Path | None = None
) -> dict[str, Any] | None:
    workspace = _research_workspace_path(research_dir, study_id)
    if workspace is None:
        return None
    try:
        validation = validate_research_workspace(workspace)
        manifest = _strict_json_object(
            workspace / "manifest.json", artifact="research workspace manifest"
        )
        plan = validate_research_plan(
            _strict_json_object(workspace / "plan.json", artifact="research plan")
        )
        state = read_research_state(workspace / "state.json", plan)
        experiments = _strict_json_object(
            workspace / "experiments.json", artifact="research experiment manifest"
        )
        expected_experiments = build_research_experiment_manifest(
            plan, state.get("frozen_confirmation_spec")
        )
        if experiments != expected_experiments:
            raise _ResearchWorkspaceInvalid(
                "research experiment manifest differs from the sealed state"
            )
        execution, execution_sha256 = _verify_execution_record(workspace, plan, state)
        pipeline_root = research_dir.parent.resolve()
        resolved_runs = (runs_dir or pipeline_root / "runs").resolve()
        if resolved_runs.parent != pipeline_root:
            raise _ResearchWorkspaceInvalid("research runs root is unsafe")
        base_config = _runtime_research_config(workspace, pipeline_root, plan)
    except _ResearchWorkspaceInvalid:
        raise
    except Exception as exc:
        raise _ResearchWorkspaceInvalid("research workspace verification failed") from exc

    frozen = state.get("frozen_confirmation_spec")
    frozen_sha256 = frozen.get("sha256") if isinstance(frozen, dict) else None
    exposure = _conservative_exposure(plan)
    stages = []
    for stage in RESEARCH_STAGES:
        record = state["stages"][stage]
        result = _validate_stage_result(stage, record, plan, frozen_sha256)
        partition = plan["partitions"][stage]
        public_stage: dict[str, Any] = {
            "name": stage,
            "label": RESEARCH_STAGE_LABELS[stage],
            "status": record["status"],
            "status_label": RESEARCH_STAGE_STATUS_LABELS[record["status"]],
            "attempts": record["attempts"],
            "start": partition["start"],
            "end": partition["end"],
            "sessions": partition["observations"],
            "source_data_end": partition["source_data_end"],
            **exposure["stages"][stage],
        }
        if result is not None:
            public_stage["result_verified"] = True
            public_stage["result_sha256"] = record["result_sha256"]
            public_stage["result"] = (
                _public_discovery_result(result, state, plan)
                if stage == "discovery"
                else _public_later_stage_result(result, stage, record)
            )
            actual = _load_actual_stage_evidence(
                stage=stage,
                result=result,
                execution=execution,
                experiments=experiments,
                plan=plan,
                base_config=base_config,
                runs_dir=resolved_runs,
            )
            if actual is not None:
                _cross_check_stage_actual(stage, result, actual)
                public_stage["actual_evidence"] = _public_actual_stage_evidence(
                    actual, plan
                )
        stages.append(public_stage)

    execution_status = execution.get("status") if execution is not None else None
    selected = (
        list(frozen.get("selected_factor_names", []))
        if isinstance(frozen, dict)
        else list(state.get("discovery_eligible_factor_names") or [])
    )
    statistics = plan["statistics"]
    evidence = plan["execution_evidence"]
    factor_multiplicity = statistics["factor_multiplicity"]
    return {
        "study_id": study_id,
        "status": validation["status"],
        "claim_status": "research_only",
        "execution_status": execution_status,
        "execution_status_label": RESEARCH_EXECUTION_STATUS_LABELS.get(
            execution_status, "未知状态"
        ),
        "exposure": exposure,
        "verification": {
            "verified": True,
            "sealed": True,
            "workspace_manifest_sha256": manifest.get("manifest_sha256"),
            "plan_sha256": plan["plan_sha256"],
            "state_sha256": state["state_sha256"],
            "execution_sha256": execution_sha256,
        },
        "pre_registration": {
            "protocol_version": plan["protocol_version"],
            "plan_id": plan["plan_id"],
            "catalog_sha256": plan["catalog_sha256"],
            "base_config_sha256": plan["base_config_sha256"],
            "calendar_sha256": plan["calendar_sha256"],
            "label_horizon_bars": plan["label_horizon_bars"],
            "family_ablation_count": len(plan["family_ablations"]),
            "factor_hypothesis_count": len(plan["factor_hypotheses"]),
            "primary_metrics": [
                metric for metric in statistics["primary_metrics"] if isinstance(metric, str)
            ],
            "alternative": statistics["alternative"],
            "hac_max_lag": statistics["hac_max_lag"],
            "bh": _select_scalars(
                factor_multiplicity, ("method", "family_size", "fdr_q")
            ),
            "confirmation_alpha": statistics["confirmation_alpha"],
            "terminal_rule": statistics["terminal_rule"],
        },
        "account_contract": _select_scalars(
            evidence,
            (
                "account_currency",
                "initial_account",
                "required_stress_slippage_bps_per_side",
                "engine",
                "daily_metric",
                "alignment_method",
                "comparison",
                "minimum_candidate_terminal_account",
                "minimum_intent_fill_rate",
                "maximum_zero_fill_intent_rate",
            ),
        ),
        "selected_factor_names": selected,
        "frozen_candidate": {
            "status": validation["frozen_candidate_status"],
            "spec_sha256": frozen_sha256,
            "selected_factor_names": selected,
        },
        "stages": stages,
        "limitations": [
            limitation for limitation in plan.get("limitations", []) if isinstance(limitation, str)
        ],
    }


def _research_payloads(
    research_dir: Path, runs_dir: Path | None = None
) -> list[dict[str, Any]]:
    if not research_dir.exists():
        return []
    if research_dir.is_symlink() or not research_dir.is_dir():
        raise _ResearchWorkspaceInvalid("research root is unsafe")
    try:
        children = sorted(research_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise _ResearchWorkspaceInvalid("research root cannot be read") from exc
    payloads = []
    for child in children:
        if child.is_file() and child.name == "exposure_registry.json":
            continue
        if not child.is_dir() and not child.is_symlink():
            continue
        try:
            validate_study_id(child.name)
        except (TypeError, ValueError) as exc:
            raise _ResearchWorkspaceInvalid("research root contains an unsafe study id") from exc
        payload = _research_payload(research_dir, child.name, runs_dir)
        if payload is None:
            raise _ResearchWorkspaceInvalid("research workspace disappeared during verification")
        payloads.append(payload)
    return payloads


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
    trusted = _trusted_run(runs_dir, candidate_id)
    if trusted is None:
        return {}
    _, manifest, _ = trusted
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
<h2>冻结原创研究候选目录</h2><p class='meta'>目录版本 {_text(factor_catalog.get('catalog_version'))} · SHA-256 <code>{_text(factor_catalog.get('sha256'))}</code></p><div class='table'><table><thead><tr><th>因子</th><th>族群</th><th>方向</th><th>Lookback</th><th>预注册假设</th></tr></thead><tbody>{factor_content}</tbody></table></div>
<p class='meta' style='margin:24px 0'>比较编号 {_text(comparison_id)} · 生成时间 {_text(public.get('generated_at'))}</p></main></body></html>"""
    )


def _research_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_id": payload["study_id"],
        "claim_status": payload["claim_status"],
        "execution_status": payload["execution_status"],
        "execution_status_label": payload["execution_status_label"],
        "exposure": payload["exposure"],
        "verification": payload["verification"],
        "selected_factor_names": payload["selected_factor_names"],
        "stages": [
            _select_scalars(
                stage,
                (
                    "name",
                    "label",
                    "status",
                    "status_label",
                    "attempts",
                    "start",
                    "end",
                    "sessions",
                    "evidence_class",
                    "evidence_label",
                    "claim_classification",
                    "result_verified",
                ),
            )
            for stage in payload["stages"]
        ],
    }


def _research_integrity_error() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail="research workspace exists but failed sealed-artifact verification",
    )


def _criterion_rows(criteria: Any) -> str:
    if not isinstance(criteria, dict) or not criteria:
        return "<tr><td colspan='2'>尚无已密封判定</td></tr>"
    return "".join(
        "<tr>"
        f"<td>{_text(RESEARCH_CRITERION_LABELS.get(name, name))}</td>"
        f"<td><span class='check {'pass' if passed is True else 'fail'}'>"
        f"{'通过' if passed is True else '未通过'}</span></td></tr>"
        for name, passed in criteria.items()
    )


def _research_metric_rows(tests: Any) -> str:
    if not isinstance(tests, dict):
        return "<tr><td colspan='7'>尚无已密封配对证据</td></tr>"
    labels = {"rank_ic": "Rank IC", "strategy_net": "10 bps 压力净收益"}
    rows = []
    for name in ("rank_ic", "strategy_net"):
        metric = tests.get(name)
        if not isinstance(metric, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{labels[name]}</td><td class='num'>{_text(metric.get('observations'))}</td>"
            f"<td class='num'>{_num(metric.get('baseline_mean'), 6)}</td>"
            f"<td class='num'>{_num(metric.get('candidate_mean'), 6)}</td>"
            f"<td class='num'>{_num(metric.get('mean_difference'), 6)}</td>"
            f"<td class='num'>{_num(metric.get('hac_t_stat'), 3)}</td>"
            f"<td class='num'>{_num(metric.get('one_sided_p_value'), 5)}</td></tr>"
        )
    return "".join(rows) or "<tr><td colspan='7'>尚无已密封配对证据</td></tr>"


def _research_terminal_rows(tests: Any) -> str:
    terminal = tests.get("terminal") if isinstance(tests, dict) else None
    if not isinstance(terminal, dict):
        return "<tr><td colspan='2'>尚无已密封账户证据</td></tr>"
    rows = (
        ("初始本金", _money(terminal.get("initial_account"))),
        ("压力滑点（单边）", f"{_num(terminal.get('stress_slippage_bps_per_side'), 0)} bps"),
        ("基准终值", _money(terminal.get("baseline_terminal_account"))),
        ("候选终值", _money(terminal.get("candidate_terminal_account"))),
        ("候选 - 基准", _money(terminal.get("account_improvement"), signed=True)),
        ("相对财富增量", _pct(terminal.get("relative_wealth_improvement"))),
        (
            "基准成交质量",
            "通过" if terminal.get("baseline_execution_quality_passed") is True else "未通过",
        ),
        (
            "候选成交质量",
            "通过" if terminal.get("candidate_execution_quality_passed") is True else "未通过",
        ),
    )
    return "".join(f"<tr><td>{label}</td><td class='num'>{_text(value)}</td></tr>" for label, value in rows)


def _render_research_index(payloads: list[dict[str, Any]]) -> HTMLResponse:
    rows = []
    for payload in payloads:
        stages = payload["stages"]
        encoded = html.escape(quote(payload["study_id"], safe=""), quote=True)
        rows.append(
            "<tr>"
            f"<td><a href='/research/{encoded}'>{_text(payload['study_id'])}</a></td>"
            "<td><span class='badge exposed'>历史已暴露</span></td>"
            "<td><span class='badge research'>research_only</span></td>"
            f"<td>{_text(payload['execution_status_label'])}</td>"
            + "".join(
                f"<td>{_text(stage['status_label'])}</td>" for stage in stages
            )
            + f"<td class='num'>{len(payload['selected_factor_names'])}</td>"
            "<td><span class='badge verified'>已密封并验证</span></td></tr>"
        )
    content = "".join(rows) or "<tr><td colspan='9'>暂无已验证研究工作区</td></tr>"
    return HTMLResponse(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>因子研究工作区</title><style>
body{{margin:0;color:#17212a;font-family:"Microsoft YaHei","Segoe UI",sans-serif;letter-spacing:0;background:#fff}}header{{border-bottom:1px solid #dce2e5}}.inner,main{{width:min(1260px,calc(100% - 40px));margin:0 auto}}.inner{{padding:20px 0}}h1{{font-size:22px;margin:5px 0}}p{{margin:5px 0;line-height:1.6}}a{{color:#215e83;text-decoration:none;font-weight:600}}.meta{{color:#64727b;font-size:12px}}.notice{{border-left:4px solid #aa3d34;background:#fff4f1;padding:12px 14px;margin:22px 0;line-height:1.6}}.table{{overflow:auto;border:1px solid #dce2e5}}table{{width:100%;border-collapse:collapse;min-width:980px;font-size:12px}}th{{text-align:left;background:#eef1f2;padding:10px 12px;border-bottom:1px solid #cbd1d5;white-space:nowrap}}td{{padding:10px 12px;border-bottom:1px solid #edf0f2;white-space:nowrap}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}.badge{{display:inline-block;padding:3px 7px;border-radius:4px;font-size:11px;font-weight:700}}.exposed,.research{{color:#9e3434;background:#fdebea}}.verified{{color:#0d6248;background:#e8f5ef}}@media(max-width:560px){{.inner,main{{width:calc(100% - 24px)}}}}
</style></head><body><header><div class='inner'><div class='meta'><a href='/'>返回运行列表</a> · 只读审计视图</div><h1>因子研究工作区</h1><p>只列出控制文件和研究状态均通过摘要校验的预注册研究。</p></div></header><main><div class='notice'><strong>历史暴露声明：</strong>当前研究所用历史数据已经暴露，全部证据仅限研究，不构成新增样本验证或实盘结论。</div><div class='table'><table><thead><tr><th>Study</th><th>证据属性</th><th>结论等级</th><th>执行状态</th><th>发现期</th><th>确认期</th><th>锁定留出集</th><th>选中因子</th><th>完整性</th></tr></thead><tbody>{content}</tbody></table></div></main></body></html>"""
    )


def _render_research(payload: dict[str, Any]) -> HTMLResponse:
    contract = payload["account_contract"]
    protocol = payload["pre_registration"]
    verification = payload["verification"]
    bh = protocol["bh"]
    selected = payload["selected_factor_names"]
    selected_content = "".join(f"<li><code>{_text(name)}</code></li>" for name in selected)
    if not selected_content:
        selected_content = "<li>尚无通过冻结规则的候选因子</li>"

    stage_blocks = []
    for stage in payload["stages"]:
        result = stage.get("result")
        result_badge = (
            "<span class='badge verified'>结果摘要已验证</span>"
            if stage.get("result_verified") is True
            else "<span class='badge pending'>无结果产物</span>"
        )
        stage_blocks.append(
            f"<div class='stage'><div class='stage-top'><div><div class='meta'>{_text(stage['name'])}</div>"
            f"<strong>{_text(stage['label'])}</strong></div><span class='badge {html.escape(stage['status'], quote=True)}'>"
            f"{_text(stage['status_label'])}</span></div><div class='stage-dates'>{_text(stage['start'])} 至 {_text(stage['end'])}"
            f" · {_text(stage['sessions'])} 个信号日</div><div class='stage-foot'>"
            f"<span class='badge exposed'>历史已暴露</span>{result_badge}</div></div>"
        )

    discovery = next(stage for stage in payload["stages"] if stage["name"] == "discovery")
    discovery_result = discovery.get("result") if isinstance(discovery.get("result"), dict) else {}
    discovery_bh = discovery_result.get("bh") if isinstance(discovery_result.get("bh"), dict) else {}
    factor_rows = []
    for factor in discovery_result.get("factors", []):
        terminal = factor.get("tests", {}).get("terminal", {})
        criteria = factor.get("selection_criteria", {})
        selected_passed = bool(criteria) and all(criteria.values())
        factor_rows.append(
            "<tr>"
            f"<td><code>{_text(factor.get('hypothesis_id'))}</code></td>"
            f"<td>{_text(factor.get('family'))}</td>"
            f"<td class='num'>{_num(factor.get('joint_p_value'), 5)}</td>"
            f"<td class='num'>{_num(factor.get('bh_q_value'), 5)}</td>"
            f"<td>{'是' if factor.get('bh_rejected') is True else '否'}</td>"
            f"<td class='num'>{_money(terminal.get('candidate_terminal_account'))}</td>"
            f"<td>{'通过' if terminal.get('candidate_execution_quality_passed') is True else '未通过'}</td>"
            f"<td><span class='check {'pass' if selected_passed else 'fail'}'>"
            f"{'入选' if selected_passed else '未入选'}</span></td></tr>"
        )
    factor_content = "".join(factor_rows) or "<tr><td colspan='8'>发现期尚无已密封 BH 结果</td></tr>"

    later_sections = []
    for stage_name in ("confirmation", "locked_holdout"):
        stage = next(item for item in payload["stages"] if item["name"] == stage_name)
        result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
        decision_name = "confirmation_passed" if stage_name == "confirmation" else "locked_holdout_passed"
        decision = result.get(decision_name)
        decision_label = "通过" if decision is True else "未通过" if decision is False else "尚无已密封结论"
        later_sections.append(
            f"<h2>{_text(stage['label'])} · {decision_label}</h2>"
            f"<p class='meta'>证据属性：历史已暴露 · 结论等级：research_only · 状态：{_text(stage['status_label'])}</p>"
            "<div class='split'><div><h3>预声明晋级条件</h3><div class='table'><table><thead><tr><th>条件</th><th>结果</th></tr></thead>"
            f"<tbody>{_criterion_rows(result.get('criteria'))}</tbody></table></div></div>"
            "<div><h3>小账户终值与成交质量</h3><div class='table'><table><thead><tr><th>项目</th><th>值</th></tr></thead>"
            f"<tbody>{_research_terminal_rows(result.get('tests'))}</tbody></table></div></div></div>"
            "<h3>配对统计证据</h3><div class='table'><table><thead><tr><th>指标</th><th>样本</th><th>基准均值</th><th>候选均值</th><th>增量</th><th>HAC t</th><th>单侧 p</th></tr></thead>"
            f"<tbody>{_research_metric_rows(result.get('tests'))}</tbody></table></div>"
        )

    contract_rows = (
        ("初始本金", _money(contract.get("initial_account"))),
        ("账户币种", contract.get("account_currency")),
        ("压力滑点（单边）", f"{_num(contract.get('required_stress_slippage_bps_per_side'), 0)} bps"),
        ("候选最低终值", _money(contract.get("minimum_candidate_terminal_account"))),
        ("最低意图成交率", _pct(contract.get("minimum_intent_fill_rate"))),
        ("最高零成交意图率", _pct(contract.get("maximum_zero_fill_intent_rate"))),
        ("执行引擎", contract.get("engine")),
        ("比较口径", contract.get("comparison")),
    )
    contract_content = "".join(
        f"<tr><td>{label}</td><td class='wrap'>{_text(value)}</td></tr>" for label, value in contract_rows
    )
    protocol_rows = (
        ("协议版本", protocol.get("protocol_version")),
        ("发现期电池", f"{_text(protocol.get('family_ablation_count'))} 个因子族消融 + {_text(protocol.get('factor_hypothesis_count'))} 个单因子假设"),
        ("多重检验", f"{_text(bh.get('method'))}，q={_num(bh.get('fdr_q'), 2)}，固定 {_text(bh.get('family_size'))} 项"),
        ("确认显著性", f"单侧 alpha={_num(protocol.get('confirmation_alpha'), 2)}"),
        ("HAC 最大滞后", protocol.get("hac_max_lag")),
        ("标签期限", f"{_text(protocol.get('label_horizon_bars'))} 个交易日"),
        ("终值规则", protocol.get("terminal_rule")),
    )
    protocol_content = "".join(
        f"<tr><td>{label}</td><td class='wrap'>{_text(value)}</td></tr>" for label, value in protocol_rows
    )
    integrity_rows = (
        ("Workspace manifest", verification.get("workspace_manifest_sha256")),
        ("Research plan", verification.get("plan_sha256")),
        ("One-shot state", verification.get("state_sha256")),
        ("Execution record", verification.get("execution_sha256")),
        ("Exposure registry", payload["exposure"].get("registry_sha256")),
    )
    integrity_content = "".join(
        f"<tr><td>{label}</td><td class='wrap'><code>{_text(value)}</code></td></tr>"
        for label, value in integrity_rows
    )
    return HTMLResponse(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>研究审计 · {_text(payload['study_id'])}</title><style>
body{{margin:0;color:#17212a;font-family:"Microsoft YaHei","Segoe UI",sans-serif;letter-spacing:0;background:#fff}}header{{border-bottom:1px solid #dce2e5}}.inner,main{{width:min(1260px,calc(100% - 40px));margin:0 auto}}.inner{{padding:20px 0}}h1{{font-size:22px;margin:5px 0}}h2{{font-size:17px;margin:30px 0 10px}}h3{{font-size:14px;margin:18px 0 8px}}p{{margin:5px 0;line-height:1.6}}a{{color:#215e83;text-decoration:none;font-weight:600}}code{{font-size:11px;overflow-wrap:anywhere}}.meta{{color:#64727b;font-size:12px}}.notice{{border-left:4px solid #aa3d34;background:#fff4f1;padding:12px 14px;margin:22px 0;line-height:1.6}}.summary{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border:1px solid #dce2e5}}.summary>div{{padding:13px;border-right:1px solid #e4e8ea}}.summary>div:last-child{{border:0}}.label{{color:#64727b;font-size:11px;margin-bottom:5px}}.value{{font-size:14px;font-weight:700;overflow-wrap:anywhere}}.stages{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.stage{{border-top:3px solid #39735e;background:#f7faf8;padding:13px;min-height:118px}}.stage-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}}.stage-dates{{font-size:12px;margin:15px 0;color:#43515a}}.stage-foot{{display:flex;gap:6px;flex-wrap:wrap}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.table{{overflow:auto;border:1px solid #dce2e5}}table{{width:100%;border-collapse:collapse;min-width:660px;font-size:12px}}th{{text-align:left;background:#eef1f2;padding:9px 11px;border-bottom:1px solid #cbd1d5;white-space:nowrap}}td{{padding:9px 11px;border-bottom:1px solid #edf0f2;vertical-align:top;white-space:nowrap}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}td.wrap{{white-space:normal;min-width:280px;line-height:1.5}}.badge,.check{{display:inline-block;padding:3px 7px;border-radius:4px;font-size:11px;font-weight:700}}.verified,.completed,.pass{{color:#0d6248;background:#e8f5ef}}.exposed,.research,.failed,.fail{{color:#9e3434;background:#fdebea}}.pending,.unopened,.claimed{{color:#72520b;background:#fff4d8}}ul{{line-height:1.7}}@media(max-width:820px){{.inner,main{{width:calc(100% - 24px)}}.summary{{grid-template-columns:1fr 1fr}}.summary>div{{border-bottom:1px solid #e4e8ea}}.stages,.split{{grid-template-columns:1fr}}}}
</style></head><body><header><div class='inner'><div class='meta'><a href='/research'>返回研究列表</a> · 只读密封产物审计</div><h1>{_text(payload['study_id'])}</h1><p>预注册因子发现、确认与锁定留出集进度。</p></div></header><main>
<div class='notice'><strong>历史已暴露 · research_only：</strong>该研究的历史样本不构成新增样本验证；页面中的“通过”仅表示满足已密封研究规则，不代表可实盘。</div>
<section class='summary'><div><div class='label'>完整性</div><div class='value'>已密封并验证</div></div><div><div class='label'>执行状态</div><div class='value'>{_text(payload['execution_status_label'])}</div></div><div><div class='label'>初始本金</div><div class='value'>{_money(contract.get('initial_account'))}</div></div><div><div class='label'>压力滑点</div><div class='value'>{_num(contract.get('required_stress_slippage_bps_per_side'), 0)} bps / 单边</div></div><div><div class='label'>BH</div><div class='value'>q={_num(bh.get('fdr_q'), 2)}</div></div><div><div class='label'>冻结因子</div><div class='value'>{len(selected)}</div></div></section>
<h2>研究阶段</h2><div class='stages'>{''.join(stage_blocks)}</div>
<div class='split'><div><h2>预注册协议</h2><div class='table'><table><thead><tr><th>项目</th><th>冻结值</th></tr></thead><tbody>{protocol_content}</tbody></table></div></div><div><h2>小账户与成交契约</h2><div class='table'><table><thead><tr><th>项目</th><th>冻结值</th></tr></thead><tbody>{contract_content}</tbody></table></div></div></div>
<h2>发现期 BH 筛选</h2><p class='meta'>方法 {_text(discovery_bh.get('method', bh.get('method')))} · 固定 {_text(discovery_bh.get('hypothesis_count', bh.get('family_size')))} 项 · 拒绝 {_text(discovery_bh.get('rejected_count'))} 项 · q={_num(discovery_bh.get('fdr_q', bh.get('fdr_q')), 2)}</p><div class='table'><table><thead><tr><th>因子</th><th>族群</th><th>联合 p</th><th>BH q 值</th><th>BH 拒绝</th><th>10 bps 候选终值</th><th>成交质量</th><th>联合选择</th></tr></thead><tbody>{factor_content}</tbody></table></div>
<h2>冻结候选</h2><ul>{selected_content}</ul>
{''.join(later_sections)}
<h2>密封与验证</h2><p class='meta'>本页每次请求都会重新验证 workspace manifest、预注册 plan、实验清单、一次性 state，以及存在时的 execution record。任何缺失或摘要不一致都会拒绝展示。</p><div class='table'><table><thead><tr><th>产物</th><th>SHA-256</th></tr></thead><tbody>{integrity_content}</tbody></table></div>
</main></body></html>"""
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
    research_dir = (pipeline_root / "research").resolve()
    app = FastAPI(title="My Quant Pipeline", docs_url="/api/docs")

    @app.get("/api/runs")
    def runs_api():
        return {"runs": [_public_run(run) for run in _trusted_registry_runs(registry_path, runs_dir)]}

    @app.get("/api/runs/{run_id}")
    def run_api(run_id: str):
        trusted = _trusted_run(runs_dir, run_id)
        if trusted is None:
            raise HTTPException(404, "run not found")
        run_dir, manifest, _ = trusted
        metrics = _read_json_object(run_dir / "metrics.json")
        gates = _read_json_object(run_dir / "gates.json")
        if metrics is None or gates is None:
            raise HTTPException(404, "run not found")
        return {
            "manifest": _public_manifest(manifest),
            "metrics": _public_metrics(metrics),
            "gates": _public_gates(gates),
        }

    @app.get("/api/comparisons")
    def comparisons_api():
        return {
            "comparisons": [
                _comparison_payload(comparison_id, comparison, runs_dir)
                for comparison_id, comparison in _trusted_comparisons(comparisons_dir, runs_dir)
            ]
        }

    @app.get("/api/comparisons/{comparison_id}")
    def comparison_api(comparison_id: str):
        path = _comparison_path(comparisons_dir, comparison_id)
        comparison = (
            dict(_trusted_comparisons(comparisons_dir, runs_dir)).get(comparison_id)
            if path is not None
            else None
        )
        if comparison is None:
            raise HTTPException(404, "comparison not found")
        return _comparison_payload(comparison_id, comparison, runs_dir)

    @app.get("/api/research")
    def research_api():
        try:
            payloads = _research_payloads(research_dir)
        except _ResearchWorkspaceInvalid:
            raise _research_integrity_error() from None
        return {"studies": [_research_summary(payload) for payload in payloads]}

    @app.get("/api/research/{study_id}")
    def research_detail_api(study_id: str):
        try:
            payload = _research_payload(research_dir, study_id)
        except _ResearchWorkspaceInvalid:
            raise _research_integrity_error() from None
        if payload is None:
            raise HTTPException(404, "research workspace not found")
        return payload

    @app.get("/research", response_class=HTMLResponse)
    def research_index_page():
        try:
            return _render_research_index(_research_payloads(research_dir))
        except _ResearchWorkspaceInvalid:
            raise _research_integrity_error() from None

    @app.get("/research/{study_id}", response_class=HTMLResponse)
    def research_detail_page(study_id: str):
        try:
            payload = _research_payload(research_dir, study_id)
        except _ResearchWorkspaceInvalid:
            raise _research_integrity_error() from None
        if payload is None:
            raise HTTPException(404, "research workspace not found")
        return _render_research(payload)

    @app.get("/api/runs/{run_id}/comparison")
    def run_comparison_api(run_id: str):
        if _trusted_run(runs_dir, run_id) is None:
            raise HTTPException(404, "run not found")
        match = _find_candidate_comparison(_trusted_comparisons(comparisons_dir, runs_dir), run_id)
        if match is None:
            raise HTTPException(404, "comparison not found")
        return _comparison_payload(match[0], match[1], runs_dir)

    @app.get("/comparisons/{comparison_id}", response_class=HTMLResponse)
    def comparison_page(comparison_id: str):
        path = _comparison_path(comparisons_dir, comparison_id)
        comparison = (
            dict(_trusted_comparisons(comparisons_dir, runs_dir)).get(comparison_id)
            if path is not None
            else None
        )
        if comparison is None:
            raise HTTPException(404, "comparison not found")
        return _render_comparison(
            comparison_id,
            comparison,
            _factor_catalog_for_comparison(runs_dir, comparison),
        )

    @app.get("/runs/{run_id}/comparison", response_class=HTMLResponse)
    def run_comparison_page(run_id: str):
        if _trusted_run(runs_dir, run_id) is None:
            raise HTTPException(404, "run not found")
        match = _find_candidate_comparison(_trusted_comparisons(comparisons_dir, runs_dir), run_id)
        if match is None:
            raise HTTPException(404, "comparison not found")
        return _render_comparison(match[0], match[1], _factor_catalog_for_comparison(runs_dir, match[1]))

    @app.get("/runs/{run_id}")
    def run_report(run_id: str):
        trusted = _trusted_run(runs_dir, run_id)
        if trusted is None:
            raise HTTPException(404, "report not found")
        run_dir, manifest, _ = trusted
        report = (run_dir / "report.html").resolve()
        if report.parent != run_dir or not report.is_file():
            raise HTTPException(404, "report not found")
        try:
            document = report.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise HTTPException(404, "report not found") from None
        sanitized = document
        local_paths = {str(pipeline_root), pipeline_root.as_posix(), str(report.parent), report.parent.as_posix()}
        if manifest.get("config"):
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
        runs = _trusted_registry_runs(registry_path, runs_dir)
        comparisons_by_candidate = {}
        for comparison_id, comparison in _trusted_comparisons(comparisons_dir, runs_dir):
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
</style></head><body><header><div class='inner'><div><h1>My Quant Pipeline</h1><div class='sub'>ETF 研究、审计、滚动训练与回测</div></div><div class='meta'><a href='/research'>研究工作区</a> · API <a href='/api/docs'>/api/docs</a></div></div></header><main><h2>实验运行</h2><div class='table'><table><thead><tr><th>Run</th><th>分类</th><th>状态</th><th>数据快照</th><th>净收益</th><th>基准</th><th>IC</th><th>回撤</th><th>增量审计</th><th>完成时间</th></tr></thead><tbody>{content}</tbody></table></div></main></body></html>"""
        )

    return app


def _pct(value):
    return "-" if not _finite_real(value) else f"{float(value) * 100:.2f}%"


def _num(value, digits=2):
    return "-" if not _finite_real(value) else f"{float(value):.{digits}f}"


def _money(value, *, signed=False):
    if not _finite_real(value):
        return "-"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}CNY {float(value):,.2f}"


def _finite_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))
