"""Pre-registered, fail-closed factor discovery and confirmation controls.

This module does not train a model and does not infer that an experiment has
already happened.  It produces an immutable experiment plan, evaluates paired
daily metrics on one declared chronological partition, and maintains a local
one-shot access ledger for discovery, confirmation, and the locked holdout.

The ledger is an audit control rather than a security boundary.  Its purpose is
to make accidental or casual repeated holdout inspection fail closed.  A user
with permission to rewrite both code and state can still bypass it; durable
external experiment tracking is required for stronger enforcement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from statistics import NormalDist
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from .factors import FACTOR_FAMILIES, ORIGINAL_RESEARCH_CANDIDATES, factor_catalog_manifest
from .exposure import (
    build_exposure_provenance,
    stage_exposure_fields,
    validate_stage_exposure_fields,
)
from .io import now_shanghai, read_json, sha256_file, write_json_atomic
from .metrics import hac_t_stat, max_drawdown


RESEARCH_PLAN_SCHEMA_VERSION = 5
RESEARCH_STATE_SCHEMA_VERSION = 4
RESEARCH_PROTOCOL_VERSION = "etf_factor_discovery_confirmation_v5"
RESEARCH_STAGES = ("discovery", "confirmation", "locked_holdout")
FAMILY_ABLATION_COUNT = 5
CATALOG_HYPOTHESIS_COUNT = 18
DISCOVERY_FDR_Q = 0.10
CONFIRMATION_ALPHA = 0.05
DISCOVERY_MULTIPLICITY_APPLIED_TO = (
    "eighteen joint model_rank_ic_strategy_net_and_signed_raw_factor_rank_ic hypotheses"
)
DEFAULT_HAC_MAX_LAG = 5
DEFAULT_LABEL_HORIZON_BARS = 2
PORTFOLIO_EXECUTION_LAG_BARS = 1
PORTFOLIO_REALIZATION_LAG_BARS = 2
REQUIRED_STRESS_SLIPPAGE_BPS = 10
RESEARCH_ACCOUNT_CNY = 20_000.0
MIN_INTENT_FILL_RATE = 0.95
MIN_NOTIONAL_FILL_RATE = 0.95
MAX_ZERO_FILL_INTENT_RATE = 0.05
MIN_RAW_FACTOR_RANK_IC_COVERAGE = 0.90
MAX_STRATEGY_DRAWDOWN = 0.25
RESEARCH_FOLD_SIGNAL_SESSIONS = 21
MIN_RESEARCH_COMPLETE_FOLDS = 3
MIN_RESEARCH_FOLD_WIN_RATIO = 0.60
MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE = 0.35
MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE = 0.50
RESEARCH_BENCHMARK = "SH510300"
RAW_SHARE_ENGINE = "raw_share_daily_v1"
EVALUATION_ALIGNMENT_METHOD = "initial_cost_compounded_into_first_realized_return"
MIN_DISCOVERY_SESSIONS = 126
MIN_CONFIRMATION_SESSIONS = 63
MIN_LOCKED_HOLDOUT_SESSIONS = 63

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_STATUS = {"unopened", "claimed", "completed", "failed"}
_DISCOVERY_SELECTION_CRITERIA_FIELDS = {
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
_CONFIRMATION_CRITERIA_FIELDS = {
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
_T = TypeVar("_T")


def _canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject NaN, infinity, and custom objects."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


MIN_PAIRED_OBSERVATIONS = 30
_BETA_EPS = 3e-14
_BETA_FP_MIN = 1e-300
_BETA_MAX_IT = 300


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via Lentz continued fraction."""

    if not all(math.isfinite(value) for value in (x, a, b)) or a <= 0.0 or b <= 0.0:
        raise ValueError("beta parameters must be finite and positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def betacf(xx: float, aa: float, bb: float) -> float:
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < _BETA_FP_MIN:
            d = _BETA_FP_MIN
        d = 1.0 / d
        h = d
        for m in range(1, _BETA_MAX_IT + 1):
            m2 = 2 * m
            term = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + term * d
            if abs(d) < _BETA_FP_MIN:
                d = _BETA_FP_MIN
            c = 1.0 + term / c
            if abs(c) < _BETA_FP_MIN:
                c = _BETA_FP_MIN
            d = 1.0 / d
            h *= d * c
            term = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + term * d
            if abs(d) < _BETA_FP_MIN:
                d = _BETA_FP_MIN
            c = 1.0 + term / c
            if abs(c) < _BETA_FP_MIN:
                c = _BETA_FP_MIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < _BETA_EPS:
                break
        return h

    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) + log_beta)
    return front * betacf(x, a, b) / a


def _student_t_cdf(x: float, df: float) -> float:
    """CDF of Student's t with positive degrees of freedom."""

    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("df must be finite and positive")
    if df >= 1e6:
        return float(NormalDist().cdf(x))
    value = float(x)
    z = df / (df + value * value)
    ib = _regularized_incomplete_beta(z, df / 2.0, 0.5)
    if value >= 0.0:
        return 0.5 * (2.0 - ib)
    return 0.5 * ib


def _one_sided_hac_p_value(statistic: float, observations: int, max_lag: int) -> float:
    """Student-t one-sided p-value with Newey-West small-sample degrees of freedom."""

    df = max(1.0, float(observations) - float(max_lag) - 1.0)
    return _student_t_cdf(-float(statistic), df)


def _finite_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return number


def _normalize_sessions(session_dates: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(session_dates, (str, bytes)):
        raise TypeError("session_dates must be an iterable of individual dates")
    normalized: list[pd.Timestamp] = []
    for position, value in enumerate(session_dates):
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"session date {position} is invalid") from exc
        if pd.isna(timestamp):
            raise ValueError(f"session date {position} is invalid")
        if timestamp.tz is not None:
            raise ValueError("session dates must be timezone-naive exchange dates")
        if timestamp != timestamp.normalize():
            raise ValueError("session dates must not contain intraday times")
        normalized.append(timestamp)
    if not normalized:
        raise ValueError("at least one session date is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("session dates must be unique")
    if normalized != sorted(normalized):
        raise ValueError("session dates must be strictly chronological")
    return tuple(timestamp.date().isoformat() for timestamp in normalized)


def _partition_record(
    name: str,
    sessions: Sequence[str],
    maturity_sessions: Sequence[str],
    *,
    label_horizon_bars: int,
) -> dict[str, Any]:
    if label_horizon_bars != PORTFOLIO_REALIZATION_LAG_BARS:
        raise ValueError(
            "the research portfolio evidence contract requires a two-session realization lag"
        )
    raw_report_sessions = [
        *sessions[PORTFOLIO_EXECUTION_LAG_BARS:],
        *maturity_sessions,
    ]
    evaluation_sessions = raw_report_sessions[1:]
    stage_calendar = [*sessions, *maturity_sessions]
    research_folds = []
    for fold_number, start in enumerate(
        range(0, len(sessions), RESEARCH_FOLD_SIGNAL_SESSIONS), start=1
    ):
        end = min(start + RESEARCH_FOLD_SIGNAL_SESSIONS, len(sessions))
        fold_sessions = list(sessions[start:end])
        fold_raw = stage_calendar[start + 1 : end + PORTFOLIO_REALIZATION_LAG_BARS]
        fold_evaluation = fold_raw[1:]
        research_folds.append(
            {
                "fold": fold_number,
                "signal_start": fold_sessions[0],
                "signal_end": fold_sessions[-1],
                "signal_observations": len(fold_sessions),
                "signal_sessions": fold_sessions,
                "signal_sessions_sha256": _sha256_json(fold_sessions),
                "raw_report_start": fold_raw[0],
                "raw_report_end": fold_raw[-1],
                "raw_report_sessions": fold_raw,
                "raw_report_sessions_sha256": _sha256_json(fold_raw),
                "evaluation_start": fold_evaluation[0],
                "evaluation_end": fold_evaluation[-1],
                "evaluation_sessions": fold_evaluation,
                "evaluation_sessions_sha256": _sha256_json(fold_evaluation),
                "complete_for_gate": len(fold_sessions)
                == RESEARCH_FOLD_SIGNAL_SESSIONS,
            }
        )
    return {
        "name": name,
        "start": sessions[0],
        "end": sessions[-1],
        "observations": len(sessions),
        "sessions": list(sessions),
        "sessions_sha256": _sha256_json(list(sessions)),
        "label_horizon_bars": label_horizon_bars,
        "label_maturity_start": maturity_sessions[0],
        "label_maturity_end": maturity_sessions[-1],
        "label_maturity_sessions": list(maturity_sessions),
        "label_maturity_sessions_sha256": _sha256_json(list(maturity_sessions)),
        "source_data_end": maturity_sessions[-1],
        "portfolio_execution_lag_bars": PORTFOLIO_EXECUTION_LAG_BARS,
        "portfolio_realization_lag_bars": PORTFOLIO_REALIZATION_LAG_BARS,
        "portfolio_raw_report_start": raw_report_sessions[0],
        "portfolio_raw_report_end": raw_report_sessions[-1],
        "portfolio_raw_report_sessions": raw_report_sessions,
        "portfolio_raw_report_sessions_sha256": _sha256_json(raw_report_sessions),
        "portfolio_evaluation_start": evaluation_sessions[0],
        "portfolio_evaluation_end": evaluation_sessions[-1],
        "portfolio_evaluation_sessions": evaluation_sessions,
        "portfolio_evaluation_sessions_sha256": _sha256_json(evaluation_sessions),
        "research_fold_signal_sessions": RESEARCH_FOLD_SIGNAL_SESSIONS,
        "research_folds": research_folds,
    }


def build_time_partitions(
    session_dates: Iterable[Any],
    *,
    discovery_end: Any,
    confirmation_end: Any,
    min_discovery_sessions: int = MIN_DISCOVERY_SESSIONS,
    min_confirmation_sessions: int = MIN_CONFIRMATION_SESSIONS,
    min_locked_holdout_sessions: int = MIN_LOCKED_HOLDOUT_SESSIONS,
    label_horizon_bars: int = DEFAULT_LABEL_HORIZON_BARS,
) -> dict[str, dict[str, Any]]:
    """Split one frozen calendar into three chronological signal partitions.

    Both cut dates are the last *signal* session of their stage.  The following
    ``label_horizon_bars`` sessions are reserved solely to mature that stage's
    final forward label.  The next stage starts after that embargo.  The final
    horizon sessions in the supplied calendar mature the locked-holdout labels.
    This prevents a discovery metric dated at its boundary from using returns
    assigned to the confirmation partition.
    """

    if (
        isinstance(label_horizon_bars, bool)
        or not isinstance(label_horizon_bars, int)
        or label_horizon_bars < 1
    ):
        raise ValueError("label_horizon_bars must be a positive integer")
    minimums = {
        "discovery": min_discovery_sessions,
        "confirmation": min_confirmation_sessions,
        "locked_holdout": min_locked_holdout_sessions,
    }
    for name, minimum in minimums.items():
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError(f"min_{name}_sessions must be a positive integer")

    sessions = _normalize_sessions(session_dates)
    try:
        discovery_cut = pd.Timestamp(discovery_end).date().isoformat()
        confirmation_cut = pd.Timestamp(confirmation_end).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("partition cut dates must be valid dates") from exc
    if discovery_cut not in sessions or confirmation_cut not in sessions:
        raise ValueError("partition cut dates must be members of the frozen session calendar")
    discovery_position = sessions.index(discovery_cut)
    confirmation_position = sessions.index(confirmation_cut)
    confirmation_start = discovery_position + 1 + label_horizon_bars
    holdout_start = confirmation_position + 1 + label_horizon_bars
    holdout_end_exclusive = len(sessions) - label_horizon_bars
    if (
        discovery_position >= confirmation_position
        or confirmation_start > confirmation_position
        or holdout_start >= holdout_end_exclusive
    ):
        raise ValueError(
            "partitions and label-maturity embargoes must be non-empty and ordered "
            "discovery, confirmation, locked_holdout"
        )

    slices = {
        "discovery": sessions[: discovery_position + 1],
        "confirmation": sessions[confirmation_start : confirmation_position + 1],
        "locked_holdout": sessions[holdout_start:holdout_end_exclusive],
    }
    maturity = {
        "discovery": sessions[
            discovery_position + 1 : discovery_position + 1 + label_horizon_bars
        ],
        "confirmation": sessions[
            confirmation_position + 1 : confirmation_position + 1 + label_horizon_bars
        ],
        "locked_holdout": sessions[holdout_end_exclusive:],
    }
    for name, values in slices.items():
        if len(values) < minimums[name]:
            raise ValueError(
                f"{name} requires at least {minimums[name]} sessions; received {len(values)}"
            )
    return {
        name: _partition_record(
            name,
            slices[name],
            maturity[name],
            label_horizon_bars=label_horizon_bars,
        )
        for name in RESEARCH_STAGES
    }


def _unsigned_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in plan.items() if key != "plan_sha256"}


def build_research_plan(
    session_dates: Iterable[Any],
    *,
    discovery_end: Any,
    confirmation_end: Any,
    plan_id: str,
    base_config_sha256: str,
    min_discovery_sessions: int = MIN_DISCOVERY_SESSIONS,
    min_confirmation_sessions: int = MIN_CONFIRMATION_SESSIONS,
    min_locked_holdout_sessions: int = MIN_LOCKED_HOLDOUT_SESSIONS,
    label_horizon_bars: int = DEFAULT_LABEL_HORIZON_BARS,
    required_stress_slippage_bps: int = REQUIRED_STRESS_SLIPPAGE_BPS,
    account_cny: float = RESEARCH_ACCOUNT_CNY,
    specification_frozen_at: str | None = None,
) -> dict[str, Any]:
    """Build a reproducible plan without claiming that any run exists.

    The five family ablations and eighteen one-factor hypotheses are all marked
    ``not_run``.  Run identifiers and result fields are deliberately absent.
    """

    if not isinstance(plan_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", plan_id):
        raise ValueError("plan_id must be a portable 1-128 character identifier")
    if not isinstance(base_config_sha256, str) or not _DIGEST_PATTERN.fullmatch(
        base_config_sha256
    ):
        raise ValueError("base_config_sha256 must be a lowercase SHA-256 digest")
    if len(FACTOR_FAMILIES) != FAMILY_ABLATION_COUNT:
        raise RuntimeError("the frozen catalog no longer contains exactly five factor families")
    if len(ORIGINAL_RESEARCH_CANDIDATES) != CATALOG_HYPOTHESIS_COUNT:
        raise RuntimeError("the frozen catalog no longer contains exactly eighteen hypotheses")
    if label_horizon_bars != PORTFOLIO_REALIZATION_LAG_BARS:
        raise ValueError("the research protocol requires label_horizon_bars=2")
    if (
        isinstance(required_stress_slippage_bps, bool)
        or not isinstance(required_stress_slippage_bps, int)
        or required_stress_slippage_bps != REQUIRED_STRESS_SLIPPAGE_BPS
    ):
        raise ValueError("the research protocol requires 10 bps per-side stress slippage")
    if isinstance(account_cny, bool) or not isinstance(
        account_cny, (int, float, np.integer, np.floating)
    ):
        raise TypeError("research account_cny must be a finite number")
    account_value = float(account_cny)
    if not math.isfinite(account_value) or account_value != RESEARCH_ACCOUNT_CNY:
        raise ValueError("the research protocol requires an initial CNY 20,000 account")
    declared_minimums = {
        "discovery": min_discovery_sessions,
        "confirmation": min_confirmation_sessions,
        "locked_holdout": min_locked_holdout_sessions,
    }
    protocol_minimums = {
        "discovery": MIN_DISCOVERY_SESSIONS,
        "confirmation": MIN_CONFIRMATION_SESSIONS,
        "locked_holdout": MIN_LOCKED_HOLDOUT_SESSIONS,
    }
    for stage, minimum in declared_minimums.items():
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < protocol_minimums[stage]:
            raise ValueError(
                f"{stage} minimum cannot be lower than the protocol minimum "
                f"of {protocol_minimums[stage]} sessions"
            )

    partitions = build_time_partitions(
        session_dates,
        discovery_end=discovery_end,
        confirmation_end=confirmation_end,
        min_discovery_sessions=min_discovery_sessions,
        min_confirmation_sessions=min_confirmation_sessions,
        min_locked_holdout_sessions=min_locked_holdout_sessions,
        label_horizon_bars=label_horizon_bars,
    )
    catalog = factor_catalog_manifest()
    exposure_provenance = build_exposure_provenance(
        partitions,
        RESEARCH_STAGES,
        specification_frozen_at=specification_frozen_at,
    )
    plan: dict[str, Any] = {
        "schema_version": RESEARCH_PLAN_SCHEMA_VERSION,
        "protocol_version": RESEARCH_PROTOCOL_VERSION,
        "plan_id": plan_id,
        "claim_status": "pre_registered_plan_only",
        "experiment_status": "not_run",
        "catalog_sha256": catalog["sha256"],
        "base_config_sha256": base_config_sha256,
        "calendar_sha256": _sha256_json(
            [
                session
                for stage in RESEARCH_STAGES
                for field in ("sessions", "label_maturity_sessions")
                for session in partitions[stage][field]
            ]
        ),
        "exposure_provenance": exposure_provenance,
        "label_horizon_bars": label_horizon_bars,
        "execution_evidence": {
            "benchmark": RESEARCH_BENCHMARK,
            "account_currency": "CNY",
            "initial_account": account_value,
            "required_stress_slippage_bps_per_side": required_stress_slippage_bps,
            "engine": RAW_SHARE_ENGINE,
            "daily_metric": "strategy_net",
            "alignment_method": EVALUATION_ALIGNMENT_METHOD,
            "comparison": "candidate_minus_unchanged_alpha158_baseline",
            "minimum_candidate_terminal_account": account_value,
            "minimum_intent_fill_rate": MIN_INTENT_FILL_RATE,
            "minimum_notional_fill_rate": MIN_NOTIONAL_FILL_RATE,
            "maximum_zero_fill_intent_rate": MAX_ZERO_FILL_INTENT_RATE,
            "maximum_strategy_drawdown": MAX_STRATEGY_DRAWDOWN,
            "research_fold_signal_sessions": RESEARCH_FOLD_SIGNAL_SESSIONS,
            "minimum_complete_research_folds": MIN_RESEARCH_COMPLETE_FOLDS,
            "minimum_research_fold_win_ratio": MIN_RESEARCH_FOLD_WIN_RATIO,
            "maximum_single_etf_abs_contribution_share": (
                MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE
            ),
            "maximum_single_fold_abs_incremental_pnl_share": (
                MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE
            ),
        },
        "partitions": partitions,
        "family_ablations": [
            {
                "experiment_id": f"family_ablation__{family}",
                "family": family,
                "features_mode": "alpha158_plus_original",
                "families": [family],
                "stage": "discovery",
                "status": "not_run",
            }
            for family in FACTOR_FAMILIES
        ],
        "factor_hypotheses": [
            {
                "hypothesis_id": factor.name,
                "family": factor.family,
                "expected_direction": factor.direction,
                "stage": "discovery",
                "status": "not_run",
            }
            for factor in ORIGINAL_RESEARCH_CANDIDATES
        ],
        "statistics": {
            "primary_metrics": [
                "paired daily rank_ic difference versus the unchanged Alpha158 baseline",
                "paired daily 10 bps stress raw-share strategy_net difference versus the unchanged Alpha158 baseline",
                "expected-direction signed raw-factor daily rank_ic",
            ],
            "metric_session_semantics": (
                "signal date T only; the following label_horizon_bars sessions may be read "
                "only to mature T's forward label"
            ),
            "alternative": "candidate_minus_baseline_greater_than_zero",
            "hac_max_lag": DEFAULT_HAC_MAX_LAG,
            "joint_hypothesis": {
                "method": "intersection-union max component p-value",
                "component_nulls": [
                    "rank_ic_mean_difference_less_than_or_equal_to_zero",
                    "strategy_net_mean_difference_less_than_or_equal_to_zero",
                    "signed_raw_factor_rank_ic_mean_less_than_or_equal_to_zero",
                ],
                "joint_p_value": (
                    "max(rank_ic_one_sided_p, strategy_net_one_sided_p, "
                    "signed_raw_factor_rank_ic_one_sided_p)"
                ),
            },
            "factor_multiplicity": {
                "method": "Benjamini-Hochberg",
                "family_size": CATALOG_HYPOTHESIS_COUNT,
                "fdr_q": DISCOVERY_FDR_Q,
            },
            "family_ablations": "screening evidence only; no standalone confirmation claim",
            "confirmation_alpha": CONFIRMATION_ALPHA,
            "terminal_rule": (
                "candidate terminal account must not lose absolute capital, must improve versus baseline, "
                "and must beat SH510300 at 10 bps stress"
            ),
            "robustness_rule": (
                "10 bps cash-reset 21-signal-session folds, 25 percent maximum drawdown, "
                "and frozen instrument/fold contribution concentration limits"
            ),
        },
        "stage_policy": {
            "discovery": "Screen only on discovery sessions; freeze one candidate specification afterward.",
            "confirmation": "Evaluate the frozen specification once on confirmation sessions without retuning.",
            "locked_holdout": "Evaluate the same frozen specification once only after confirmation passes.",
            "one_shot_ledger": "A claimed stage is consumed even when evaluation fails.",
        },
        "limitations": [
            "No experiment, statistical result, or improvement is represented by this plan.",
            "Evidence remains research-only while the ETF universe is not point-in-time.",
            "Multiple-testing control does not repair data leakage, survivorship bias, or execution-model error.",
            "Benjamini-Hochberg FDR control relies on its standard dependence conditions across factor tests.",
            (
                "Dates on or before the fixed exposure cutoff are retrospective_exposed and can never "
                "support an unseen, pristine, blind, or promotion claim."
            ),
        ],
    }
    plan["plan_sha256"] = _sha256_json(plan)
    validate_research_plan(plan)
    return plan


def validate_research_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a defensive copy of a plan."""

    if not isinstance(plan, Mapping):
        raise TypeError("research plan must be a mapping")
    value = deepcopy(dict(plan))
    if value.get("schema_version") != RESEARCH_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported research plan schema")
    if value.get("protocol_version") != RESEARCH_PROTOCOL_VERSION:
        raise ValueError("unsupported research protocol version")
    digest = value.get("plan_sha256")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("research plan has an invalid plan_sha256")
    if _sha256_json(_unsigned_plan(value)) != digest:
        raise ValueError("research plan SHA-256 does not match its content")
    if value.get("catalog_sha256") != factor_catalog_manifest()["sha256"]:
        raise ValueError("research plan does not match the current frozen factor catalog")
    if not isinstance(value.get("base_config_sha256"), str) or not _DIGEST_PATTERN.fullmatch(
        value["base_config_sha256"]
    ):
        raise ValueError("research plan base configuration identity is invalid")
    if value.get("experiment_status") != "not_run":
        raise ValueError("a research plan must not claim that experiments have run")
    if value.get("claim_status") != "pre_registered_plan_only":
        raise ValueError("research plan claim status is invalid")

    partitions = value.get("partitions")
    if not isinstance(partitions, dict) or tuple(partitions) != RESEARCH_STAGES:
        raise ValueError("research plan must contain the three ordered partitions")
    label_horizon = value.get("label_horizon_bars")
    if (
        isinstance(label_horizon, bool)
        or not isinstance(label_horizon, int)
        or label_horizon != PORTFOLIO_REALIZATION_LAG_BARS
    ):
        raise ValueError("research plan label_horizon_bars is invalid")
    expected_execution_evidence = {
        "benchmark": RESEARCH_BENCHMARK,
        "account_currency": "CNY",
        "initial_account": RESEARCH_ACCOUNT_CNY,
        "required_stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
        "engine": RAW_SHARE_ENGINE,
        "daily_metric": "strategy_net",
        "alignment_method": EVALUATION_ALIGNMENT_METHOD,
        "comparison": "candidate_minus_unchanged_alpha158_baseline",
        "minimum_candidate_terminal_account": RESEARCH_ACCOUNT_CNY,
        "minimum_intent_fill_rate": MIN_INTENT_FILL_RATE,
        "minimum_notional_fill_rate": MIN_NOTIONAL_FILL_RATE,
        "maximum_zero_fill_intent_rate": MAX_ZERO_FILL_INTENT_RATE,
        "maximum_strategy_drawdown": MAX_STRATEGY_DRAWDOWN,
        "research_fold_signal_sessions": RESEARCH_FOLD_SIGNAL_SESSIONS,
        "minimum_complete_research_folds": MIN_RESEARCH_COMPLETE_FOLDS,
        "minimum_research_fold_win_ratio": MIN_RESEARCH_FOLD_WIN_RATIO,
        "maximum_single_etf_abs_contribution_share": (
            MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE
        ),
        "maximum_single_fold_abs_incremental_pnl_share": (
            MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE
        ),
    }
    if value.get("execution_evidence") != expected_execution_evidence:
        raise ValueError("research plan execution evidence contract is invalid")
    combined: list[str] = []
    previous_end: pd.Timestamp | None = None
    minimums = {
        "discovery": MIN_DISCOVERY_SESSIONS,
        "confirmation": MIN_CONFIRMATION_SESSIONS,
        "locked_holdout": MIN_LOCKED_HOLDOUT_SESSIONS,
    }
    for stage in RESEARCH_STAGES:
        partition = partitions[stage]
        if not isinstance(partition, dict) or partition.get("name") != stage:
            raise ValueError(f"invalid {stage} partition")
        sessions = _normalize_sessions(partition.get("sessions", []))
        if partition.get("start") != sessions[0] or partition.get("end") != sessions[-1]:
            raise ValueError(f"{stage} partition boundaries do not match its sessions")
        if partition.get("observations") != len(sessions):
            raise ValueError(f"{stage} partition observation count is invalid")
        if len(sessions) < minimums[stage]:
            raise ValueError(f"{stage} partition is below the protocol minimum")
        if partition.get("sessions_sha256") != _sha256_json(list(sessions)):
            raise ValueError(f"{stage} partition session digest is invalid")
        maturity = _normalize_sessions(partition.get("label_maturity_sessions", []))
        if len(maturity) != label_horizon:
            raise ValueError(f"{stage} label-maturity session count is invalid")
        if partition.get("label_horizon_bars") != label_horizon:
            raise ValueError(f"{stage} label horizon conflicts with the research plan")
        if (
            partition.get("label_maturity_start") != maturity[0]
            or partition.get("label_maturity_end") != maturity[-1]
            or partition.get("source_data_end") != maturity[-1]
        ):
            raise ValueError(f"{stage} label-maturity boundaries are invalid")
        if partition.get("label_maturity_sessions_sha256") != _sha256_json(list(maturity)):
            raise ValueError(f"{stage} label-maturity session digest is invalid")
        raw_report_sessions = list(sessions[PORTFOLIO_EXECUTION_LAG_BARS:]) + list(maturity)
        evaluation_sessions = raw_report_sessions[1:]
        portfolio_contract = {
            "portfolio_execution_lag_bars": PORTFOLIO_EXECUTION_LAG_BARS,
            "portfolio_realization_lag_bars": PORTFOLIO_REALIZATION_LAG_BARS,
            "portfolio_raw_report_start": raw_report_sessions[0],
            "portfolio_raw_report_end": raw_report_sessions[-1],
            "portfolio_raw_report_sessions": raw_report_sessions,
            "portfolio_raw_report_sessions_sha256": _sha256_json(raw_report_sessions),
            "portfolio_evaluation_start": evaluation_sessions[0],
            "portfolio_evaluation_end": evaluation_sessions[-1],
            "portfolio_evaluation_sessions": evaluation_sessions,
            "portfolio_evaluation_sessions_sha256": _sha256_json(evaluation_sessions),
            "research_fold_signal_sessions": RESEARCH_FOLD_SIGNAL_SESSIONS,
            "research_folds": _partition_record(
                stage,
                sessions,
                maturity,
                label_horizon_bars=label_horizon,
            )["research_folds"],
        }
        if any(partition.get(key) != expected for key, expected in portfolio_contract.items()):
            raise ValueError(f"{stage} portfolio evidence dates are invalid")
        if previous_end is not None and pd.Timestamp(sessions[0]) <= previous_end:
            raise ValueError("research partitions overlap or are out of order")
        if pd.Timestamp(maturity[0]) <= pd.Timestamp(sessions[-1]):
            raise ValueError(f"{stage} label maturity must follow its final signal")
        previous_end = pd.Timestamp(maturity[-1])
        combined.extend((*sessions, *maturity))
    _normalize_sessions(combined)
    if value.get("calendar_sha256") != _sha256_json(combined):
        raise ValueError("research plan calendar digest is invalid")

    exposure = value.get("exposure_provenance")
    if not isinstance(exposure, dict):
        raise ValueError("research plan exposure provenance is missing")
    expected_exposure = build_exposure_provenance(
        partitions,
        RESEARCH_STAGES,
        specification_frozen_at=exposure.get("specification_frozen_at"),
    )
    if exposure != expected_exposure:
        raise ValueError("research plan exposure provenance differs from the fixed registry")

    family_records = value.get("family_ablations")
    expected_family_records = [
        {
            "experiment_id": f"family_ablation__{family}",
            "family": family,
            "features_mode": "alpha158_plus_original",
            "families": [family],
            "stage": "discovery",
            "status": "not_run",
        }
        for family in FACTOR_FAMILIES
    ]
    if family_records != expected_family_records:
        raise ValueError("research plan must contain the five frozen family ablations")
    hypothesis_records = value.get("factor_hypotheses")
    expected_hypothesis_records = [
        {
            "hypothesis_id": factor.name,
            "family": factor.family,
            "expected_direction": factor.direction,
            "stage": "discovery",
            "status": "not_run",
        }
        for factor in ORIGINAL_RESEARCH_CANDIDATES
    ]
    if hypothesis_records != expected_hypothesis_records:
        raise ValueError("research plan must contain the eighteen frozen factor hypotheses")
    statistics = value.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("research plan statistics are missing")
    expected_primary_metrics = [
        "paired daily rank_ic difference versus the unchanged Alpha158 baseline",
        "paired daily 10 bps stress raw-share strategy_net difference versus the unchanged Alpha158 baseline",
        "expected-direction signed raw-factor daily rank_ic",
    ]
    if statistics.get("primary_metrics") != expected_primary_metrics:
        raise ValueError("research plan must retain both paired primary metrics")
    if statistics.get("joint_hypothesis") != {
        "method": "intersection-union max component p-value",
        "component_nulls": [
            "rank_ic_mean_difference_less_than_or_equal_to_zero",
            "strategy_net_mean_difference_less_than_or_equal_to_zero",
            "signed_raw_factor_rank_ic_mean_less_than_or_equal_to_zero",
        ],
        "joint_p_value": (
            "max(rank_ic_one_sided_p, strategy_net_one_sided_p, "
            "signed_raw_factor_rank_ic_one_sided_p)"
        ),
    }:
        raise ValueError("research plan joint hypothesis rule is invalid")
    if statistics.get("terminal_rule") != (
        "candidate terminal account must not lose absolute capital, must improve versus baseline, "
        "and must beat SH510300 at 10 bps stress"
    ):
        raise ValueError("research plan terminal account rule is invalid")
    if statistics.get("robustness_rule") != (
        "10 bps cash-reset 21-signal-session folds, 25 percent maximum drawdown, "
        "and frozen instrument/fold contribution concentration limits"
    ):
        raise ValueError("research plan robustness rule is invalid")
    multiplicity = statistics.get("factor_multiplicity")
    if multiplicity != {
        "method": "Benjamini-Hochberg",
        "family_size": CATALOG_HYPOTHESIS_COUNT,
        "fdr_q": DISCOVERY_FDR_Q,
    }:
        raise ValueError("research plan must retain the frozen eighteen-test BH q=0.10 rule")
    if (
        statistics.get("alternative") != "candidate_minus_baseline_greater_than_zero"
        or statistics.get("hac_max_lag") != DEFAULT_HAC_MAX_LAG
        or statistics.get("confirmation_alpha") != CONFIRMATION_ALPHA
        or statistics.get("family_ablations")
        != "screening evidence only; no standalone confirmation claim"
    ):
        raise ValueError("research plan statistical thresholds are invalid")
    return value


def benjamini_hochberg(
    p_values: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    q: float = DISCOVERY_FDR_Q,
) -> dict[str, Any]:
    """Apply the step-up Benjamini-Hochberg procedure deterministically."""

    q_value = _finite_probability(q, name="q")
    if q_value <= 0.0:
        raise ValueError("q must be greater than zero")
    raw_items = sorted(p_values.items()) if isinstance(p_values, Mapping) else list(p_values)
    if not raw_items:
        raise ValueError("at least one hypothesis is required")
    identifiers: list[str] = []
    values: list[float] = []
    for position, item in enumerate(raw_items):
        if not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError("p_values must contain (hypothesis_id, p_value) pairs")
        identifier, raw_value = item
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"hypothesis {position} has an invalid identifier")
        if identifier in identifiers:
            raise ValueError(f"duplicate hypothesis identifier: {identifier}")
        identifiers.append(identifier)
        values.append(_finite_probability(raw_value, name=f"p_value[{identifier}]"))

    order = sorted(range(len(values)), key=lambda index: (values[index], identifiers[index]))
    m = len(order)
    rejection_rank = 0
    for rank, index in enumerate(order, start=1):
        if values[index] <= q_value * rank / m:
            rejection_rank = rank

    adjusted_by_index: dict[int, float] = {}
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * m / rank)
        adjusted_by_index[index] = min(1.0, running)
    rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
    records = [
        {
            "hypothesis_id": identifier,
            "p_value": values[index],
            "q_value": adjusted_by_index[index],
            "rank": rank_by_index[index],
            "rejected": rank_by_index[index] <= rejection_rank,
        }
        for index, identifier in enumerate(identifiers)
    ]
    return {
        "method": "Benjamini-Hochberg",
        "fdr_q": q_value,
        "hypothesis_count": m,
        "rejected_count": rejection_rank,
        "cutoff_p_value": values[order[rejection_rank - 1]] if rejection_rank else None,
        "results": records,
    }


def catalog_benjamini_hochberg(
    p_values: Mapping[str, float], *, q: float = DISCOVERY_FDR_Q
) -> dict[str, Any]:
    """Apply BH to exactly the eighteen pre-registered catalog hypotheses."""

    if not isinstance(p_values, Mapping):
        raise TypeError("catalog p_values must be a mapping")
    expected = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
    missing = sorted(set(expected) - set(p_values))
    unexpected = sorted(set(p_values) - set(expected))
    if missing or unexpected or len(p_values) != CATALOG_HYPOTHESIS_COUNT:
        raise ValueError(
            "catalog BH requires exactly the eighteen frozen hypotheses"
            f"; missing={missing}; unexpected={unexpected}"
        )
    result = benjamini_hochberg([(name, p_values[name]) for name in expected], q=q)
    if result["hypothesis_count"] != CATALOG_HYPOTHESIS_COUNT:
        raise RuntimeError("catalog multiplicity family size changed unexpectedly")
    return result


def _series_with_exact_partition(
    values: pd.Series,
    partition: Mapping[str, Any],
    *,
    artifact: str,
) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{artifact} must be a pandas Series")
    expected = pd.DatetimeIndex(_normalize_sessions(partition.get("sessions", [])), name="datetime")
    try:
        actual = pd.DatetimeIndex(values.index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{artifact} must have a datetime index") from exc
    if actual.tz is not None:
        raise ValueError(f"{artifact} index must be timezone-naive")
    if actual.has_duplicates or not actual.is_monotonic_increasing:
        raise ValueError(f"{artifact} index must be unique and chronological")
    if not actual.equals(expected):
        raise ValueError(
            f"{artifact} dates must equal the declared {partition.get('name')} partition exactly"
        )
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.equals(numeric.notna()):
        raise ValueError(f"{artifact} contains infinite values")
    numeric.index = expected
    return numeric


def paired_hac_test(
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    partition: Mapping[str, Any],
    max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> dict[str, Any]:
    """Test the one-sided paired alternative candidate minus baseline > 0."""

    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")
    baseline_values = _series_with_exact_partition(baseline, partition, artifact="baseline metric")
    candidate_values = _series_with_exact_partition(candidate, partition, artifact="candidate metric")
    if not baseline_values.isna().equals(candidate_values.isna()):
        raise ValueError("paired baseline and candidate missing-value masks differ")
    paired = pd.concat(
        [baseline_values.rename("baseline"), candidate_values.rename("candidate")], axis=1
    ).dropna()
    if len(paired) < MIN_PAIRED_OBSERVATIONS:
        raise ValueError(f"paired metric needs at least {MIN_PAIRED_OBSERVATIONS} finite observations")
    difference = paired["candidate"] - paired["baseline"]
    statistic = float(hac_t_stat(difference, max_lag=max_lag))
    if not math.isfinite(statistic):
        raise ValueError("paired HAC statistic is not finite")
    one_sided_p = _one_sided_hac_p_value(statistic, len(paired), max_lag)
    return {
        "observations": len(paired),
        "baseline_mean": float(paired["baseline"].mean()),
        "candidate_mean": float(paired["candidate"].mean()),
        "mean_difference": float(difference.mean()),
        "hac_max_lag": max_lag,
        "hac_t_stat": statistic,
        "one_sided_p_value": one_sided_p,
        "alternative": "candidate_minus_baseline_greater_than_zero",
    }


def _catalog_factor_names(factor_names: Iterable[str], *, artifact: str) -> tuple[str, ...]:
    if isinstance(factor_names, (str, bytes)):
        raise TypeError(f"{artifact} factor names must be an iterable of individual names")
    supplied = list(factor_names)
    if not supplied or any(not isinstance(name, str) or not name for name in supplied):
        raise ValueError(f"{artifact} factor names must be a non-empty list")
    if len(supplied) != len(set(supplied)):
        raise ValueError(f"{artifact} factor names must be unique")
    catalog_order = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
    unexpected = sorted(set(supplied) - set(catalog_order))
    canonical = tuple(name for name in catalog_order if name in set(supplied))
    if unexpected or tuple(supplied) != canonical:
        raise ValueError(
            f"{artifact} factor names must be known and follow the frozen catalog order"
        )
    return canonical


def _one_sided_signed_raw_factor_hac(
    raw_rank_ic: pd.Series,
    partition: Mapping[str, Any],
    *,
    factor_name: str,
    expected_direction: int,
    max_lag: int,
) -> dict[str, Any]:
    """Test ``expected_direction * raw RankIC > 0`` with a one-sided HAC test."""

    values = _series_with_exact_partition(
        raw_rank_ic,
        partition,
        artifact=f"raw factor {factor_name} rank_ic",
    )
    finite = values.dropna()
    coverage = len(finite) / len(values)
    if coverage < MIN_RAW_FACTOR_RANK_IC_COVERAGE:
        raise ValueError(
            f"raw factor {factor_name} RankIC coverage {coverage:.2%} is below "
            f"{MIN_RAW_FACTOR_RANK_IC_COVERAGE:.0%}"
        )
    if len(finite) < MIN_PAIRED_OBSERVATIONS:
        raise ValueError(
            f"raw factor {factor_name} RankIC needs at least {MIN_PAIRED_OBSERVATIONS} finite observations"
        )
    signed = finite * expected_direction
    statistic = float(hac_t_stat(signed, max_lag=max_lag))
    if not math.isfinite(statistic):
        raise ValueError(f"raw factor {factor_name} signed RankIC HAC statistic is not finite")
    one_sided_p = _one_sided_hac_p_value(statistic, len(finite), max_lag)

    fold_records: list[dict[str, Any]] = []
    complete_fold_count = 0
    eligible_fold_count = 0
    positive_fold_count = 0
    for fold in partition["research_folds"]:
        fold_index = pd.DatetimeIndex(fold["signal_sessions"], name="datetime")
        fold_values = values.loc[fold_index]
        fold_finite = fold_values.dropna()
        fold_coverage = len(fold_finite) / len(fold_values)
        complete = bool(fold["complete_for_gate"])
        eligible = complete and fold_coverage >= MIN_RAW_FACTOR_RANK_IC_COVERAGE
        signed_mean = (
            float((fold_finite * expected_direction).mean()) if len(fold_finite) else None
        )
        positive = bool(eligible and signed_mean is not None and signed_mean > 0.0)
        if complete:
            complete_fold_count += 1
        if eligible:
            eligible_fold_count += 1
            positive_fold_count += int(positive)
        fold_records.append(
            {
                "fold": fold["fold"],
                "signal_start": fold["signal_start"],
                "signal_end": fold["signal_end"],
                "signal_observations": fold["signal_observations"],
                "finite_observations": len(fold_finite),
                "coverage": fold_coverage,
                "complete_for_gate": complete,
                "included_in_gate": eligible,
                "raw_rank_ic_mean": (
                    float(fold_finite.mean()) if len(fold_finite) else None
                ),
                "signed_rank_ic_mean": signed_mean,
                "positive_signed_rank_ic": positive if eligible else None,
            }
        )
    win_ratio = (
        positive_fold_count / eligible_fold_count if eligible_fold_count else None
    )
    fold_majority_passed = (
        complete_fold_count >= MIN_RESEARCH_COMPLETE_FOLDS
        and eligible_fold_count == complete_fold_count
        and win_ratio is not None
        and win_ratio >= MIN_RESEARCH_FOLD_WIN_RATIO
    )
    return {
        "factor_name": factor_name,
        "expected_direction": expected_direction,
        "observations": len(finite),
        "total_sessions": len(values),
        "coverage": coverage,
        "minimum_coverage": MIN_RAW_FACTOR_RANK_IC_COVERAGE,
        "raw_rank_ic_mean": float(finite.mean()),
        "signed_rank_ic_mean": float(signed.mean()),
        "hac_max_lag": max_lag,
        "hac_t_stat": statistic,
        "one_sided_p_value": one_sided_p,
        "alternative": "expected_direction_times_raw_factor_rank_ic_mean_greater_than_zero",
        "folds": {
            "signal_sessions_per_fold": RESEARCH_FOLD_SIGNAL_SESSIONS,
            "minimum_complete_folds": MIN_RESEARCH_COMPLETE_FOLDS,
            "minimum_positive_ratio": MIN_RESEARCH_FOLD_WIN_RATIO,
            "complete_folds": complete_fold_count,
            "eligible_complete_folds": eligible_fold_count,
            "positive_folds": positive_fold_count,
            "positive_ratio": win_ratio,
            "all_complete_folds_have_required_coverage": (
                eligible_fold_count == complete_fold_count
            ),
            "majority_positive_passed": fold_majority_passed,
            "records": fold_records,
        },
    }


def _plan_partition(plan: Mapping[str, Any], stage: str) -> dict[str, Any]:
    validated = validate_research_plan(plan)
    if stage not in RESEARCH_STAGES:
        raise ValueError(f"unknown research stage: {stage}")
    return validated["partitions"][stage]


def _portfolio_evaluation_partition(
    plan: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    partition = _plan_partition(plan, stage)
    return {
        "name": f"{stage} portfolio evaluation",
        "sessions": deepcopy(partition["portfolio_evaluation_sessions"]),
    }


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validated_execution_rates(value: Mapping[str, Any], *, artifact: str) -> dict[str, Any]:
    rates = {
        name: _finite_real(value.get(name), name=f"{artifact} {name}")
        for name in ("intent_fill_rate", "notional_fill_rate", "zero_fill_intent_rate")
    }
    for name, rate in rates.items():
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{artifact} {name} must be between zero and one")
    return {
        **rates,
        "passed": (
            rates["intent_fill_rate"] >= MIN_INTENT_FILL_RATE
            and rates["notional_fill_rate"] >= MIN_NOTIONAL_FILL_RATE
            and rates["zero_fill_intent_rate"] <= MAX_ZERO_FILL_INTENT_RATE
        ),
    }


def _validated_etf_concentration(
    value: Mapping[str, Any], *, artifact: str
) -> dict[str, Any]:
    share = _finite_real(
        value.get("single_etf_abs_contribution_share"),
        name=f"{artifact} single_etf_abs_contribution_share",
    )
    numerator = _finite_real(
        value.get("single_etf_abs_contribution_numerator_cny"),
        name=f"{artifact} single_etf_abs_contribution_numerator_cny",
    )
    denominator = _finite_real(
        value.get("single_etf_abs_contribution_denominator_cny"),
        name=f"{artifact} single_etf_abs_contribution_denominator_cny",
    )
    symbol = value.get("single_etf_abs_contribution_symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"{artifact} single ETF contribution symbol is invalid")
    if (
        not 0.0 <= share <= 1.0
        or numerator < 0.0
        or denominator <= 0.0
        or numerator > denominator + 1e-9
        or not math.isclose(share, numerator / denominator, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError(
            f"{artifact} single ETF gross-absolute contribution concentration does not reconcile"
        )
    return {
        "method": "gross_abs_daily_symbol_net_pnl",
        "symbol": symbol,
        "numerator_cny": numerator,
        "denominator_cny": denominator,
        "share": share,
    }


def _validated_stage_evidence(
    evidence: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage: str,
    *,
    artifact: str,
    expected_factor_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError(f"{artifact} evidence must be a mapping")
    if set(evidence) != {
        "rank_ic",
        "raw_factor_rank_ic",
        "strategy_net",
        "benchmark",
        "portfolio",
    }:
        raise ValueError(f"{artifact} evidence fields are invalid")
    signal_partition = _plan_partition(plan, stage)
    portfolio_partition = _portfolio_evaluation_partition(plan, stage)
    rank_ic = _series_with_exact_partition(
        evidence["rank_ic"], signal_partition, artifact=f"{artifact} rank_ic"
    )
    rank_ic_coverage = float(rank_ic.notna().mean())
    if rank_ic_coverage < MIN_RAW_FACTOR_RANK_IC_COVERAGE:
        raise ValueError(
            f"{artifact} rank_ic coverage {rank_ic_coverage:.2%} is below "
            f"{MIN_RAW_FACTOR_RANK_IC_COVERAGE:.0%}"
        )
    strategy_net = _series_with_exact_partition(
        evidence["strategy_net"],
        portfolio_partition,
        artifact=f"{artifact} strategy_net",
    )
    if strategy_net.isna().any():
        raise ValueError(f"{artifact} strategy_net must be complete")
    if (strategy_net <= -1.0).any():
        raise ValueError(f"{artifact} strategy_net would make wealth non-positive")
    benchmark = _series_with_exact_partition(
        evidence["benchmark"],
        portfolio_partition,
        artifact=f"{artifact} benchmark",
    )
    if benchmark.isna().any() or (benchmark <= -1.0).any():
        raise ValueError(f"{artifact} benchmark must be complete and preserve positive wealth")
    raw_factor_rank_ic = evidence["raw_factor_rank_ic"]
    if not isinstance(raw_factor_rank_ic, Mapping):
        raise TypeError(f"{artifact} raw_factor_rank_ic must be a mapping")
    expected_names = tuple(expected_factor_names)
    if tuple(raw_factor_rank_ic) != expected_names:
        raise ValueError(
            f"{artifact} raw_factor_rank_ic must exactly match its experiment factors; "
            f"expected={list(expected_names)} actual={list(raw_factor_rank_ic)}"
        )
    raw_metrics = {
        name: _series_with_exact_partition(
            values,
            signal_partition,
            artifact=f"{artifact} raw factor {name} rank_ic",
        )
        for name, values in raw_factor_rank_ic.items()
    }
    for name, values in raw_metrics.items():
        coverage = float(values.notna().mean())
        if coverage < MIN_RAW_FACTOR_RANK_IC_COVERAGE:
            raise ValueError(
                f"{artifact} raw factor {name} RankIC coverage {coverage:.2%} is below "
                f"{MIN_RAW_FACTOR_RANK_IC_COVERAGE:.0%}"
            )
    portfolio = deepcopy(evidence["portfolio"])
    if not isinstance(portfolio, dict):
        raise TypeError(f"{artifact} portfolio evidence must be a mapping")
    required = {
        "benchmark": RESEARCH_BENCHMARK,
        "account_currency": "CNY",
        "initial_account": RESEARCH_ACCOUNT_CNY,
        "stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
        "engine": RAW_SHARE_ENGINE,
        "alignment_method": EVALUATION_ALIGNMENT_METHOD,
        "evaluation_sessions_sha256": signal_partition[
            "portfolio_evaluation_sessions_sha256"
        ],
    }
    if any(portfolio.get(key) != expected for key, expected in required.items()):
        raise ValueError(f"{artifact} portfolio evidence contract is invalid")
    terminal = portfolio.get("terminal_account")
    terminal_value = _finite_real(terminal, name=f"{artifact} terminal account")
    if terminal_value <= 0.0:
        raise ValueError(f"{artifact} terminal account must be positive and finite")
    implied_terminal = RESEARCH_ACCOUNT_CNY * float((1.0 + strategy_net).prod())
    if not math.isclose(terminal_value, implied_terminal, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f"{artifact} terminal account does not reconcile to strategy_net")
    portfolio["terminal_account"] = terminal_value
    execution = _validated_execution_rates(portfolio, artifact=artifact)
    portfolio.update({key: execution[key] for key in execution if key != "passed"})
    portfolio["execution_quality_passed"] = execution["passed"]
    implied_benchmark_terminal = RESEARCH_ACCOUNT_CNY * float((1.0 + benchmark).prod())
    benchmark_terminal = portfolio.get("benchmark_terminal_account")
    benchmark_terminal_value = _finite_real(
        benchmark_terminal, name=f"{artifact} benchmark terminal account"
    )
    if benchmark_terminal_value <= 0.0 or not math.isclose(
        benchmark_terminal_value, implied_benchmark_terminal, rel_tol=1e-10, abs_tol=1e-6
    ):
        raise ValueError(f"{artifact} benchmark terminal account does not reconcile")
    drawdown = float(max_drawdown(strategy_net))
    declared_drawdown = portfolio.get("strategy_max_drawdown")
    declared_drawdown_value = _finite_real(
        declared_drawdown, name=f"{artifact} strategy maximum drawdown"
    )
    if not math.isclose(
        declared_drawdown_value, drawdown, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(f"{artifact} strategy maximum drawdown does not reconcile")
    concentration = _validated_etf_concentration(portfolio, artifact=artifact)
    folds = portfolio.get("research_folds")
    expected_folds = signal_partition["research_folds"]
    if not isinstance(folds, list) or len(folds) != len(expected_folds):
        raise ValueError(f"{artifact} research fold evidence is incomplete")
    normalized_folds = []
    for expected, record in zip(expected_folds, folds):
        if not isinstance(record, Mapping):
            raise TypeError(f"{artifact} research fold evidence must contain mappings")
        for key in (
            "fold",
            "signal_start",
            "signal_end",
            "evaluation_start",
            "evaluation_end",
            "complete_for_gate",
        ):
            if record.get(key) != expected[key]:
                raise ValueError(f"{artifact} research fold {key} differs from the plan")
        normalized = dict(record)
        required_fold = {
            "initial_account": RESEARCH_ACCOUNT_CNY,
            "stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
            "engine": RAW_SHARE_ENGINE,
        }
        if any(normalized.get(key) != expected_value for key, expected_value in required_fold.items()):
            raise ValueError(f"{artifact} research fold does not use the frozen cash engine")
        for key in (
            "terminal_account",
            "benchmark_terminal_account",
        ):
            normalized[key] = _finite_real(
                normalized.get(key), name=f"{artifact} research fold {key}"
            )
        if normalized["terminal_account"] <= 0.0 or normalized["benchmark_terminal_account"] <= 0.0:
            raise ValueError(f"{artifact} research fold terminal wealth must be positive")
        fold_concentration = _validated_etf_concentration(
            normalized, artifact=f"{artifact} research fold {normalized['fold']}"
        )
        normalized["single_etf_abs_contribution"] = fold_concentration
        normalized_folds.append(normalized)
    portfolio["benchmark_terminal_account"] = benchmark_terminal_value
    portfolio["strategy_max_drawdown"] = drawdown
    portfolio["single_etf_abs_contribution"] = concentration
    portfolio["research_folds"] = normalized_folds
    return {
        "rank_ic": rank_ic,
        "raw_factor_rank_ic": raw_metrics,
        "strategy_net": strategy_net,
        "benchmark": benchmark,
        "portfolio": portfolio,
    }


def _paired_stage_evidence(
    baseline_evidence: Mapping[str, Any],
    candidate_evidence: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage: str,
    *,
    candidate_factor_names: Sequence[str],
    max_lag: int,
) -> dict[str, Any]:
    factor_names = tuple(candidate_factor_names)
    baseline = _validated_stage_evidence(
        baseline_evidence,
        plan,
        stage,
        artifact="baseline",
        expected_factor_names=(),
    )
    candidate = _validated_stage_evidence(
        candidate_evidence,
        plan,
        stage,
        artifact="candidate",
        expected_factor_names=factor_names,
    )
    rank_ic = paired_hac_test(
        baseline["rank_ic"],
        candidate["rank_ic"],
        partition=_plan_partition(plan, stage),
        max_lag=max_lag,
    )
    strategy_net = paired_hac_test(
        baseline["strategy_net"],
        candidate["strategy_net"],
        partition=_portfolio_evaluation_partition(plan, stage),
        max_lag=max_lag,
    )
    if not baseline["benchmark"].equals(candidate["benchmark"]):
        raise ValueError("baseline and candidate benchmark return paths differ")
    direction_by_name = {
        factor.name: factor.direction for factor in ORIGINAL_RESEARCH_CANDIDATES
    }
    raw_factor_tests = {
        name: _one_sided_signed_raw_factor_hac(
            candidate["raw_factor_rank_ic"][name],
            _plan_partition(plan, stage),
            factor_name=name,
            expected_direction=direction_by_name[name],
            max_lag=max_lag,
        )
        for name in factor_names
    }
    component_p_values = {
        "model_rank_ic": rank_ic["one_sided_p_value"],
        "strategy_net": strategy_net["one_sided_p_value"],
        **{
            f"signed_raw_factor_rank_ic::{name}": test["one_sided_p_value"]
            for name, test in raw_factor_tests.items()
        },
    }
    joint_p = max(component_p_values.values())
    baseline_terminal = baseline["portfolio"]["terminal_account"]
    candidate_terminal = candidate["portfolio"]["terminal_account"]
    terminal_improvement = candidate_terminal - baseline_terminal
    relative_improvement = candidate_terminal / baseline_terminal - 1.0
    return {
        "rank_ic": rank_ic,
        "strategy_net": strategy_net,
        "signed_raw_factor_rank_ic": raw_factor_tests,
        "joint_iut": {
            "method": "intersection-union max component p-value",
            "one_sided_p_value": joint_p,
            "component_count": len(component_p_values),
            "component_one_sided_p_values": component_p_values,
            "alternative": (
                "model_rank_ic_and_strategy_net_improvements_and_all_expected_direction_"
                "raw_factor_rank_ic_means_greater_than_zero"
            ),
        },
        "terminal": {
            "account_currency": "CNY",
            "initial_account": RESEARCH_ACCOUNT_CNY,
            "stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
            "baseline_terminal_account": baseline_terminal,
            "candidate_terminal_account": candidate_terminal,
            "account_improvement": terminal_improvement,
            "relative_wealth_improvement": relative_improvement,
            "account_improvement_positive": terminal_improvement > 0.0,
            "relative_wealth_improvement_positive": relative_improvement > 0.0,
            "candidate_terminal_account_not_below_initial": (
                candidate_terminal >= RESEARCH_ACCOUNT_CNY
            ),
            "baseline_execution_quality_passed": baseline["portfolio"][
                "execution_quality_passed"
            ],
            "candidate_execution_quality_passed": candidate["portfolio"][
                "execution_quality_passed"
            ],
            "comparison": "candidate_minus_baseline",
        },
        "benchmark": {
            "symbol": RESEARCH_BENCHMARK,
            "baseline_terminal_account": baseline["portfolio"][
                "benchmark_terminal_account"
            ],
            "candidate_terminal_account": candidate["portfolio"][
                "benchmark_terminal_account"
            ],
            "baseline_beats_benchmark": (
                baseline_terminal > baseline["portfolio"]["benchmark_terminal_account"]
            ),
            "candidate_beats_benchmark": (
                candidate_terminal > candidate["portfolio"]["benchmark_terminal_account"]
            ),
        },
        "drawdown": {
            "maximum_allowed": MAX_STRATEGY_DRAWDOWN,
            "baseline": baseline["portfolio"]["strategy_max_drawdown"],
            "candidate": candidate["portfolio"]["strategy_max_drawdown"],
            "candidate_within_limit": (
                abs(candidate["portfolio"]["strategy_max_drawdown"])
                <= MAX_STRATEGY_DRAWDOWN
            ),
        },
        "execution_quality": {
            "thresholds": {
                "minimum_intent_fill_rate": MIN_INTENT_FILL_RATE,
                "minimum_notional_fill_rate": MIN_NOTIONAL_FILL_RATE,
                "maximum_zero_fill_intent_rate": MAX_ZERO_FILL_INTENT_RATE,
            },
            "baseline": {
                key: baseline["portfolio"][key]
                for key in (
                    "intent_fill_rate",
                    "notional_fill_rate",
                    "zero_fill_intent_rate",
                    "execution_quality_passed",
                )
            },
            "candidate": {
                key: candidate["portfolio"][key]
                for key in (
                    "intent_fill_rate",
                    "notional_fill_rate",
                    "zero_fill_intent_rate",
                    "execution_quality_passed",
                )
            },
        },
        "concentration": {
            "maximum_single_etf_abs_contribution_share": (
                MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE
            ),
            "baseline_single_etf": baseline["portfolio"][
                "single_etf_abs_contribution"
            ],
            "candidate_single_etf": candidate["portfolio"][
                "single_etf_abs_contribution"
            ],
            "candidate_single_etf_within_limit": (
                candidate["portfolio"]["single_etf_abs_contribution"]["share"]
                <= MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE
            ),
        },
        "portfolio_inputs": {
            "baseline_research_folds": baseline["portfolio"]["research_folds"],
            "candidate_research_folds": candidate["portfolio"]["research_folds"],
        },
    }


def _paired_research_fold_evidence(tests: Mapping[str, Any]) -> dict[str, Any]:
    baseline_records = tests["portfolio_inputs"]["baseline_research_folds"]
    candidate_records = tests["portfolio_inputs"]["candidate_research_folds"]
    records: list[dict[str, Any]] = []
    complete_count = 0
    wins = 0
    complete_increments: list[tuple[int, float]] = []
    for baseline, candidate in zip(baseline_records, candidate_records):
        identity_fields = (
            "fold",
            "signal_start",
            "signal_end",
            "evaluation_start",
            "evaluation_end",
            "complete_for_gate",
        )
        if any(baseline[key] != candidate[key] for key in identity_fields):
            raise ValueError("baseline and candidate research fold identities differ")
        if not math.isclose(
            baseline["benchmark_terminal_account"],
            candidate["benchmark_terminal_account"],
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ValueError("baseline and candidate research fold benchmark wealth differs")
        complete = bool(candidate["complete_for_gate"])
        incremental_pnl = candidate["terminal_account"] - baseline["terminal_account"]
        candidate_relative_to_benchmark = (
            candidate["terminal_account"] / candidate["benchmark_terminal_account"] - 1.0
        )
        if complete:
            complete_count += 1
            wins += int(incremental_pnl > 0.0)
            complete_increments.append((int(candidate["fold"]), incremental_pnl))
        records.append(
            {
                "fold": candidate["fold"],
                "signal_start": candidate["signal_start"],
                "signal_end": candidate["signal_end"],
                "evaluation_start": candidate["evaluation_start"],
                "evaluation_end": candidate["evaluation_end"],
                "complete_for_gate": complete,
                "included_in_gate": complete,
                "initial_account": RESEARCH_ACCOUNT_CNY,
                "stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
                "engine": RAW_SHARE_ENGINE,
                "baseline_terminal_account": baseline["terminal_account"],
                "candidate_terminal_account": candidate["terminal_account"],
                "candidate_benchmark_terminal_account": candidate[
                    "benchmark_terminal_account"
                ],
                "candidate_relative_to_benchmark": candidate_relative_to_benchmark,
                "incremental_pnl_cny": incremental_pnl,
                "candidate_minus_baseline_positive": (
                    incremental_pnl > 0.0 if complete else None
                ),
                "baseline_single_etf_abs_contribution": baseline[
                    "single_etf_abs_contribution"
                ],
                "candidate_single_etf_abs_contribution": candidate[
                    "single_etf_abs_contribution"
                ],
            }
        )
    if complete_count < MIN_RESEARCH_COMPLETE_FOLDS:
        raise ValueError(
            f"paired research evidence needs at least {MIN_RESEARCH_COMPLETE_FOLDS} complete folds"
        )
    win_ratio = wins / complete_count
    denominator = float(sum(abs(value) for _, value in complete_increments))
    if denominator <= 0.0:
        concentration = {
            "fold": None,
            "numerator_cny": 0.0,
            "denominator_cny": 0.0,
            "share": None,
            "passed": False,
        }
    else:
        dominant_fold, dominant_increment = max(
            complete_increments, key=lambda item: (abs(item[1]), -item[0])
        )
        numerator = abs(dominant_increment)
        share = numerator / denominator
        concentration = {
            "fold": dominant_fold,
            "numerator_cny": numerator,
            "denominator_cny": denominator,
            "share": share,
            "passed": share <= MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE,
        }
    return {
        "signal_sessions_per_fold": RESEARCH_FOLD_SIGNAL_SESSIONS,
        "minimum_complete_folds": MIN_RESEARCH_COMPLETE_FOLDS,
        "minimum_win_ratio": MIN_RESEARCH_FOLD_WIN_RATIO,
        "complete_folds": complete_count,
        "wins": wins,
        "losses_or_ties": complete_count - wins,
        "win_ratio": win_ratio,
        "majority_positive_passed": win_ratio >= MIN_RESEARCH_FOLD_WIN_RATIO,
        "maximum_single_fold_abs_incremental_pnl_share": (
            MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE
        ),
        "single_fold_abs_incremental_pnl_concentration": concentration,
        "records": records,
    }


def _complete_stage_tests(tests: dict[str, Any]) -> dict[str, Any]:
    result = dict(tests)
    result["research_folds"] = _paired_research_fold_evidence(result)
    result["concentration"] = {
        **result["concentration"],
        "single_fold_abs_incremental_pnl": result["research_folds"][
            "single_fold_abs_incremental_pnl_concentration"
        ],
    }
    result.pop("portfolio_inputs")
    return result


def _result_rate(value: Any, *, name: str) -> float:
    rate = _finite_real(value, name=name)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return rate


def _recomputed_paired_hac_result(
    value: Mapping[str, Any],
    *,
    artifact: str,
    minimum_observations: int,
    exact_observations: int | None = None,
) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{artifact} result must be a mapping")
    observations = value.get("observations")
    if (
        isinstance(observations, bool)
        or not isinstance(observations, int)
        or observations < minimum_observations
        or (exact_observations is not None and observations != exact_observations)
    ):
        raise ValueError(f"{artifact} observation count is invalid")
    baseline_mean = _finite_real(
        value.get("baseline_mean"), name=f"{artifact} baseline mean"
    )
    candidate_mean = _finite_real(
        value.get("candidate_mean"), name=f"{artifact} candidate mean"
    )
    mean_difference = _finite_real(
        value.get("mean_difference"), name=f"{artifact} mean difference"
    )
    statistic = _finite_real(value.get("hac_t_stat"), name=f"{artifact} HAC t-stat")
    p_value = _result_rate(
        value.get("one_sided_p_value"), name=f"{artifact} one-sided p-value"
    )
    implied_p_value = _one_sided_hac_p_value(statistic, observations, DEFAULT_HAC_MAX_LAG)
    if (
        value.get("hac_max_lag") != DEFAULT_HAC_MAX_LAG
        or value.get("alternative") != "candidate_minus_baseline_greater_than_zero"
        or not math.isclose(
            mean_difference,
            candidate_mean - baseline_mean,
            rel_tol=1e-10,
            abs_tol=1e-15,
        )
        or not math.isclose(p_value, implied_p_value, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ValueError(f"{artifact} HAC result does not reconcile")
    return mean_difference, p_value


def _recomputed_concentration_record(
    value: Mapping[str, Any], *, artifact: str
) -> float:
    if not isinstance(value, Mapping):
        raise TypeError(f"{artifact} concentration must be a mapping")
    symbol = value.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"{artifact} concentration symbol is invalid")
    share = _result_rate(value.get("share"), name=f"{artifact} concentration share")
    numerator = _finite_real(
        value.get("numerator_cny"), name=f"{artifact} concentration numerator"
    )
    denominator = _finite_real(
        value.get("denominator_cny"), name=f"{artifact} concentration denominator"
    )
    if (
        value.get("method") != "gross_abs_daily_symbol_net_pnl"
        or numerator < 0.0
        or denominator <= 0.0
        or numerator > denominator + 1e-9
        or not math.isclose(share, numerator / denominator, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError(f"{artifact} concentration does not reconcile")
    return share


def _recomputed_execution_passed(record: Mapping[str, Any], *, artifact: str) -> bool:
    if not isinstance(record, Mapping):
        raise TypeError(f"{artifact} execution quality must be a mapping")
    intent = _result_rate(record.get("intent_fill_rate"), name=f"{artifact} intent fill rate")
    notional = _result_rate(
        record.get("notional_fill_rate"), name=f"{artifact} notional fill rate"
    )
    zero = _result_rate(
        record.get("zero_fill_intent_rate"), name=f"{artifact} zero-fill rate"
    )
    passed = (
        intent >= MIN_INTENT_FILL_RATE
        and notional >= MIN_NOTIONAL_FILL_RATE
        and zero <= MAX_ZERO_FILL_INTENT_RATE
    )
    if record.get("execution_quality_passed") is not passed:
        raise ValueError(f"{artifact} execution-quality decision does not reconcile")
    return passed


def _recomputed_research_fold_gates(
    value: Mapping[str, Any], expected_folds: Sequence[Mapping[str, Any]]
) -> tuple[bool, bool]:
    if not isinstance(value, Mapping) or not isinstance(value.get("records"), list):
        raise TypeError("research fold result must contain records")
    if len(value["records"]) != len(expected_folds):
        raise ValueError("research fold result does not match the frozen fold plan")
    complete: list[tuple[int, float]] = []
    seen_folds: set[int] = set()
    wins = 0
    for record, expected_fold in zip(value["records"], expected_folds):
        if not isinstance(record, Mapping):
            raise TypeError("research fold result records must be mappings")
        fold = record.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 1 or fold in seen_folds:
            raise ValueError("research fold result has an invalid fold number")
        seen_folds.add(fold)
        for key in (
            "fold",
            "signal_start",
            "signal_end",
            "evaluation_start",
            "evaluation_end",
            "complete_for_gate",
        ):
            if record.get(key) != expected_fold.get(key):
                raise ValueError("research fold result differs from the frozen fold plan")
        is_complete = record.get("complete_for_gate")
        included = record.get("included_in_gate")
        if not isinstance(is_complete, bool) or included is not is_complete:
            raise ValueError("only complete research folds may be included in gates")
        for key, expected in (
            ("initial_account", RESEARCH_ACCOUNT_CNY),
            ("stress_slippage_bps_per_side", REQUIRED_STRESS_SLIPPAGE_BPS),
            ("engine", RAW_SHARE_ENGINE),
        ):
            if record.get(key) != expected:
                raise ValueError("research fold result conflicts with the frozen cash engine")
        baseline_terminal = _finite_real(
            record.get("baseline_terminal_account"),
            name=f"research fold {fold} baseline terminal account",
        )
        candidate_terminal = _finite_real(
            record.get("candidate_terminal_account"),
            name=f"research fold {fold} candidate terminal account",
        )
        benchmark_terminal = _finite_real(
            record.get("candidate_benchmark_terminal_account"),
            name=f"research fold {fold} benchmark terminal account",
        )
        if min(baseline_terminal, candidate_terminal, benchmark_terminal) <= 0.0:
            raise ValueError("research fold result contains non-positive terminal wealth")
        _recomputed_concentration_record(
            record.get("baseline_single_etf_abs_contribution"),
            artifact=f"research fold {fold} baseline ETF",
        )
        _recomputed_concentration_record(
            record.get("candidate_single_etf_abs_contribution"),
            artifact=f"research fold {fold} candidate ETF",
        )
        increment = candidate_terminal - baseline_terminal
        declared_increment = _finite_real(
            record.get("incremental_pnl_cny"),
            name=f"research fold {fold} incremental P&L",
        )
        declared_relative = _finite_real(
            record.get("candidate_relative_to_benchmark"),
            name=f"research fold {fold} relative benchmark wealth",
        )
        if (
            not math.isclose(declared_increment, increment, rel_tol=1e-10, abs_tol=1e-6)
            or not math.isclose(
                declared_relative,
                candidate_terminal / benchmark_terminal - 1.0,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
            or record.get("candidate_minus_baseline_positive")
            is not (increment > 0.0 if is_complete else None)
        ):
            raise ValueError("research fold result does not reconcile")
        if is_complete:
            complete.append((fold, increment))
            wins += int(increment > 0.0)
    if len(complete) < MIN_RESEARCH_COMPLETE_FOLDS:
        raise ValueError("research fold result has too few complete folds")
    win_ratio = wins / len(complete)
    if (
        value.get("signal_sessions_per_fold") != RESEARCH_FOLD_SIGNAL_SESSIONS
        or value.get("minimum_complete_folds") != MIN_RESEARCH_COMPLETE_FOLDS
        or value.get("minimum_win_ratio") != MIN_RESEARCH_FOLD_WIN_RATIO
        or value.get("maximum_single_fold_abs_incremental_pnl_share")
        != MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE
        or
        value.get("complete_folds") != len(complete)
        or value.get("wins") != wins
        or value.get("losses_or_ties") != len(complete) - wins
        or not math.isclose(
            _result_rate(value.get("win_ratio"), name="research fold win ratio"),
            win_ratio,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or value.get("majority_positive_passed")
        is not (win_ratio >= MIN_RESEARCH_FOLD_WIN_RATIO)
    ):
        raise ValueError("research fold majority decision does not reconcile")
    denominator = float(sum(abs(increment) for _, increment in complete))
    concentration = value.get("single_fold_abs_incremental_pnl_concentration")
    if not isinstance(concentration, Mapping):
        raise TypeError("research fold concentration must be a mapping")
    if denominator <= 0.0:
        expected_fold = None
        numerator = 0.0
        share = None
        concentration_passed = False
    else:
        expected_fold, increment = max(complete, key=lambda item: (abs(item[1]), -item[0]))
        numerator = abs(increment)
        share = numerator / denominator
        concentration_passed = share <= MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE
    declared_numerator = _finite_real(
        concentration.get("numerator_cny"), name="research fold concentration numerator"
    )
    declared_denominator = _finite_real(
        concentration.get("denominator_cny"), name="research fold concentration denominator"
    )
    if (
        concentration.get("fold") != expected_fold
        or not math.isclose(declared_numerator, numerator, rel_tol=1e-10, abs_tol=1e-6)
        or not math.isclose(declared_denominator, denominator, rel_tol=1e-10, abs_tol=1e-6)
        or (
            share is None
            and concentration.get("share") is not None
        )
        or (
            share is not None
            and not math.isclose(
                _result_rate(concentration.get("share"), name="research fold concentration share"),
                share,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        )
        or concentration.get("passed") is not concentration_passed
    ):
        raise ValueError("research fold concentration does not reconcile")
    return win_ratio >= MIN_RESEARCH_FOLD_WIN_RATIO, concentration_passed


def _recomputed_signed_raw_factor_gates(
    value: Mapping[str, Any],
    factor_names: Sequence[str],
    *,
    alpha: float | None,
    expected_folds: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, bool]:
    if (
        not isinstance(value, Mapping)
        or len(value) != len(factor_names)
        or set(value) != set(factor_names)
    ):
        raise ValueError("signed raw factor result does not match the frozen factor set")
    means_positive = True
    p_values_passed = True
    fold_majorities = True
    for name in factor_names:
        record = value[name]
        if not isinstance(record, Mapping) or record.get("factor_name") != name:
            raise ValueError("signed raw factor result has an invalid factor identity")
        direction = next(
            factor.direction for factor in ORIGINAL_RESEARCH_CANDIDATES if factor.name == name
        )
        raw_mean = _finite_real(
            record.get("raw_rank_ic_mean"), name=f"{name} raw RankIC mean"
        )
        signed_mean = _finite_real(
            record.get("signed_rank_ic_mean"), name=f"{name} signed RankIC mean"
        )
        p_value = _result_rate(
            record.get("one_sided_p_value"), name=f"{name} one-sided p-value"
        )
        statistic = _finite_real(
            record.get("hac_t_stat"), name=f"{name} signed RankIC HAC t-stat"
        )
        coverage = _result_rate(record.get("coverage"), name=f"{name} coverage")
        observations = record.get("observations")
        total_sessions = record.get("total_sessions")
        if (
            record.get("expected_direction") != direction
            or isinstance(observations, bool)
            or not isinstance(observations, int)
            or isinstance(total_sessions, bool)
            or not isinstance(total_sessions, int)
            or total_sessions <= 0
            or observations < MIN_PAIRED_OBSERVATIONS
            or observations > total_sessions
            or total_sessions != sum(
                int(fold.get("signal_observations", 0)) for fold in expected_folds
            )
            or not math.isclose(
                coverage, observations / total_sessions, rel_tol=0.0, abs_tol=1e-15
            )
            or coverage < MIN_RAW_FACTOR_RANK_IC_COVERAGE
            or not math.isclose(signed_mean, direction * raw_mean, rel_tol=1e-10, abs_tol=1e-15)
            or record.get("minimum_coverage") != MIN_RAW_FACTOR_RANK_IC_COVERAGE
            or record.get("hac_max_lag") != DEFAULT_HAC_MAX_LAG
            or record.get("alternative")
            != "expected_direction_times_raw_factor_rank_ic_mean_greater_than_zero"
            or not math.isclose(
                p_value,
                _one_sided_hac_p_value(statistic, observations, DEFAULT_HAC_MAX_LAG),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("signed raw factor result does not reconcile to its frozen direction")
        folds = record.get("folds")
        if not isinstance(folds, Mapping) or not isinstance(folds.get("records"), list):
            raise TypeError("signed raw factor fold result is invalid")
        if len(folds["records"]) != len(expected_folds):
            raise ValueError("signed raw factor folds differ from the frozen fold plan")
        eligible = 0
        positive = 0
        complete = 0
        finite_total = 0
        weighted_raw_sum = 0.0
        weighted_signed_sum = 0.0
        for fold, expected_fold in zip(folds["records"], expected_folds):
            if not isinstance(fold, Mapping):
                raise TypeError("signed raw factor fold record must be a mapping")
            for key in (
                "fold",
                "signal_start",
                "signal_end",
                "signal_observations",
                "complete_for_gate",
            ):
                if fold.get(key) != expected_fold.get(key):
                    raise ValueError("signed raw factor fold differs from the frozen fold plan")
            is_complete = fold.get("complete_for_gate")
            included = fold.get("included_in_gate")
            fold_coverage = _result_rate(
                fold.get("coverage"), name=f"{name} fold coverage"
            )
            finite_observations = fold.get("finite_observations")
            signal_observations = fold.get("signal_observations")
            if (
                isinstance(finite_observations, bool)
                or not isinstance(finite_observations, int)
                or finite_observations < 0
                or finite_observations > signal_observations
                or not math.isclose(
                    fold_coverage,
                    finite_observations / signal_observations,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError("signed raw factor fold coverage does not reconcile")
            expected_included = bool(
                is_complete is True and fold_coverage >= MIN_RAW_FACTOR_RANK_IC_COVERAGE
            )
            if not isinstance(is_complete, bool) or included is not expected_included:
                raise ValueError("signed raw factor fold inclusion does not reconcile")
            complete += int(is_complete)
            raw_fold_mean = fold.get("raw_rank_ic_mean")
            signed_fold_mean = fold.get("signed_rank_ic_mean")
            if finite_observations:
                raw_fold_mean = _finite_real(
                    raw_fold_mean, name=f"{name} fold raw RankIC"
                )
                signed_fold_mean = _finite_real(
                    signed_fold_mean, name=f"{name} fold signed RankIC"
                )
                if not math.isclose(
                    signed_fold_mean,
                    direction * raw_fold_mean,
                    rel_tol=1e-10,
                    abs_tol=1e-15,
                ):
                    raise ValueError("signed raw factor fold mean does not reconcile")
                finite_total += finite_observations
                weighted_raw_sum += finite_observations * raw_fold_mean
                weighted_signed_sum += finite_observations * signed_fold_mean
            elif raw_fold_mean is not None or signed_fold_mean is not None:
                raise ValueError("empty signed raw factor fold cannot record a mean")
            if included:
                eligible += 1
                fold_signed = float(signed_fold_mean)
                fold_positive = fold_signed > 0.0
                if fold.get("positive_signed_rank_ic") is not fold_positive:
                    raise ValueError("signed raw factor fold direction does not reconcile")
                positive += int(fold_positive)
            elif fold.get("positive_signed_rank_ic") is not None:
                raise ValueError("ineligible signed raw factor fold cannot enter the direction gate")
        ratio = positive / eligible if eligible else None
        majority = (
            complete >= MIN_RESEARCH_COMPLETE_FOLDS
            and eligible == complete
            and ratio is not None
            and ratio >= MIN_RESEARCH_FOLD_WIN_RATIO
        )
        if (
            folds.get("signal_sessions_per_fold") != RESEARCH_FOLD_SIGNAL_SESSIONS
            or folds.get("minimum_complete_folds") != MIN_RESEARCH_COMPLETE_FOLDS
            or folds.get("minimum_positive_ratio") != MIN_RESEARCH_FOLD_WIN_RATIO
            or
            folds.get("complete_folds") != complete
            or folds.get("eligible_complete_folds") != eligible
            or folds.get("positive_folds") != positive
            or (
                ratio is None
                and folds.get("positive_ratio") is not None
            )
            or (
                ratio is not None
                and not math.isclose(
                    _result_rate(
                        folds.get("positive_ratio"),
                        name=f"{name} positive fold ratio",
                    ),
                    ratio,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
            or folds.get("all_complete_folds_have_required_coverage")
            is not (eligible == complete)
            or folds.get("majority_positive_passed") is not majority
            or finite_total != observations
            or not math.isclose(
                raw_mean,
                weighted_raw_sum / finite_total,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
            or not math.isclose(
                signed_mean,
                weighted_signed_sum / finite_total,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("signed raw factor fold majority does not reconcile")
        means_positive = means_positive and signed_mean > 0.0
        p_values_passed = p_values_passed and (alpha is None or p_value <= alpha)
        fold_majorities = fold_majorities and majority
    return means_positive, p_values_passed, fold_majorities


def _recomputed_common_criteria(
    tests: Mapping[str, Any],
    factor_names: Sequence[str],
    *,
    alpha: float | None,
    expected_folds: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    terminal = tests.get("terminal")
    benchmark = tests.get("benchmark")
    drawdown = tests.get("drawdown")
    execution = tests.get("execution_quality")
    concentration = tests.get("concentration")
    if not all(
        isinstance(value, Mapping)
        for value in (terminal, benchmark, drawdown, execution, concentration)
    ):
        raise TypeError("stage result is missing auditable gate values")
    baseline_terminal = _finite_real(
        terminal.get("baseline_terminal_account"), name="baseline terminal account"
    )
    candidate_terminal = _finite_real(
        terminal.get("candidate_terminal_account"), name="candidate terminal account"
    )
    benchmark_terminal = _finite_real(
        benchmark.get("candidate_terminal_account"), name="benchmark terminal account"
    )
    baseline_benchmark_terminal = _finite_real(
        benchmark.get("baseline_terminal_account"),
        name="baseline benchmark terminal account",
    )
    improvement = candidate_terminal - baseline_terminal
    relative = candidate_terminal / baseline_terminal - 1.0
    if (
        min(baseline_terminal, candidate_terminal, benchmark_terminal) <= 0.0
        or terminal.get("account_currency") != "CNY"
        or terminal.get("initial_account") != RESEARCH_ACCOUNT_CNY
        or terminal.get("stress_slippage_bps_per_side")
        != REQUIRED_STRESS_SLIPPAGE_BPS
        or terminal.get("comparison") != "candidate_minus_baseline"
        or not math.isclose(
            _finite_real(terminal.get("account_improvement"), name="terminal improvement"),
            improvement,
            rel_tol=1e-10,
            abs_tol=1e-6,
        )
        or not math.isclose(
            _finite_real(
                terminal.get("relative_wealth_improvement"), name="relative wealth improvement"
            ),
            relative,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
        or terminal.get("account_improvement_positive") is not (improvement > 0.0)
        or terminal.get("relative_wealth_improvement_positive") is not (relative > 0.0)
        or terminal.get("candidate_terminal_account_not_below_initial")
        is not (candidate_terminal >= RESEARCH_ACCOUNT_CNY)
        or not math.isclose(
            baseline_benchmark_terminal,
            benchmark_terminal,
            rel_tol=1e-10,
            abs_tol=1e-6,
        )
        or benchmark.get("symbol") != RESEARCH_BENCHMARK
        or benchmark.get("baseline_beats_benchmark")
        is not (baseline_terminal > baseline_benchmark_terminal)
        or benchmark.get("candidate_beats_benchmark")
        is not (candidate_terminal > benchmark_terminal)
    ):
        raise ValueError("terminal result does not reconcile")
    candidate_drawdown = _finite_real(drawdown.get("candidate"), name="candidate drawdown")
    baseline_drawdown = _finite_real(drawdown.get("baseline"), name="baseline drawdown")
    if (
        drawdown.get("maximum_allowed") != MAX_STRATEGY_DRAWDOWN
        or not -1.0 < candidate_drawdown <= 0.0
        or not -1.0 < baseline_drawdown <= 0.0
        or drawdown.get("candidate_within_limit")
        is not (abs(candidate_drawdown) <= MAX_STRATEGY_DRAWDOWN)
    ):
        raise ValueError("drawdown result does not reconcile")
    thresholds = execution.get("thresholds")
    if thresholds != {
        "minimum_intent_fill_rate": MIN_INTENT_FILL_RATE,
        "minimum_notional_fill_rate": MIN_NOTIONAL_FILL_RATE,
        "maximum_zero_fill_intent_rate": MAX_ZERO_FILL_INTENT_RATE,
    }:
        raise ValueError("execution-quality thresholds differ from the frozen rule")
    if (
        terminal.get("baseline_execution_quality_passed")
        is not execution.get("baseline", {}).get("execution_quality_passed")
        or terminal.get("candidate_execution_quality_passed")
        is not execution.get("candidate", {}).get("execution_quality_passed")
    ):
        raise ValueError("terminal and execution-quality decisions conflict")
    etf = concentration.get("candidate_single_etf")
    etf_share = _recomputed_concentration_record(etf, artifact="candidate ETF")
    _recomputed_concentration_record(
        concentration.get("baseline_single_etf"), artifact="baseline ETF"
    )
    if (
        concentration.get("maximum_single_etf_abs_contribution_share")
        != MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE
        or concentration.get("candidate_single_etf_within_limit")
        is not (etf_share <= MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE)
    ):
        raise ValueError("candidate ETF concentration does not reconcile")
    fold_majority, fold_concentration = _recomputed_research_fold_gates(
        tests.get("research_folds"), expected_folds
    )
    if concentration.get("single_fold_abs_incremental_pnl") != tests.get(
        "research_folds", {}
    ).get("single_fold_abs_incremental_pnl_concentration"):
        raise ValueError("fold concentration records conflict")
    raw_positive, raw_p_values, raw_folds = _recomputed_signed_raw_factor_gates(
        tests.get("signed_raw_factor_rank_ic"),
        factor_names,
        alpha=alpha,
        expected_folds=expected_folds,
    )
    baseline_execution = _recomputed_execution_passed(
        execution.get("baseline"), artifact="baseline"
    )
    candidate_execution = _recomputed_execution_passed(
        execution.get("candidate"), artifact="candidate"
    )
    if (
        terminal.get("baseline_execution_quality_passed") is not baseline_execution
        or terminal.get("candidate_execution_quality_passed") is not candidate_execution
    ):
        raise ValueError("terminal execution-quality decisions do not reconcile")
    return {
        "terminal_account_improvement_positive": improvement > 0.0,
        "terminal_relative_wealth_improvement_positive": relative > 0.0,
        "candidate_terminal_account_not_below_initial": candidate_terminal
        >= RESEARCH_ACCOUNT_CNY,
        "candidate_execution_quality_passed": candidate_execution,
        "baseline_execution_quality_passed": baseline_execution,
        "candidate_beats_benchmark_at_10bps": candidate_terminal > benchmark_terminal,
        "candidate_max_drawdown_within_limit": abs(candidate_drawdown)
        <= MAX_STRATEGY_DRAWDOWN,
        "paired_complete_fold_majority": fold_majority,
        "single_etf_abs_contribution_share_within_limit": etf_share
        <= MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE,
        "single_fold_abs_incremental_pnl_share_within_limit": fold_concentration,
        "signed_raw_factor_rank_ic_positive": raw_positive,
        "signed_raw_factor_fold_majority": raw_folds,
        "all_signed_raw_factor_rank_ic_positive": raw_positive,
        "all_signed_raw_factor_rank_ic_p_values_below_alpha": raw_p_values,
        "all_signed_raw_factor_fold_majorities": raw_folds,
    }


def _recomputed_joint_iut(
    tests: Mapping[str, Any], factor_names: Sequence[str], *, artifact: str
) -> float:
    joint = tests.get("joint_iut")
    if not isinstance(joint, Mapping):
        raise TypeError(f"{artifact} joint IUT must be a mapping")
    rank = tests.get("rank_ic")
    strategy = tests.get("strategy_net")
    raw = tests.get("signed_raw_factor_rank_ic")
    if not isinstance(rank, Mapping) or not isinstance(strategy, Mapping) or not isinstance(raw, Mapping):
        raise TypeError(f"{artifact} joint IUT components are incomplete")
    expected_components = {
        "model_rank_ic": _result_rate(
            rank.get("one_sided_p_value"), name=f"{artifact} model RankIC p-value"
        ),
        "strategy_net": _result_rate(
            strategy.get("one_sided_p_value"), name=f"{artifact} strategy p-value"
        ),
        **{
            f"signed_raw_factor_rank_ic::{name}": _result_rate(
                raw[name].get("one_sided_p_value"),
                name=f"{artifact} {name} raw RankIC p-value",
            )
            for name in factor_names
        },
    }
    components = joint.get("component_one_sided_p_values")
    if not isinstance(components, Mapping) or set(components) != set(expected_components):
        raise ValueError(f"{artifact} joint IUT component set is invalid")
    for name, expected in expected_components.items():
        actual = _result_rate(
            components.get(name), name=f"{artifact} joint component {name}"
        )
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{artifact} joint IUT component p-value does not reconcile")
    joint_p = _result_rate(
        joint.get("one_sided_p_value"), name=f"{artifact} joint p-value"
    )
    if (
        joint.get("method") != "intersection-union max component p-value"
        or joint.get("component_count") != 2 + len(factor_names)
        or joint.get("alternative")
        != (
            "model_rank_ic_and_strategy_net_improvements_and_all_expected_direction_"
            "raw_factor_rank_ic_means_greater_than_zero"
        )
        or not math.isclose(
            joint_p, max(expected_components.values()), rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise ValueError(f"{artifact} joint IUT does not reconcile")
    return joint_p


def analyze_family_ablations(
    baseline_evidence: Mapping[str, Any],
    family_evidence: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> dict[str, Any]:
    """Describe all five family ablations on discovery dates only.

    These records deliberately do not select a deployable candidate.  The exact
    eighteen-hypothesis BH family remains the formal discovery selection step.
    """

    if max_lag != DEFAULT_HAC_MAX_LAG:
        raise ValueError("family descriptions must use the pre-registered HAC lag")
    if not isinstance(family_evidence, Mapping):
        raise TypeError("family_evidence must be a mapping")
    missing = sorted(set(FACTOR_FAMILIES) - set(family_evidence))
    unexpected = sorted(set(family_evidence) - set(FACTOR_FAMILIES))
    if missing or unexpected or len(family_evidence) != FAMILY_ABLATION_COUNT:
        raise ValueError(
            "family ablation requires exactly the five frozen families"
            f"; missing={missing}; unexpected={unexpected}"
        )
    partition = _plan_partition(plan, "discovery")
    records = []
    for family in FACTOR_FAMILIES:
        factor_names = tuple(
            factor.name
            for factor in ORIGINAL_RESEARCH_CANDIDATES
            if factor.family == family
        )
        records.append(
            {
                "family": family,
                **_complete_stage_tests(
                    _paired_stage_evidence(
                        baseline_evidence,
                        family_evidence[family],
                        plan,
                        "discovery",
                        candidate_factor_names=factor_names,
                        max_lag=max_lag,
                    )
                ),
            }
        )
    exposure = stage_exposure_fields(plan, "discovery")
    return {
        "analysis_status": "completed",
        "stage": "discovery",
        "scope": "discovery_only_family_screen",
        "claim": "Family results are screening evidence only and cannot confirm improvement.",
        "plan_sha256": validate_research_plan(plan)["plan_sha256"],
        "partition_sha256": partition["sessions_sha256"],
        **exposure,
        "family_count": len(records),
        "results": records,
    }


def analyze_factor_discovery(
    baseline_evidence: Mapping[str, Any],
    candidate_evidence: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    q: float = DISCOVERY_FDR_Q,
    max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> dict[str, Any]:
    """Run exactly eighteen paired discovery tests and the pre-registered BH rule."""

    if q != DISCOVERY_FDR_Q or max_lag != DEFAULT_HAC_MAX_LAG:
        raise ValueError("discovery must use the pre-registered BH q and HAC lag")
    if not isinstance(candidate_evidence, Mapping):
        raise TypeError("candidate_evidence must be a mapping")
    expected = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
    missing = sorted(set(expected) - set(candidate_evidence))
    unexpected = sorted(set(candidate_evidence) - set(expected))
    if missing or unexpected or len(candidate_evidence) != CATALOG_HYPOTHESIS_COUNT:
        raise ValueError(
            "factor discovery requires exactly the eighteen frozen candidates"
            f"; missing={missing}; unexpected={unexpected}"
        )
    partition = _plan_partition(plan, "discovery")
    tests: dict[str, dict[str, Any]] = {}
    for name in expected:
        tests[name] = _complete_stage_tests(
            _paired_stage_evidence(
                baseline_evidence,
                candidate_evidence[name],
                plan,
                "discovery",
                candidate_factor_names=(name,),
                max_lag=max_lag,
            )
        )
    bh = catalog_benjamini_hochberg(
        {name: tests[name]["joint_iut"]["one_sided_p_value"] for name in expected}, q=q
    )
    bh_by_name = {record["hypothesis_id"]: record for record in bh["results"]}
    records = [
        {
            "hypothesis_id": name,
            "family": next(factor.family for factor in ORIGINAL_RESEARCH_CANDIDATES if factor.name == name),
            **tests[name],
            "joint_p_value": tests[name]["joint_iut"]["one_sided_p_value"],
            "bh_q_value": bh_by_name[name]["q_value"],
            "bh_rejected": bh_by_name[name]["rejected"],
            "selection_criteria": {
                "joint_bh_rejected": bh_by_name[name]["rejected"],
                "rank_ic_mean_difference_positive": tests[name]["rank_ic"][
                    "mean_difference"
                ]
                > 0.0,
                "strategy_net_mean_difference_positive": tests[name]["strategy_net"][
                    "mean_difference"
                ]
                > 0.0,
                "terminal_account_improvement_positive": tests[name]["terminal"][
                    "account_improvement_positive"
                ],
                "terminal_relative_wealth_improvement_positive": tests[name]["terminal"][
                    "relative_wealth_improvement_positive"
                ],
                "candidate_terminal_account_not_below_initial": tests[name]["terminal"][
                    "candidate_terminal_account_not_below_initial"
                ],
                "candidate_execution_quality_passed": tests[name]["terminal"][
                    "candidate_execution_quality_passed"
                ],
                "baseline_execution_quality_passed": tests[name]["terminal"][
                    "baseline_execution_quality_passed"
                ],
                "candidate_beats_benchmark_at_10bps": tests[name]["benchmark"][
                    "candidate_beats_benchmark"
                ],
                "candidate_max_drawdown_within_limit": tests[name]["drawdown"][
                    "candidate_within_limit"
                ],
                "paired_complete_fold_majority": tests[name]["research_folds"][
                    "majority_positive_passed"
                ],
                "single_etf_abs_contribution_share_within_limit": tests[name][
                    "concentration"
                ]["candidate_single_etf_within_limit"],
                "single_fold_abs_incremental_pnl_share_within_limit": tests[name][
                    "concentration"
                ]["single_fold_abs_incremental_pnl"]["passed"],
                "signed_raw_factor_rank_ic_positive": tests[name][
                    "signed_raw_factor_rank_ic"
                ][name]["signed_rank_ic_mean"]
                > 0.0,
                "signed_raw_factor_fold_majority": tests[name][
                    "signed_raw_factor_rank_ic"
                ][name]["folds"]["majority_positive_passed"],
            },
        }
        for name in expected
    ]
    selected = [
        record["hypothesis_id"]
        for record in records
        if all(record["selection_criteria"].values())
    ]
    exposure = stage_exposure_fields(plan, "discovery")
    return {
        "analysis_status": "completed",
        "stage": "discovery",
        "scope": "discovery_only_paired_incremental_evidence",
        "claim": "Selections are discovery candidates only; confirmation has not occurred.",
        "plan_sha256": validate_research_plan(plan)["plan_sha256"],
        "partition_sha256": partition["sessions_sha256"],
        **exposure,
        "hypothesis_count": len(records),
        "joint_test_method": "intersection-union max component p-value",
        "multiplicity_applied_to": DISCOVERY_MULTIPLICITY_APPLIED_TO,
        "bh": bh,
        "selected_factor_names": selected,
        "results": records,
    }


def analyze_discovery(
    baseline_evidence: Mapping[str, Any],
    family_evidence: Mapping[str, Mapping[str, Any]],
    candidate_evidence: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    q: float = DISCOVERY_FDR_Q,
    max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> dict[str, Any]:
    """Run the complete pre-registered discovery battery on one partition."""

    families = analyze_family_ablations(
        baseline_evidence, family_evidence, plan, max_lag=max_lag
    )
    factors = analyze_factor_discovery(
        baseline_evidence, candidate_evidence, plan, q=q, max_lag=max_lag
    )
    if families["partition_sha256"] != factors["partition_sha256"]:
        raise RuntimeError("family and factor discovery partitions differ")
    exposure = stage_exposure_fields(plan, "discovery")
    return {
        "analysis_status": "completed",
        "stage": "discovery",
        "scope": "complete_pre_registered_discovery_battery",
        "claim": "Selections are discovery candidates only; confirmation has not occurred.",
        "plan_sha256": factors["plan_sha256"],
        "partition_sha256": factors["partition_sha256"],
        **exposure,
        "family_ablations": families,
        "factor_discovery": factors,
        "selected_factor_names": factors["selected_factor_names"],
    }


def analyze_confirmation(
    baseline_evidence: Mapping[str, Any],
    candidate_evidence: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    frozen_spec_sha256: str,
    candidate_factor_names: Sequence[str],
    minimum_mean_difference: float = 0.0,
    alpha: float = CONFIRMATION_ALPHA,
    max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> dict[str, Any]:
    """Evaluate one already-frozen candidate on the confirmation partition."""

    threshold = float(minimum_mean_difference)
    if not math.isfinite(threshold) or threshold != 0.0:
        raise ValueError("confirmation minimum_mean_difference must remain zero")
    alpha_value = _finite_probability(alpha, name="alpha")
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if alpha_value != CONFIRMATION_ALPHA or max_lag != DEFAULT_HAC_MAX_LAG:
        raise ValueError("confirmation must use the pre-registered alpha and HAC lag")
    if not isinstance(frozen_spec_sha256, str) or not _DIGEST_PATTERN.fullmatch(
        frozen_spec_sha256
    ):
        raise ValueError("frozen_spec_sha256 must be a lowercase SHA-256 digest")
    factor_names = _catalog_factor_names(
        candidate_factor_names, artifact="confirmation candidate"
    )
    partition = _plan_partition(plan, "confirmation")
    tests = _complete_stage_tests(
        _paired_stage_evidence(
            baseline_evidence,
            candidate_evidence,
            plan,
            "confirmation",
            candidate_factor_names=factor_names,
            max_lag=max_lag,
        )
    )
    exposure = stage_exposure_fields(plan, "confirmation")
    criteria = {
        "rank_ic_mean_difference_above_minimum": tests["rank_ic"]["mean_difference"]
        > threshold,
        "rank_ic_one_sided_p_value_below_alpha": tests["rank_ic"]["one_sided_p_value"]
        <= alpha_value,
        "strategy_net_mean_difference_above_minimum": tests["strategy_net"][
            "mean_difference"
        ]
        > threshold,
        "strategy_net_one_sided_p_value_below_alpha": tests["strategy_net"][
            "one_sided_p_value"
        ]
        <= alpha_value,
        "terminal_account_improvement_positive": tests["terminal"][
            "account_improvement_positive"
        ],
        "terminal_relative_wealth_improvement_positive": tests["terminal"][
            "relative_wealth_improvement_positive"
        ],
        "candidate_terminal_account_not_below_initial": tests["terminal"][
            "candidate_terminal_account_not_below_initial"
        ],
        "candidate_execution_quality_passed": tests["terminal"][
            "candidate_execution_quality_passed"
        ],
        "baseline_execution_quality_passed": tests["terminal"][
            "baseline_execution_quality_passed"
        ],
        "candidate_beats_benchmark_at_10bps": tests["benchmark"][
            "candidate_beats_benchmark"
        ],
        "candidate_max_drawdown_within_limit": tests["drawdown"][
            "candidate_within_limit"
        ],
        "paired_complete_fold_majority": tests["research_folds"][
            "majority_positive_passed"
        ],
        "single_etf_abs_contribution_share_within_limit": tests["concentration"][
            "candidate_single_etf_within_limit"
        ],
        "single_fold_abs_incremental_pnl_share_within_limit": tests[
            "concentration"
        ]["single_fold_abs_incremental_pnl"]["passed"],
        "all_signed_raw_factor_rank_ic_positive": all(
            test["signed_rank_ic_mean"] > 0.0
            for test in tests["signed_raw_factor_rank_ic"].values()
        ),
        "all_signed_raw_factor_rank_ic_p_values_below_alpha": all(
            test["one_sided_p_value"] <= alpha_value
            for test in tests["signed_raw_factor_rank_ic"].values()
        ),
        "all_signed_raw_factor_fold_majorities": all(
            test["folds"]["majority_positive_passed"]
            for test in tests["signed_raw_factor_rank_ic"].values()
        ),
    }
    passed = all(criteria.values())
    return {
        "analysis_status": "completed",
        "stage": "confirmation",
        "scope": "single_frozen_candidate_confirmation",
        "claim": (
            "The frozen candidate passed the pre-registered retrospective confirmation rule; "
            "this is exposed research-only evidence, not a blind or promotion result."
            if passed and exposure["evidence_class"] == "retrospective_exposed"
            else (
                "The frozen candidate passed the pre-registered confirmation rule; any later "
                "stage remains separately classified by the exposure registry."
                if passed
                else "Confirmation failed; the locked holdout must remain unopened."
            )
        ),
        "plan_sha256": validate_research_plan(plan)["plan_sha256"],
        "frozen_spec_sha256": frozen_spec_sha256,
        "candidate_factor_names": list(factor_names),
        "partition_sha256": partition["sessions_sha256"],
        **exposure,
        "alpha": alpha_value,
        "minimum_mean_difference": threshold,
        "account_currency": "CNY",
        "initial_account": RESEARCH_ACCOUNT_CNY,
        "stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
        "confirmation_passed": passed,
        "criteria": criteria,
        "tests": tests,
    }


def analyze_locked_holdout(
    baseline_evidence: Mapping[str, Any],
    candidate_evidence: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    frozen_spec_sha256: str,
    candidate_factor_names: Sequence[str],
    minimum_mean_difference: float = 0.0,
    alpha: float = CONFIRMATION_ALPHA,
    max_lag: int = DEFAULT_HAC_MAX_LAG,
) -> dict[str, Any]:
    """Evaluate the unchanged frozen candidate on the final one-shot partition."""

    threshold = float(minimum_mean_difference)
    if not math.isfinite(threshold) or threshold != 0.0:
        raise ValueError("locked holdout minimum_mean_difference must remain zero")
    alpha_value = _finite_probability(alpha, name="alpha")
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if alpha_value != CONFIRMATION_ALPHA or max_lag != DEFAULT_HAC_MAX_LAG:
        raise ValueError("locked holdout must use the pre-registered alpha and HAC lag")
    if not isinstance(frozen_spec_sha256, str) or not _DIGEST_PATTERN.fullmatch(
        frozen_spec_sha256
    ):
        raise ValueError("frozen_spec_sha256 must be a lowercase SHA-256 digest")
    factor_names = _catalog_factor_names(
        candidate_factor_names, artifact="locked holdout candidate"
    )
    partition = _plan_partition(plan, "locked_holdout")
    tests = _complete_stage_tests(
        _paired_stage_evidence(
            baseline_evidence,
            candidate_evidence,
            plan,
            "locked_holdout",
            candidate_factor_names=factor_names,
            max_lag=max_lag,
        )
    )
    exposure = stage_exposure_fields(plan, "locked_holdout")
    criteria = {
        "rank_ic_mean_difference_above_minimum": tests["rank_ic"]["mean_difference"]
        > threshold,
        "rank_ic_one_sided_p_value_below_alpha": tests["rank_ic"]["one_sided_p_value"]
        <= alpha_value,
        "strategy_net_mean_difference_above_minimum": tests["strategy_net"][
            "mean_difference"
        ]
        > threshold,
        "strategy_net_one_sided_p_value_below_alpha": tests["strategy_net"][
            "one_sided_p_value"
        ]
        <= alpha_value,
        "terminal_account_improvement_positive": tests["terminal"][
            "account_improvement_positive"
        ],
        "terminal_relative_wealth_improvement_positive": tests["terminal"][
            "relative_wealth_improvement_positive"
        ],
        "candidate_terminal_account_not_below_initial": tests["terminal"][
            "candidate_terminal_account_not_below_initial"
        ],
        "candidate_execution_quality_passed": tests["terminal"][
            "candidate_execution_quality_passed"
        ],
        "baseline_execution_quality_passed": tests["terminal"][
            "baseline_execution_quality_passed"
        ],
        "candidate_beats_benchmark_at_10bps": tests["benchmark"][
            "candidate_beats_benchmark"
        ],
        "candidate_max_drawdown_within_limit": tests["drawdown"][
            "candidate_within_limit"
        ],
        "paired_complete_fold_majority": tests["research_folds"][
            "majority_positive_passed"
        ],
        "single_etf_abs_contribution_share_within_limit": tests["concentration"][
            "candidate_single_etf_within_limit"
        ],
        "single_fold_abs_incremental_pnl_share_within_limit": tests[
            "concentration"
        ]["single_fold_abs_incremental_pnl"]["passed"],
        "all_signed_raw_factor_rank_ic_positive": all(
            test["signed_rank_ic_mean"] > 0.0
            for test in tests["signed_raw_factor_rank_ic"].values()
        ),
        "all_signed_raw_factor_rank_ic_p_values_below_alpha": all(
            test["one_sided_p_value"] <= alpha_value
            for test in tests["signed_raw_factor_rank_ic"].values()
        ),
        "all_signed_raw_factor_fold_majorities": all(
            test["folds"]["majority_positive_passed"]
            for test in tests["signed_raw_factor_rank_ic"].values()
        ),
    }
    passed = all(criteria.values())
    return {
        "analysis_status": "completed",
        "stage": "locked_holdout",
        "scope": "one_shot_locked_holdout_estimate",
        "claim": (
            "The unchanged frozen candidate passed the one-shot retrospective estimate; "
            "the period was historically exposed and remains research_only."
            if passed and exposure["evidence_class"] == "retrospective_exposed"
            else (
                "The locked holdout supports the unchanged frozen candidate under the declared rule."
                if passed
                else "The locked holdout does not establish out-of-sample improvement."
            )
        ),
        "plan_sha256": validate_research_plan(plan)["plan_sha256"],
        "frozen_spec_sha256": frozen_spec_sha256,
        "candidate_factor_names": list(factor_names),
        "partition_sha256": partition["sessions_sha256"],
        **exposure,
        "alpha": alpha_value,
        "minimum_mean_difference": threshold,
        "account_currency": "CNY",
        "initial_account": RESEARCH_ACCOUNT_CNY,
        "stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
        "locked_holdout_passed": passed,
        "criteria": criteria,
        "tests": tests,
    }


def _read_json_object(path: Path, *, artifact: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{artifact} must contain a JSON object")
    return value


def _validate_completed_pair(
    baseline_run: Path, candidate_run: Path, *, expected_family: str | None
) -> tuple[str, str]:
    """Reuse the pipeline's frozen-run identity checks without evaluating holdout values."""

    from .comparison import (
        _code_identity,
        _normalized_experiment_config,
        _source_identity,
        _validate_feature_roles,
        _value_differences,
        _verify_run_integrity,
    )

    baseline_manifest = _read_json_object(baseline_run / "manifest.json", artifact="baseline manifest")
    candidate_manifest = _read_json_object(candidate_run / "manifest.json", artifact="candidate manifest")
    baseline_config = _read_json_object(baseline_run / "config.json", artifact="baseline config")
    candidate_config = _read_json_object(candidate_run / "config.json", artifact="candidate config")
    if baseline_manifest.get("status") != "completed" or candidate_manifest.get("status") != "completed":
        raise ValueError("both run manifests must have status completed")
    if baseline_run == candidate_run or baseline_manifest.get("run_id") == candidate_manifest.get("run_id"):
        raise ValueError("baseline and candidate runs must be distinct")
    _verify_run_integrity(baseline_run, baseline_manifest, "baseline")
    _verify_run_integrity(candidate_run, candidate_manifest, "candidate")
    if _source_identity(baseline_manifest) != _source_identity(candidate_manifest):
        raise ValueError("baseline and candidate data identities differ")
    if _code_identity(baseline_manifest, "baseline") != _code_identity(candidate_manifest, "candidate"):
        raise ValueError("baseline and candidate source tree identities differ")
    _validate_feature_roles(baseline_config, baseline_manifest, candidate_config, candidate_manifest)
    differences = _value_differences(
        _normalized_experiment_config(baseline_config),
        _normalized_experiment_config(candidate_config),
    )
    if differences:
        raise ValueError("non-feature experiment configuration differs: " + ", ".join(differences[:20]))
    if expected_family is not None:
        features = candidate_config.get("features", {})
        if features.get("families") != [expected_family]:
            raise ValueError(f"candidate run is not the declared single-family ablation: {expected_family}")
    return (
        _expected_run_artifact_sha256(baseline_run, baseline_manifest, "signal_metrics.parquet"),
        _expected_run_artifact_sha256(candidate_run, candidate_manifest, "signal_metrics.parquet"),
    )


def _expected_run_artifact_sha256(
    run_dir: Path, manifest: Mapping[str, Any], relative_path: str
) -> str:
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("run manifest is missing integrity metadata")
    checksum_name = integrity.get("checksum_manifest")
    if not isinstance(checksum_name, str):
        raise ValueError("run checksum manifest path is invalid")
    checksum = _read_json_object(run_dir / checksum_name, artifact="artifact checksum manifest")
    records = checksum.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("artifact checksum manifest has no artifact records")
    matches = [record for record in records if isinstance(record, dict) and record.get("path") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"artifact checksum manifest does not uniquely identify {relative_path}")
    digest = matches[0].get("sha256")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError(f"artifact checksum manifest has an invalid SHA-256 for {relative_path}")
    return digest


def load_partition_signal_metric(
    parquet_path: Path,
    partition: Mapping[str, Any],
    *,
    metric: str = "rank_ic",
    expected_sha256: str,
) -> pd.Series:
    """Read only one declared date partition from a frozen signal artifact.

    Standalone artifacts require an externally frozen SHA-256.  Completed-run
    callers obtain that digest from the verified run checksum manifest.
    PyArrow predicate pushdown is mandatory; the function never falls back to
    loading an entire file that could include a locked holdout.
    """

    path = Path(parquet_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"signal metric artifact is missing or unsafe: {path}")
    if not isinstance(expected_sha256, str) or not _DIGEST_PATTERN.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if sha256_file(path) != expected_sha256:
        raise ValueError("signal metric artifact SHA-256 does not match the frozen digest")
    if not isinstance(metric, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", metric):
        raise ValueError("metric must be a simple column name")
    sessions = _normalize_sessions(partition.get("sessions", []))
    filters = [
        ("datetime", ">=", pd.Timestamp(sessions[0])),
        ("datetime", "<=", pd.Timestamp(sessions[-1])),
    ]
    try:
        frame = pd.read_parquet(path, columns=[metric], filters=filters, engine="pyarrow")
    except Exception as exc:
        raise ValueError(f"could not predicate-read the declared partition from {path.name}") from exc
    if metric not in frame:
        raise ValueError(f"signal metric artifact is missing {metric}")
    return _series_with_exact_partition(frame[metric], partition, artifact=str(path.name))


def load_completed_pair_metric(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    plan: Mapping[str, Any],
    *,
    stage: str,
    metric: str = "rank_ic",
    expected_family: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Load a stage-local paired metric from two comparable completed runs."""

    baseline_run = Path(baseline_run_dir).resolve()
    candidate_run = Path(candidate_run_dir).resolve()
    if not baseline_run.is_dir() or not candidate_run.is_dir():
        raise ValueError("both completed run directories must exist")
    if expected_family is not None and expected_family not in FACTOR_FAMILIES:
        raise ValueError(f"unknown factor family: {expected_family}")
    baseline_checksum, candidate_checksum = _validate_completed_pair(
        baseline_run, candidate_run, expected_family=expected_family
    )
    partition = _plan_partition(plan, stage)
    baseline = load_partition_signal_metric(
        baseline_run / "signal_metrics.parquet",
        partition,
        metric=metric,
        expected_sha256=baseline_checksum,
    )
    candidate = load_partition_signal_metric(
        candidate_run / "signal_metrics.parquet",
        partition,
        metric=metric,
        expected_sha256=candidate_checksum,
    )
    if not baseline.isna().equals(candidate.isna()):
        raise ValueError("paired run metric missing-value masks differ")
    return baseline, candidate


def _unsigned_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in state.items() if key != "state_sha256"}


def _seal_state(state: dict[str, Any]) -> dict[str, Any]:
    sealed = deepcopy(state)
    sealed["state_sha256"] = _sha256_json(_unsigned_state(sealed))
    return sealed


def _validate_state(state: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    validated_plan = validate_research_plan(plan)
    if not isinstance(state, Mapping):
        raise TypeError("research state must be a mapping")
    value = deepcopy(dict(state))
    if value.get("schema_version") != RESEARCH_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported research state schema")
    digest = value.get("state_sha256")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("research state has an invalid state_sha256")
    if _sha256_json(_unsigned_state(value)) != digest:
        raise ValueError("research state SHA-256 does not match its content")
    if value.get("plan_sha256") != validated_plan["plan_sha256"]:
        raise ValueError("research state belongs to a different plan")
    if value.get("catalog_sha256") != validated_plan["catalog_sha256"]:
        raise ValueError("research state catalog identity differs from the plan")
    exposure = validated_plan["exposure_provenance"]
    if value.get("exposure_registry_sha256") != exposure["registry_sha256"]:
        raise ValueError("research state exposure registry identity differs from the plan")
    revision = value.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("research state revision is invalid")
    stages = value.get("stages")
    if not isinstance(stages, dict) or tuple(stages) != RESEARCH_STAGES:
        raise ValueError("research state stages are invalid")
    for stage in RESEARCH_STAGES:
        record = stages[stage]
        if not isinstance(record, dict) or record.get("status") not in _STAGE_STATUS:
            raise ValueError(f"research state {stage} record is invalid")
        attempts = record.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts not in (0, 1):
            raise ValueError(f"research state {stage} attempts are invalid")
        if (record["status"] == "unopened") != (attempts == 0):
            raise ValueError(f"research state {stage} status conflicts with attempts")
        validate_stage_exposure_fields(
            record, validated_plan, stage, artifact=f"research state {stage}"
        )
    return value


@contextmanager
def _exclusive_state_lock(state_path: Path):
    lock_path = state_path.with_name(state_path.name + ".lock")
    if lock_path.is_symlink():
        raise RuntimeError("research state lock path must not be a symbolic link")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "research state is locked; a stale lock is fail-closed and requires an explicit audit"
        ) from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def initialize_research_state(state_path: Path, plan: Mapping[str, Any]) -> Path:
    """Create a new access ledger; an existing path is never overwritten."""

    validated_plan = validate_research_plan(plan)
    path = Path(state_path).resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"research state already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_state_lock(path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"research state already exists: {path}")
        state = {
            "schema_version": RESEARCH_STATE_SCHEMA_VERSION,
            "protocol_version": RESEARCH_PROTOCOL_VERSION,
            "plan_id": validated_plan["plan_id"],
            "plan_sha256": validated_plan["plan_sha256"],
            "catalog_sha256": validated_plan["catalog_sha256"],
            "exposure_registry_sha256": validated_plan["exposure_provenance"][
                "registry_sha256"
            ],
            "created_at": now_shanghai().isoformat(),
            "revision": 0,
            "stages": {
                stage: {
                    "status": "unopened",
                    "attempts": 0,
                    **stage_exposure_fields(validated_plan, stage),
                }
                for stage in RESEARCH_STAGES
            },
            "discovery_eligible_factor_names": None,
            "frozen_confirmation_spec": None,
        }
        write_json_atomic(path, _seal_state(state))
    return path


def read_research_state(state_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(state_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"research state is missing or unsafe: {path}")
    return _validate_state(_read_json_object(path, artifact="research state"), plan)


def _write_state_revision(path: Path, state: dict[str, Any]) -> None:
    state["revision"] += 1
    state["updated_at"] = now_shanghai().isoformat()
    write_json_atomic(path, _seal_state(state))


def freeze_confirmation_spec(
    state_path: Path,
    plan: Mapping[str, Any],
    *,
    selected_factor_names: Sequence[str],
    frozen_spec: Mapping[str, Any],
) -> str:
    """Freeze the sole candidate specification allowed in later partitions."""

    if isinstance(selected_factor_names, (str, bytes)):
        raise TypeError("selected_factor_names must be a sequence")
    selected = list(selected_factor_names)
    if not selected or len(selected) != len(set(selected)) or not all(isinstance(item, str) for item in selected):
        raise ValueError("selected_factor_names must be a non-empty unique string list")
    catalog_order = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
    if any(item not in catalog_order for item in selected):
        raise ValueError("frozen selection contains a factor outside the catalog")
    selected.sort(key=catalog_order.index)
    if not isinstance(frozen_spec, Mapping) or not frozen_spec:
        raise ValueError("frozen_spec must be a non-empty JSON mapping")
    canonical_spec = json.loads(_canonical_json(dict(frozen_spec)))
    spec_record = {
        "selected_factor_names": selected,
        "specification": canonical_spec,
    }
    spec_record["sha256"] = _sha256_json(spec_record)

    path = Path(state_path).resolve()
    with _exclusive_state_lock(path):
        state = read_research_state(path, plan)
        if state["stages"]["discovery"]["status"] != "completed":
            raise RuntimeError("discovery must complete before freezing confirmation")
        if state["stages"]["confirmation"]["status"] != "unopened":
            raise RuntimeError("confirmation has already been consumed")
        if state["frozen_confirmation_spec"] is not None:
            raise RuntimeError("confirmation specification is already frozen")
        eligible = state.get("discovery_eligible_factor_names")
        if not isinstance(eligible, list) or selected != eligible:
            raise ValueError("frozen factors must equal the complete recorded joint-selected set")
        state["frozen_confirmation_spec"] = spec_record
        _write_state_revision(path, state)
    return spec_record["sha256"]


def _claim_stage(state_path: Path, plan: Mapping[str, Any], stage: str) -> str:
    if stage not in RESEARCH_STAGES:
        raise ValueError(f"unknown research stage: {stage}")
    path = Path(state_path).resolve()
    with _exclusive_state_lock(path):
        state = read_research_state(path, plan)
        record = state["stages"][stage]
        if record["status"] != "unopened":
            raise RuntimeError(f"{stage} has already been consumed with status {record['status']}")
        position = RESEARCH_STAGES.index(stage)
        if position and state["stages"][RESEARCH_STAGES[position - 1]]["status"] != "completed":
            raise RuntimeError(f"{RESEARCH_STAGES[position - 1]} must complete before {stage}")
        if stage != "discovery" and state["frozen_confirmation_spec"] is None:
            raise RuntimeError("the confirmation candidate specification has not been frozen")
        if stage == "locked_holdout" and state["stages"]["confirmation"].get(
            "confirmation_passed"
        ) is not True:
            raise RuntimeError("locked holdout must remain unopened because confirmation did not pass")
        token = secrets.token_hex(32)
        record.update(
            {
                "status": "claimed",
                "attempts": 1,
                "claim_token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
                "claimed_at": now_shanghai().isoformat(),
                "partition_sha256": _plan_partition(plan, stage)["sessions_sha256"],
                "frozen_spec_sha256": (
                    state["frozen_confirmation_spec"]["sha256"] if stage != "discovery" else None
                ),
            }
        )
        _write_state_revision(path, state)
    return token


def _finish_stage(
    state_path: Path,
    plan: Mapping[str, Any],
    stage: str,
    token: str,
    *,
    result: Any = None,
    error: BaseException | None = None,
) -> None:
    path = Path(state_path).resolve()
    with _exclusive_state_lock(path):
        state = read_research_state(path, plan)
        record = state["stages"][stage]
        if record["status"] != "claimed":
            raise RuntimeError(f"{stage} is not in claimed state")
        token_digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        if not secrets.compare_digest(record.get("claim_token_sha256", ""), token_digest):
            raise PermissionError("stage claim token does not match")
        if error is not None:
            record.update(
                {
                    "status": "failed",
                    "finished_at": now_shanghai().isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:500],
                }
            )
        else:
            canonical_result = json.loads(_canonical_json(result))
            if not isinstance(canonical_result, dict):
                raise TypeError("stage evaluator must return a JSON object")
            if canonical_result.get("stage") != stage:
                raise ValueError("stage evaluator result does not match the claimed stage")
            if canonical_result.get("plan_sha256") != validate_research_plan(plan)["plan_sha256"]:
                raise ValueError("stage evaluator result does not match the research plan")
            if canonical_result.get("partition_sha256") != record["partition_sha256"]:
                raise ValueError("stage evaluator result does not match the claimed partition")
            validate_stage_exposure_fields(
                canonical_result,
                plan,
                stage,
                artifact="stage evaluator result",
            )
            record.update(
                {
                    "status": "completed",
                    "finished_at": now_shanghai().isoformat(),
                    "result_sha256": _sha256_json(canonical_result),
                    "result": canonical_result,
                }
            )
            if stage == "discovery":
                if canonical_result.get("scope") != "complete_pre_registered_discovery_battery":
                    raise ValueError("discovery state requires the complete pre-registered battery")
                family_result = canonical_result.get("family_ablations")
                factor_result = canonical_result.get("factor_discovery")
                if not isinstance(family_result, dict) or family_result.get("family_count") != FAMILY_ABLATION_COUNT:
                    raise ValueError("discovery result must contain all five family ablations")
                if (
                    not isinstance(factor_result, dict)
                    or factor_result.get("hypothesis_count") != CATALOG_HYPOTHESIS_COUNT
                ):
                    raise ValueError("discovery result must contain all eighteen factor tests")
                if (
                    factor_result.get("joint_test_method")
                    != "intersection-union max component p-value"
                    or factor_result.get("multiplicity_applied_to")
                    != DISCOVERY_MULTIPLICITY_APPLIED_TO
                ):
                    raise ValueError("discovery result must contain the frozen joint evidence rule")
                bh = factor_result.get("bh")
                if (
                    not isinstance(bh, dict)
                    or bh.get("method") != "Benjamini-Hochberg"
                    or bh.get("hypothesis_count") != CATALOG_HYPOTHESIS_COUNT
                    or bh.get("fdr_q") != DISCOVERY_FDR_Q
                ):
                    raise ValueError("discovery result must contain the frozen BH q=0.10 decision")
                selected = canonical_result.get("selected_factor_names")
                catalog_names = {factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES}
                if not isinstance(selected, list) or len(selected) != len(set(selected)) or not set(selected).issubset(
                    catalog_names
                ):
                    raise ValueError("discovery result must record valid selected_factor_names")
                if selected != factor_result.get("selected_factor_names"):
                    raise ValueError("discovery selected factors conflict with the joint discovery result")
                result_records = factor_result.get("results")
                if not isinstance(result_records, list) or len(result_records) != CATALOG_HYPOTHESIS_COUNT:
                    raise ValueError("discovery result must preserve all joint hypothesis records")
                expected_hypothesis_ids = [
                    factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES
                ]
                if [
                    item.get("hypothesis_id") if isinstance(item, dict) else None
                    for item in result_records
                ] != expected_hypothesis_ids:
                    raise ValueError(
                        "discovery result must preserve the frozen hypothesis order"
                    )
                partition = _plan_partition(plan, "discovery")
                expected_folds = partition["research_folds"]
                recomputed_records: dict[str, dict[str, Any]] = {}
                for item in result_records:
                    criteria = item.get("selection_criteria")
                    if (
                        not isinstance(criteria, dict)
                        or set(criteria) != _DISCOVERY_SELECTION_CRITERIA_FIELDS
                        or not all(isinstance(value, bool) for value in criteria.values())
                    ):
                        raise ValueError(
                            "discovery result has invalid joint selection criteria"
                        )
                    hypothesis_id = item["hypothesis_id"]
                    expected_family = next(
                        factor.family
                        for factor in ORIGINAL_RESEARCH_CANDIDATES
                        if factor.name == hypothesis_id
                    )
                    if item.get("family") != expected_family:
                        raise ValueError("discovery result factor family is invalid")
                    raw_tests = item.get("signed_raw_factor_rank_ic")
                    joint = item.get("joint_iut")
                    components = joint.get("component_one_sided_p_values") if isinstance(joint, dict) else None
                    expected_component_keys = {
                        "model_rank_ic",
                        "strategy_net",
                        f"signed_raw_factor_rank_ic::{hypothesis_id}",
                    }
                    if (
                        not isinstance(raw_tests, dict)
                        or tuple(raw_tests) != (hypothesis_id,)
                        or not isinstance(components, dict)
                        or set(components) != expected_component_keys
                        or joint.get("component_count") != 3
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            or not 0.0 <= float(value) <= 1.0
                            for value in components.values()
                        )
                        or not math.isclose(
                            float(joint.get("one_sided_p_value", float("nan"))),
                            max(float(value) for value in components.values()),
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        )
                        or not math.isclose(
                            float(item.get("joint_p_value", float("nan"))),
                            float(joint["one_sided_p_value"]),
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        )
                    ):
                        raise ValueError("discovery result has invalid three-component joint evidence")
                    common = _recomputed_common_criteria(
                        item,
                        (hypothesis_id,),
                        alpha=None,
                        expected_folds=expected_folds,
                    )
                    rank_difference, _ = _recomputed_paired_hac_result(
                        item.get("rank_ic"),
                        artifact=f"discovery {hypothesis_id} RankIC",
                        minimum_observations=math.ceil(
                            MIN_RAW_FACTOR_RANK_IC_COVERAGE
                            * len(partition["sessions"])
                        ),
                    )
                    strategy_difference, _ = _recomputed_paired_hac_result(
                        item.get("strategy_net"),
                        artifact=f"discovery {hypothesis_id} strategy net",
                        minimum_observations=MIN_PAIRED_OBSERVATIONS,
                        exact_observations=len(
                            partition["portfolio_evaluation_sessions"]
                        ),
                    )
                    joint_p = _recomputed_joint_iut(
                        item, (hypothesis_id,), artifact=f"discovery {hypothesis_id}"
                    )
                    recorded_joint_p = _result_rate(
                        item.get("joint_p_value"),
                        name=f"discovery {hypothesis_id} joint p-value",
                    )
                    if not math.isclose(
                        recorded_joint_p, joint_p, rel_tol=0.0, abs_tol=1e-15
                    ):
                        raise ValueError("discovery joint p-value does not reconcile")
                    recomputed_records[hypothesis_id] = {
                        "joint_p_value": joint_p,
                        "criteria_without_bh": {
                            "rank_ic_mean_difference_positive": rank_difference > 0.0,
                            "strategy_net_mean_difference_positive": strategy_difference
                            > 0.0,
                            **{
                                name: common[name]
                                for name in _DISCOVERY_SELECTION_CRITERIA_FIELDS
                                if name
                                not in {
                                    "joint_bh_rejected",
                                    "rank_ic_mean_difference_positive",
                                    "strategy_net_mean_difference_positive",
                                }
                            },
                        },
                    }
                recomputed_bh = catalog_benjamini_hochberg(
                    {
                        name: recomputed_records[name]["joint_p_value"]
                        for name in expected_hypothesis_ids
                    },
                    q=DISCOVERY_FDR_Q,
                )
                if bh != recomputed_bh:
                    raise ValueError("discovery BH result does not reconcile to all eighteen joint p-values")
                recomputed_bh_by_name = {
                    item["hypothesis_id"]: item
                    for item in recomputed_bh["results"]
                }
                for item in result_records:
                    hypothesis_id = item["hypothesis_id"]
                    expected_bh = recomputed_bh_by_name[hypothesis_id]
                    recorded_q = _result_rate(
                        item.get("bh_q_value"),
                        name=f"discovery {hypothesis_id} BH q-value",
                    )
                    if (
                        not math.isclose(
                            recorded_q,
                            expected_bh["q_value"],
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        )
                        or item.get("bh_rejected") is not expected_bh["rejected"]
                    ):
                        raise ValueError("discovery record BH decision does not reconcile")
                    recomputed_criteria = {
                        "joint_bh_rejected": expected_bh["rejected"],
                        **recomputed_records[hypothesis_id]["criteria_without_bh"],
                    }
                    if item["selection_criteria"] != recomputed_criteria:
                        raise ValueError(
                            "discovery selection criteria do not reconcile to evidence"
                        )
                selected_from_records = [
                    item["hypothesis_id"]
                    for item in result_records
                    if all(item["selection_criteria"].values())
                ]
                if selected != selected_from_records:
                    raise ValueError("discovery selected factors conflict with joint selection criteria")
                state["discovery_eligible_factor_names"] = selected
            elif stage == "confirmation":
                if canonical_result.get("frozen_spec_sha256") != record["frozen_spec_sha256"]:
                    raise ValueError("confirmation result does not match the frozen specification")
                frozen_names = state["frozen_confirmation_spec"]["selected_factor_names"]
                if canonical_result.get("candidate_factor_names") != frozen_names:
                    raise ValueError("confirmation result factor names differ from the frozen specification")
                passed = canonical_result.get("confirmation_passed")
                if not isinstance(passed, bool):
                    raise ValueError("confirmation result must record confirmation_passed")
                criteria = canonical_result.get("criteria")
                if (
                    not isinstance(criteria, dict)
                    or set(criteria) != _CONFIRMATION_CRITERIA_FIELDS
                    or not all(isinstance(value, bool) for value in criteria.values())
                    or passed is not all(criteria.values())
                ):
                    raise ValueError("confirmation result conflicts with the frozen joint rule")
                if (
                    canonical_result.get("analysis_status") != "completed"
                    or canonical_result.get("scope")
                    != "single_frozen_candidate_confirmation"
                    or canonical_result.get("alpha") != CONFIRMATION_ALPHA
                    or canonical_result.get("minimum_mean_difference") != 0.0
                    or canonical_result.get("account_currency") != "CNY"
                    or canonical_result.get("initial_account") != RESEARCH_ACCOUNT_CNY
                    or canonical_result.get("stress_slippage_bps_per_side")
                    != REQUIRED_STRESS_SLIPPAGE_BPS
                ):
                    raise ValueError("confirmation result conflicts with the frozen protocol")
                tests = canonical_result.get("tests")
                if not isinstance(tests, Mapping):
                    raise TypeError("confirmation result must contain complete tests")
                partition = _plan_partition(plan, "confirmation")
                rank_difference, rank_p = _recomputed_paired_hac_result(
                    tests.get("rank_ic"),
                    artifact="confirmation RankIC",
                    minimum_observations=math.ceil(
                        MIN_RAW_FACTOR_RANK_IC_COVERAGE
                        * len(partition["sessions"])
                    ),
                )
                strategy_difference, strategy_p = _recomputed_paired_hac_result(
                    tests.get("strategy_net"),
                    artifact="confirmation strategy net",
                    minimum_observations=MIN_PAIRED_OBSERVATIONS,
                    exact_observations=len(partition["portfolio_evaluation_sessions"]),
                )
                common = _recomputed_common_criteria(
                    tests,
                    frozen_names,
                    alpha=CONFIRMATION_ALPHA,
                    expected_folds=partition["research_folds"],
                )
                _recomputed_joint_iut(
                    tests, frozen_names, artifact="confirmation"
                )
                recomputed_criteria = {
                    "rank_ic_mean_difference_above_minimum": rank_difference > 0.0,
                    "rank_ic_one_sided_p_value_below_alpha": rank_p
                    <= CONFIRMATION_ALPHA,
                    "strategy_net_mean_difference_above_minimum": strategy_difference
                    > 0.0,
                    "strategy_net_one_sided_p_value_below_alpha": strategy_p
                    <= CONFIRMATION_ALPHA,
                    **{
                        name: common[name]
                        for name in _CONFIRMATION_CRITERIA_FIELDS
                        if name
                        not in {
                            "rank_ic_mean_difference_above_minimum",
                            "rank_ic_one_sided_p_value_below_alpha",
                            "strategy_net_mean_difference_above_minimum",
                            "strategy_net_one_sided_p_value_below_alpha",
                        }
                    },
                }
                if criteria != recomputed_criteria or passed is not all(
                    recomputed_criteria.values()
                ):
                    raise ValueError(
                        "confirmation criteria do not reconcile to complete tests"
                    )
                record["confirmation_passed"] = passed
            elif stage == "locked_holdout":
                if canonical_result.get("frozen_spec_sha256") != record["frozen_spec_sha256"]:
                    raise ValueError("locked holdout result does not match the frozen specification")
                frozen_names = state["frozen_confirmation_spec"]["selected_factor_names"]
                if canonical_result.get("candidate_factor_names") != frozen_names:
                    raise ValueError("locked holdout result factor names differ from the frozen specification")
                passed = canonical_result.get("locked_holdout_passed")
                if not isinstance(passed, bool):
                    raise ValueError("locked holdout result must record locked_holdout_passed")
                criteria = canonical_result.get("criteria")
                if (
                    not isinstance(criteria, dict)
                    or set(criteria) != _CONFIRMATION_CRITERIA_FIELDS
                    or not all(isinstance(value, bool) for value in criteria.values())
                    or passed is not all(criteria.values())
                ):
                    raise ValueError("locked holdout result conflicts with the frozen joint rule")
                if (
                    canonical_result.get("analysis_status") != "completed"
                    or canonical_result.get("scope") != "one_shot_locked_holdout_estimate"
                    or canonical_result.get("alpha") != CONFIRMATION_ALPHA
                    or canonical_result.get("minimum_mean_difference") != 0.0
                    or canonical_result.get("account_currency") != "CNY"
                    or canonical_result.get("initial_account") != RESEARCH_ACCOUNT_CNY
                    or canonical_result.get("stress_slippage_bps_per_side")
                    != REQUIRED_STRESS_SLIPPAGE_BPS
                ):
                    raise ValueError("locked holdout result conflicts with the frozen protocol")
                tests = canonical_result.get("tests")
                if not isinstance(tests, Mapping):
                    raise TypeError("locked holdout result must contain complete tests")
                partition = _plan_partition(plan, "locked_holdout")
                rank_difference, rank_p = _recomputed_paired_hac_result(
                    tests.get("rank_ic"),
                    artifact="locked holdout RankIC",
                    minimum_observations=math.ceil(
                        MIN_RAW_FACTOR_RANK_IC_COVERAGE
                        * len(partition["sessions"])
                    ),
                )
                strategy_difference, strategy_p = _recomputed_paired_hac_result(
                    tests.get("strategy_net"),
                    artifact="locked holdout strategy net",
                    minimum_observations=MIN_PAIRED_OBSERVATIONS,
                    exact_observations=len(partition["portfolio_evaluation_sessions"]),
                )
                common = _recomputed_common_criteria(
                    tests,
                    frozen_names,
                    alpha=CONFIRMATION_ALPHA,
                    expected_folds=partition["research_folds"],
                )
                _recomputed_joint_iut(
                    tests, frozen_names, artifact="locked holdout"
                )
                recomputed_criteria = {
                    "rank_ic_mean_difference_above_minimum": rank_difference > 0.0,
                    "rank_ic_one_sided_p_value_below_alpha": rank_p
                    <= CONFIRMATION_ALPHA,
                    "strategy_net_mean_difference_above_minimum": strategy_difference
                    > 0.0,
                    "strategy_net_one_sided_p_value_below_alpha": strategy_p
                    <= CONFIRMATION_ALPHA,
                    **{
                        name: common[name]
                        for name in _CONFIRMATION_CRITERIA_FIELDS
                        if name
                        not in {
                            "rank_ic_mean_difference_above_minimum",
                            "rank_ic_one_sided_p_value_below_alpha",
                            "strategy_net_mean_difference_above_minimum",
                            "strategy_net_one_sided_p_value_below_alpha",
                        }
                    },
                }
                if criteria != recomputed_criteria or passed is not all(
                    recomputed_criteria.values()
                ):
                    raise ValueError(
                        "locked holdout criteria do not reconcile to complete tests"
                    )
                record["locked_holdout_passed"] = passed
        _write_state_revision(path, state)


def evaluate_stage_once(
    state_path: Path,
    plan: Mapping[str, Any],
    stage: str,
    evaluator: Callable[[dict[str, Any]], _T],
) -> _T:
    """Claim a partition before invoking ``evaluator`` and consume it forever.

    The callback receives the declared partition.  It should predicate-read only
    those dates and return one of this module's JSON analysis objects.  A raised
    exception records ``failed`` and is re-raised; neither failed nor completed
    stages can be claimed again.
    """

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    token = _claim_stage(Path(state_path), plan, stage)
    try:
        result = evaluator(deepcopy(_plan_partition(plan, stage)))
        _finish_stage(Path(state_path), plan, stage, token, result=result)
        return result
    except BaseException as exc:
        try:
            current = read_research_state(Path(state_path), plan)
            if current["stages"][stage]["status"] == "claimed":
                _finish_stage(Path(state_path), plan, stage, token, error=exc)
        except Exception as ledger_error:
            raise RuntimeError(
                f"stage evaluation failed and its consumed state could not be finalized: {ledger_error}"
            ) from exc
        raise
