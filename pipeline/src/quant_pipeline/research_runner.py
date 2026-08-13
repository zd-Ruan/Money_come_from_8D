"""Deterministic experiment orchestration for the frozen factor protocol.

This module deliberately does not train models or inspect research artifacts on
import.  It turns a validated :mod:`factor_research` plan into JSON-friendly,
content-addressed experiment specifications and stage-bound run requests.  A
caller can therefore connect its own runner without silently widening a stage's
prediction or metric dates.

The one-shot state ledger remains the authority for opening confirmation and
the locked holdout.  ``build_stage_run_request`` is intended to be called with
the partition supplied to ``evaluate_stage_once``; its partition checks are an
audit guard, not a security boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .factor_research import (
    EVALUATION_ALIGNMENT_METHOD,
    MAX_SINGLE_ETF_ABS_CONTRIBUTION_SHARE,
    MAX_SINGLE_FOLD_ABS_INCREMENTAL_PNL_SHARE,
    MAX_STRATEGY_DRAWDOWN,
    MAX_ZERO_FILL_INTENT_RATE,
    MIN_INTENT_FILL_RATE,
    MIN_NOTIONAL_FILL_RATE,
    MIN_RESEARCH_COMPLETE_FOLDS,
    MIN_RESEARCH_FOLD_WIN_RATIO,
    RAW_SHARE_ENGINE,
    RESEARCH_ACCOUNT_CNY,
    RESEARCH_BENCHMARK,
    RESEARCH_FOLD_SIGNAL_SESSIONS,
    RESEARCH_STAGES,
    REQUIRED_STRESS_SLIPPAGE_BPS,
    load_partition_signal_metric,
    read_research_state,
    validate_research_plan,
)
from .config import json_ready_config
from .exposure import stage_exposure_fields, validate_stage_exposure_fields
from .io import read_json, sha256_file
from .metrics import evaluation_frame, max_drawdown
from .factors import (
    FACTOR_CATALOG_VERSION,
    FACTOR_FAMILIES,
    ORIGINAL_RESEARCH_CANDIDATES,
    RESEARCH_PROTOCOL,
    FactorDefinition,
    validate_factor_definitions,
)


EXPERIMENT_SPEC_SCHEMA_VERSION = 1
EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1
STAGE_RUN_REQUEST_SCHEMA_VERSION = 3
MIN_VALID_METRIC_COVERAGE = 0.90

BASELINE_EXPERIMENT_ID = "alpha158_baseline"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SPEC_ROLES = {"baseline", "family_ablation", "single_factor", "frozen_candidate"}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _seal(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    payload = _json_copy(dict(value), name="record")
    payload[digest_field] = _sha256_json(payload)
    return payload


def _verify_digest(value: Mapping[str, Any], digest_field: str, *, name: str) -> dict[str, Any]:
    payload = _json_copy(dict(value), name=name)
    digest = payload.pop(digest_field, None)
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError(f"{name} has an invalid {digest_field}")
    if _sha256_json(payload) != digest:
        raise ValueError(f"{name} {digest_field} does not match its content")
    payload[digest_field] = digest
    return payload


def select_factor_definitions_by_name(
    factor_names: Iterable[str],
) -> tuple[FactorDefinition, ...]:
    """Select exact factors by name and return them in frozen catalog order.

    Family selection is intentionally not used here: a one-factor experiment
    must never acquire the other factors in the same family by accident.
    """

    if isinstance(factor_names, (str, bytes)):
        raise TypeError("factor_names must be an iterable of individual names")
    requested = list(factor_names)
    if not requested:
        raise ValueError("at least one factor name is required")
    if not all(isinstance(name, str) and name for name in requested):
        raise ValueError("factor names must be non-empty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("factor names must be unique")

    catalog = validate_factor_definitions(ORIGINAL_RESEARCH_CANDIDATES)
    by_name = {factor.name: factor for factor in catalog}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"unknown factor names: {unknown}")
    requested_set = set(requested)
    return tuple(factor for factor in catalog if factor.name in requested_set)


def factor_config_by_name(factor_names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return exact Qlib ``(fields, names)`` lists for named candidates."""

    selected = select_factor_definitions_by_name(factor_names)
    return [factor.expression for factor in selected], [factor.name for factor in selected]


def factor_catalog_manifest_by_name(factor_names: Iterable[str]) -> dict[str, Any]:
    """Build a content-addressed catalog subset without expanding families."""

    selected = select_factor_definitions_by_name(factor_names)
    payload = {
        "catalog_version": FACTOR_CATALOG_VERSION,
        "protocol": asdict(RESEARCH_PROTOCOL),
        "families": list(dict.fromkeys(factor.family for factor in selected)),
        "factors": [asdict(factor) for factor in selected],
    }
    return _seal(payload, "sha256")


def combined_alpha158_named_feature_config(
    factor_names: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return Alpha158 followed by only the explicitly named candidates."""

    from qlib.contrib.data.loader import Alpha158DL

    alpha_fields, alpha_names = Alpha158DL.get_feature_config(
        {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {},
        }
    )
    candidate_fields, candidate_names = factor_config_by_name(factor_names)
    names = [*alpha_names, *candidate_names]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Alpha158 and named candidate features overlap: {duplicates}")
    return [*alpha_fields, *candidate_fields], names


def build_alpha158_named_factor_handler(
    *,
    factor_names: Iterable[str],
    instruments: str = "csi500",
    start_time: str | None = None,
    end_time: str | None = None,
    fit_start_time: str | None = None,
    fit_end_time: str | None = None,
    label: tuple[list[str], list[str]] | None = None,
    **handler_kwargs: Any,
):
    """Build an Alpha158 handler containing exactly the named factors."""

    from qlib.contrib.data.handler import Alpha158

    fields, names = combined_alpha158_named_feature_config(factor_names)

    class _Alpha158WithNamedCandidates(Alpha158):
        def get_feature_config(self):
            return fields, names

    kwargs: dict[str, Any] = {
        "instruments": instruments,
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        **handler_kwargs,
    }
    if label is not None:
        kwargs["label"] = label
    return _Alpha158WithNamedCandidates(**kwargs)


def _feature_selection(definitions: Sequence[FactorDefinition]) -> dict[str, Any]:
    if not definitions:
        return {
            "mode": "alpha158",
            "selection": "none",
            "families": [],
            "factor_names": [],
            "handler_factory": "qlib.contrib.data.handler.Alpha158",
        }
    return {
        "mode": "alpha158_plus_original",
        "selection": "factor_names",
        "families": list(dict.fromkeys(factor.family for factor in definitions)),
        "factor_names": [factor.name for factor in definitions],
        "handler_factory": "quant_pipeline.research_runner.build_alpha158_named_factor_handler",
    }


def _build_experiment_spec(
    plan: Mapping[str, Any],
    *,
    experiment_id: str,
    role: str,
    definitions: Sequence[FactorDefinition],
    allowed_stages: Sequence[str],
    role_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated_plan = validate_research_plan(plan)
    if role not in _SPEC_ROLES:
        raise ValueError(f"unknown experiment role: {role}")
    payload: dict[str, Any] = {
        "schema_version": EXPERIMENT_SPEC_SCHEMA_VERSION,
        "protocol_version": validated_plan["protocol_version"],
        "plan_id": validated_plan["plan_id"],
        "plan_sha256": validated_plan["plan_sha256"],
        "base_config_sha256": validated_plan["base_config_sha256"],
        "catalog_sha256": validated_plan["catalog_sha256"],
        "exposure_registry_sha256": validated_plan["exposure_provenance"][
            "registry_sha256"
        ],
        "experiment_id": experiment_id,
        "role": role,
        "status": "not_run",
        "claim_status": "specification_only",
        "allowed_stages": list(allowed_stages),
        "features": _feature_selection(definitions),
        "factor_catalog": (
            factor_catalog_manifest_by_name([factor.name for factor in definitions])
            if definitions
            else None
        ),
    }
    if role_metadata:
        payload.update(_json_copy(dict(role_metadata), name="role metadata"))
    return _seal(payload, "spec_sha256")


def build_baseline_experiment_spec(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build the unchanged Alpha158 comparator shared by all three stages."""

    return _build_experiment_spec(
        plan,
        experiment_id=BASELINE_EXPERIMENT_ID,
        role="baseline",
        definitions=(),
        allowed_stages=RESEARCH_STAGES,
    )


def _build_family_experiment_spec(plan: Mapping[str, Any], family: str) -> dict[str, Any]:
    if family not in FACTOR_FAMILIES:
        raise ValueError(f"unknown factor family: {family}")
    definitions = tuple(factor for factor in ORIGINAL_RESEARCH_CANDIDATES if factor.family == family)
    return _build_experiment_spec(
        plan,
        experiment_id=f"family_ablation__{family}",
        role="family_ablation",
        definitions=definitions,
        allowed_stages=("discovery",),
        role_metadata={"family": family},
    )


def _build_single_factor_experiment_spec(
    plan: Mapping[str, Any], factor_name: str
) -> dict[str, Any]:
    definition = select_factor_definitions_by_name([factor_name])[0]
    return _build_experiment_spec(
        plan,
        experiment_id=f"single_factor__{definition.name}",
        role="single_factor",
        definitions=(definition,),
        allowed_stages=("discovery",),
        role_metadata={
            "factor_name": definition.name,
            "family": definition.family,
            "expected_direction": definition.direction,
        },
    )


def _validate_frozen_confirmation_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("frozen_confirmation_spec must be a mapping")
    record = _json_copy(dict(value), name="frozen confirmation specification")
    if set(record) != {"selected_factor_names", "specification", "sha256"}:
        raise ValueError("frozen confirmation specification fields are invalid")
    digest = record.get("sha256")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("frozen confirmation specification has an invalid SHA-256")
    unsigned = {"selected_factor_names": record.get("selected_factor_names"), "specification": record.get("specification")}
    if _sha256_json(unsigned) != digest:
        raise ValueError("frozen confirmation specification SHA-256 does not match its content")
    names = record.get("selected_factor_names")
    if not isinstance(names, list):
        raise TypeError("frozen selected_factor_names must be a list")
    selected = select_factor_definitions_by_name(names)
    canonical_names = [factor.name for factor in selected]
    if names != canonical_names:
        raise ValueError("frozen selected_factor_names must follow catalog order")
    specification = record.get("specification")
    if not isinstance(specification, dict) or not specification:
        raise ValueError("frozen candidate specification must be a non-empty JSON object")
    return record


def build_frozen_candidate_experiment_spec(
    plan: Mapping[str, Any], frozen_confirmation_spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the sole candidate permitted in confirmation and holdout.

    ``frozen_confirmation_spec`` is the record stored by
    ``freeze_confirmation_spec`` in the one-shot research state, not an ad-hoc
    list assembled after confirmation results are known.
    """

    frozen = _validate_frozen_confirmation_spec(frozen_confirmation_spec)
    definitions = select_factor_definitions_by_name(frozen["selected_factor_names"])
    return _build_experiment_spec(
        plan,
        experiment_id=f"frozen_candidate__{frozen['sha256'][:16]}",
        role="frozen_candidate",
        definitions=definitions,
        allowed_stages=("confirmation", "locked_holdout"),
        role_metadata={"frozen_confirmation_spec": frozen},
    )


def validate_experiment_spec(
    plan: Mapping[str, Any], experiment_spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an experiment spec against the frozen plan and catalog."""

    if not isinstance(experiment_spec, Mapping):
        raise TypeError("experiment_spec must be a mapping")
    value = _verify_digest(experiment_spec, "spec_sha256", name="experiment specification")
    validated_plan = validate_research_plan(plan)
    if value.get("schema_version") != EXPERIMENT_SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported experiment specification schema")
    if value.get("plan_sha256") != validated_plan["plan_sha256"]:
        raise ValueError("experiment specification belongs to a different research plan")
    if value.get("catalog_sha256") != validated_plan["catalog_sha256"]:
        raise ValueError("experiment specification catalog differs from the research plan")
    if value.get("exposure_registry_sha256") != validated_plan[
        "exposure_provenance"
    ]["registry_sha256"]:
        raise ValueError("experiment specification exposure registry differs from the plan")
    if value.get("status") != "not_run" or value.get("claim_status") != "specification_only":
        raise ValueError("experiment specification must not claim a completed run")

    role = value.get("role")
    if role == "baseline":
        expected = build_baseline_experiment_spec(validated_plan)
    elif role == "family_ablation":
        expected = _build_family_experiment_spec(validated_plan, value.get("family"))
    elif role == "single_factor":
        expected = _build_single_factor_experiment_spec(validated_plan, value.get("factor_name"))
    elif role == "frozen_candidate":
        expected = build_frozen_candidate_experiment_spec(
            validated_plan, value.get("frozen_confirmation_spec")
        )
    else:
        raise ValueError(f"unknown experiment role: {role}")
    if value != expected:
        raise ValueError("experiment specification does not match its declared frozen role")
    return value


def build_discovery_experiment_specs(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build the 1 baseline + 5 family + 18 single-factor discovery battery."""

    validated_plan = validate_research_plan(plan)
    family_specs = [_build_family_experiment_spec(validated_plan, family) for family in FACTOR_FAMILIES]
    single_specs = [
        _build_single_factor_experiment_spec(validated_plan, factor.name)
        for factor in ORIGINAL_RESEARCH_CANDIDATES
    ]
    payload = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "protocol_version": validated_plan["protocol_version"],
        "plan_id": validated_plan["plan_id"],
        "plan_sha256": validated_plan["plan_sha256"],
        "catalog_sha256": validated_plan["catalog_sha256"],
        **stage_exposure_fields(validated_plan, "discovery"),
        "status": "not_run",
        "claim_status": "pre_registered_specs_only",
        "stage": "discovery",
        "experiment_count": 1 + len(family_specs) + len(single_specs),
        "baseline": build_baseline_experiment_spec(validated_plan),
        "family_ablations": family_specs,
        "single_factor_tests": single_specs,
    }
    return _seal(payload, "manifest_sha256")


def validate_discovery_experiment_specs(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("discovery experiment manifest must be a mapping")
    value = _verify_digest(manifest, "manifest_sha256", name="discovery experiment manifest")
    expected = build_discovery_experiment_specs(plan)
    if value != expected:
        raise ValueError("discovery experiment manifest differs from the frozen plan")
    return value


def build_research_experiment_manifest(
    plan: Mapping[str, Any],
    frozen_confirmation_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one self-contained CLI-friendly manifest for orchestration.

    Before discovery is consumed, ``frozen_candidate`` is explicitly ``None``.
    Supplying the ledger's frozen record adds the unchanged confirmation and
    holdout candidate while every spec remains marked ``not_run``.
    """

    validated_plan = validate_research_plan(plan)
    discovery = build_discovery_experiment_specs(validated_plan)
    candidate = (
        build_frozen_candidate_experiment_spec(validated_plan, frozen_confirmation_spec)
        if frozen_confirmation_spec is not None
        else None
    )
    payload = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "protocol_version": discovery["protocol_version"],
        "plan_id": discovery["plan_id"],
        "plan_sha256": discovery["plan_sha256"],
        "catalog_sha256": discovery["catalog_sha256"],
        "exposure_registry_sha256": validated_plan["exposure_provenance"][
            "registry_sha256"
        ],
        "status": "not_run",
        "claim_status": "specifications_only",
        "discovery": discovery,
        "frozen_candidate_status": "frozen_not_run" if candidate else "awaiting_discovery_freeze",
        "frozen_candidate": candidate,
    }
    return _seal(payload, "manifest_sha256")


def _claimed_plan_partition(
    plan: Mapping[str, Any], claimed_partition: Mapping[str, Any]
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    validated_plan = validate_research_plan(plan)
    if not isinstance(claimed_partition, Mapping):
        raise TypeError("claimed_partition must be the mapping supplied by evaluate_stage_once")
    supplied = _json_copy(dict(claimed_partition), name="claimed partition")
    stage = supplied.get("name")
    if stage not in RESEARCH_STAGES:
        raise ValueError(f"unknown claimed research stage: {stage}")
    expected = validated_plan["partitions"][stage]
    if supplied != expected:
        raise ValueError("claimed partition does not exactly match the frozen research plan")
    return validated_plan, stage, expected


def build_stage_run_request(
    plan: Mapping[str, Any],
    experiment_spec: Mapping[str, Any],
    claimed_partition: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one experiment to exactly one ledger-opened stage partition."""

    validated_plan, stage, partition = _claimed_plan_partition(plan, claimed_partition)
    spec = validate_experiment_spec(validated_plan, experiment_spec)
    if stage not in spec["allowed_stages"]:
        raise ValueError(f"{spec['role']} experiment is not allowed in {stage}")
    stage_position = RESEARCH_STAGES.index(stage)
    discovery_end = validated_plan["partitions"]["discovery"]["end"]
    exposure = stage_exposure_fields(validated_plan, stage)
    payload = {
        "schema_version": STAGE_RUN_REQUEST_SCHEMA_VERSION,
        "protocol_version": validated_plan["protocol_version"],
        "plan_id": validated_plan["plan_id"],
        "plan_sha256": validated_plan["plan_sha256"],
        "status": "not_run",
        "claim_status": "stage_bound_request_only",
        "stage": stage,
        **exposure,
        "experiment": spec,
        "partition": deepcopy(partition),
        "training_contract": {
            "method": "purged_expanding_walk_forward",
            "source_data_end_not_after": partition["source_data_end"],
            "label_horizon_bars": partition["label_horizon_bars"],
            "label_maturity_sessions": deepcopy(partition["label_maturity_sessions"]),
            "label_maturity_sessions_sha256": partition[
                "label_maturity_sessions_sha256"
            ],
            "prediction_start": partition["start"],
            "prediction_end": partition["end"],
            "prediction_sessions": deepcopy(partition["sessions"]),
            "prediction_sessions_sha256": partition["sessions_sha256"],
            "feature_selection_and_hyperparameter_tuning_date_not_after": discovery_end,
            "each_training_label_must_mature_before_its_prediction_fold": True,
            "later_partition_training_forbidden": list(RESEARCH_STAGES[stage_position + 1 :]),
            "frozen_hyperparameters_required": stage != "discovery",
            "fold_purge_and_label_maturity_checks_required": True,
        },
        "metric_contract": {
            "signal": {
                "artifact": "signal_metrics.parquet",
                "format": "parquet",
                "column": "rank_ic",
                "datetime_column": "datetime",
                "read_mode": "pyarrow_predicate_pushdown_exact_sessions",
                "start": partition["start"],
                "end": partition["end"],
                "sessions": deepcopy(partition["sessions"]),
                "sessions_sha256": partition["sessions_sha256"],
                "raw_factor_artifact": "raw_factor_metrics.parquet",
                "raw_factor_column_suffix": "__rank_ic",
                "expected_direction_required": True,
            },
            "portfolio": {
                "artifact": (
                    f"backtests/slippage_{REQUIRED_STRESS_SLIPPAGE_BPS:02d}bps/"
                    "report.parquet"
                ),
                "summary_artifact": (
                    f"backtests/slippage_{REQUIRED_STRESS_SLIPPAGE_BPS:02d}bps/"
                    "summary.json"
                ),
                "attribution_artifact": (
                    f"backtests/slippage_{REQUIRED_STRESS_SLIPPAGE_BPS:02d}bps/"
                    "symbol_attribution.parquet"
                ),
                "format": "raw_share_report_parquet",
                "daily_metric": "strategy_net",
                "engine": RAW_SHARE_ENGINE,
                "account_currency": "CNY",
                "benchmark": RESEARCH_BENCHMARK,
                "initial_account": RESEARCH_ACCOUNT_CNY,
                "stress_slippage_bps_per_side": REQUIRED_STRESS_SLIPPAGE_BPS,
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
                "research_folds": deepcopy(partition["research_folds"]),
                "alignment_method": EVALUATION_ALIGNMENT_METHOD,
                "raw_report_start": partition["portfolio_raw_report_start"],
                "raw_report_end": partition["portfolio_raw_report_end"],
                "raw_report_sessions": deepcopy(
                    partition["portfolio_raw_report_sessions"]
                ),
                "raw_report_sessions_sha256": partition[
                    "portfolio_raw_report_sessions_sha256"
                ],
                "evaluation_start": partition["portfolio_evaluation_start"],
                "evaluation_end": partition["portfolio_evaluation_end"],
                "evaluation_sessions": deepcopy(
                    partition["portfolio_evaluation_sessions"]
                ),
                "evaluation_sessions_sha256": partition[
                    "portfolio_evaluation_sessions_sha256"
                ],
            },
            "later_partition_metric_access_forbidden": list(RESEARCH_STAGES[stage_position + 1 :]),
        },
    }
    return _seal(payload, "request_sha256")


def prepare_stage_pipeline_config(
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the only pipeline configuration authorized by a stage request.

    The caller must supply the unchanged pre-registered base configuration.
    Stage dates and feature selection are taken exclusively from the signed
    request.  The frozen ``data.end_date`` remains the complete provider
    snapshot boundary required by the data audit.  ``_research_stage`` carries
    the stricter source-data upper bound enforced by :func:`run_pipeline`;
    ``test_start_date`` is the first authorized signal session, so the rolling
    runner cannot emit an earlier test.
    """

    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    if "_research_stage" in base_config:
        raise ValueError("_research_stage is reserved for a derived stage-bound config")
    validated_plan = validate_research_plan(plan)
    validated_request = validate_stage_run_request(validated_plan, request)
    config = deepcopy(dict(base_config))
    actual_digest = _sha256_json(json_ready_config(config))
    if actual_digest != validated_plan["base_config_sha256"]:
        raise ValueError("pipeline configuration differs from the pre-registered base config")

    partition = validated_request["partition"]
    configured_horizon = config.get("data", {}).get("label_horizon_bars")
    if (
        isinstance(configured_horizon, bool)
        or not isinstance(configured_horizon, int)
        or configured_horizon != partition["label_horizon_bars"]
    ):
        raise ValueError("pipeline label horizon differs from the research plan")
    if pd.Timestamp(config["data"]["start_date"]) > pd.Timestamp(partition["start"]):
        raise ValueError("pipeline training history starts after the requested stage")
    if pd.Timestamp(config["data"]["end_date"]) < pd.Timestamp(partition["source_data_end"]):
        raise ValueError("pipeline provider snapshot ends before the requested label maturity")
    evidence = validated_plan["execution_evidence"]
    execution = config.get("execution", {})
    gates = config.get("gates", {})
    if float(execution.get("account", float("nan"))) != evidence["initial_account"]:
        raise ValueError("pipeline account differs from the frozen CNY 20,000 research account")
    if config.get("data", {}).get("benchmark") != evidence["benchmark"]:
        raise ValueError("pipeline benchmark differs from the frozen research benchmark")
    required_stress = evidence["required_stress_slippage_bps_per_side"]
    if int(gates.get("required_stress_slippage_bps", -1)) != required_stress:
        raise ValueError("pipeline required stress slippage differs from the research plan")
    configured_stress = execution.get("stress_slippage_bps_per_side", [])
    if required_stress not in configured_stress:
        raise ValueError("pipeline does not produce the required stress-slippage ledger")
    frozen_gate_values = {
        "research_fold_days": evidence["research_fold_signal_sessions"],
        "min_research_fold_win_ratio": evidence["minimum_research_fold_win_ratio"],
        "max_strategy_drawdown": evidence["maximum_strategy_drawdown"],
        "max_single_etf_abs_contribution_share": evidence[
            "maximum_single_etf_abs_contribution_share"
        ],
        "max_single_fold_abs_incremental_pnl_share": evidence[
            "maximum_single_fold_abs_incremental_pnl_share"
        ],
    }
    for key, expected in frozen_gate_values.items():
        actual = gates.get(key)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or float(actual) != float(expected):
            raise ValueError(f"pipeline {key} differs from the frozen research plan")

    config["data"]["test_start_date"] = partition["start"]
    config["_research_stage"] = {
        "stage": validated_request["stage"],
        "request_sha256": validated_request["request_sha256"],
        "prediction_end": partition["end"],
        "source_data_end": partition["source_data_end"],
        **stage_exposure_fields(validated_plan, validated_request["stage"]),
    }
    features = validated_request["experiment"]["features"]
    config["features"] = {
        "mode": features["mode"],
        "families": [],
        "factor_names": deepcopy(features["factor_names"]),
    }
    return config, validated_request


def enforce_valid_metric_coverage(
    values: pd.Series,
    *,
    experiment_id: str,
    minimum: float = MIN_VALID_METRIC_COVERAGE,
) -> dict[str, Any]:
    """Fail closed when a stage metric has too few finite daily observations."""

    if not isinstance(values, pd.Series) or values.empty:
        raise ValueError(f"{experiment_id} metric must be a non-empty pandas Series")
    threshold = float(minimum)
    if not 0.90 <= threshold <= 1.0:
        raise ValueError("minimum valid metric coverage cannot be below 0.90")
    numeric = pd.to_numeric(values, errors="coerce")
    finite_count = int(
        numeric.map(lambda value: pd.notna(value) and math.isfinite(float(value))).sum()
    )
    coverage = finite_count / len(numeric)
    if coverage < threshold:
        raise RuntimeError(
            f"{experiment_id} valid metric coverage {coverage:.2%} is below {threshold:.2%}"
        )
    return {
        "experiment_id": experiment_id,
        "observations": len(numeric),
        "finite_observations": finite_count,
        "valid_metric_coverage": coverage,
        "minimum_valid_metric_coverage": threshold,
        "passed": True,
    }


def _expected_research_control(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": request["protocol_version"],
        "plan_id": request["plan_id"],
        "plan_sha256": request["plan_sha256"],
        "request_sha256": request["request_sha256"],
        "stage": request["stage"],
        "experiment_id": request["experiment"]["experiment_id"],
        "experiment_spec_sha256": request["experiment"]["spec_sha256"],
        "partition_sha256": request["partition"]["sessions_sha256"],
        "portfolio_evaluation_sessions_sha256": request["partition"][
            "portfolio_evaluation_sessions_sha256"
        ],
        "label_maturity_sessions_sha256": request["partition"][
            "label_maturity_sessions_sha256"
        ],
        "source_data_end": request["partition"]["source_data_end"],
        "exposure_registry_sha256": request["exposure_registry_sha256"],
        "evidence_class": request["evidence_class"],
        "claim_classification": request["claim_classification"],
    }


def load_completed_stage_signal_metric(
    run_dir: Path,
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    metric: str = "rank_ic",
) -> tuple[pd.Series, tuple[str, str, str]]:
    """Verify one completed stage run and load only its authorized metric dates.

    The returned identity is ``(source_fingerprint, snapshot_id,
    runtime_code_sha256)`` and must match across every run in a stage battery.
    """

    from .comparison import _code_identity, _source_identity, _verify_run_integrity

    root = Path(run_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"completed research run is missing or unsafe: {root}")
    validated_request = validate_stage_run_request(plan, request)
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError(f"research run {root.name} is not completed")
    if manifest.get("run_id") != root.name:
        raise ValueError("research run directory and manifest run_id differ")
    if manifest.get("research") != _expected_research_control(validated_request):
        raise ValueError("completed run research identity differs from its stage request")
    recorded_request = read_json(root / "research_request.json")
    if recorded_request != validated_request:
        raise ValueError("completed run did not preserve the exact stage request")

    expected_config, _ = prepare_stage_pipeline_config(base_config, plan, validated_request)
    recorded_config = read_json(root / "config.json")
    if recorded_config != json_ready_config(expected_config):
        raise ValueError("completed run configuration differs from the stage-bound configuration")

    _verify_run_integrity(root, manifest, "research")
    integrity = manifest["integrity"]
    seal_name = integrity.get("seal_manifest")
    if not isinstance(seal_name, str) or not seal_name:
        raise ValueError("research run manifest is missing its detached integrity seal")
    checksum_path = root / integrity["checksum_manifest"]
    if sha256_file(checksum_path) != integrity["checksum_sha256"]:
        raise ValueError("research run checksum manifest digest is invalid")
    checksum = read_json(checksum_path)
    records = checksum.get("artifacts") if isinstance(checksum, dict) else None
    matches = [
        record
        for record in records or []
        if isinstance(record, dict) and record.get("path") == "signal_metrics.parquet"
    ]
    if len(matches) != 1:
        raise ValueError("research run does not uniquely checksum signal_metrics.parquet")
    expected_sha256 = matches[0].get("sha256")
    values = load_stage_signal_metric(
        root / "signal_metrics.parquet",
        plan,
        validated_request,
        metric=metric,
        expected_sha256=expected_sha256,
    )
    source_fingerprint, snapshot_id = _source_identity(manifest)
    identity = (source_fingerprint, snapshot_id, _code_identity(manifest, "research"))
    return values, identity


def _checksum_artifact_sha256(
    records: Any, relative_path: str, *, artifact: str
) -> str:
    matches = [
        record
        for record in records or []
        if isinstance(record, dict) and record.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"research run does not uniquely checksum {relative_path}")
    digest = matches[0].get("sha256")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError(f"research run checksum for {artifact} is invalid")
    return digest


def _sealed_artifact_path(
    root: Path,
    records: Any,
    relative_path: str,
    *,
    artifact: str,
) -> tuple[Path, str]:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or Path(relative_path).is_absolute()
    ):
        raise ValueError(f"{artifact} path is invalid")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{artifact} resolves outside the completed run") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{artifact} is missing or unsafe")
    expected_sha256 = _checksum_artifact_sha256(
        records, relative_path, artifact=artifact
    )
    if sha256_file(path) != expected_sha256:
        raise ValueError(
            f"{artifact} SHA-256 does not match the verified checksum manifest"
        )
    return path, expected_sha256


def _finite_number(value: Any, *, artifact: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{artifact} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{artifact} must be a finite number")
    return result


def _finite_integer(value: Any, *, artifact: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{artifact} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{artifact} must be a non-negative integer")
    return result


def _assert_close(
    actual: Any,
    expected: float,
    *,
    artifact: str,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-8,
) -> float:
    value = _finite_number(actual, artifact=artifact)
    if not math.isclose(value, float(expected), rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(f"{artifact} does not reconcile to sealed portfolio data")
    return value


def _read_sealed_parquet(
    root: Path,
    records: Any,
    relative_path: str,
    *,
    artifact: str,
) -> tuple[pd.DataFrame, str]:
    path, digest = _sealed_artifact_path(
        root, records, relative_path, artifact=artifact
    )
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise ValueError(f"could not read {artifact}") from exc
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{artifact} must contain a dataframe")
    return frame, digest


def _portfolio_report_evidence(
    report: pd.DataFrame,
    *,
    raw_sessions: Sequence[str],
    evaluation_sessions: Sequence[str],
    initial_account: float,
    artifact: str,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    try:
        actual_index = pd.DatetimeIndex(report.index, name="datetime")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{artifact} index must be datetime-like") from exc
    expected_raw = pd.DatetimeIndex(raw_sessions, name="datetime")
    if (
        actual_index.tz is not None
        or actual_index.has_duplicates
        or not actual_index.is_monotonic_increasing
        or not actual_index.equals(expected_raw)
    ):
        raise ValueError(f"{artifact} dates must exactly equal the portfolio contract")
    required_columns = ["return", "cost", "bench", "account"]
    missing = set(required_columns) - set(report.columns)
    if missing:
        raise ValueError(f"{artifact} is missing columns {sorted(missing)}")
    numeric = report[required_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{artifact} contains non-finite portfolio data")
    report_net = numeric["return"] - numeric["cost"]
    if (report_net <= -1.0).any() or (numeric["bench"] <= -1.0).any():
        raise ValueError(f"{artifact} contains returns that make wealth non-positive")
    expected_accounts = float(initial_account) * (1.0 + report_net).cumprod()
    if (numeric["account"] <= 0.0).any() or not np.allclose(
        numeric["account"].to_numpy(dtype=float),
        expected_accounts.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError(f"{artifact} account path does not reconcile to daily net returns")

    normalized_report = report.copy()
    normalized_report[required_columns] = numeric
    try:
        aligned = evaluation_frame(normalized_report)
    except Exception as exc:
        raise ValueError(f"{artifact} cannot be aligned to realizable returns") from exc
    expected_evaluation = pd.DatetimeIndex(evaluation_sessions, name="datetime")
    aligned.index = pd.DatetimeIndex(aligned.index, name="datetime")
    if not aligned.index.equals(expected_evaluation):
        raise ValueError(f"{artifact} evaluation dates differ from the portfolio contract")
    strategy_net = pd.to_numeric(aligned["strategy_net"], errors="coerce").astype(float)
    benchmark = pd.to_numeric(aligned["benchmark"], errors="coerce").astype(float)
    if (
        not np.isfinite(strategy_net.to_numpy(dtype=float)).all()
        or not np.isfinite(benchmark.to_numpy(dtype=float)).all()
        or (strategy_net <= -1.0).any()
        or (benchmark <= -1.0).any()
    ):
        raise ValueError(f"{artifact} contains invalid aligned returns")
    terminal_account = float(numeric["account"].iloc[-1])
    terminal_from_returns = float(initial_account) * float((1.0 + strategy_net).prod())
    if not math.isclose(
        terminal_from_returns, terminal_account, rel_tol=1e-10, abs_tol=1e-6
    ):
        raise ValueError(f"{artifact} terminal account does not reconcile to strategy_net")
    benchmark_terminal = float(initial_account) * float((1.0 + benchmark).prod())
    return (
        strategy_net.rename("strategy_net"),
        benchmark.rename("benchmark"),
        {
            "terminal_account": terminal_account,
            "benchmark_terminal_account": benchmark_terminal,
            "strategy_max_drawdown": float(max_drawdown(strategy_net)),
        },
    )


def _execution_quality(summary: Mapping[str, Any], *, artifact: str) -> dict[str, float]:
    execution = summary.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"{artifact} is missing its execution ledger summary")
    intent_count = _finite_integer(
        execution.get("intent_count"), artifact=f"{artifact} intent_count"
    )
    filled_count = _finite_integer(
        execution.get("filled_intent_count"),
        artifact=f"{artifact} filled_intent_count",
    )
    zero_count = _finite_integer(
        execution.get("zero_fill_intent_count"),
        artifact=f"{artifact} zero_fill_intent_count",
    )
    if filled_count + zero_count != intent_count:
        raise ValueError(f"{artifact} execution intent counts do not reconcile")
    target_notional = _finite_number(
        execution.get("target_notional"),
        artifact=f"{artifact} target_notional",
        minimum=0.0,
    )
    fill_notional = _finite_number(
        execution.get("fill_notional"),
        artifact=f"{artifact} fill_notional",
        minimum=0.0,
    )
    intent_fill_rate = filled_count / intent_count if intent_count else 0.0
    zero_fill_intent_rate = zero_count / intent_count if intent_count else 0.0
    notional_fill_rate = fill_notional / target_notional if target_notional > 0.0 else 0.0
    if not 0.0 <= notional_fill_rate <= 1.0 + 1e-10:
        raise ValueError(f"{artifact} notional fill rate is outside zero and one")
    notional_fill_rate = min(notional_fill_rate, 1.0)
    recomputed = {
        "intent_fill_rate": intent_fill_rate,
        "zero_fill_intent_rate": zero_fill_intent_rate,
        "notional_fill_rate": notional_fill_rate,
    }
    for field, expected in recomputed.items():
        _assert_close(
            execution.get(field),
            expected,
            artifact=f"{artifact} execution {field}",
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    for field in ("fill_rate", "notional_fill_rate"):
        if field in summary:
            _assert_close(
                summary.get(field),
                notional_fill_rate,
                artifact=f"{artifact} {field}",
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
    return recomputed


def _attribution_evidence(
    attribution: pd.DataFrame,
    report: pd.DataFrame,
    *,
    raw_sessions: Sequence[str],
    initial_account: float,
    artifact: str,
) -> dict[str, Any]:
    required = {"date", "symbol", "net_pnl"}
    missing = required - set(attribution.columns)
    if missing:
        raise ValueError(f"{artifact} is missing columns {sorted(missing)}")
    values = attribution.copy()
    try:
        values["date"] = pd.to_datetime(values["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{artifact} dates must be datetime-like") from exc
    expected_dates = pd.DatetimeIndex(raw_sessions)
    dates = pd.DatetimeIndex(values["date"])
    if dates.tz is not None or not dates.isin(expected_dates).all():
        raise ValueError(f"{artifact} contains dates outside the portfolio contract")
    if values.duplicated(["date", "symbol"]).any():
        raise ValueError(f"{artifact} contains duplicate date-symbol rows")
    symbols = values["symbol"]
    if symbols.isna().any() or (symbols.astype(str).str.strip() == "").any():
        raise ValueError(f"{artifact} contains invalid symbols")
    net_pnl = pd.to_numeric(values["net_pnl"], errors="coerce").astype(float)
    if not np.isfinite(net_pnl.to_numpy(dtype=float)).all():
        raise ValueError(f"{artifact} contains non-finite net P&L")
    if "net_pnl_cny" in values:
        net_pnl_cny = pd.to_numeric(values["net_pnl_cny"], errors="coerce")
        if not np.isfinite(net_pnl_cny.to_numpy(dtype=float)).all() or not np.allclose(
            net_pnl_cny.to_numpy(dtype=float),
            net_pnl.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{artifact} net_pnl and net_pnl_cny differ")
    values["net_pnl"] = net_pnl
    attributed_by_date = (
        values.groupby("date", sort=True)["net_pnl"]
        .sum()
        .reindex(expected_dates, fill_value=0.0)
    )
    account = pd.to_numeric(report["account"], errors="coerce").astype(float)
    account_change = account.diff()
    account_change.iloc[0] = float(account.iloc[0]) - float(initial_account)
    if not np.allclose(
        attributed_by_date.to_numpy(dtype=float),
        account_change.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-8,
    ):
        raise ValueError(f"{artifact} net P&L does not reconcile to account NAV")
    by_symbol = values.groupby(symbols.astype(str), sort=True)["net_pnl"].apply(
        lambda series: float(series.abs().sum())
    )
    denominator = float(by_symbol.sum())
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError(f"{artifact} has zero gross absolute P&L concentration denominator")
    symbol = str(by_symbol.idxmax())
    numerator = float(by_symbol.loc[symbol])
    return {
        "single_etf_abs_contribution_share": numerator / denominator,
        "single_etf_abs_contribution_symbol": symbol,
        "single_etf_abs_contribution_numerator_cny": numerator,
        "single_etf_abs_contribution_denominator_cny": denominator,
    }


def _validate_declared_concentration(
    summary: Mapping[str, Any], concentration: Mapping[str, Any], *, artifact: str
) -> None:
    share = float(concentration["single_etf_abs_contribution_share"])
    for source, field in (
        (summary, "single_etf_abs_contribution_share"),
        (summary.get("execution", {}), "max_single_etf_gross_abs_contribution_share"),
    ):
        if not isinstance(source, Mapping) or field not in source:
            raise ValueError(f"{artifact} is missing declared concentration {field}")
        _assert_close(
            source.get(field),
            share,
            artifact=f"{artifact} {field}",
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    for source_name, source in (
        ("summary", summary.get("symbol_attribution_concentration")),
        ("execution", summary.get("execution", {}).get("symbol_attribution_concentration")),
    ):
        if not isinstance(source, Mapping):
            raise ValueError(f"{artifact} {source_name} concentration is missing")
        gross = source.get("gross_abs")
        if not isinstance(gross, Mapping):
            raise ValueError(f"{artifact} {source_name} concentration is invalid")
        if gross.get("symbol") != concentration["single_etf_abs_contribution_symbol"]:
            raise ValueError(f"{artifact} {source_name} concentration symbol differs")
        for field, expected in (
            ("share", share),
            ("numerator_cny", concentration["single_etf_abs_contribution_numerator_cny"]),
            ("denominator_cny", concentration["single_etf_abs_contribution_denominator_cny"]),
        ):
            _assert_close(
                gross.get(field),
                float(expected),
                artifact=f"{artifact} {source_name} concentration {field}",
                rel_tol=1e-10,
                abs_tol=1e-8,
            )


def _load_portfolio_bundle(
    root: Path,
    records: Any,
    *,
    report_relative: str,
    summary_relative: str,
    attribution_relative: str,
    raw_sessions: Sequence[str],
    evaluation_sessions: Sequence[str],
    initial_account: float,
    stress_slippage_bps: int,
    engine: str,
    alignment_method: str,
    expected_summary: Mapping[str, Any],
    artifact: str,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    report, report_sha256 = _read_sealed_parquet(
        root, records, report_relative, artifact=f"{artifact} report"
    )
    summary_path, summary_sha256 = _sealed_artifact_path(
        root, records, summary_relative, artifact=f"{artifact} summary"
    )
    try:
        summary = read_json(summary_path)
    except Exception as exc:
        raise ValueError(f"could not read {artifact} summary") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"{artifact} summary must contain a JSON object")
    attribution, attribution_sha256 = _read_sealed_parquet(
        root, records, attribution_relative, artifact=f"{artifact} attribution"
    )
    strategy_net, benchmark, portfolio = _portfolio_report_evidence(
        report,
        raw_sessions=raw_sessions,
        evaluation_sessions=evaluation_sessions,
        initial_account=initial_account,
        artifact=f"{artifact} report",
    )
    for key, expected in expected_summary.items():
        actual = summary.get(key)
        if isinstance(expected, float):
            _assert_close(actual, expected, artifact=f"{artifact} summary {key}")
        elif actual != expected:
            raise ValueError(f"{artifact} summary {key} differs from the portfolio contract")
    for key, expected in (
        ("slippage_bps_per_side", stress_slippage_bps),
        ("alignment_method", alignment_method),
        ("raw_execution_days", len(raw_sessions)),
        ("days", len(evaluation_sessions)),
        ("terminal_account", portfolio["terminal_account"]),
    ):
        actual = summary.get(key)
        if isinstance(expected, (float, np.floating)):
            _assert_close(actual, float(expected), artifact=f"{artifact} summary {key}")
        elif actual != expected:
            raise ValueError(f"{artifact} summary {key} differs from the portfolio contract")
    execution = summary.get("execution")
    if not isinstance(execution, Mapping) or execution.get("engine") != engine:
        raise ValueError(f"{artifact} summary does not identify the raw-share engine")
    if execution.get("nav_reconciled") is not True:
        raise ValueError(f"{artifact} summary does not record successful NAV reconciliation")
    _assert_close(
        execution.get("initial_account"),
        initial_account,
        artifact=f"{artifact} execution initial_account",
    )
    _assert_close(
        execution.get("final_account"),
        portfolio["terminal_account"],
        artifact=f"{artifact} execution final_account",
        abs_tol=1e-6,
    )
    execution_config = execution.get("config")
    if not isinstance(execution_config, Mapping):
        raise ValueError(f"{artifact} execution is missing its raw-share configuration")
    _assert_close(
        execution_config.get("initial_cash"),
        initial_account,
        artifact=f"{artifact} execution initial_cash",
    )
    _assert_close(
        execution_config.get("slippage_bps_per_side"),
        stress_slippage_bps,
        artifact=f"{artifact} execution slippage_bps_per_side",
    )
    quality = _execution_quality(summary, artifact=f"{artifact} summary")
    concentration = _attribution_evidence(
        attribution,
        report,
        raw_sessions=raw_sessions,
        initial_account=initial_account,
        artifact=f"{artifact} attribution",
    )
    _validate_declared_concentration(summary, concentration, artifact=artifact)
    for field in ("strategy_max_drawdown", "max_drawdown"):
        if field in summary:
            _assert_close(
                summary.get(field),
                portfolio["strategy_max_drawdown"],
                artifact=f"{artifact} summary {field}",
                abs_tol=1e-12,
            )
    benchmark_declared = summary.get("benchmark_terminal_account")
    if benchmark_declared is not None:
        _assert_close(
            benchmark_declared,
            portfolio["benchmark_terminal_account"],
            artifact=f"{artifact} summary benchmark_terminal_account",
            abs_tol=1e-6,
        )
    for field, expected in (
        ("net_cumulative_return", portfolio["terminal_account"] / initial_account - 1.0),
        (
            "benchmark_cumulative_return",
            portfolio["benchmark_terminal_account"] / initial_account - 1.0,
        ),
    ):
        _assert_close(
            summary.get(field),
            expected,
            artifact=f"{artifact} summary {field}",
            abs_tol=1e-12,
        )
    if "fold" in expected_summary:
        for field, expected in (
            ("terminal_account_value", portfolio["terminal_account"]),
            (
                "benchmark_terminal_account_value",
                portfolio["benchmark_terminal_account"],
            ),
        ):
            _assert_close(
                summary.get(field),
                expected,
                artifact=f"{artifact} summary {field}",
                abs_tol=1e-6,
            )
    portfolio.update(quality)
    portfolio.update(concentration)
    portfolio.update(
        {
            "report_sha256": report_sha256,
            "summary_sha256": summary_sha256,
            "attribution_sha256": attribution_sha256,
        }
    )
    return strategy_net, benchmark, portfolio


def _load_research_folds(
    root: Path,
    records: Any,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    expected_folds = contract.get("research_folds")
    if not isinstance(expected_folds, list) or not expected_folds:
        raise ValueError("portfolio contract has no research folds")
    for expected in expected_folds:
        if not isinstance(expected, Mapping):
            raise ValueError("portfolio contract contains an invalid research fold")
        fold = expected.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold != len(folds) + 1:
            raise ValueError("portfolio research folds must be numbered consecutively")
        prefix = f"folds/research_fold_{fold:02d}/backtest"
        _, _, calculated = _load_portfolio_bundle(
            root,
            records,
            report_relative=f"{prefix}/report.parquet",
            summary_relative=f"{prefix}/summary.json",
            attribution_relative=f"{prefix}/symbol_attribution.parquet",
            raw_sessions=expected["raw_report_sessions"],
            evaluation_sessions=expected["evaluation_sessions"],
            initial_account=float(contract["initial_account"]),
            stress_slippage_bps=int(contract["stress_slippage_bps_per_side"]),
            engine=str(contract["engine"]),
            alignment_method=str(contract["alignment_method"]),
            expected_summary={
                "fold": fold,
                "signal_start": expected["signal_start"],
                "signal_end": expected["signal_end"],
                "signal_observations": expected["signal_observations"],
                "signal_sessions": expected["signal_sessions"],
                "raw_report_start": expected["raw_report_start"],
                "raw_report_end": expected["raw_report_end"],
                "raw_report_sessions": expected["raw_report_sessions"],
                "evaluation_start": expected["evaluation_start"],
                "evaluation_end": expected["evaluation_end"],
                "evaluation_sessions": expected["evaluation_sessions"],
                "complete_for_gate": expected["complete_for_gate"],
                "initial_account_value": float(contract["initial_account"]),
            },
            artifact=f"research fold {fold}",
        )
        folds.append(
            {
                "fold": fold,
                "signal_start": expected["signal_start"],
                "signal_end": expected["signal_end"],
                "evaluation_start": expected["evaluation_start"],
                "evaluation_end": expected["evaluation_end"],
                "complete_for_gate": expected["complete_for_gate"],
                "initial_account": float(contract["initial_account"]),
                "stress_slippage_bps_per_side": int(
                    contract["stress_slippage_bps_per_side"]
                ),
                "engine": contract["engine"],
                **calculated,
            }
        )
    return folds


def _load_stage_portfolio_evidence(
    root: Path,
    records: Any,
    request: Mapping[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    contract = request["metric_contract"]["portfolio"]
    strategy_net, benchmark, portfolio = _load_portfolio_bundle(
        root,
        records,
        report_relative=contract["artifact"],
        summary_relative=contract["summary_artifact"],
        attribution_relative=contract["attribution_artifact"],
        raw_sessions=contract["raw_report_sessions"],
        evaluation_sessions=contract["evaluation_sessions"],
        initial_account=float(contract["initial_account"]),
        stress_slippage_bps=int(contract["stress_slippage_bps_per_side"]),
        engine=str(contract["engine"]),
        alignment_method=str(contract["alignment_method"]),
        expected_summary={
            "initial_execution_date": contract["raw_report_start"],
            "evaluation_start_date": contract["evaluation_start"],
            "evaluation_end_date": contract["evaluation_end"],
        },
        artifact="10 bps raw-share backtest",
    )
    portfolio["research_folds"] = _load_research_folds(root, records, contract)
    portfolio.update({
        "benchmark": contract["benchmark"],
        "account_currency": contract["account_currency"],
        "initial_account": contract["initial_account"],
        "stress_slippage_bps_per_side": contract["stress_slippage_bps_per_side"],
        "engine": contract["engine"],
        "alignment_method": contract["alignment_method"],
        "evaluation_sessions_sha256": contract["evaluation_sessions_sha256"],
    })
    portfolio["_benchmark"] = benchmark
    return strategy_net, portfolio


def _load_stage_raw_factor_evidence(
    root: Path,
    records: Any,
    request: Mapping[str, Any],
) -> dict[str, pd.Series]:
    contract = request["metric_contract"]["signal"]
    relative = contract["raw_factor_artifact"]
    path, _ = _sealed_artifact_path(
        root, records, relative, artifact="raw factor metrics"
    )
    expected_names = list(request["experiment"]["features"]["factor_names"])
    suffix = contract["raw_factor_column_suffix"]
    expected_columns = [f"{name}{suffix}" for name in expected_names]
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        schema_names = list(parquet.schema_arrow.names)
        row_count = int(parquet.metadata.num_rows)
    except Exception as exc:
        raise ValueError("could not inspect raw factor metrics parquet schema") from exc
    data_columns = [name for name in schema_names if name != "datetime"]
    if data_columns != expected_columns:
        raise ValueError("raw factor metrics columns differ from the experiment factor set")
    if not expected_names:
        if row_count != 0:
            raise ValueError("baseline raw factor metrics must be empty")
        return {}
    expected_sessions = pd.DatetimeIndex(contract["sessions"], name="datetime")
    if row_count != len(expected_sessions):
        raise ValueError("raw factor metrics row count differs from the stage contract")
    filters = [
        ("datetime", ">=", expected_sessions[0]),
        ("datetime", "<=", expected_sessions[-1]),
    ]
    try:
        frame = pd.read_parquet(
            path, columns=expected_columns, filters=filters, engine="pyarrow"
        )
    except Exception as exc:
        raise ValueError("could not predicate-read raw factor metrics") from exc
    actual_index = pd.DatetimeIndex(frame.index, name="datetime")
    if (
        actual_index.tz is not None
        or actual_index.has_duplicates
        or not actual_index.is_monotonic_increasing
        or not actual_index.equals(expected_sessions)
    ):
        raise ValueError("raw factor metric dates differ from the stage contract")
    result: dict[str, pd.Series] = {}
    for factor_name, column in zip(expected_names, expected_columns):
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        finite_or_missing = values.isna() | np.isfinite(values.to_numpy(dtype=float))
        if not bool(finite_or_missing.all()):
            raise ValueError(f"raw factor {factor_name} rank_ic contains infinite values")
        values.index = expected_sessions
        enforce_valid_metric_coverage(
            values,
            experiment_id=f"{request['experiment']['experiment_id']} raw factor {factor_name}",
        )
        result[factor_name] = values.rename(column)
    return result


def load_completed_stage_evidence(
    run_dir: Path,
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, str, str]]:
    """Verify a completed run and load paired signal plus raw-share evidence."""

    values, identity = load_completed_stage_signal_metric(
        run_dir, base_config, plan, request, metric="rank_ic"
    )
    root = Path(run_dir).resolve()
    manifest = read_json(root / "manifest.json")
    integrity = manifest["integrity"]
    checksum = read_json(root / integrity["checksum_manifest"])
    records = checksum.get("artifacts") if isinstance(checksum, dict) else None
    validated_request = validate_stage_run_request(plan, request)
    strategy_net, portfolio = _load_stage_portfolio_evidence(
        root, records, validate_stage_run_request(plan, request)
    )
    benchmark = portfolio.pop("_benchmark")
    raw_factor_rank_ic = _load_stage_raw_factor_evidence(
        root, records, validated_request
    )
    return {
        "rank_ic": values,
        "raw_factor_rank_ic": raw_factor_rank_ic,
        "strategy_net": strategy_net,
        "benchmark": benchmark,
        "portfolio": portfolio,
    }, identity


def validate_stage_run_request(
    plan: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise TypeError("stage run request must be a mapping")
    value = _verify_digest(request, "request_sha256", name="stage run request")
    if value.get("schema_version") != STAGE_RUN_REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported stage run request schema")
    if value.get("status") != "not_run" or value.get("claim_status") != "stage_bound_request_only":
        raise ValueError("stage run request must not claim that training occurred")
    stage = value.get("stage")
    if stage not in RESEARCH_STAGES:
        raise ValueError("stage run request has an invalid research stage")
    validate_stage_exposure_fields(
        value, plan, stage, artifact="stage run request"
    )
    expected = build_stage_run_request(plan, value.get("experiment"), value.get("partition"))
    if value != expected:
        raise ValueError("stage run request differs from the frozen experiment and partition")
    return value


def validate_claimed_stage_run(
    state_path: Path,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a request to be executed while its one-shot stage is claimed."""

    validated_request = validate_stage_run_request(plan, request)
    state = read_research_state(Path(state_path), plan)
    stage = validated_request["stage"]
    record = state["stages"][stage]
    if record.get("status") != "claimed" or record.get("attempts") != 1:
        raise RuntimeError(f"{stage} must be claimed exactly once before training")
    if record.get("partition_sha256") != validated_request["partition"]["sessions_sha256"]:
        raise RuntimeError("claimed research partition differs from the run request")
    if stage != "discovery":
        requested_frozen = validated_request["experiment"].get(
            "frozen_confirmation_spec", {}
        ).get("sha256")
        if requested_frozen != record.get("frozen_spec_sha256"):
            raise RuntimeError("run request differs from the claimed frozen candidate")
    return validated_request


def build_handler_for_experiment(
    plan: Mapping[str, Any], experiment_spec: Mapping[str, Any], **handler_kwargs: Any
):
    """Instantiate the exact Qlib handler declared by an experiment spec."""

    spec = validate_experiment_spec(plan, experiment_spec)
    if spec["role"] == "baseline":
        from qlib.contrib.data.handler import Alpha158

        return Alpha158(**handler_kwargs)
    return build_alpha158_named_factor_handler(
        factor_names=spec["features"]["factor_names"], **handler_kwargs
    )


def load_stage_signal_metric(
    parquet_path: Path,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    metric: str = "rank_ic",
    expected_sha256: str,
) -> pd.Series:
    """Predicate-read only the exact metric dates authorized by a run request."""

    validated_request = validate_stage_run_request(plan, request)
    return load_partition_signal_metric(
        Path(parquet_path),
        validated_request["partition"],
        metric=metric,
        expected_sha256=expected_sha256,
    )


def load_stage_metric_pair(
    baseline_parquet_path: Path,
    candidate_parquet_path: Path,
    plan: Mapping[str, Any],
    baseline_request: Mapping[str, Any],
    candidate_request: Mapping[str, Any],
    *,
    baseline_sha256: str,
    candidate_sha256: str,
    metric: str = "rank_ic",
) -> tuple[pd.Series, pd.Series]:
    """Load a comparable baseline/candidate pair without widening the stage."""

    baseline = validate_stage_run_request(plan, baseline_request)
    candidate = validate_stage_run_request(plan, candidate_request)
    if baseline["experiment"]["role"] != "baseline":
        raise ValueError("baseline_request must contain the Alpha158 baseline")
    if candidate["experiment"]["role"] == "baseline":
        raise ValueError("candidate_request must contain a non-baseline experiment")
    if baseline["stage"] != candidate["stage"] or baseline["partition"] != candidate["partition"]:
        raise ValueError("paired stage requests must use the same exact partition")
    if Path(baseline_parquet_path).resolve() == Path(candidate_parquet_path).resolve():
        raise ValueError("baseline and candidate metric artifacts must be distinct")

    baseline_metric = load_stage_signal_metric(
        baseline_parquet_path,
        plan,
        baseline,
        metric=metric,
        expected_sha256=baseline_sha256,
    )
    candidate_metric = load_stage_signal_metric(
        candidate_parquet_path,
        plan,
        candidate,
        metric=metric,
        expected_sha256=candidate_sha256,
    )
    if not baseline_metric.isna().equals(candidate_metric.isna()):
        raise ValueError("paired stage metric missing-value masks differ")
    return baseline_metric, candidate_metric
