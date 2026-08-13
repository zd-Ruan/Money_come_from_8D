"""Filesystem-safe CLI helpers for pre-registering and executing factor research."""

from __future__ import annotations

import hashlib
import json
import re
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from .config import json_ready_config, load_config
from .factor_research import (
    RESEARCH_STAGES,
    analyze_confirmation,
    analyze_discovery,
    analyze_locked_holdout,
    build_research_plan,
    evaluate_stage_once,
    freeze_confirmation_spec,
    initialize_research_state,
    read_research_state,
    validate_research_plan,
)
from .exposure import (
    exposure_fields_from_request,
    stage_exposure_fields,
    validate_stage_exposure_fields,
)
from .io import now_shanghai, read_json, sha256_file, write_json_atomic
from .research_runner import (
    build_baseline_experiment_spec,
    build_discovery_experiment_specs,
    build_frozen_candidate_experiment_spec,
    build_research_experiment_manifest,
    build_stage_run_request,
    enforce_valid_metric_coverage,
    load_completed_stage_evidence,
    prepare_stage_pipeline_config,
    validate_discovery_experiment_specs,
)


RESEARCH_WORKSPACE_SCHEMA_VERSION = 3
RESEARCH_EXECUTION_SCHEMA_VERSION = 3
_SAFE_STUDY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _read_json_object(path: Path, *, artifact: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{artifact} must contain a JSON object")
    return value


def validate_study_id(study_id: str) -> str:
    if not isinstance(study_id, str) or not _SAFE_STUDY_ID.fullmatch(study_id):
        raise ValueError("study_id must be a portable 1-128 character identifier")
    if study_id in {".", ".."} or study_id.endswith("."):
        raise ValueError("study_id must be a portable 1-128 character identifier")
    return study_id


def resolve_research_workspace(research_root: Path, study_id: str) -> Path:
    """Resolve one direct study directory beneath the configured root."""

    root = Path(research_root).resolve()
    target = (root / validate_study_id(study_id)).resolve()
    if target.parent != root:
        raise ValueError("study_id resolves outside the research root")
    return target


def load_session_calendar(calendar_path: Path) -> pd.DatetimeIndex:
    """Load a strict, timezone-naive exchange calendar from a one-column file."""

    path = Path(calendar_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"calendar is missing or unsafe: {path}")
    try:
        values = pd.read_csv(path, header=None).iloc[:, 0]
        sessions = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"calendar is invalid: {path}") from exc
    if sessions.empty:
        raise ValueError("calendar must contain at least one session")
    if sessions.tz is not None or any(timestamp != timestamp.normalize() for timestamp in sessions):
        raise ValueError("calendar sessions must be timezone-naive dates")
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("calendar must be unique and strictly chronological")
    return sessions


def _configuration_record(config_path: Path) -> dict[str, Any]:
    loaded = load_config(Path(config_path))
    config = json_ready_config(loaded)
    workspace = Path(loaded["_meta"]["workspace_root"]).resolve()
    source = Path(loaded["_meta"]["config_path"]).resolve().relative_to(workspace).as_posix()
    return {
        "config": config,
        "config_sha256": _sha256_json(config),
        "config_source": source,
    }


def _workspace_manifest(
    plan: Mapping[str, Any], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": RESEARCH_WORKSPACE_SCHEMA_VERSION,
        "status": "initialized_not_run",
        "claim_status": "pre_registration_only",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "catalog_sha256": plan["catalog_sha256"],
        "exposure_registry_sha256": plan["exposure_provenance"]["registry_sha256"],
        "known_through_session": plan["exposure_provenance"]["known_through_session"],
        "base_config_sha256": configuration["config_sha256"],
        "config_source": configuration["config_source"],
        "artifacts": {
            "plan": "plan.json",
            "state": "state.json",
            "experiments": "experiments.json",
            "base_config": "base_config.json",
        },
        "limitations": [
            "Initialization does not mean any model was trained or any stage was opened.",
            "No discovery, confirmation, holdout, or prior run metric is read by this command.",
            "The current-snapshot ETF universe keeps every eventual result research_only.",
            "Dates on or before the fixed exposure cutoff are retrospective_exposed, never pristine.",
        ],
    }
    payload["manifest_sha256"] = _sha256_json(payload)
    return payload


def initialize_research_workspace(
    *,
    research_root: Path,
    study_id: str,
    config_path: Path,
    calendar_path: Path,
    discovery_end: str,
    confirmation_end: str,
) -> Path:
    """Atomically pre-register a study directory without opening any stage."""

    workspace = resolve_research_workspace(research_root, study_id)
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError(f"research workspace already exists: {workspace}")

    configuration = _configuration_record(config_path)
    full_calendar = load_session_calendar(calendar_path)
    configured_data = configuration["config"]["data"]
    sessions = full_calendar[
        (full_calendar >= pd.Timestamp(configured_data["test_start_date"]))
        & (full_calendar <= pd.Timestamp(configured_data["end_date"]))
    ]
    if sessions.empty:
        raise ValueError("base config out-of-sample dates do not overlap the supplied calendar")
    plan = build_research_plan(
        sessions,
        discovery_end=discovery_end,
        confirmation_end=confirmation_end,
        plan_id=study_id,
        base_config_sha256=configuration["config_sha256"],
        label_horizon_bars=int(configuration["config"]["data"]["label_horizon_bars"]),
        required_stress_slippage_bps=int(
            configuration["config"]["gates"]["required_stress_slippage_bps"]
        ),
        account_cny=float(configuration["config"]["execution"]["account"]),
        specification_frozen_at=now_shanghai().isoformat(),
    )
    experiments = build_research_experiment_manifest(plan)
    manifest = _workspace_manifest(plan, configuration)

    workspace.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(exist_ok=False)
    try:
        write_json_atomic(workspace / "plan.json", plan)
        write_json_atomic(workspace / "base_config.json", configuration)
        write_json_atomic(workspace / "experiments.json", experiments)
        initialize_research_state(workspace / "state.json", plan)
        write_json_atomic(workspace / "manifest.json", manifest)
        validate_research_workspace(workspace)
    except BaseException:
        # Preserve a partial directory as audit evidence; a retry must use a new
        # study id or be cleaned up explicitly after inspection.
        raise
    return workspace


def _validate_manifest_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    digest = manifest.pop("manifest_sha256", None)
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("research workspace manifest has an invalid SHA-256")
    if _sha256_json(manifest) != digest:
        raise ValueError("research workspace manifest SHA-256 does not match its content")
    manifest["manifest_sha256"] = digest
    return manifest


def validate_research_workspace(workspace_path: Path) -> dict[str, Any]:
    """Validate control artifacts without reading any metric or run result."""

    workspace = Path(workspace_path).resolve()
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError(f"research workspace is missing or unsafe: {workspace}")
    manifest = _validate_manifest_digest(
        _read_json_object(workspace / "manifest.json", artifact="research workspace manifest")
    )
    if manifest.get("schema_version") != RESEARCH_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("unsupported research workspace schema")
    if manifest.get("status") != "initialized_not_run":
        raise ValueError("research workspace manifest must not claim a completed experiment")
    expected_artifacts = {
        "plan": "plan.json",
        "state": "state.json",
        "experiments": "experiments.json",
        "base_config": "base_config.json",
    }
    if manifest.get("artifacts") != expected_artifacts:
        raise ValueError("research workspace artifact mapping is invalid")

    plan = validate_research_plan(
        _read_json_object(workspace / "plan.json", artifact="research plan")
    )
    configuration = _read_json_object(
        workspace / "base_config.json", artifact="frozen base configuration"
    )
    config = configuration.get("config")
    config_digest = configuration.get("config_sha256")
    config_source = configuration.get("config_source")
    if (
        not isinstance(config, dict)
        or not isinstance(config_digest, str)
        or not isinstance(config_source, str)
        or not config_source
        or "\\" in config_source
        or Path(config_source).is_absolute()
        or any(part in {"", ".", ".."} for part in config_source.split("/"))
    ):
        raise ValueError("frozen base configuration record is invalid")
    if _sha256_json(config) != config_digest:
        raise ValueError("frozen base configuration SHA-256 does not match its content")
    if config_digest != manifest.get("base_config_sha256"):
        raise ValueError("workspace manifest and frozen base configuration differ")
    if config_source != manifest.get("config_source"):
        raise ValueError("workspace manifest and frozen config source differ")
    if plan["plan_id"] != manifest.get("plan_id") or plan["plan_sha256"] != manifest.get(
        "plan_sha256"
    ):
        raise ValueError("workspace manifest and research plan differ")
    if plan["catalog_sha256"] != manifest.get("catalog_sha256"):
        raise ValueError("workspace manifest and factor catalog differ")
    if (
        plan["exposure_provenance"]["registry_sha256"]
        != manifest.get("exposure_registry_sha256")
        or plan["exposure_provenance"]["known_through_session"]
        != manifest.get("known_through_session")
    ):
        raise ValueError("workspace manifest and exposure provenance differ")

    experiments = _read_json_object(
        workspace / "experiments.json", artifact="research experiment manifest"
    )
    discovery = experiments.get("discovery")
    if not isinstance(discovery, dict):
        raise ValueError("research experiment manifest has no discovery battery")
    validate_discovery_experiment_specs(plan, discovery)
    if experiments.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("research experiment manifest and plan differ")
    state = read_research_state(workspace / "state.json", plan)
    execution_status = None
    execution_path = workspace / "execution.json"
    if execution_path.exists() or execution_path.is_symlink():
        execution_status = _load_execution_record(execution_path, plan).get("status")
    return {
        "valid": True,
        "status": manifest["status"],
        "claim_status": manifest["claim_status"],
        "workspace": str(workspace),
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "catalog_sha256": plan["catalog_sha256"],
        "base_config_sha256": config_digest,
        "exposure_provenance": deepcopy(plan["exposure_provenance"]),
        "experiment_count": discovery["experiment_count"],
        "frozen_candidate_status": (
            "frozen_not_run"
            if state.get("frozen_confirmation_spec") is not None
            else experiments.get("frozen_candidate_status")
        ),
        "stages": {
            stage: {
                "status": record["status"],
                "attempts": record["attempts"],
                **stage_exposure_fields(plan, stage),
            }
            for stage, record in state["stages"].items()
        },
        "execution_status": execution_status,
    }


def write_research_experiment_manifest(workspace_path: Path, output_path: Path) -> Path:
    """Export validated specs to a new JSON path without overwriting files."""

    workspace = Path(workspace_path).resolve()
    validate_research_workspace(workspace)
    output = Path(output_path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"research manifest output already exists: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("research manifest output must use a .json filename")
    plan = _read_json_object(workspace / "plan.json", artifact="research plan")
    state = read_research_state(workspace / "state.json", plan)
    manifest = build_research_experiment_manifest(plan, state.get("frozen_confirmation_spec"))
    write_json_atomic(output, manifest)
    return output


def _runtime_base_config(
    configuration: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repository root is missing or unsafe: {root}")
    config = deepcopy(configuration["config"])
    for key, raw_value in config.get("paths", {}).items():
        path = Path(raw_value)
        config["paths"][key] = str(path if path.is_absolute() else (root / path).resolve())
    source = (root / configuration["config_source"]).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("frozen config source resolves outside the repository") from exc
    config["_meta"] = {"config_path": str(source), "workspace_root": str(root)}
    if _sha256_json(json_ready_config(config)) != configuration["config_sha256"]:
        raise ValueError("rehydrated base configuration differs from its frozen digest")
    return config


def _run_id(plan: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    experiment = request["experiment"]["experiment_id"]
    prefix = f"research-{plan['plan_id'][:24]}-{request['stage'][:12]}-{experiment[:48]}"
    return f"{prefix}-{request['request_sha256'][:12]}"[:128].rstrip(".")


def _stage_preflight_specs(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    stage: str,
) -> list[Mapping[str, Any]]:
    if stage == "discovery":
        manifest = build_discovery_experiment_specs(plan)
        return [
            manifest["baseline"],
            *manifest["family_ablations"],
            *manifest["single_factor_tests"],
        ]
    frozen = state.get("frozen_confirmation_spec")
    if not isinstance(frozen, Mapping):
        raise RuntimeError(f"{stage} preflight requires the frozen candidate specification")
    return [
        build_baseline_experiment_spec(plan),
        build_frozen_candidate_experiment_spec(plan, frozen),
    ]


def _preflight_stage_requests(
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    stage: str,
) -> None:
    """Validate every deterministic request and output path before a stage claim."""

    partition = plan["partitions"][stage]
    runs_root = Path(base_config["paths"]["runs"])
    from .integrity import resolve_run_directory

    seen: set[str] = set()
    for spec in _stage_preflight_specs(plan, state, stage):
        request = build_stage_run_request(plan, spec, partition)
        prepare_stage_pipeline_config(base_config, plan, request)
        run_id = _run_id(plan, request)
        if run_id in seen:
            raise RuntimeError(f"deterministic research run ID collision: {run_id}")
        seen.add(run_id)
        run_dir = resolve_run_directory(runs_root, run_id)
        if run_dir.exists() or run_dir.is_symlink():
            raise FileExistsError(f"research run already exists before stage claim: {run_dir}")


def _preflight_research_execution(
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Perform the production research gates without persisting any artifact."""

    from .audit import audit_and_snapshot
    from .environment import validate_locked_environment
    from .runner import (
        run_pretraining_corporate_action_audit,
        validate_frozen_factor_provider,
        validate_lightgbm_device,
    )

    root = Path(repository_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repository root is missing or unsafe: {root}")
    environment = validate_locked_environment()
    validate_lightgbm_device(dict(base_config))
    audit = audit_and_snapshot(dict(base_config), persist=False)
    if not audit.report["data_valid"]:
        raise RuntimeError(f"data quality gate failed: {audit.report['blocking_issues']}")
    action_audit = run_pretraining_corporate_action_audit(dict(base_config), None)
    factor_smoke = validate_frozen_factor_provider(dict(base_config))
    return {
        "environment_lock_sha256": environment["lock"]["sha256"],
        "snapshot_id": audit.snapshot_id,
        "source_fingerprint": audit.report["source_fingerprint"],
        "corporate_action_audit_passed": bool(action_audit.passed),
        "frozen_factor_provider": factor_smoke,
        "plan_sha256": plan["plan_sha256"],
    }


def _new_execution_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESEARCH_EXECUTION_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "exposure_registry_sha256": plan["exposure_provenance"]["registry_sha256"],
        "status": "running",
        "started_at": now_shanghai().isoformat(),
        "updated_at": now_shanghai().isoformat(),
        "stages": {},
    }


def _load_execution_record(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(path)
    if value is None:
        return _new_execution_record(plan)
    if not isinstance(value, dict):
        raise ValueError("research execution evidence must contain a JSON object")
    digest = value.get("execution_sha256")
    unsigned = {key: deepcopy(item) for key, item in value.items() if key != "execution_sha256"}
    if (
        not isinstance(digest, str)
        or not _DIGEST_PATTERN.fullmatch(digest)
        or _sha256_json(unsigned) != digest
        or value.get("schema_version") != RESEARCH_EXECUTION_SCHEMA_VERSION
        or value.get("plan_sha256") != plan["plan_sha256"]
        or value.get("exposure_registry_sha256")
        != plan["exposure_provenance"]["registry_sha256"]
        or not isinstance(value.get("stages"), dict)
    ):
        raise ValueError("research execution evidence is invalid or belongs to another plan")
    for stage, record in value["stages"].items():
        if stage not in RESEARCH_STAGES or not isinstance(record, dict):
            raise ValueError("research execution contains an invalid stage record")
        validate_stage_exposure_fields(
            record, plan, stage, artifact=f"research execution {stage}"
        )
        experiments = record.get("experiments", [])
        if not isinstance(experiments, list):
            raise ValueError("research execution stage experiments must be a list")
        for experiment in experiments:
            if not isinstance(experiment, dict):
                raise ValueError("research execution experiment record is invalid")
            validate_stage_exposure_fields(
                experiment,
                plan,
                stage,
                artifact="research execution experiment",
            )
    return unsigned


def _write_execution_record(path: Path, record: dict[str, Any]) -> None:
    record.pop("execution_sha256", None)
    record["updated_at"] = now_shanghai().isoformat()
    sealed = deepcopy(record)
    sealed["execution_sha256"] = _sha256_json(sealed)
    write_json_atomic(path, sealed)


def _record_experiment(
    execution_path: Path,
    execution: dict[str, Any],
    stage: str,
    request: Mapping[str, Any],
    *,
    status: str,
    run_id: str,
    error: BaseException | None = None,
) -> None:
    stage_record = execution["stages"].setdefault(
        stage,
        {
            "status": "running",
            "experiments": [],
            **exposure_fields_from_request(request),
        },
    )
    records = stage_record["experiments"]
    match = next(
        (item for item in records if item.get("request_sha256") == request["request_sha256"]),
        None,
    )
    if match is None:
        match = {
            "experiment_id": request["experiment"]["experiment_id"],
            "request_sha256": request["request_sha256"],
            "run_id": run_id,
            **exposure_fields_from_request(request),
        }
        records.append(match)
    match["status"] = status
    match[f"{status}_at"] = now_shanghai().isoformat()
    if error is not None:
        match["error_type"] = type(error).__name__
        match["error_message"] = str(error)[:500]
    _write_execution_record(execution_path, execution)


def _execute_experiment(
    *,
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    state_path: Path,
    partition: Mapping[str, Any],
    experiment_spec: Mapping[str, Any],
    execution_path: Path,
    execution: dict[str, Any],
    pipeline_runner: Callable[..., Path],
) -> tuple[dict[str, Any], tuple[str, str, str], dict[str, Any]]:
    request = build_stage_run_request(plan, experiment_spec, partition)
    run_id = _run_id(plan, request)
    _record_experiment(
        execution_path, execution, request["stage"], request, status="running", run_id=run_id
    )
    try:
        run_dir = pipeline_runner(
            deepcopy(dict(base_config)),
            run_id=run_id,
            research_plan=dict(plan),
            research_request=request,
            research_state_path=state_path,
        )
        evidence, identity = load_completed_stage_evidence(
            run_dir, base_config, plan, request
        )
        coverage = {
            "rank_ic": enforce_valid_metric_coverage(
                evidence["rank_ic"], experiment_id=experiment_spec["experiment_id"]
            ),
            "strategy_net": enforce_valid_metric_coverage(
                evidence["strategy_net"], experiment_id=experiment_spec["experiment_id"]
            ),
        }
    except BaseException as exc:
        _record_experiment(
            execution_path,
            execution,
            request["stage"],
            request,
            status="failed",
            run_id=run_id,
            error=exc,
        )
        raise
    _record_experiment(
        execution_path, execution, request["stage"], request, status="completed", run_id=run_id
    )
    return evidence, identity, coverage


def _execute_stage_battery(
    *,
    stage: str,
    specs: list[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    state_path: Path,
    partition: Mapping[str, Any],
    execution_path: Path,
    execution: dict[str, Any],
    pipeline_runner: Callable[..., Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    evidence_by_experiment: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    run_ids: dict[str, str] = {}
    expected_identity: tuple[str, str, str] | None = None
    for spec in specs:
        evidence, identity, metric_coverage = _execute_experiment(
            base_config=base_config,
            plan=plan,
            state_path=state_path,
            partition=partition,
            experiment_spec=spec,
            execution_path=execution_path,
            execution=execution,
            pipeline_runner=pipeline_runner,
        )
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise RuntimeError("research stage runs do not share one data snapshot and source tree")
        experiment_id = spec["experiment_id"]
        evidence_by_experiment[experiment_id] = evidence
        coverage[experiment_id] = metric_coverage
        run_ids[experiment_id] = _run_id(
            plan, build_stage_run_request(plan, spec, partition)
        )
    return evidence_by_experiment, coverage, run_ids


def execute_research_workspace(
    workspace_path: Path,
    *,
    repository_root: Path | None = None,
    pipeline_runner: Callable[..., Path] | None = None,
    preflight_runner: Callable[[Mapping[str, Any], Mapping[str, Any], Path], Any]
    | None = None,
) -> dict[str, Any]:
    """Execute or resume the next unopened stages of one frozen research study."""

    workspace = Path(workspace_path).resolve()
    validate_research_workspace(workspace)
    plan = validate_research_plan(
        _read_json_object(workspace / "plan.json", artifact="research plan")
    )
    configuration = _read_json_object(
        workspace / "base_config.json", artifact="frozen base configuration"
    )
    resolved_repository_root = (
        repository_root or Path(__file__).resolve().parents[3]
    )
    base_config = _runtime_base_config(
        configuration,
        resolved_repository_root,
    )
    state_path = workspace / "state.json"
    state = read_research_state(state_path, plan)
    for stage in RESEARCH_STAGES:
        status = state["stages"][stage]["status"]
        if status in {"claimed", "failed"}:
            raise RuntimeError(
                f"{stage} was already consumed with status {status}; automatic replay is forbidden"
            )

    if pipeline_runner is None:
        from .runner import run_pipeline

        pipeline_runner = run_pipeline
        production_pipeline = True
    else:
        production_pipeline = False
    if preflight_runner is None:
        preflight_runner = _preflight_research_execution if production_pipeline else (
            lambda _config, _plan, _root: None
        )
    preflight_runner(base_config, plan, Path(resolved_repository_root).resolve())

    state = read_research_state(state_path, plan)
    for stage in RESEARCH_STAGES:
        if state["stages"][stage]["status"] == "unopened":
            if stage != "discovery" and state.get("frozen_confirmation_spec") is None:
                break
            _preflight_stage_requests(base_config, plan, state, stage)
            break
    execution_path = workspace / "execution.json"
    execution = _load_execution_record(execution_path, plan)
    execution["status"] = "running"
    _write_execution_record(execution_path, execution)

    def mark_stage(stage: str, status: str, result: Mapping[str, Any] | None = None) -> None:
        record = execution["stages"].setdefault(stage, {"experiments": []})
        record.update(stage_exposure_fields(plan, stage))
        record["status"] = status
        record[f"{status}_at"] = now_shanghai().isoformat()
        if result is not None:
            record["result_sha256"] = _sha256_json(result)
        _write_execution_record(execution_path, execution)

    try:
        state = read_research_state(state_path, plan)
        if state["stages"]["discovery"]["status"] == "unopened":
            _preflight_stage_requests(base_config, plan, state, "discovery")
            discovery_manifest = build_discovery_experiment_specs(plan)

            def evaluate_discovery(partition):
                specs = [
                    discovery_manifest["baseline"],
                    *discovery_manifest["family_ablations"],
                    *discovery_manifest["single_factor_tests"],
                ]
                evidence, coverage, run_ids = _execute_stage_battery(
                    stage="discovery",
                    specs=specs,
                    base_config=base_config,
                    plan=plan,
                    state_path=state_path,
                    partition=partition,
                    execution_path=execution_path,
                    execution=execution,
                    pipeline_runner=pipeline_runner,
                )
                baseline = evidence[discovery_manifest["baseline"]["experiment_id"]]
                families = {
                    spec["family"]: evidence[spec["experiment_id"]]
                    for spec in discovery_manifest["family_ablations"]
                }
                factors = {
                    spec["factor_name"]: evidence[spec["experiment_id"]]
                    for spec in discovery_manifest["single_factor_tests"]
                }
                result = analyze_discovery(baseline, families, factors, plan)
                result["metric_coverage"] = coverage
                result["run_ids"] = run_ids
                return result

            try:
                discovery_result = evaluate_stage_once(
                    state_path, plan, "discovery", evaluate_discovery
                )
            except BaseException:
                mark_stage("discovery", "failed")
                raise
            mark_stage("discovery", "completed", discovery_result)
        else:
            discovery_result = state["stages"]["discovery"]["result"]

        selected = discovery_result["selected_factor_names"]
        if not selected:
            execution["status"] = "stopped_no_bh_candidate"
            execution["finished_at"] = now_shanghai().isoformat()
            _write_execution_record(execution_path, execution)
            return validate_research_workspace(workspace) | {
                "execution_status": execution["status"],
                "selected_factor_names": [],
            }

        state = read_research_state(state_path, plan)
        if state.get("frozen_confirmation_spec") is None:
            frozen_specification = {
                "selection_rule": (
                    "all_and_only_joint_model_rank_ic_strategy_net_and_signed_raw_factor_"
                    "rank_ic_iut_bh_q_0.10_factors_passing_all_economic_gates"
                ),
                "base_config_sha256": plan["base_config_sha256"],
                "model": deepcopy(configuration["config"]["model"]),
                "rolling": deepcopy(configuration["config"]["rolling"]),
                "label": configuration["config"]["data"]["label"],
                "label_horizon_bars": plan["label_horizon_bars"],
                "execution_evidence": deepcopy(plan["execution_evidence"]),
            }
            freeze_confirmation_spec(
                state_path,
                plan,
                selected_factor_names=selected,
                frozen_spec=frozen_specification,
            )
            state = read_research_state(state_path, plan)
            write_json_atomic(
                workspace / "experiments.json",
                build_research_experiment_manifest(plan, state["frozen_confirmation_spec"]),
            )
        frozen = state["frozen_confirmation_spec"]
        if frozen["selected_factor_names"] != selected:
            raise RuntimeError("frozen candidate is not exactly the complete joint-selected set")

        baseline_spec = build_baseline_experiment_spec(plan)
        candidate_spec = build_frozen_candidate_experiment_spec(plan, frozen)
        if state["stages"]["confirmation"]["status"] == "unopened":
            _preflight_stage_requests(base_config, plan, state, "confirmation")

            def evaluate_confirmation(partition):
                evidence, coverage, run_ids = _execute_stage_battery(
                    stage="confirmation",
                    specs=[baseline_spec, candidate_spec],
                    base_config=base_config,
                    plan=plan,
                    state_path=state_path,
                    partition=partition,
                    execution_path=execution_path,
                    execution=execution,
                    pipeline_runner=pipeline_runner,
                )
                result = analyze_confirmation(
                    evidence[baseline_spec["experiment_id"]],
                    evidence[candidate_spec["experiment_id"]],
                    plan,
                    frozen_spec_sha256=frozen["sha256"],
                    candidate_factor_names=frozen["selected_factor_names"],
                )
                result["metric_coverage"] = coverage
                result["run_ids"] = run_ids
                return result

            try:
                confirmation_result = evaluate_stage_once(
                    state_path, plan, "confirmation", evaluate_confirmation
                )
            except BaseException:
                mark_stage("confirmation", "failed")
                raise
            mark_stage("confirmation", "completed", confirmation_result)
        else:
            confirmation_result = state["stages"]["confirmation"]["result"]

        if confirmation_result["confirmation_passed"] is not True:
            execution["status"] = "stopped_confirmation_failed"
            execution["finished_at"] = now_shanghai().isoformat()
            _write_execution_record(execution_path, execution)
            return validate_research_workspace(workspace) | {
                "execution_status": execution["status"],
                "selected_factor_names": selected,
            }

        state = read_research_state(state_path, plan)
        if state["stages"]["locked_holdout"]["status"] == "unopened":
            _preflight_stage_requests(base_config, plan, state, "locked_holdout")

            def evaluate_holdout(partition):
                evidence, coverage, run_ids = _execute_stage_battery(
                    stage="locked_holdout",
                    specs=[baseline_spec, candidate_spec],
                    base_config=base_config,
                    plan=plan,
                    state_path=state_path,
                    partition=partition,
                    execution_path=execution_path,
                    execution=execution,
                    pipeline_runner=pipeline_runner,
                )
                result = analyze_locked_holdout(
                    evidence[baseline_spec["experiment_id"]],
                    evidence[candidate_spec["experiment_id"]],
                    plan,
                    frozen_spec_sha256=frozen["sha256"],
                    candidate_factor_names=frozen["selected_factor_names"],
                )
                result["metric_coverage"] = coverage
                result["run_ids"] = run_ids
                return result

            try:
                holdout_result = evaluate_stage_once(
                    state_path, plan, "locked_holdout", evaluate_holdout
                )
            except BaseException:
                mark_stage("locked_holdout", "failed")
                raise
            mark_stage("locked_holdout", "completed", holdout_result)

        execution["status"] = "completed"
        execution["finished_at"] = now_shanghai().isoformat()
        _write_execution_record(execution_path, execution)
        return validate_research_workspace(workspace) | {
            "execution_status": "completed",
            "selected_factor_names": selected,
        }
    except BaseException as exc:
        execution["status"] = "failed"
        execution["finished_at"] = now_shanghai().isoformat()
        execution["error_type"] = type(exc).__name__
        execution["error_message"] = str(exc)[:500]
        execution["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-8000:]
        _write_execution_record(execution_path, execution)
        raise
