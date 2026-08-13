"""Immutable provenance for historical research-result exposure.

The repository already contains inspected CPU factor comparisons through
2026-08-11.  A one-shot ledger cannot make those dates unseen again.  This
module binds every later research artifact to that fact and fails closed when
the repository registry is missing, modified, or contradicted.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import read_json


EXPOSURE_REGISTRY_SCHEMA_VERSION = 1
EXPOSURE_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "research" / "exposure_registry.json"
EXPECTED_EXPOSURE_REGISTRY_SHA256 = "9806baa6d566cdc90748897420be189dcc53033e5bdfd46bf68740f97c2cb5e2"
RETROSPECTIVE_EXPOSED = "retrospective_exposed"
PROSPECTIVE_UNSEEN = "prospective_unseen"
RESEARCH_ONLY = "research_only"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


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


def load_exposure_registry() -> dict[str, Any]:
    """Load the repository-pinned registry; self-consistent rewrites still fail."""

    path = EXPOSURE_REGISTRY_PATH.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"fixed exposure registry is missing or unsafe: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("fixed exposure registry must contain a JSON object")
    registry = deepcopy(value)
    digest = registry.pop("registry_sha256", None)
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("fixed exposure registry has an invalid registry_sha256")
    actual = _sha256_json(registry)
    if digest != actual:
        raise ValueError("fixed exposure registry SHA-256 does not match its content")
    if digest != EXPECTED_EXPOSURE_REGISTRY_SHA256:
        raise ValueError("fixed exposure registry differs from the code-pinned registry")
    if registry.get("schema_version") != EXPOSURE_REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported exposure registry schema")
    if registry.get("policy_version") != "historical_exposure_registry_v1":
        raise ValueError("unsupported exposure registry policy")

    cutoff = registry.get("known_through_session")
    try:
        cutoff_timestamp = pd.Timestamp(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("exposure registry known_through_session is invalid") from exc
    if cutoff_timestamp.tz is not None or cutoff_timestamp != cutoff_timestamp.normalize():
        raise ValueError("exposure registry cutoff must be a timezone-naive session date")
    evidence = registry.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("exposure registry evidence is missing")
    commit = evidence.get("git_commit")
    snapshot = evidence.get("data_snapshot")
    published = evidence.get("published_research")
    if (
        not isinstance(commit, dict)
        or not _COMMIT_PATTERN.fullmatch(str(commit.get("commit_oid", "")))
        or not isinstance(snapshot, dict)
        or not _DIGEST_PATTERN.fullmatch(str(snapshot.get("source_fingerprint", "")))
        or snapshot.get("calendar_end") != cutoff
        or not isinstance(published, dict)
        or published.get("portfolio_evaluation_end") != cutoff
        or not isinstance(published.get("run_ids"), list)
        or not published["run_ids"]
        or not isinstance(published.get("comparison_artifacts"), list)
        or not published["comparison_artifacts"]
    ):
        raise ValueError("exposure registry evidence contract is invalid")
    if registry.get("policy") != {
        "on_or_before_known_through": RETROSPECTIVE_EXPOSED,
        "exposed_claim_classification": RESEARCH_ONLY,
        "promotion_allowed_for_retrospective_exposed": False,
        "prospective_requirement": (
            "stage start must be after known_through_session and "
            "specification_frozen_at must precede stage start"
        ),
    }:
        raise ValueError("exposure registry claim policy is invalid")
    registry["registry_sha256"] = digest
    return registry


def _validated_frozen_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("specification_frozen_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("specification_frozen_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("specification_frozen_at must include an explicit timezone")
    return parsed


def classify_stage_exposure(
    partition: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    specification_frozen_at: str | None = None,
) -> dict[str, Any]:
    """Classify one stage without allowing an exposed period to become pristine."""

    if not isinstance(partition, Mapping):
        raise TypeError("research partition must be a mapping")
    try:
        start = pd.Timestamp(partition["start"])
        end = pd.Timestamp(partition["end"])
        cutoff = pd.Timestamp(registry["known_through_session"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research partition exposure dates are invalid") from exc
    if any(value.tz is not None or value != value.normalize() for value in (start, end, cutoff)):
        raise ValueError("research exposure dates must be timezone-naive session dates")
    if start > end:
        raise ValueError("research partition starts after it ends")

    frozen = _validated_frozen_at(specification_frozen_at)
    if start <= cutoff:
        evidence_class = RETROSPECTIVE_EXPOSED
        reason = "stage begins on or before the fixed historical-exposure cutoff"
    else:
        if frozen is None:
            raise ValueError(
                "a stage after the exposure cutoff requires specification_frozen_at"
            )
        start_with_zone = start.tz_localize(frozen.tzinfo)
        if frozen >= start_with_zone.to_pydatetime():
            raise ValueError(
                "specification_frozen_at must precede a prospective stage start"
            )
        evidence_class = PROSPECTIVE_UNSEEN
        reason = "stage begins after the fixed cutoff and the specification was frozen first"
    return {
        "evidence_class": evidence_class,
        "claim_classification": RESEARCH_ONLY,
        "promotion_eligible": False,
        "reason": reason,
    }


def build_exposure_provenance(
    partitions: Mapping[str, Mapping[str, Any]],
    stage_order: tuple[str, ...],
    *,
    specification_frozen_at: str | None = None,
) -> dict[str, Any]:
    registry = load_exposure_registry()
    if not isinstance(partitions, Mapping) or tuple(partitions) != stage_order:
        raise ValueError("research partitions do not match the exposure stage order")
    classifications = {
        stage: classify_stage_exposure(
            partitions[stage],
            registry,
            specification_frozen_at=specification_frozen_at,
        )
        for stage in stage_order
    }
    return {
        "registry_schema_version": registry["schema_version"],
        "registry_sha256": registry["registry_sha256"],
        "known_through_session": registry["known_through_session"],
        "specification_frozen_at": specification_frozen_at,
        "provenance_status": "verified",
        "stage_classification": classifications,
    }


def stage_exposure_fields(plan: Mapping[str, Any], stage: str) -> dict[str, Any]:
    provenance = plan.get("exposure_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("research plan is missing exposure provenance")
    stages = provenance.get("stage_classification")
    if not isinstance(stages, Mapping) or stage not in stages:
        raise ValueError(f"research plan is missing {stage} exposure classification")
    classification = stages[stage]
    if not isinstance(classification, Mapping):
        raise ValueError(f"research plan has invalid {stage} exposure classification")
    return {
        "exposure_registry_sha256": provenance.get("registry_sha256"),
        "evidence_class": classification.get("evidence_class"),
        "claim_classification": classification.get("claim_classification"),
    }


def exposure_fields_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Extract already-validated exposure fields for execution journaling."""

    if not isinstance(request, Mapping):
        raise TypeError("stage run request must be a mapping")
    fields = {
        key: request.get(key)
        for key in (
            "exposure_registry_sha256",
            "evidence_class",
            "claim_classification",
        )
    }
    if not _DIGEST_PATTERN.fullmatch(str(fields["exposure_registry_sha256"] or "")):
        raise ValueError("stage run request exposure registry identity is invalid")
    if fields["evidence_class"] not in {RETROSPECTIVE_EXPOSED, PROSPECTIVE_UNSEEN}:
        raise ValueError("stage run request evidence_class is invalid")
    if fields["claim_classification"] != RESEARCH_ONLY:
        raise ValueError("stage run request claim classification must remain research_only")
    return fields


def validate_stage_exposure_fields(
    value: Mapping[str, Any], plan: Mapping[str, Any], stage: str, *, artifact: str
) -> dict[str, Any]:
    expected = stage_exposure_fields(plan, stage)
    actual = {key: value.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"{artifact} exposure provenance differs from the research plan")
    if actual["evidence_class"] == RETROSPECTIVE_EXPOSED and actual[
        "claim_classification"
    ] != RESEARCH_ONLY:
        raise ValueError(f"{artifact} exposed evidence must remain research_only")
    return expected
