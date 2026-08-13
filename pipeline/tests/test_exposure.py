import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.exposure import (
    EXPECTED_EXPOSURE_REGISTRY_SHA256,
    classify_stage_exposure,
    load_exposure_registry,
    stage_exposure_fields,
    validate_stage_exposure_fields,
)
from quant_pipeline.factor_research import (
    build_research_plan,
    initialize_research_state,
    read_research_state,
    validate_research_plan,
)
from quant_pipeline.research_runner import (
    build_baseline_experiment_spec,
    build_stage_run_request,
    validate_stage_run_request,
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _historical_plan():
    dates = pd.bdate_range("2025-01-02", periods=258)
    return build_research_plan(
        dates,
        discovery_end=dates[125],
        confirmation_end=dates[190],
        plan_id="exposure-contract-study",
        base_config_sha256="f" * 64,
        specification_frozen_at="2026-08-13T05:00:00+08:00",
    )


class FixedExposureRegistryTests(unittest.TestCase):
    def test_repository_registry_is_self_validating_and_code_pinned(self):
        registry = load_exposure_registry()
        self.assertEqual(registry["registry_sha256"], EXPECTED_EXPOSURE_REGISTRY_SHA256)
        self.assertEqual(registry["known_through_session"], "2026-08-11")
        self.assertEqual(
            registry["evidence"]["published_research"]["portfolio_evaluation_end"],
            "2026-08-11",
        )

    def test_missing_registry_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "quant_pipeline.exposure.EXPOSURE_REGISTRY_PATH",
            Path(directory) / "missing.json",
        ):
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                load_exposure_registry()

    def test_self_consistent_registry_rewrite_still_fails_code_pin(self):
        registry = load_exposure_registry()
        registry["evidence"]["git_commit"]["subject"] = "rewritten history"
        registry.pop("registry_sha256")
        registry["registry_sha256"] = _digest(registry)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exposure_registry.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            with patch("quant_pipeline.exposure.EXPOSURE_REGISTRY_PATH", path):
                with self.assertRaisesRegex(ValueError, "code-pinned"):
                    load_exposure_registry()


class ExposureClassificationTests(unittest.TestCase):
    def test_stage_touching_known_history_is_retrospective_and_never_promotable(self):
        result = classify_stage_exposure(
            {"start": "2026-08-11", "end": "2026-09-30"},
            load_exposure_registry(),
            specification_frozen_at="2026-08-01T12:00:00+08:00",
        )
        self.assertEqual(result["evidence_class"], "retrospective_exposed")
        self.assertEqual(result["claim_classification"], "research_only")
        self.assertFalse(result["promotion_eligible"])

    def test_prospective_stage_requires_a_pre_start_freeze(self):
        registry = load_exposure_registry()
        partition = {"start": "2026-08-12", "end": "2026-12-31"}
        with self.assertRaisesRegex(ValueError, "requires specification_frozen_at"):
            classify_stage_exposure(partition, registry)
        with self.assertRaisesRegex(ValueError, "must precede"):
            classify_stage_exposure(
                partition,
                registry,
                specification_frozen_at="2026-08-12T00:00:00+08:00",
            )
        result = classify_stage_exposure(
            partition,
            registry,
            specification_frozen_at="2026-08-11T23:59:59+08:00",
        )
        self.assertEqual(result["evidence_class"], "prospective_unseen")
        self.assertEqual(result["claim_classification"], "research_only")


class ExposureArtifactBindingTests(unittest.TestCase):
    def setUp(self):
        self.plan = _historical_plan()

    def test_all_current_stages_are_explicitly_retrospective(self):
        provenance = self.plan["exposure_provenance"]
        self.assertEqual(provenance["known_through_session"], "2026-08-11")
        for record in provenance["stage_classification"].values():
            self.assertEqual(record["evidence_class"], "retrospective_exposed")
            self.assertEqual(record["claim_classification"], "research_only")
            self.assertFalse(record["promotion_eligible"])

    def test_rehashed_plan_cannot_relabel_exposed_history(self):
        tampered = json.loads(json.dumps(self.plan))
        tampered["exposure_provenance"]["stage_classification"]["confirmation"][
            "claim_classification"
        ] = "promotion"
        tampered.pop("plan_sha256")
        tampered["plan_sha256"] = _digest(tampered)
        with self.assertRaisesRegex(ValueError, "fixed registry"):
            validate_research_plan(tampered)

    def test_rehashed_request_cannot_relabel_exposed_history(self):
        spec = build_baseline_experiment_spec(self.plan)
        request = build_stage_run_request(
            self.plan, spec, self.plan["partitions"]["confirmation"]
        )
        request["evidence_class"] = "prospective_unseen"
        request.pop("request_sha256")
        request["request_sha256"] = _digest(request)
        with self.assertRaisesRegex(ValueError, "exposure provenance"):
            validate_stage_run_request(self.plan, request)

    def test_result_fields_are_exactly_plan_bound(self):
        result = {
            **stage_exposure_fields(self.plan, "locked_holdout"),
            "stage": "locked_holdout",
        }
        validate_stage_exposure_fields(
            result, self.plan, "locked_holdout", artifact="result"
        )
        del result["exposure_registry_sha256"]
        with self.assertRaisesRegex(ValueError, "exposure provenance"):
            validate_stage_exposure_fields(
                result, self.plan, "locked_holdout", artifact="result"
            )

    def test_rehashed_state_cannot_relabel_exposed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            initialize_research_state(path, self.plan)
            state = json.loads(path.read_text(encoding="utf-8"))
            state["stages"]["locked_holdout"]["evidence_class"] = "prospective_unseen"
            state.pop("state_sha256")
            state["state_sha256"] = _digest(state)
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exposure provenance"):
                read_research_state(path, self.plan)


if __name__ == "__main__":
    unittest.main()
