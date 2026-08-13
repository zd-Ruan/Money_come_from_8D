import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline import cli
from quant_pipeline.factor_research import (
    RESEARCH_STAGES,
    analyze_confirmation,
    analyze_discovery,
    analyze_locked_holdout,
    evaluate_stage_once,
    freeze_confirmation_spec,
    read_research_state,
)
from quant_pipeline.exposure import stage_exposure_fields
from quant_pipeline.factors import FACTOR_FAMILIES, ORIGINAL_RESEARCH_CANDIDATES
from quant_pipeline.io import write_json_atomic
from quant_pipeline.metrics import max_drawdown
from quant_pipeline.research_cli import (
    RESEARCH_EXECUTION_SCHEMA_VERSION,
    initialize_research_workspace,
)
from quant_pipeline.research_runner import (
    build_research_experiment_manifest,
    build_stage_run_request,
)
from quant_pipeline.web import create_app


class ResearchWebTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.pipeline_root = Path(temporary.name).resolve()
        (self.pipeline_root / "runs").mkdir()
        (self.pipeline_root / "comparisons").mkdir()
        write_json_atomic(self.pipeline_root / "registry.json", {"runs": []})
        self.research_root = self.pipeline_root / "research"
        calendar = self.pipeline_root / "calendar.txt"
        self.sessions = pd.bdate_range("2025-01-02", periods=258)
        calendar.write_text(
            "\n".join(session.date().isoformat() for session in self.sessions) + "\n",
            encoding="utf-8",
        )
        self.workspace = initialize_research_workspace(
            research_root=self.research_root,
            study_id="web-study",
            config_path=cli.DEFAULT_CONFIG,
            calendar_path=calendar,
            discovery_end=self.sessions[125].date().isoformat(),
            confirmation_end=self.sessions[190].date().isoformat(),
        )
        self.client = TestClient(create_app(self.pipeline_root))
        self.addCleanup(self.client.close)

    @staticmethod
    def _stage_series(plan, stage, values):
        index = pd.DatetimeIndex(plan["partitions"][stage]["sessions"], name="datetime")
        if np.isscalar(values):
            values = np.full(len(index), float(values))
        return pd.Series(values, index=index, dtype=float)

    @classmethod
    def _stage_evidence(
        cls,
        plan,
        stage,
        signal_values,
        *,
        net_values=None,
        factor_names=(),
    ):
        signal = cls._stage_series(plan, stage, signal_values)
        evaluation_index = pd.DatetimeIndex(
            plan["partitions"][stage]["portfolio_evaluation_sessions"], name="datetime"
        )
        if net_values is None:
            net_values = np.sin(np.arange(len(evaluation_index)) / 9.0) * 0.0001
        if np.isscalar(net_values):
            net_values = np.full(len(evaluation_index), float(net_values))
        strategy_net = pd.Series(
            net_values, index=evaluation_index, dtype=float, name="strategy_net"
        )
        benchmark = pd.Series(
            np.full(len(evaluation_index), -0.0001),
            index=evaluation_index,
            dtype=float,
            name="benchmark",
        )
        factor_by_name = {factor.name: factor for factor in ORIGINAL_RESEARCH_CANDIDATES}
        raw_metrics = {
            name: cls._stage_series(
                plan,
                stage,
                factor_by_name[name].direction
                * (0.02 + np.sin(np.arange(len(signal)) / (7.0 + position)) * 0.001),
            )
            for position, name in enumerate(factor_names)
        }
        folds = []
        for fold in plan["partitions"][stage]["research_folds"]:
            fold_index = pd.DatetimeIndex(fold["evaluation_sessions"], name="datetime")
            fold_net = strategy_net.loc[fold_index]
            fold_benchmark = benchmark.loc[fold_index]
            folds.append(
                {
                    "fold": fold["fold"],
                    "signal_start": fold["signal_start"],
                    "signal_end": fold["signal_end"],
                    "evaluation_start": fold["evaluation_start"],
                    "evaluation_end": fold["evaluation_end"],
                    "complete_for_gate": fold["complete_for_gate"],
                    "initial_account": 20_000.0,
                    "stress_slippage_bps_per_side": 10,
                    "engine": "raw_share_daily_v1",
                    "terminal_account": 20_000.0 * float((1.0 + fold_net).prod()),
                    "benchmark_terminal_account": 20_000.0
                    * float((1.0 + fold_benchmark).prod()),
                    "single_etf_abs_contribution_share": 0.20,
                    "single_etf_abs_contribution_symbol": "SH510300",
                    "single_etf_abs_contribution_numerator_cny": 20.0,
                    "single_etf_abs_contribution_denominator_cny": 100.0,
                }
            )
        return {
            "rank_ic": signal,
            "raw_factor_rank_ic": raw_metrics,
            "strategy_net": strategy_net,
            "benchmark": benchmark,
            "portfolio": {
                "benchmark": "SH510300",
                "account_currency": "CNY",
                "initial_account": 20_000.0,
                "stress_slippage_bps_per_side": 10,
                "engine": "raw_share_daily_v1",
                "alignment_method": "initial_cost_compounded_into_first_realized_return",
                "evaluation_sessions_sha256": plan["partitions"][stage][
                    "portfolio_evaluation_sessions_sha256"
                ],
                "terminal_account": 20_000.0 * float((1.0 + strategy_net).prod()),
                "benchmark_terminal_account": 20_000.0 * float((1.0 + benchmark).prod()),
                "intent_fill_rate": 1.0,
                "notional_fill_rate": 1.0,
                "zero_fill_intent_rate": 0.0,
                "strategy_max_drawdown": max_drawdown(strategy_net),
                "single_etf_abs_contribution_share": 0.20,
                "single_etf_abs_contribution_symbol": "SH510300",
                "single_etf_abs_contribution_numerator_cny": 20.0,
                "single_etf_abs_contribution_denominator_cny": 100.0,
                "research_folds": folds,
            },
        }

    @staticmethod
    def _stage_specs(stage, manifest):
        if stage == "discovery":
            discovery = manifest["discovery"]
            return [
                discovery["baseline"],
                *discovery["family_ablations"],
                *discovery["single_factor_tests"],
            ]
        return [manifest["discovery"]["baseline"], manifest["frozen_candidate"]]

    @staticmethod
    def _sealed(value):
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def _write_execution(self, plan, manifest):
        state = read_research_state(self.workspace / "state.json", plan)
        stages = {}
        for stage in RESEARCH_STAGES:
            result = state["stages"][stage]["result"]
            records = []
            for spec in self._stage_specs(stage, manifest):
                request = build_stage_run_request(plan, spec, plan["partitions"][stage])
                records.append(
                    {
                        "experiment_id": spec["experiment_id"],
                        "request_sha256": request["request_sha256"],
                        "run_id": result["run_ids"][spec["experiment_id"]],
                        "status": "completed",
                        "completed_at": "2026-08-13T12:00:00+08:00",
                        **stage_exposure_fields(plan, stage),
                    }
                )
            stages[stage] = {
                "status": "completed",
                "completed_at": "2026-08-13T12:00:00+08:00",
                "result_sha256": state["stages"][stage]["result_sha256"],
                "experiments": records,
                **stage_exposure_fields(plan, stage),
            }
        execution = {
            "schema_version": RESEARCH_EXECUTION_SCHEMA_VERSION,
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["plan_sha256"],
            "exposure_registry_sha256": plan["exposure_provenance"]["registry_sha256"],
            "status": "completed",
            "started_at": "2026-08-13T12:00:00+08:00",
            "updated_at": "2026-08-13T12:00:00+08:00",
            "finished_at": "2026-08-13T12:00:00+08:00",
            "stages": stages,
        }
        execution["execution_sha256"] = self._sealed(execution)
        write_json_atomic(self.workspace / "execution.json", execution)

    def _complete_study(self, *, with_runs=False, confirmation_terminal_delta=0.0):
        plan = json.loads((self.workspace / "plan.json").read_text(encoding="utf-8"))
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        discovery_signal = self._stage_series(
            plan, "discovery", np.sin(np.arange(126) / 7.0) * 0.01
        )
        baseline = self._stage_evidence(plan, "discovery", discovery_signal)
        base_net = baseline["strategy_net"].to_numpy()
        family_evidence = {
            family: self._stage_evidence(
                plan,
                "discovery",
                discovery_signal + 0.0003 + np.cos(np.arange(126) / 11.0) * 0.00002,
                net_values=base_net + 0.00003 + np.cos(np.arange(126) / 13.0) * 0.00001,
                factor_names=tuple(
                    factor.name
                    for factor in ORIGINAL_RESEARCH_CANDIDATES
                    if factor.family == family
                ),
            )
            for family in FACTOR_FAMILIES
        }
        candidate_evidence = {}
        for position, factor in enumerate(ORIGINAL_RESEARCH_CANDIDATES):
            passed = position == 0
            candidate_evidence[factor.name] = self._stage_evidence(
                plan,
                "discovery",
                discovery_signal
                + (0.002 if passed else -0.0003)
                + np.cos(np.arange(126) / (11.0 + position)) * 0.00002,
                net_values=base_net
                + (0.00025 if passed else -0.00005)
                + np.cos(np.arange(126) / (13.0 + position)) * 0.00001,
                factor_names=(factor.name,),
            )
        discovery = analyze_discovery(baseline, family_evidence, candidate_evidence, plan)
        self.assertEqual(discovery["selected_factor_names"], [selected])
        initial_manifest = build_research_experiment_manifest(plan)
        self.run_evidence = {}
        if with_runs:
            discovery["run_ids"] = {
                spec["experiment_id"]: f"run-discovery-{position:02d}"
                for position, spec in enumerate(self._stage_specs("discovery", initial_manifest))
            }
            self.run_evidence[("discovery", initial_manifest["discovery"]["baseline"]["experiment_id"])] = baseline
            for spec in initial_manifest["discovery"]["family_ablations"]:
                self.run_evidence[("discovery", spec["experiment_id"])] = family_evidence[
                    spec["family"]
                ]
            for spec in initial_manifest["discovery"]["single_factor_tests"]:
                self.run_evidence[("discovery", spec["experiment_id"])] = candidate_evidence[
                    spec["factor_name"]
                ]
        evaluate_stage_once(
            self.workspace / "state.json", plan, "discovery", lambda _partition: discovery
        )
        spec_sha256 = freeze_confirmation_spec(
            self.workspace / "state.json",
            plan,
            selected_factor_names=[selected],
            frozen_spec={"model": "lightgbm", "account": 20_000, "stress_bps": 10},
        )
        state = read_research_state(self.workspace / "state.json", plan)
        manifest = build_research_experiment_manifest(plan, state["frozen_confirmation_spec"])
        write_json_atomic(self.workspace / "experiments.json", manifest)
        for stage, passed_field in (
            ("confirmation", "confirmation_passed"),
            ("locked_holdout", "locked_holdout_passed"),
        ):
            observations = len(plan["partitions"][stage]["sessions"])
            signal = self._stage_series(
                plan, stage, np.sin(np.arange(observations) / 7.0) * 0.01
            )
            stage_baseline = self._stage_evidence(plan, stage, signal)
            stage_candidate = self._stage_evidence(
                plan,
                stage,
                signal + 0.002 + np.cos(np.arange(observations) / 11.0) * 0.00002,
                net_values=stage_baseline["strategy_net"].to_numpy()
                + 0.00025
                + np.cos(np.arange(observations) / 13.0) * 0.00001,
                factor_names=(selected,),
            )
            analyzer = analyze_confirmation if stage == "confirmation" else analyze_locked_holdout
            result = analyzer(
                stage_baseline,
                stage_candidate,
                plan,
                frozen_spec_sha256=spec_sha256,
                candidate_factor_names=(selected,),
            )
            self.assertTrue(result[passed_field])
            if stage == "confirmation" and confirmation_terminal_delta:
                terminal = result["tests"]["terminal"]
                terminal["candidate_terminal_account"] += confirmation_terminal_delta
                terminal["account_improvement"] += confirmation_terminal_delta
                terminal["relative_wealth_improvement"] = (
                    terminal["candidate_terminal_account"]
                    / terminal["baseline_terminal_account"]
                    - 1.0
                )
            if with_runs:
                specs = self._stage_specs(stage, manifest)
                result["run_ids"] = {
                    spec["experiment_id"]: f"run-{stage}-{position:02d}"
                    for position, spec in enumerate(specs)
                }
                self.run_evidence[(stage, specs[0]["experiment_id"])] = stage_baseline
                self.run_evidence[(stage, specs[1]["experiment_id"])] = stage_candidate
            evaluate_stage_once(
                self.workspace / "state.json", plan, stage, lambda _partition, value=result: value
            )
        if with_runs:
            self._write_execution(plan, manifest)
        return plan

    def test_initialized_workspace_is_verified_and_conservatively_classified(self):
        detail = self.client.get("/api/research/web-study")
        self.assertEqual(detail.status_code, 200)
        payload = detail.json()
        self.assertTrue(payload["verification"]["verified"])
        self.assertTrue(payload["verification"]["sealed"])
        self.assertEqual(payload["claim_status"], "research_only")
        self.assertEqual(payload["exposure"]["evidence_class"], "retrospective_exposed")
        self.assertEqual(payload["exposure"]["evidence_label"], "历史已暴露")
        self.assertEqual(payload["account_contract"]["initial_account"], 20_000.0)
        self.assertEqual(payload["account_contract"]["required_stress_slippage_bps_per_side"], 10)
        self.assertEqual([stage["name"] for stage in payload["stages"]], list(RESEARCH_STAGES))
        self.assertTrue(all(stage["status"] == "unopened" for stage in payload["stages"]))
        self.assertNotIn(str(self.pipeline_root), detail.text)

        listing = self.client.get("/api/research")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual([item["study_id"] for item in listing.json()["studies"]], ["web-study"])
        page = self.client.get("/research")
        self.assertEqual(page.status_code, 200)
        self.assertIn("历史已暴露", page.text)
        self.assertIn("research_only", page.text)
        for forbidden in ("未见", "盲测", "unseen", "blind", "pristine"):
            self.assertNotIn(forbidden, page.text)

    def test_completed_workspace_renders_bh_confirmation_holdout_and_small_account(self):
        self._complete_study()
        response = self.client.get("/api/research/web-study")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["selected_factor_names"], [ORIGINAL_RESEARCH_CANDIDATES[0].name])
        self.assertEqual(payload["stages"][0]["result"]["bh"]["rejected_count"], 1)
        self.assertTrue(payload["stages"][1]["result"]["confirmation_passed"])
        self.assertTrue(payload["stages"][2]["result"]["locked_holdout_passed"])
        self.assertGreater(
            payload["stages"][2]["result"]["tests"]["terminal"][
                "candidate_terminal_account"
            ],
            20_000.0,
        )
        serialized = response.text
        for private in ("run_ids", "metric_coverage", "claim_token_sha256", "specification"):
            self.assertNotIn(private, serialized)

        page = self.client.get("/research/web-study")
        self.assertEqual(page.status_code, 200)
        for expected in (
            "发现期 BH 筛选",
            "Benjamini-Hochberg",
            ORIGINAL_RESEARCH_CANDIDATES[0].name,
            "确认期 · 通过",
            "锁定留出集 · 通过",
            "CNY 20,000.00",
            "10 bps",
            "最低意图成交率",
            "95.00%",
            "候选成交质量",
            "已密封并验证",
            "历史已暴露",
            "research_only",
        ):
            self.assertIn(expected, page.text)
        for forbidden in ("未见", "盲测", "unseen", "blind", "pristine"):
            self.assertNotIn(forbidden, page.text)

    def test_declared_runs_are_recomputed_without_exposing_run_identity(self):
        self._complete_study(with_runs=True)

        def load_evidence(_run_dir, _base_config, _plan, request):
            key = (request["stage"], request["experiment"]["experiment_id"])
            return self.run_evidence[key], ("snapshot", "source", "config")

        with patch(
            "quant_pipeline.web.load_completed_stage_evidence",
            side_effect=load_evidence,
        ):
            response = self.client.get("/api/research/web-study")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        confirmation = payload["stages"][1]["actual_evidence"]
        experiments = confirmation["experiments"]
        candidate_id = next(
            identifier for identifier in experiments if identifier != "alpha158_baseline"
        )
        candidate = experiments[candidate_id]
        declared_terminal = payload["stages"][1]["result"]["tests"]["terminal"][
            "candidate_terminal_account"
        ]
        self.assertAlmostEqual(candidate["terminal_account"], declared_terminal)
        self.assertLess(candidate["benchmark_terminal_account"], candidate["terminal_account"])
        self.assertLessEqual(abs(candidate["strategy_max_drawdown"]), 0.25)
        self.assertEqual(candidate["intent_fill_rate"], 1.0)
        self.assertEqual(candidate["notional_fill_rate"], 1.0)
        self.assertEqual(candidate["zero_fill_intent_rate"], 0.0)
        self.assertEqual(candidate["single_etf_abs_contribution_share"], 0.20)
        factor = candidate["raw_factor_rank_ic"][0]
        self.assertEqual(
            factor["expected_direction"], ORIGINAL_RESEARCH_CANDIDATES[0].direction
        )
        self.assertGreater(factor["signed_mean_rank_ic"], 0.0)
        folds = confirmation["fold_comparisons"][candidate_id]
        self.assertGreaterEqual(folds["win_ratio"], 0.60)
        self.assertGreaterEqual(folds["complete_folds"], 3)
        self.assertNotIn("run_ids", response.text)
        self.assertNotIn("run-confirmation", response.text)

    def test_declared_missing_run_fails_closed(self):
        self._complete_study(with_runs=True)
        response = self.client.get("/api/research/web-study")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            "research workspace exists but failed sealed-artifact verification",
        )

    def test_declared_terminal_disagreement_with_recomputed_run_fails_closed(self):
        self._complete_study(with_runs=True, confirmation_terminal_delta=10.0)

        def load_evidence(_run_dir, _base_config, _plan, request):
            key = (request["stage"], request["experiment"]["experiment_id"])
            return self.run_evidence[key], ("snapshot", "source", "config")

        with patch(
            "quant_pipeline.web.load_completed_stage_evidence",
            side_effect=load_evidence,
        ):
            response = self.client.get("/api/research/web-study")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            "research workspace exists but failed sealed-artifact verification",
        )

    def test_missing_or_tampered_sealed_artifacts_fail_explicitly(self):
        self.assertEqual(self.client.get("/api/research/missing").status_code, 404)
        self.assertEqual(self.client.get("/research/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/research/%2e%2e%5cregistry").status_code, 404)

        experiments_path = self.workspace / "experiments.json"
        experiments_document = experiments_path.read_text(encoding="utf-8")
        experiments_path.unlink()
        missing = self.client.get("/api/research/web-study")
        self.assertEqual(missing.status_code, 409)
        self.assertEqual(
            missing.json()["detail"],
            "research workspace exists but failed sealed-artifact verification",
        )
        experiments_path.write_text(experiments_document, encoding="utf-8")

        state_path = self.workspace / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["revision"] = 999
        state_path.write_text(json.dumps(state), encoding="utf-8")
        for path in ("/api/research/web-study", "/research/web-study", "/api/research"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 409)
            self.assertEqual(
                response.json()["detail"],
                "research workspace exists but failed sealed-artifact verification",
            )
            self.assertNotIn(str(self.pipeline_root), response.text)


if __name__ == "__main__":
    unittest.main()
