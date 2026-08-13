import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline import cli
from quant_pipeline.factor_research import read_research_state
from quant_pipeline.factors import ORIGINAL_RESEARCH_CANDIDATES
from quant_pipeline.io import read_json
from quant_pipeline.metrics import max_drawdown


class ComparisonOutputTests(unittest.TestCase):
    def test_default_output_is_a_direct_comparisons_child(self):
        output = cli.resolve_comparison_output(None, "baseline", "candidate")
        self.assertEqual(output, (cli.PIPELINE_ROOT / "comparisons" / "baseline__vs__candidate.json").resolve())

    def test_rejects_output_outside_comparisons_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "direct child"):
                cli.resolve_comparison_output(Path(directory) / "result.json", "baseline", "candidate")

    def test_accepts_explicit_direct_child_json(self):
        target = cli.PIPELINE_ROOT / "comparisons" / "paired.json"
        with patch.object(Path, "cwd", return_value=cli.PIPELINE_ROOT.parent):
            output = cli.resolve_comparison_output(target, "baseline", "candidate")
        self.assertEqual(output, target.resolve())


class ResearchCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.calendar = self.root / "day.txt"
        sessions = pd.bdate_range("2025-01-02", periods=258)
        self.calendar.write_text(
            "\n".join(timestamp.date().isoformat() for timestamp in sessions) + "\n",
            encoding="utf-8",
        )
        self.sessions = sessions
        self.research_root = self.root / "research"
        self.config = cli.PIPELINE_ROOT / "configs" / "baseline.yaml"

    def run_cli(self, *arguments):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["quant-pipeline", *arguments]), patch(
            "sys.stdout", stdout
        ):
            cli.main()
        return json.loads(stdout.getvalue())

    def init_arguments(self):
        return (
            "research",
            "init",
            "--study-id",
            "frozen-study",
            "--config",
            str(self.config),
            "--calendar",
            str(self.calendar),
            "--discovery-end",
            self.sessions[125].date().isoformat(),
            "--confirmation-end",
            self.sessions[190].date().isoformat(),
            "--research-root",
            str(self.research_root),
        )

    def test_research_init_and_status_create_only_unopened_control_artifacts(self):
        initialized = self.run_cli(*self.init_arguments())
        self.assertTrue(initialized["valid"])
        self.assertEqual(initialized["experiment_count"], 24)
        self.assertEqual(initialized["frozen_candidate_status"], "awaiting_discovery_freeze")
        self.assertTrue(all(stage["status"] == "unopened" for stage in initialized["stages"].values()))

        workspace = self.research_root / "frozen-study"
        self.assertEqual(
            {path.name for path in workspace.iterdir()},
            {"base_config.json", "experiments.json", "manifest.json", "plan.json", "state.json"},
        )
        status = self.run_cli(
            "research",
            "status",
            "--study-id",
            "frozen-study",
            "--research-root",
            str(self.research_root),
        )
        self.assertEqual(status["plan_sha256"], initialized["plan_sha256"])
        self.assertEqual(status["stages"], initialized["stages"])

    def test_research_init_refuses_overwrite_and_export_is_not_a_run_claim(self):
        self.run_cli(*self.init_arguments())
        with self.assertRaises(SystemExit):
            self.run_cli(*self.init_arguments())

        output = self.root / "exported_experiments.json"
        result = self.run_cli(
            "research",
            "manifest",
            "--study-id",
            "frozen-study",
            "--research-root",
            str(self.research_root),
            "--output",
            str(output),
        )
        self.assertTrue(result["valid"])
        manifest = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "not_run")
        self.assertNotIn("run_id", json.dumps(manifest))

    def test_research_status_detects_frozen_config_tampering(self):
        self.run_cli(*self.init_arguments())
        path = self.research_root / "frozen-study" / "base_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["config"]["execution"]["account"] = 999999
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_cli(
                "research",
                "status",
                "--study-id",
                "frozen-study",
                "--research-root",
                str(self.research_root),
            )


class ResearchExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name).resolve()
        self.workspace = self.root / "research" / "study"
        self.repository = cli.PIPELINE_ROOT.parent.resolve()
        calendar = self.root / "day.txt"
        self.sessions = pd.bdate_range("2025-01-02", periods=258)
        calendar.write_text(
            "\n".join(value.date().isoformat() for value in self.sessions) + "\n",
            encoding="utf-8",
        )
        from quant_pipeline.research_cli import initialize_research_workspace

        initialize_research_workspace(
            research_root=self.root / "research",
            study_id="study",
            config_path=cli.DEFAULT_CONFIG,
            calendar_path=calendar,
            discovery_end=self.sessions[125].date().isoformat(),
            confirmation_end=self.sessions[190].date().isoformat(),
        )

    @staticmethod
    def metric_for_request(request, *, effect=0.0, missing=0):
        index = pd.DatetimeIndex(request["partition"]["sessions"], name="datetime")
        values = np.sin(np.arange(len(index)) / 7.0) * 0.01 + effect
        result = pd.Series(values, index=index, name="rank_ic")
        if missing:
            result.iloc[:missing] = np.nan
        return result

    @staticmethod
    def evidence_for_request(request, *, effect=0.0, net_effect=None, missing=0):
        rank_ic = ResearchExecutionTests.metric_for_request(
            request, effect=effect, missing=missing
        )
        index = pd.DatetimeIndex(
            request["partition"]["portfolio_evaluation_sessions"], name="datetime"
        )
        baseline_net = np.sin(np.arange(len(index)) / 9.0) * 0.001
        strategy_net = pd.Series(
            baseline_net + (effect / 10.0 if net_effect is None else net_effect),
            index=index,
            name="strategy_net",
        )
        benchmark = pd.Series(0.0, index=index, name="benchmark")
        factor_names = request["experiment"]["features"]["factor_names"]
        directions = {factor.name: factor.direction for factor in ORIGINAL_RESEARCH_CANDIDATES}
        raw_factor_rank_ic = {
            name: pd.Series(
                directions[name] * (0.004 + np.sin(np.arange(len(rank_ic)) / 11.0) * 0.0001),
                index=rank_ic.index,
                name=f"{name}__rank_ic",
            )
            for name in factor_names
        }
        fold_records = []
        for fold in request["partition"]["research_folds"]:
            fold_effect = effect * (fold["fold"] + 1) / 10.0
            fold_records.append(
                {
                    **{
                        key: fold[key]
                        for key in (
                            "fold", "signal_start", "signal_end", "evaluation_start",
                            "evaluation_end", "complete_for_gate",
                        )
                    },
                    "initial_account": 20_000.0,
                    "stress_slippage_bps_per_side": 10,
                    "engine": "raw_share_daily_v1",
                    "terminal_account": 20_000.0 + fold_effect,
                    "benchmark_terminal_account": 20_000.0,
                    "single_etf_abs_contribution_symbol": "SH510300",
                    "single_etf_abs_contribution_numerator_cny": 1.0,
                    "single_etf_abs_contribution_denominator_cny": 10.0,
                    "single_etf_abs_contribution_share": 0.1,
                }
            )
        return {
            "rank_ic": rank_ic,
            "raw_factor_rank_ic": raw_factor_rank_ic,
            "strategy_net": strategy_net,
            "benchmark": benchmark,
            "portfolio": {
                "benchmark": "SH510300",
                "account_currency": "CNY",
                "initial_account": 20_000.0,
                "stress_slippage_bps_per_side": 10,
                "engine": "raw_share_daily_v1",
                "alignment_method": "initial_cost_compounded_into_first_realized_return",
                "evaluation_sessions_sha256": request["partition"][
                    "portfolio_evaluation_sessions_sha256"
                ],
                "terminal_account": 20_000.0 * float((1.0 + strategy_net).prod()),
                "benchmark_terminal_account": 20_000.0,
                "intent_fill_rate": 1.0,
                "notional_fill_rate": 1.0,
                "zero_fill_intent_rate": 0.0,
                "strategy_max_drawdown": float(max_drawdown(strategy_net)),
                "single_etf_abs_contribution_symbol": "SH510300",
                "single_etf_abs_contribution_numerator_cny": 1.0,
                "single_etf_abs_contribution_denominator_cny": 10.0,
                "single_etf_abs_contribution_share": 0.1,
                "research_folds": fold_records,
            },
        }

    def run_protocol(self, *, confirmation_effect=0.002, missing_baseline=0):
        from quant_pipeline.research_cli import execute_research_workspace

        calls = []

        def fake_pipeline(config, *, run_id, research_plan, research_request, research_state_path):
            calls.append(research_request)
            return Path(config["paths"]["runs"]) / run_id

        def fake_load(run_dir, base_config, plan, request):
            role = request["experiment"]["role"]
            stage = request["stage"]
            effect = 0.0
            frequency = 1.0
            if role == "single_factor":
                position = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES].index(
                    request["experiment"]["factor_name"]
                )
                effect = 0.002 if position < 2 else -0.0002
                frequency = position + 1.0
            elif role == "family_ablation":
                effect = 0.0001
                frequency = 2.0
            elif role == "frozen_candidate":
                effect = confirmation_effect
                frequency = 3.0
            missing = missing_baseline if role == "baseline" and stage == "discovery" else 0
            evidence = self.evidence_for_request(request, effect=effect, missing=missing)
            if role != "baseline":
                evidence["rank_ic"] = evidence["rank_ic"] + np.cos(
                    np.arange(len(evidence["rank_ic"])) * frequency / 17.0
                ) * 0.0002
                evidence["strategy_net"] = evidence["strategy_net"] + np.cos(
                    np.arange(len(evidence["strategy_net"])) * frequency / 19.0
                ) * 0.00002
                evidence["portfolio"]["terminal_account"] = 20_000.0 * float(
                    (1.0 + evidence["strategy_net"]).prod()
                )
                evidence["portfolio"]["strategy_max_drawdown"] = float(
                    max_drawdown(evidence["strategy_net"])
                )
            return evidence, (
                "a" * 64,
                "snapshot",
                "b" * 64,
            )

        with patch(
            "quant_pipeline.research_cli.load_completed_stage_evidence",
            side_effect=fake_load,
        ):
            result = execute_research_workspace(
                self.workspace,
                repository_root=self.repository,
                pipeline_runner=fake_pipeline,
            )
        return result, calls

    def test_execute_runs_24_then_confirmation_and_holdout_and_is_not_replayable(self):
        result, calls = self.run_protocol()
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(len(calls), 28)
        self.assertEqual([request["stage"] for request in calls[:24]], ["discovery"] * 24)
        discovery = calls[:24]
        self.assertEqual(sum(request["experiment"]["role"] == "baseline" for request in discovery), 1)
        self.assertEqual(sum(request["experiment"]["role"] == "family_ablation" for request in discovery), 5)
        self.assertEqual(sum(request["experiment"]["role"] == "single_factor" for request in discovery), 18)
        plan = read_json(self.workspace / "plan.json")
        state = read_research_state(self.workspace / "state.json", plan)
        expected = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES[:2]]
        self.assertEqual(state["frozen_confirmation_spec"]["selected_factor_names"], expected)
        self.assertEqual(
            calls[25]["experiment"]["features"]["factor_names"], expected
        )
        self.assertEqual(
            calls[27]["experiment"]["spec_sha256"],
            calls[25]["experiment"]["spec_sha256"],
        )
        resumed, repeated_calls = self.run_protocol()
        self.assertEqual(resumed["execution_status"], "completed")
        self.assertEqual(repeated_calls, [])

    def test_confirmation_failure_never_opens_holdout(self):
        result, calls = self.run_protocol(confirmation_effect=0.0)
        self.assertEqual(result["execution_status"], "stopped_confirmation_failed")
        self.assertEqual(len(calls), 26)
        plan = read_json(self.workspace / "plan.json")
        state = read_research_state(self.workspace / "state.json", plan)
        self.assertEqual(state["stages"]["locked_holdout"]["status"], "unopened")

    def test_coverage_failure_consumes_discovery_and_persists_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "below 90.00%"):
            self.run_protocol(missing_baseline=13)
        plan = read_json(self.workspace / "plan.json")
        state = read_research_state(self.workspace / "state.json", plan)
        self.assertEqual(state["stages"]["discovery"]["status"], "failed")
        evidence = read_json(self.workspace / "execution.json")
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(
            evidence["stages"]["discovery"]["experiments"][0]["status"], "failed"
        )
        with self.assertRaisesRegex(RuntimeError, "automatic replay is forbidden"):
            self.run_protocol()

    def test_preflight_failure_leaves_stage_and_files_unclaimed(self):
        from quant_pipeline.research_cli import execute_research_workspace

        calls = []

        def pipeline(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("pipeline must not run")

        def fail_preflight(config, plan, root):
            raise RuntimeError("preflight failed")

        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            execute_research_workspace(
                self.workspace,
                repository_root=self.repository,
                pipeline_runner=pipeline,
                preflight_runner=fail_preflight,
            )
        plan = read_json(self.workspace / "plan.json")
        state = read_research_state(self.workspace / "state.json", plan)
        self.assertEqual(state["stages"]["discovery"]["status"], "unopened")
        self.assertEqual(state["stages"]["discovery"]["attempts"], 0)
        self.assertFalse((self.workspace / "execution.json").exists())
        self.assertEqual(calls, [])

    def test_run_id_collision_is_detected_before_stage_claim(self):
        from quant_pipeline.research_cli import (
            _run_id,
            execute_research_workspace,
        )
        from quant_pipeline.research_runner import (
            build_baseline_experiment_spec,
            build_stage_run_request,
        )

        plan = read_json(self.workspace / "plan.json")
        request = build_stage_run_request(
            plan,
            build_baseline_experiment_spec(plan),
            plan["partitions"]["discovery"],
        )
        run_dir = self.repository / "pipeline" / "runs" / _run_id(plan, request)
        run_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: run_dir.rmdir() if run_dir.exists() else None)

        with self.assertRaisesRegex(FileExistsError, "before stage claim"):
            execute_research_workspace(
                self.workspace,
                repository_root=self.repository,
                pipeline_runner=lambda *args, **kwargs: None,
            )
        state = read_research_state(self.workspace / "state.json", plan)
        self.assertEqual(state["stages"]["discovery"]["status"], "unopened")
        self.assertEqual(state["stages"]["discovery"]["attempts"], 0)
        self.assertFalse((self.workspace / "execution.json").exists())


if __name__ == "__main__":
    unittest.main()
