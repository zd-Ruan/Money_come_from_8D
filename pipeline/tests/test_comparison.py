import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.comparison import compare_completed_runs, generate_comparison_json
from quant_pipeline.factors import factor_catalog_manifest
from quant_pipeline.integrity import generate_artifact_checksums
from quant_pipeline.io import sha256_file
from quant_pipeline.metrics import independent_portfolio_performance


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.baseline = root / "baseline-run"
        self.candidate = root / "candidate-run"
        self.baseline.mkdir()
        self.candidate.mkdir()
        self.dates = pd.bdate_range("2026-01-05", periods=42)
        self.signal_dates = self.dates[:-2]
        self.fingerprint = "a" * 64
        self._create_run(self.baseline, candidate=False)
        self._create_run(self.candidate, candidate=True)
        self._seal_run(self.baseline)
        self._seal_run(self.candidate)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _config(self, candidate):
        return {
            "project": {
                "name": "candidate" if candidate else "baseline",
                "description": "candidate features" if candidate else "Alpha158",
                "random_seeds": [11, 12, 13],
            },
            "paths": {
                "qlib_provider": "shared/provider",
                "universe": "shared/universe.csv",
                "instruments": "shared/t1_etf.txt",
                "validation_report": "shared/validation.json",
                "runs": "candidate/runs" if candidate else "baseline/runs",
                "snapshots": "shared/snapshots",
                "registry": "candidate/registry.json" if candidate else "baseline/registry.json",
            },
            "data": {
                "region": "cn",
                "market": "t1_etf",
                "benchmark": "SH510300",
                "start_date": "2020-01-01",
                "end_date": "2026-03-03",
                "test_start_date": "2026-01-05",
                "label": "Ref($close,-2)/Ref($close,-1)-1",
                "label_horizon_bars": 2,
                "liquidity_expression": "Mean($close*$volume,20)>10000000",
                "universe_mode": "current_snapshot",
            },
            "rolling": {"train_mode": "expanding", "validation_days": 126, "test_days": 20, "purge_bars": 2},
            "model": {
                "name": "lightgbm",
                "objective": "regression",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "num_boost_round": 500,
            },
            "strategy": {"topk": 20, "n_drop": 4, "hold_thresh": 1, "risk_degree": 0.95},
            "execution": {
                "account": 10_000_000,
                "deal_price": "close",
                "commission_bps_per_side": 3,
                "base_slippage_bps_per_side": 5,
                "min_cost": 5,
                "limit_threshold": 0.1,
                "max_daily_volume_participation": 0.05,
                "stress_slippage_bps_per_side": [0, 5, 10],
            },
            "gates": {"min_hac_t_stat": 1.96},
            "features": {
                "mode": "alpha158_plus_original" if candidate else "alpha158",
                "families": ["trend_crowding"] if candidate else [],
            },
            "report": {"title": "Candidate" if candidate else "Baseline", "language": "zh-CN"},
            "_meta": {
                "config_path": "candidate.yaml" if candidate else "baseline.yaml",
                "workspace_root": "ignored/runtime/path",
            },
        }

    def _create_run(self, run_dir, candidate):
        config = self._config(candidate)
        catalog = factor_catalog_manifest(["trend_crowding"]) if candidate else None
        manifest = {
            "run_id": run_dir.name,
            "status": "completed",
            "snapshot_id": "snapshot-1",
            "data": {"source_fingerprint": self.fingerprint, "snapshot_id": "snapshot-1"},
            "environment": {"qlib": "0.9.7", "lightgbm": "4.7.0"},
        }
        if catalog is not None:
            manifest["factor_catalog"] = catalog
        metrics = {
            "last_realized_signal_date": self.signal_dates[-1].date().isoformat(),
            "backtest_end_date": self.dates[-1].date().isoformat(),
        }
        folds = [
            {
                "fold": 1,
                "train_start": "2020-01-01",
                "train_end": "2025-01-01",
                "valid_start": "2025-01-06",
                "valid_end": "2025-12-31",
                "test_start": self.dates[0].date().isoformat(),
                "test_end": self.dates[20].date().isoformat(),
                "purge_bars": 2,
            },
            {
                "fold": 2,
                "train_start": "2020-01-01",
                "train_end": "2025-02-01",
                "valid_start": "2025-02-06",
                "valid_end": "2026-01-31",
                "test_start": self.dates[21].date().isoformat(),
                "test_end": self.dates[-1].date().isoformat(),
                "purge_bars": 2,
            },
        ]
        for filename, value in (
            ("manifest.json", manifest),
            ("config.json", config),
            ("metrics.json", metrics),
            ("folds.json", folds),
        ):
            (run_dir / filename).write_text(json.dumps(value), encoding="utf-8")

        day = np.arange(len(self.dates), dtype=float)
        baseline_return = 0.0005 + 0.0002 * np.sin(day)
        improvement = 0.0015 + 0.0003 * np.cos(day * 0.7)
        strategy_return = baseline_return + improvement if candidate else baseline_return
        strategy_return[0] = 0.0
        report = pd.DataFrame(
            {
                "return": strategy_return,
                "cost": np.zeros(len(day)),
                "bench": 0.0003 + 0.0001 * np.sin(day * 0.5),
            },
            index=self.dates,
        )
        report.index.name = "datetime"
        scenario = run_dir / "backtests" / "slippage_05bps"
        scenario.mkdir(parents=True)
        report.to_parquet(scenario / "report.parquet")

        # Each fold is a separate cash-reset portfolio. Signal t is executed at
        # t+1 close and becomes a realized return in the following report row.
        fold_ranges = (
            (self.dates[1:23], True),
            (self.dates[22:], False),
        )
        for fold, (fold_dates, complete) in zip(folds, fold_ranges):
            fold_day = np.arange(len(fold_dates), dtype=float)
            fold_return = 0.0005 + 0.0002 * np.sin(fold_day)
            if candidate:
                fold_return += 0.0015 + 0.0003 * np.cos(fold_day * 0.7)
            fold_return[0] = 0.0
            fold_report = pd.DataFrame(
                {
                    "return": fold_return,
                    "cost": np.zeros(len(fold_dates)),
                    "bench": 0.0003 + 0.0001 * np.sin(fold_day * 0.5),
                },
                index=fold_dates,
            )
            fold_report.index.name = "datetime"
            fold_root = run_dir / "folds" / f"fold_{fold['fold']:02d}"
            (fold_root / "backtest").mkdir(parents=True)
            fold_report.to_parquet(fold_root / "backtest" / "report.parquet")
            portfolio = independent_portfolio_performance(fold_report)
            portfolio.update(
                {
                    "start": fold_dates[0].date().isoformat(),
                    "end": fold_dates[-1].date().isoformat(),
                    "complete_for_gate": complete,
                }
            )
            (fold_root / "summary.json").write_text(
                json.dumps({**fold, "portfolio": portfolio}), encoding="utf-8"
            )

        signal_day = np.arange(len(self.signal_dates), dtype=float)
        baseline_ic = 0.01 + 0.003 * np.sin(signal_day)
        baseline_rank = 0.02 + 0.003 * np.cos(signal_day)
        signal = pd.DataFrame(
            {
                "ic": baseline_ic + (0.008 + 0.001 * np.cos(signal_day * 0.6) if candidate else 0),
                "rank_ic": baseline_rank + (0.009 + 0.001 * np.sin(signal_day * 0.4) if candidate else 0),
            },
            index=self.signal_dates,
        )
        signal.index.name = "datetime"
        signal.to_parquet(run_dir / "signal_metrics.parquet")

    def _seal_run(self, run_dir):
        manifest = self._read_json(run_dir, "manifest.json")
        manifest["code"] = {"source_tree_sha256": "c" * 64}
        manifest["artifacts"] = {"artifact_checksums": "artifact_checksums.json"}
        self._write_json(run_dir, "manifest.json", manifest)
        checksum_path = generate_artifact_checksums(run_dir)
        checksum_payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        manifest["integrity"] = {
            "checksum_manifest": checksum_path.name,
            "checksum_sha256": sha256_file(checksum_path),
            "artifact_count": len(checksum_payload["artifacts"]),
            "verified": True,
        }
        self._write_json(run_dir, "manifest.json", manifest)

    def _read_json(self, run_dir, filename):
        return json.loads((run_dir / filename).read_text(encoding="utf-8"))

    def _write_json(self, run_dir, filename, value):
        (run_dir / filename).write_text(json.dumps(value), encoding="utf-8")

    def _refresh_fold_summary(self, run_dir, fold_id):
        root = run_dir / "folds" / f"fold_{fold_id:02d}"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        report = pd.read_parquet(root / "backtest" / "report.parquet")
        complete = summary["portfolio"]["complete_for_gate"]
        portfolio = independent_portfolio_performance(report)
        portfolio.update(
            {
                "start": report.index.min().date().isoformat(),
                "end": report.index.max().date().isoformat(),
                "complete_for_gate": complete,
            }
        )
        summary["portfolio"] = portfolio
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def test_significant_paired_gain_is_improved(self):
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "improved")
        self.assertTrue(result["comparable"])
        self.assertGreater(result["deltas"]["terminal_relative_wealth"]["difference"], 0)
        self.assertGreater(result["deltas"]["daily_strategy_return"]["hac_t_stat"], 1.96)
        self.assertGreater(result["deltas"]["ic"]["hac_t_stat"], 1.96)
        self.assertEqual(result["deltas"]["folds"]["win_rate"], 1.0)
        self.assertEqual(result["deltas"]["folds"]["complete_folds"], 1)
        self.assertEqual(result["deltas"]["folds"]["residual_folds"], 1)
        self.assertFalse(result["deltas"]["folds"]["records"][1]["included_in_win_rate"])
        self.assertEqual(
            result["deltas"]["folds"]["records"][0]["start"],
            self.dates[2].date().isoformat(),
        )
        self.assertIn("supported", result["decision"]["claim"])

    def test_residual_fold_loss_is_displayed_but_excluded_from_win_rate(self):
        path = self.candidate / "folds" / "fold_02" / "backtest" / "report.parquet"
        report = pd.read_parquet(path)
        report.loc[report.index[1:], "return"] = -0.02
        report.to_parquet(path)
        self._refresh_fold_summary(self.candidate, 2)
        self._seal_run(self.candidate)

        result = compare_completed_runs(self.baseline, self.candidate)
        folds = result["deltas"]["folds"]
        self.assertEqual(result["status"], "improved")
        self.assertEqual(folds["wins"], 1)
        self.assertEqual(folds["losses"], 0)
        self.assertEqual(folds["win_rate"], 1.0)
        self.assertEqual(folds["records"][1]["outcome"], "loss")
        self.assertFalse(folds["records"][1]["included_in_win_rate"])

    def test_fold_summary_boundaries_must_match_independent_report(self):
        path = self.candidate / "folds" / "fold_01" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["portfolio"]["evaluation_end_date"] = "2026-12-31"
        path.write_text(json.dumps(summary), encoding="utf-8")
        self._seal_run(self.candidate)

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("evaluation_end_date" in reason for reason in result["reasons"]))

    def test_fold_benchmark_must_pair_exactly(self):
        path = self.candidate / "folds" / "fold_01" / "backtest" / "report.parquet"
        report = pd.read_parquet(path)
        report.loc[report.index[-1], "bench"] += 0.001
        report.to_parquet(path)
        self._refresh_fold_summary(self.candidate, 1)
        self._seal_run(self.candidate)

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("independent benchmark returns differ" in reason for reason in result["reasons"]))

    def test_fold_must_be_a_cash_reset_backtest(self):
        path = self.candidate / "folds" / "fold_01" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["portfolio"]["reset_cash"] = False
        path.write_text(json.dumps(summary), encoding="utf-8")
        self._seal_run(self.candidate)

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("not an independent cash-reset" in reason for reason in result["reasons"]))

    def test_fold_report_boundaries_must_follow_signal_timing(self):
        path = self.candidate / "folds" / "fold_01" / "backtest" / "report.parquet"
        report = pd.read_parquet(path).iloc[1:]
        report.iloc[0, report.columns.get_loc("return")] = 0.0
        report.to_parquet(path)
        self._refresh_fold_summary(self.candidate, 1)
        self._seal_run(self.candidate)

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("report dates differ" in reason for reason in result["reasons"]))

    def test_positive_terminal_gain_without_significance_is_not_improved(self):
        report_path = self.candidate / "backtests" / "slippage_05bps" / "report.parquet"
        report = pd.read_parquet(report_path)
        baseline = pd.read_parquet(self.baseline / "backtests" / "slippage_05bps" / "report.parquet")
        generator = np.random.default_rng(20260812)
        noise = generator.normal(0.0, 0.006, len(report))
        noise[1:] = noise[1:] - noise[1:].mean() + 0.0004
        noise[0] = 0.0
        report["return"] = baseline["return"].to_numpy() + noise
        report.iloc[0, report.columns.get_loc("return")] = 0.0
        report.to_parquet(report_path)
        self._seal_run(self.candidate)

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "not_improved")
        self.assertGreater(result["deltas"]["terminal_relative_wealth"]["difference"], 0)
        self.assertFalse(result["decision"]["criteria"]["daily_return_hac_significant"])
        self.assertIn("no positive claim", result["decision"]["claim"])

    def test_strategy_or_model_change_is_incomparable(self):
        config = self._read_json(self.candidate, "config.json")
        config["strategy"]["topk"] = 30
        config["model"]["num_leaves"] = 63
        self._write_json(self.candidate, "config.json", config)
        self._seal_run(self.candidate)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertIn("config.model.num_leaves", result["reasons"][0])
        self.assertIn("config.strategy.topk", result["reasons"][0])

    def test_account_or_cost_change_is_incomparable(self):
        config = self._read_json(self.candidate, "config.json")
        config["execution"]["account"] = 20_000_000
        config["execution"]["commission_bps_per_side"] = 4
        self._write_json(self.candidate, "config.json", config)
        self._seal_run(self.candidate)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertIn("config.execution.account", result["reasons"][0])
        self.assertIn("config.execution.commission_bps_per_side", result["reasons"][0])

    def test_data_fingerprint_or_test_date_change_is_incomparable(self):
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["data"]["source_fingerprint"] = "b" * 64
        self._write_json(self.candidate, "manifest.json", manifest)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("source_fingerprint differs" in reason for reason in result["reasons"]))

    def test_report_or_signal_dates_must_pair_exactly(self):
        signal_path = self.candidate / "signal_metrics.parquet"
        signal = pd.read_parquet(signal_path).iloc[1:]
        signal.to_parquet(signal_path)
        self._seal_run(self.candidate)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("signal metric test dates differ" in reason for reason in result["reasons"]))

    def test_metrics_dates_must_match_paired_artifacts(self):
        metrics = self._read_json(self.candidate, "metrics.json")
        metrics["backtest_end_date"] = "2026-12-31"
        self._write_json(self.candidate, "metrics.json", metrics)
        baseline_metrics = self._read_json(self.baseline, "metrics.json")
        baseline_metrics["backtest_end_date"] = "2026-12-31"
        self._write_json(self.baseline, "metrics.json", baseline_metrics)
        self._seal_run(self.baseline)
        self._seal_run(self.candidate)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("does not match the base report" in reason for reason in result["reasons"]))

    def test_overlapping_folds_are_incomparable(self):
        for run_dir in (self.baseline, self.candidate):
            folds = self._read_json(run_dir, "folds.json")
            folds[1]["test_start"] = folds[0]["test_end"]
            self._write_json(run_dir, "folds.json", folds)
            self._seal_run(run_dir)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("overlaps" in reason for reason in result["reasons"]))

    def test_catalog_tampering_is_incomparable(self):
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["factor_catalog"]["factors"][0]["direction"] *= -1
        self._write_json(self.candidate, "manifest.json", manifest)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("sha256 does not match" in reason for reason in result["reasons"]))

    def test_corrupt_parquet_is_incomparable_instead_of_raising(self):
        path = self.candidate / "signal_metrics.parquet"
        path.write_bytes(b"not a parquet file")
        self._seal_run(self.candidate)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("could not read candidate signal_metrics" in reason for reason in result["reasons"]))

    def test_modified_artifact_fails_checksum_verification(self):
        path = self.candidate / "metrics.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("modified: metrics.json" in reason for reason in result["reasons"]))

    def test_missing_artifact_fails_checksum_verification(self):
        (self.candidate / "signal_metrics.parquet").unlink()

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(
            any("missing: signal_metrics.parquet" in reason for reason in result["reasons"])
        )

    def test_unexpected_artifact_fails_checksum_verification(self):
        (self.candidate / "untracked.txt").write_text("extra", encoding="utf-8")

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("unexpected: untracked.txt" in reason for reason in result["reasons"]))

    def test_checksum_manifest_sha256_is_anchored_by_manifest(self):
        path = self.candidate / "artifact_checksums.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")

        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("checksum manifest SHA-256" in reason for reason in result["reasons"]))

    def test_source_tree_hash_must_be_valid_and_equal(self):
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["code"]["source_tree_sha256"] = "d" * 64
        self._write_json(self.candidate, "manifest.json", manifest)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertIn("code.source_tree_sha256 differs", result["reasons"])

        manifest["code"]["source_tree_sha256"] = "not-a-digest"
        self._write_json(self.candidate, "manifest.json", manifest)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("64-character hexadecimal" in reason for reason in result["reasons"]))

    def test_json_generation_does_not_overwrite_completed_result_by_default(self):
        comparisons = Path(self.temporary_directory.name) / "comparisons"
        with patch("quant_pipeline.comparison.DEFAULT_COMPARISONS_ROOT", comparisons):
            output = generate_comparison_json(self.baseline, self.candidate)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["comparison_status"], "completed")
        self.assertEqual(payload["status"], "improved")
        self.assertEqual(output.parent, comparisons.resolve())
        self.assertFalse(output.is_relative_to(self.candidate))
        original = output.read_bytes()
        with patch("quant_pipeline.comparison.DEFAULT_COMPARISONS_ROOT", comparisons):
            with self.assertRaises(FileExistsError):
                generate_comparison_json(self.baseline, self.candidate)
        self.assertEqual(output.read_bytes(), original)
        with patch("quant_pipeline.comparison.DEFAULT_COMPARISONS_ROOT", comparisons):
            overwritten = generate_comparison_json(self.baseline, self.candidate, overwrite=True)
        self.assertEqual(overwritten, output)

    def test_overwrite_cannot_replace_a_noncomparison_artifact(self):
        with self.assertRaisesRegex(ValueError, "immutable run directory"):
            generate_comparison_json(
                self.baseline,
                self.candidate,
                output_path=self.candidate / "manifest.json",
                overwrite=True,
            )
        self.assertEqual(self._read_json(self.candidate, "manifest.json")["run_id"], "candidate-run")

    def test_comparison_output_cannot_be_written_inside_a_run(self):
        with self.assertRaisesRegex(ValueError, "immutable run directory"):
            generate_comparison_json(
                self.baseline,
                self.candidate,
                output_path=self.candidate / "comparison.json",
            )

    def test_non_completed_run_is_incomparable(self):
        manifest = self._read_json(self.candidate, "manifest.json")
        manifest["status"] = "reporting"
        self._write_json(self.candidate, "manifest.json", manifest)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "incomparable")
        self.assertTrue(any("not completed" in reason for reason in result["reasons"]))

    def test_manifest_runtime_metadata_is_not_an_experimental_condition(self):
        baseline = self._read_json(self.baseline, "manifest.json")
        candidate = self._read_json(self.candidate, "manifest.json")
        baseline.update({"git": {"commit": "old"}, "environment": {"qlib": "old"}})
        candidate.update({"git": {"commit": "new"}, "environment": {"qlib": "new"}})
        self._write_json(self.baseline, "manifest.json", baseline)
        self._write_json(self.candidate, "manifest.json", candidate)
        result = compare_completed_runs(self.baseline, self.candidate)
        self.assertEqual(result["status"], "improved")


if __name__ == "__main__":
    unittest.main()
