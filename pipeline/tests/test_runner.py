import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.runner import (
    _cost_only_stress_report,
    _load_factor_tables,
    _model_params,
    _qlib_manifest_version,
    _run_research_backtest_folds,
    _seal_completed_run,
    _sanitize_workspace_text,
    _workspace_relative,
    backtest_bounds,
    fold_is_complete,
    enforce_research_exposure_gate,
    run_backtest,
    run_pipeline,
    resolve_pipeline_data_bounds,
    select_backtest_predictions,
    validate_lightgbm_device,
    run_pretraining_corporate_action_audit,
)
from quant_pipeline.config import validate_config
from quant_pipeline.environment import DEFAULT_ENVIRONMENT_LOCK
from quant_pipeline.io import sha256_file
from quant_pipeline.windows import RollingFold


class _SourceQlib:
    __version__ = "0.1.dev1"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2026-01-05", periods=10)

    def test_source_qlib_version_is_recorded_without_distribution_metadata(self):
        self.assertEqual(
            _qlib_manifest_version({"packages": {"lightgbm": "4.6.0"}}, _SourceQlib),
            "0.1.dev1",
        )
        self.assertEqual(
            _qlib_manifest_version({"packages": {"pyqlib": "1.2.0"}}, _SourceQlib),
            "1.2.0",
        )

    def test_backtest_cutoff_does_not_filter_instruments_by_future_label(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2026-08-06"), "SH510300"),
                (pd.Timestamp("2026-08-06"), "SH510500"),
                (pd.Timestamp("2026-08-07"), "SH510300"),
            ],
            names=["datetime", "instrument"],
        )
        predictions = pd.DataFrame(
            {"score": [0.2, 0.1, 0.3], "label": [0.01, float("nan"), float("nan")]},
            index=index,
        )
        selected = select_backtest_predictions(predictions, "2026-08-06")
        self.assertEqual(len(selected), 2)
        self.assertIn((pd.Timestamp("2026-08-06"), "SH510500"), selected.index)

    def test_backtest_bounds_follow_signal_t_plus_one_to_realization_t_plus_two(self):
        start, end = backtest_bounds(self.calendar, "2026-01-05", "2026-01-14")
        self.assertEqual(start, "2026-01-06")
        self.assertEqual(end, "2026-01-16")

    def test_research_data_bounds_require_exact_request_and_label_maturity(self):
        config = {
            "data": {"end_date": "2026-08-12", "label_horizon_bars": 2},
            "_research_stage": {
                "stage": "discovery",
                "request_sha256": "a" * 64,
                "prediction_end": "2026-03-27",
                "source_data_end": "2026-03-31",
                "exposure_registry_sha256": "b" * 64,
                "evidence_class": "retrospective_exposed",
                "claim_classification": "research_only",
            },
        }
        request = {
            "stage": "discovery",
            "request_sha256": "a" * 64,
            "exposure_registry_sha256": "b" * 64,
            "evidence_class": "retrospective_exposed",
            "claim_classification": "research_only",
            "partition": {
                "end": "2026-03-27",
                "source_data_end": "2026-03-31",
                "label_maturity_sessions": ["2026-03-30", "2026-03-31"],
            },
        }
        self.assertEqual(
            resolve_pipeline_data_bounds(config, request),
            ("2026-03-31", "2026-03-27"),
        )
        with self.assertRaisesRegex(ValueError, "requires a validated"):
            resolve_pipeline_data_bounds(config)
        tampered = json.loads(json.dumps(request))
        tampered["partition"]["label_maturity_sessions"][-1] = "2026-04-01"
        with self.assertRaisesRegex(ValueError, "exactly mature"):
            resolve_pipeline_data_bounds(config, tampered)

    def test_retrospective_research_cannot_inherit_candidate_status(self):
        base = {
            "status": "candidate",
            "promotion_eligible": True,
            "passed": 1,
            "total": 1,
            "checks": [{"name": "ordinary_gate", "passed": True}],
        }
        result = enforce_research_exposure_gate(
            base,
            {
                "evidence_class": "retrospective_exposed",
                "claim_classification": "research_only",
            },
        )
        self.assertEqual(result["status"], "research_only")
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["checks"][-1]["name"], "research_exposure_not_historically_exposed")
        self.assertFalse(result["checks"][-1]["passed"])
        self.assertEqual(base["status"], "candidate")

    def test_only_full_length_fold_is_complete(self):
        complete = RollingFold(
            1, "2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06",
            "2026-01-07", "2026-01-13", 0,
        )
        residual = RollingFold(
            2, "2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06",
            "2026-01-14", "2026-01-16", 0,
        )
        self.assertTrue(fold_is_complete(complete, self.calendar, 5))
        self.assertFalse(fold_is_complete(residual, self.calendar, 5))
        self.assertFalse(fold_is_complete(complete, self.calendar, 5, "2026-01-12"))

    def test_gpu_parameters_are_explicit_and_auditable(self):
        config = {
            "project": {"random_seeds": [11]},
            "model": {
                "objective": "regression", "learning_rate": 0.03, "num_leaves": 31,
                "max_depth": 7, "min_data_in_leaf": 10, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 0,
                "lambda_l2": 0, "num_threads": 4, "device_type": "gpu",
                "gpu_platform_id": 0, "gpu_device_id": 0, "gpu_use_dp": True,
                "max_bin": 63, "verbosity": 1,
            },
        }
        params = _model_params(config, 11)
        self.assertEqual(params["device_type"], "gpu")
        self.assertTrue(params["gpu_use_dp"])
        self.assertEqual(params["verbosity"], 1)
        self.assertNotIn("deterministic", params)

    def test_run_backtest_uses_raw_share_ledger_and_starts_at_first_execution(self):
        dates = pd.bdate_range("2026-01-05", periods=4)
        prediction_index = pd.MultiIndex.from_product(
            [[dates[0], dates[1]], ["SH510300"]], names=["datetime", "instrument"]
        )
        predictions = pd.DataFrame({"score": [1.0, 1.0]}, index=prediction_index)
        raw = pd.DataFrame(
            {
                "date": dates,
                "symbol": "SH510300",
                "raw_open": 10.0,
                "raw_close": 10.0,
                "raw_high": 10.0,
                "raw_low": 10.0,
                "volume": 1_000_000,
            }
        )
        actions = pd.DataFrame(
            columns=[
                "symbol", "record_date", "ex_date", "cash_payment_date",
                "cash_dividend_per_old_share", "share_ratio", "source_url", "source_sha256",
                "fractional_share_treatment",
            ]
        )
        audit_summary = pd.DataFrame(
            [{"audit_passed": True, "material_factor_jump_count": 0, "corporate_action_count": 0}]
        )
        prepared = {
            "calendar": dates,
            "raw_bars": raw,
            "benchmark_close": raw[["date", "symbol", "raw_close"]],
            "corporate_actions": actions,
            "corporate_action_audit": types.SimpleNamespace(
                summary=audit_summary, details=pd.DataFrame()
            ),
        }
        config = {
            "strategy": {"topk": 1, "n_drop": 1, "hold_thresh": 1, "risk_degree": 0.5},
            "execution": {
                "account": 20_000, "commission_bps_per_side": 3, "min_cost": 5,
                "trade_unit": 100, "max_daily_volume_participation": 0.05,
                "standard_limit_ratio": 0.10, "wide_limit_ratio": 0.20, "price_tick": 0.001,
            },
            "data": {"benchmark": "SH510300"},
        }
        with patch("quant_pipeline.runner.prepare_raw_backtest_inputs", return_value=prepared):
            report, positions, indicators, metadata, executions, summary = run_backtest(
                predictions,
                config,
                5,
                dates[1].date().isoformat(),
                dates[-1].date().isoformat(),
            )
        self.assertEqual(report.index[0], dates[1])
        self.assertNotIn(dates[0], report.index)
        self.assertEqual(executions.iloc[0]["signal_date"], dates[0])
        self.assertEqual(executions.iloc[0]["execution_date"], dates[1])
        self.assertEqual(summary["engine"], "raw_share_daily_v1")
        self.assertEqual(summary["total_cost"], executions["total_cost"].sum())
        self.assertEqual(summary["zero_fill_order_rate"], summary["zero_fill_intent_rate"])
        self.assertIsInstance(positions, pd.DataFrame)
        self.assertEqual(indicators.index.tolist(), report.index.tolist())
        self.assertIn("corporate_action_ledger", metadata)
        self.assertIn("symbol_attribution", metadata)
        self.assertTrue(
            metadata["symbol_attribution"]["net_pnl_cny"].equals(
                metadata["symbol_attribution"]["net_pnl"]
            )
        )
        self.assertEqual(
            metadata["symbol_attribution"]["date"].min(),
            dates[1],
        )

    def test_run_backtest_never_calls_qlib_backtest(self):
        fake = types.ModuleType("qlib.backtest")
        fake.backtest = lambda *args, **kwargs: self.fail("Qlib account path must not run")
        dates = pd.bdate_range("2026-01-05", periods=2)
        index = pd.MultiIndex.from_tuples(
            [(dates[0], "SH510300")], names=["datetime", "instrument"]
        )
        predictions = pd.DataFrame({"score": [1.0]}, index=index)
        raw = pd.DataFrame(
            {
                "date": dates, "symbol": "SH510300", "raw_open": 10.0,
                "raw_close": 10.0, "raw_high": 10.0, "raw_low": 10.0, "volume": 1_000_000,
            }
        )
        prepared = {
            "calendar": dates,
            "raw_bars": raw,
            "benchmark_close": raw[["date", "symbol", "raw_close"]],
            "corporate_actions": pd.DataFrame(columns=[
                "symbol", "record_date", "ex_date", "cash_payment_date",
                "cash_dividend_per_old_share", "share_ratio",
                "fractional_share_treatment",
            ]),
            "corporate_action_audit": types.SimpleNamespace(
                summary=pd.DataFrame([{"audit_passed": True}]), details=pd.DataFrame()
            ),
        }
        config = {
            "strategy": {"topk": 1, "n_drop": 1, "hold_thresh": 1, "risk_degree": 0.5},
            "execution": {
                "account": 20_000, "commission_bps_per_side": 3, "min_cost": 5,
                "trade_unit": 100, "max_daily_volume_participation": 0.05,
                "standard_limit_ratio": 0.10, "wide_limit_ratio": 0.20, "price_tick": 0.001,
            },
            "data": {"benchmark": "SH510300"},
        }
        with (
            patch.dict(sys.modules, {"qlib.backtest": fake}),
            patch("quant_pipeline.runner.prepare_raw_backtest_inputs", return_value=prepared),
        ):
            run_backtest(
                predictions, config, 5, dates[1].date().isoformat(), dates[1].date().isoformat()
            )

    def test_research_folds_are_independent_twenty_one_signal_session_accounts(self):
        dates = pd.bdate_range("2026-01-05", periods=44)
        signal_dates = dates[:42]
        signal_strings = [date.date().isoformat() for date in signal_dates]
        stage_calendar = [date.date().isoformat() for date in dates]
        contracts = []
        for fold_number, start in enumerate((0, 21), start=1):
            fold_signals = signal_strings[start : start + 21]
            raw_sessions = stage_calendar[start + 1 : start + 23]
            contracts.append(
                {
                    "fold": fold_number,
                    "signal_start": fold_signals[0],
                    "signal_end": fold_signals[-1],
                    "signal_observations": 21,
                    "signal_sessions": fold_signals,
                    "raw_report_start": raw_sessions[0],
                    "raw_report_end": raw_sessions[-1],
                    "raw_report_sessions": raw_sessions,
                    "evaluation_start": raw_sessions[1],
                    "evaluation_end": raw_sessions[-1],
                    "evaluation_sessions": raw_sessions[1:],
                    "complete_for_gate": True,
                }
            )
        request = {
            "partition": {
                "sessions": signal_strings,
                "label_maturity_sessions": stage_calendar[-2:],
                "portfolio_execution_lag_bars": 1,
                "portfolio_realization_lag_bars": 2,
                "research_fold_signal_sessions": 21,
                "research_folds": contracts,
            },
            "metric_contract": {
                "portfolio": {
                    "initial_account": 20_000,
                    "stress_slippage_bps_per_side": 10,
                    "research_folds": contracts,
                }
            },
        }
        predictions = pd.DataFrame(
            {"score": np.ones(len(signal_dates))},
            index=pd.MultiIndex.from_arrays(
                [signal_dates, ["A"] * len(signal_dates)],
                names=["datetime", "instrument"],
            ),
        )
        raw = pd.DataFrame(
            {
                "date": dates,
                "symbol": "A",
                "raw_open": 10.0,
                "raw_close": 10.0,
                "raw_high": 10.0,
                "raw_low": 10.0,
                "volume": 1_000_000,
            }
        )
        audit = types.SimpleNamespace(
            summary=pd.DataFrame([{"audit_passed": True}]),
            details=pd.DataFrame(),
            factor_changes=pd.DataFrame(),
        )
        prepared = {
            "calendar": dates,
            "raw_bars": raw,
            "benchmark_close": pd.DataFrame(
                {"date": dates, "symbol": "BENCH", "raw_close": 10.0}
            ),
            "corporate_actions": pd.DataFrame(
                columns=[
                    "symbol", "record_date", "ex_date", "cash_payment_date",
                    "cash_dividend_per_old_share", "share_ratio",
                    "fractional_share_treatment",
                ]
            ),
            "corporate_action_audit": audit,
        }
        config = {
            "data": {
                "benchmark": "BENCH",
                "label_horizon_bars": 2,
                "label": "Ref($close, -2) / Ref($close, -1) - 1",
            },
            "strategy": {
                "topk": 1, "n_drop": 0, "hold_thresh": 1, "risk_degree": 0.5,
            },
            "execution": {
                "account": 20_000,
                "commission_bps_per_side": 3,
                "min_cost": 5,
                "trade_unit": 100,
                "max_daily_volume_participation": 0.05,
                "standard_limit_ratio": 0.10,
                "wide_limit_ratio": 0.20,
                "price_tick": 0.001,
                "stress_slippage_bps_per_side": [0, 5, 10],
            },
            "gates": {"research_fold_days": 21},
        }

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summaries = _run_research_backtest_folds(
                predictions,
                config,
                request,
                dates,
                prepared,
                run_dir,
            )

            self.assertEqual(len(summaries), 2)
            reports = []
            for contract, summary in zip(contracts, summaries):
                fold_dir = (
                    run_dir / "folds" / f"research_fold_{contract['fold']:02d}" / "backtest"
                )
                report = pd.read_parquet(fold_dir / "report.parquet")
                attribution = pd.read_parquet(fold_dir / "symbol_attribution.parquet")
                saved_summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
                reports.append(report)
                self.assertEqual(
                    [date.date().isoformat() for date in report.index],
                    contract["raw_report_sessions"],
                )
                self.assertEqual(saved_summary, summary)
                self.assertEqual(summary["signal_observations"], 21)
                self.assertEqual(summary["raw_execution_days"], 22)
                self.assertEqual(summary["days"], 21)
                self.assertEqual(summary["slippage_bps_per_side"], 10)
                self.assertEqual(summary["initial_account_value"], 20_000)
                self.assertEqual(summary["execution"]["initial_account"], 20_000)
                self.assertEqual(summary["execution"]["config"]["initial_cash"], 20_000)
                self.assertEqual(summary["execution"]["config"]["slippage_bps_per_side"], 10)
                self.assertTrue(summary["complete_for_gate"])
                self.assertAlmostEqual(attribution["net_pnl"].sum(), -15.0)
                self.assertTrue(attribution["net_pnl_cny"].equals(attribution["net_pnl"]))
                self.assertEqual(
                    attribution["date"].min().date().isoformat(), contract["raw_report_start"]
                )
                self.assertEqual(
                    attribution["date"].max().date().isoformat(), contract["raw_report_end"]
                )
            self.assertAlmostEqual(reports[0].iloc[0]["account"], reports[1].iloc[0]["account"])
            self.assertAlmostEqual(summaries[0]["terminal_account"], summaries[1]["terminal_account"])

            tampered = json.loads(json.dumps(request))
            tampered["metric_contract"]["portfolio"]["stress_slippage_bps_per_side"] = 5
            with self.assertRaisesRegex(ValueError, "10 bps"):
                _run_research_backtest_folds(
                    predictions,
                    config,
                    tampered,
                    dates,
                    prepared,
                    run_dir / "tampered",
                )

            bad_label = dict(config)
            bad_label["data"] = {
                **config["data"],
                "label": "Ref($close, -1) / $close - 1",
            }
            with self.assertRaisesRegex(ValueError, r"T\+1 close execution"):
                _run_research_backtest_folds(
                    predictions,
                    bad_label,
                    request,
                    dates,
                    prepared,
                    run_dir / "bad-label",
                )

    def test_incomplete_corporate_action_collection_fails_closed(self):
        from quant_pipeline.runner import _read_complete_corporate_actions

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "symbol": ["SH510300"],
                    "error": ["HTTP 514"],
                    "full_universe_scope": [True],
                    "published": [False],
                }
            ).to_csv(root / "corporate_action_report.csv", index=False)
            pd.DataFrame(
                {
                    "symbol": ["SH510300"],
                    "record_date": ["2026-01-05"],
                    "ex_date": ["2026-01-06"],
                    "cash_payment_date": ["2026-01-07"],
                    "cash_dividend_per_old_share": [0.1],
                    "share_ratio": [1.0],
                    "fractional_share_treatment": ["not_applicable_no_share_change"],
                    "source_url": ["https://example.test"],
                    "source_sha256": ["a" * 64],
                }
            ).to_csv(root / "corporate_actions.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                _read_complete_corporate_actions(root)

            report = pd.read_csv(root / "corporate_action_report.csv")
            report["error"] = ""
            report["published"] = "False"
            report.to_csv(root / "corporate_action_report.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "not published"):
                _read_complete_corporate_actions(root)

    def test_price_limit_contract_requires_explicit_conservative_tiers(self):
        config = {
            "features": {"mode": "alpha158", "families": []},
            "data": {"label_horizon_bars": 2},
            "rolling": {"purge_bars": 2, "validation_days": 126, "test_days": 63},
            "execution": {
                "account": 20_000,
                "trade_unit": 100,
                "stamp_tax_bps": 0,
                "max_daily_volume_participation": 0.05,
                "stress_slippage_bps_per_side": [0, 5, 10],
                "price_limit_mode": "threshold",
                "standard_limit_ratio": 0.10,
                "wide_limit_ratio": 0.20,
                "price_tick": 0.001,
            },
            "model": {"device_type": "gpu"},
            "strategy": {"topk": 5, "n_drop": 1, "hold_thresh": 5},
            "gates": {
                "min_complete_folds": 4,
                "research_fold_days": 21,
                "min_research_fold_win_ratio": 0.60,
                "max_single_etf_abs_contribution_share": 0.35,
                "max_single_fold_abs_incremental_pnl_share": 0.50,
            },
        }
        with self.assertRaisesRegex(ValueError, "price_limit_mode"):
            validate_config(config)

        config["execution"]["price_limit_mode"] = "ohlc_proven_tier_conservative"
        validate_config(config)

        config["features"] = {"mode": "alpha360", "families": []}
        validate_config(config)

    def test_cpu_device_check_does_not_probe_gpu(self):
        with patch("quant_pipeline.runner.lgb.train") as train:
            validate_lightgbm_device({"model": {"device_type": "cpu"}})
        train.assert_not_called()

    def test_gpu_device_check_uses_declared_device_contract(self):
        config = {
            "model": {
                "device_type": "gpu", "gpu_platform_id": 2, "gpu_device_id": 3,
                "gpu_use_dp": True, "max_bin": 31, "num_threads": 6, "verbosity": 1,
            }
        }
        with patch("quant_pipeline.runner.lgb.train") as train:
            validate_lightgbm_device(config)
        params = train.call_args.args[0]
        self.assertEqual(params["device_type"], "gpu")
        self.assertEqual(params["gpu_platform_id"], 2)
        self.assertEqual(params["gpu_device_id"], 3)
        self.assertTrue(params["gpu_use_dp"])
        self.assertEqual(params["max_bin"], 31)

    def test_pretraining_action_audit_dry_mode_does_not_create_output(self):
        output = Path("should-not-be-created")
        result = types.SimpleNamespace(passed=True)
        with (
            patch("quant_pipeline.runner._data_root", return_value=Path("data")),
            patch("quant_pipeline.runner.pd.read_csv", return_value=pd.DataFrame({"symbol": ["A"]})),
            patch("quant_pipeline.runner._read_complete_corporate_actions", return_value=pd.DataFrame(columns=["symbol", "ex_date"])),
            patch("quant_pipeline.runner._pretraining_audit_calendar", return_value=(self.calendar, self.calendar[0])),
            patch("quant_pipeline.runner._load_factor_tables", return_value=pd.DataFrame()),
            patch("quant_pipeline.runner.detect_material_factor_changes", return_value=pd.DataFrame(columns=["symbol"])),
            patch("quant_pipeline.runner.audit_corporate_actions", return_value=result),
            patch.object(Path, "mkdir") as mkdir,
        ):
            actual = run_pretraining_corporate_action_audit(
                {"paths": {"universe": "universe.csv"}, "data": {"benchmark": "A"}},
                None,
            )
        self.assertIs(actual, result)
        mkdir.assert_not_called()

    def test_manifest_paths_are_relative_and_traceback_is_sanitized(self):
        workspace = Path("C:/workspace/project").resolve()
        self.assertEqual(
            _workspace_relative(workspace / "pipeline" / "runs" / "run-1", workspace),
            "pipeline/runs/run-1",
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            _workspace_relative(Path("C:/elsewhere/secret.txt"), workspace)
        rendered = _sanitize_workspace_text(f"failed in {workspace}\\data", workspace)
        self.assertNotIn(str(workspace), rendered)

    def test_completed_manifest_is_written_before_and_protected_by_outer_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text('{"return": 0.01}', encoding="utf-8")
            manifest = {
                "run_id": "sealed-run",
                "status": "reporting",
                "artifacts": {
                    "artifact_checksums": "artifact_checksums.json",
                    "integrity_seal": "integrity_seal.json",
                },
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = _seal_completed_run(run_dir, manifest)

            frozen = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            seal = json.loads((run_dir / "integrity_seal.json").read_text(encoding="utf-8"))
            self.assertTrue(result["valid"])
            self.assertEqual(frozen["status"], "completed")
            self.assertTrue(frozen["integrity"]["verified"])
            self.assertEqual(
                {record["path"] for record in seal["protected_files"]},
                {"manifest.json", "artifact_checksums.json"},
            )

            frozen["classification"] = "tampered"
            (run_dir / "manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
            from quant_pipeline.integrity import verify_artifact_checksums

            verification = verify_artifact_checksums(run_dir)
            self.assertFalse(verification["valid"])
            self.assertIn("manifest.json", verification["modified"])

    def test_git_state_is_frozen_before_the_run_directory_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            runs = workspace / "pipeline" / "runs"
            run_id = "frozen-git-state"
            run_dir = runs / run_id
            config = {
                "_meta": {"workspace_root": str(workspace)},
                "features": {"mode": "alpha158", "families": []},
                "project": {"name": "test"},
                "data": {"end_date": "2026-08-12"},
                "paths": {
                    "runs": str(runs),
                    "registry": str(workspace / "pipeline" / "registry.json"),
                },
            }
            frozen = {
                "available": True,
                "commit": "a" * 40,
                "branch": "master",
                "dirty": False,
                "status": "",
            }

            def capture_git_state(repo):
                self.assertEqual(repo, workspace)
                self.assertFalse(run_dir.exists())
                return frozen

            fake_modules = {
                "qlib": types.ModuleType("qlib"),
                "qlib.contrib": types.ModuleType("qlib.contrib"),
                "qlib.contrib.data": types.ModuleType("qlib.contrib.data"),
                "qlib.contrib.data.handler": types.ModuleType("qlib.contrib.data.handler"),
                "qlib.data": types.ModuleType("qlib.data"),
                "qlib.data.dataset": types.ModuleType("qlib.data.dataset"),
            }
            fake_modules["qlib.contrib.data.handler"].Alpha158 = object
            fake_modules["qlib.data.dataset"].DatasetH = object
            fake_modules["qlib"].__file__ = str(workspace / "qlib" / "__init__.py")

            with (
                patch.dict(sys.modules, fake_modules),
                patch("quant_pipeline.runner.git_state", side_effect=capture_git_state) as git_state_mock,
                patch(
                    "quant_pipeline.runner.runtime_code_identity",
                    return_value={
                        "pipeline_source_sha256": "b" * 64,
                        "qlib_package_sha256": "c" * 64,
                        "runtime_code_sha256": "d" * 64,
                    },
                ),
                patch(
                    "quant_pipeline.runner.validate_locked_environment",
                    return_value={"lock": {"sha256": sha256_file(DEFAULT_ENVIRONMENT_LOCK)}},
                ),
                patch("quant_pipeline.runner.shutil.copy2", side_effect=lambda src, dst: dst.write_bytes(src.read_bytes())),
                patch("quant_pipeline.runner.validate_lightgbm_device", side_effect=RuntimeError("stop")),
                patch("quant_pipeline.runner.update_registry"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    run_pipeline(config, run_id=run_id)

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["git"], frozen)
            git_state_mock.assert_called_once_with(workspace)

    def test_pretraining_action_gate_runs_before_any_fold_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            runs = workspace / "pipeline" / "runs"
            config = {
                "_meta": {"workspace_root": str(workspace)},
                "features": {"mode": "alpha158", "families": []},
                "project": {"name": "test"},
                "data": {"end_date": "2026-08-12"},
                "paths": {
                    "runs": str(runs),
                    "registry": str(workspace / "pipeline" / "registry.json"),
                },
            }
            fake_modules = {
                "qlib": types.ModuleType("qlib"),
                "qlib.contrib": types.ModuleType("qlib.contrib"),
                "qlib.contrib.data": types.ModuleType("qlib.contrib.data"),
                "qlib.contrib.data.handler": types.ModuleType("qlib.contrib.data.handler"),
                "qlib.data": types.ModuleType("qlib.data"),
                "qlib.data.dataset": types.ModuleType("qlib.data.dataset"),
            }
            fake_modules["qlib.contrib.data.handler"].Alpha158 = object
            fake_modules["qlib.data.dataset"].DatasetH = object
            fake_modules["qlib"].__file__ = str(workspace / "qlib" / "__init__.py")
            data_audit = types.SimpleNamespace(
                report={"data_valid": True, "blocking_issues": []}
            )

            with (
                patch.dict(sys.modules, fake_modules),
                patch("quant_pipeline.runner.git_state", return_value={"available": False}),
                patch(
                    "quant_pipeline.runner.runtime_code_identity",
                    return_value={
                        "pipeline_source_sha256": "b" * 64,
                        "qlib_package_sha256": "c" * 64,
                        "runtime_code_sha256": "d" * 64,
                    },
                ),
                patch(
                    "quant_pipeline.runner.validate_locked_environment",
                    return_value={"lock": {"sha256": sha256_file(DEFAULT_ENVIRONMENT_LOCK)}},
                ),
                patch("quant_pipeline.runner.shutil.copy2", side_effect=lambda src, dst: dst.write_bytes(src.read_bytes())),
                patch("quant_pipeline.runner.validate_lightgbm_device"),
                patch("quant_pipeline.runner.audit_and_snapshot", return_value=data_audit),
                patch(
                    "quant_pipeline.runner.run_pretraining_corporate_action_audit",
                    side_effect=RuntimeError("pretraining action audit failed"),
                ) as action_gate,
                patch("quant_pipeline.runner.train_fold") as train,
                patch("quant_pipeline.runner.update_registry"),
            ):
                with self.assertRaisesRegex(RuntimeError, "pretraining action audit failed"):
                    run_pipeline(config, run_id="pretraining-gate")

            action_gate.assert_called_once()
            train.assert_not_called()
            manifest = json.loads(
                (runs / "pretraining-gate" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")


class CostOnlyStressTests(unittest.TestCase):
    def _base_fixture(self):
        dates = pd.bdate_range("2026-01-05", periods=3)
        report = pd.DataFrame(
            {
                "return": [0.01, -0.005, 0.01],
                "cost": [0.00025, 0.0, 0.0],
                "turnover": [0.5, 0.0, 0.0],
                "account": [20000.0 * 1.00975, 20000.0 * 1.00975 * 0.995, 0.0],
                "cash": [10000.0, 10000.0, 10000.0],
                "bench": [0.0, 0.0, 0.0],
                "value": [10000.0, 10000.0, 10000.0],
                "receivable": [0.0, 0.0, 0.0],
            },
            index=dates,
        )
        report.loc[dates[2], "account"] = report.loc[dates[1], "account"] * 1.01
        executions = pd.DataFrame(
            [
                {
                    "execution_date": dates[0],
                    "symbol": "SH510300",
                    "fill_shares": 100,
                    "fill_notional": 10000.0,
                    "commission": 5.0,
                    "slippage": 5.0,
                }
            ]
        )
        base_summary = {
            "initial_account": 20000.0,
            "commission_total": 5.0,
            "slippage_total": 5.0,
            "total_cost": 10.0,
            "config": {"slippage_bps_per_side": 5.0},
        }
        return report, executions, base_summary

    def test_cost_only_stress_is_monotonic_and_reconciles(self):
        report, executions, base_summary = self._base_fixture()
        stress_10, stress_10_summary = _cost_only_stress_report(
            report, executions, base_summary, 10, 5
        )
        stress_30, stress_30_summary = _cost_only_stress_report(
            report, executions, base_summary, 30, 5
        )
        self.assertEqual(
            stress_10_summary["total_cost"], 5.0 + 10000.0 * 10 / 10000.0
        )
        self.assertEqual(
            stress_30_summary["total_cost"], 5.0 + 10000.0 * 30 / 10000.0
        )
        self.assertTrue(stress_10_summary["cost_only_stress"])
        self.assertEqual(stress_10_summary["base_execution_path_slippage_bps"], 5.0)
        self.assertLessEqual(
            stress_30["account"].iloc[-1], stress_10["account"].iloc[-1]
        )
        self.assertLessEqual(
            stress_10["account"].iloc[-1], report["account"].iloc[-1]
        )
        daily_cost = stress_10_summary["total_cost"]
        ledger = float((stress_10["cost"] * stress_10["account"].shift(1).fillna(20000.0)).sum())
        self.assertAlmostEqual(ledger, daily_cost, places=6)


if __name__ == "__main__":
    unittest.main()
