import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.runner import (
    _model_params,
    _sanitize_workspace_text,
    _workspace_relative,
    backtest_bounds,
    fold_is_complete,
    run_pipeline,
    select_backtest_predictions,
    validate_lightgbm_device,
)
from quant_pipeline.windows import RollingFold


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2026-01-05", periods=10)

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

            with (
                patch.dict(sys.modules, fake_modules),
                patch("quant_pipeline.runner.git_state", side_effect=capture_git_state) as git_state_mock,
                patch("quant_pipeline.runner.source_tree_sha256", return_value="b" * 64),
                patch("quant_pipeline.runner.validate_lightgbm_device", side_effect=RuntimeError("stop")),
                patch("quant_pipeline.runner.update_registry"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    run_pipeline(config, run_id=run_id)

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["git"], frozen)
            git_state_mock.assert_called_once_with(workspace)


if __name__ == "__main__":
    unittest.main()
