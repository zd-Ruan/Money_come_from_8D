import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.web import create_app


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pipeline_root = Path(self.temp_dir.name)
        runs_dir = self.pipeline_root / "runs"
        self.completed_dir = runs_dir / "completed-run"
        self.failed_dir = runs_dir / "failed-run"
        self.completed_dir.mkdir(parents=True)
        self.failed_dir.mkdir()

        registry = {
            "updated_at": "2026-08-12T16:00:00+08:00",
            "secret": "registry-secret",
            "runs": [
                {
                    "run_id": "completed-run",
                    "created_at": "2026-08-12T15:00:00+08:00",
                    "completed_at": "2026-08-12T16:00:00+08:00",
                    "status": "completed",
                    "classification": "research_only",
                    "snapshot_id": "snapshot-1",
                    "run_dir": str(self.completed_dir.resolve()),
                    "error": "private error",
                    "traceback": "private traceback",
                    "metrics": {
                        "net_cumulative_return": 0.5,
                        "benchmark_cumulative_return": 0.2,
                        "ic": 0.01,
                        "rank_ic": 0.02,
                        "max_drawdown": -0.1,
                        "secret": "metric-secret",
                    },
                },
                {
                    "run_id": "failed-run",
                    "created_at": "2026-08-12T14:00:00+08:00",
                    "completed_at": "2026-08-12T14:01:00+08:00",
                    "status": "failed",
                    "classification": "invalid",
                    "run_dir": str(self.failed_dir.resolve()),
                    "error": f"failed under {self.pipeline_root.resolve()}",
                    "traceback": "Traceback: private",
                },
            ],
        }
        self._write_json(self.pipeline_root / "registry.json", registry)
        self._write_json(
            self.completed_dir / "manifest.json",
            {
                "run_id": "completed-run",
                "created_at": "2026-08-12T15:00:00+08:00",
                "completed_at": "2026-08-12T16:00:00+08:00",
                "status": "completed",
                "classification": "research_only",
                "snapshot_id": "snapshot-1",
                "snapshot_manifest": str(self.pipeline_root / "snapshots" / "manifest.json"),
                "config": str(self.pipeline_root / "configs" / "baseline.yaml"),
                "traceback": "Traceback: private",
                "git": {"status": "private-file.py"},
                "environment": {
                    "python": "3.11",
                    "platform": "Windows",
                    "qlib": "0.9.7",
                    "lightgbm": "4.7.0",
                    "secret": "environment-secret",
                },
            },
        )
        self._write_json(
            self.completed_dir / "metrics.json",
            {
                "base_slippage_bps_per_side": 5,
                "ic": 0.01,
                "secret": "metrics-secret",
                "base": {"days": 300, "net_cumulative_return": 0.5, "secret": "base-secret"},
                "stress": {"10": {"days": 300, "net_cumulative_return": 0.4, "secret": "stress-secret"}},
                "folds": [
                    {
                        "fold": 1,
                        "ic": 0.01,
                        "secret": "fold-secret",
                        "rows": {"train": 100, "secret": "row-secret"},
                        "portfolio": {"days": 60, "net_cumulative_return": 0.1, "secret": "portfolio-secret"},
                    }
                ],
            },
        )
        self._write_json(
            self.completed_dir / "gates.json",
            {
                "status": "research_only",
                "promotion_eligible": False,
                "passed": 1,
                "total": 2,
                "secret": "gates-secret",
                "checks": [{"name": "data_valid", "passed": True, "value": True, "secret": "check-secret"}],
            },
        )
        (self.completed_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
        self.client = TestClient(create_app(self.pipeline_root))

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_runs_api_uses_public_allowlist(self):
        response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        runs = response.json()["runs"]
        completed = runs[0]
        self.assertEqual(
            set(completed),
            {"run_id", "created_at", "completed_at", "status", "classification", "snapshot_id", "metrics"},
        )
        self.assertEqual(
            set(completed["metrics"]),
            {
                "net_cumulative_return",
                "benchmark_cumulative_return",
                "ic",
                "rank_ic",
                "max_drawdown",
            },
        )
        payload = response.text
        self.assertNotIn(str(self.pipeline_root.resolve()), payload)
        self.assertNotIn("traceback", payload.lower())
        self.assertNotIn("private error", payload)
        self.assertNotIn("secret", payload)

    def test_run_api_filters_manifest_and_nested_artifacts(self):
        response = self.client.get("/api/runs/completed-run")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload["manifest"]),
            {
                "run_id",
                "created_at",
                "completed_at",
                "status",
                "classification",
                "snapshot_id",
                "environment",
            },
        )
        self.assertEqual(set(payload["manifest"]["environment"]), {"python", "platform", "qlib", "lightgbm"})
        self.assertEqual(payload["metrics"]["base"], {"days": 300, "net_cumulative_return": 0.5})
        self.assertEqual(payload["metrics"]["stress"]["10"], {"days": 300, "net_cumulative_return": 0.4})
        self.assertEqual(payload["metrics"]["folds"][0]["rows"], {"train": 100})
        self.assertEqual(
            payload["metrics"]["folds"][0]["portfolio"],
            {"days": 60, "net_cumulative_return": 0.1},
        )
        self.assertEqual(payload["gates"]["checks"][0], {"name": "data_valid", "passed": True, "value": True})
        serialized = response.text
        self.assertNotIn(str(self.pipeline_root.resolve()), serialized)
        self.assertNotIn("traceback", serialized.lower())
        self.assertNotIn("private-file.py", serialized)
        self.assertNotIn("secret", serialized)

    def test_dashboard_translates_states_and_only_links_completed_runs(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.text
        self.assertIn("<span class='badge research_only'>仅限研究</span>", page)
        self.assertIn("<span class='badge invalid'>无效</span>", page)
        self.assertIn("<td>已完成</td>", page)
        self.assertIn("<td>失败</td>", page)
        self.assertIn("href='/runs/completed-run'", page)
        self.assertNotIn("href='/runs/failed-run'", page)
        self.assertNotIn(">research_only</span>", page)
        self.assertNotIn(">completed</td>", page)

    def test_report_and_run_paths_stay_inside_runs_directory(self):
        self.assertEqual(self.client.get("/runs/completed-run").status_code, 200)
        self.assertEqual(self.client.get("/api/runs/missing-run").status_code, 404)
        self.assertEqual(self.client.get("/runs/missing-run").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/%2e%2e%5cregistry.json").status_code, 404)
        self.assertEqual(self.client.get("/runs/%2e%2e%5cregistry.json").status_code, 404)


if __name__ == "__main__":
    unittest.main()
