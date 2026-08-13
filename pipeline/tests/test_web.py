import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.integrity import (
    generate_artifact_checksums,
    generate_integrity_seal,
    verify_artifact_checksums,
)
from quant_pipeline.io import sha256_file
from quant_pipeline.web import create_app


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pipeline_root = Path(self.temp_dir.name)
        runs_dir = self.pipeline_root / "runs"
        self.baseline_dir = runs_dir / "baseline-run"
        self.completed_dir = runs_dir / "completed-run"
        self.failed_dir = runs_dir / "failed-run"
        self.legacy_dir = runs_dir / "legacy-large-account"
        self.unverified_dir = runs_dir / "unverified-run"
        self.corrupt_dir = runs_dir / "corrupt-run"
        self.baseline_dir.mkdir(parents=True)
        self.completed_dir.mkdir(parents=True)
        self.failed_dir.mkdir()
        self.legacy_dir.mkdir()
        self.unverified_dir.mkdir()
        self.corrupt_dir.mkdir()

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
                        "relative_wealth_max_drawdown": -0.08,
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
                {
                    "run_id": "legacy-large-account",
                    "created_at": "2026-08-12T13:00:00+08:00",
                    "completed_at": "2026-08-12T13:30:00+08:00",
                    "status": "completed",
                    "classification": "research_only",
                    "snapshot_id": "legacy-snapshot",
                    "metrics": {"net_cumulative_return": 9.9},
                },
                {
                    "run_id": "unverified-run",
                    "created_at": "2026-08-12T12:00:00+08:00",
                    "completed_at": "2026-08-12T12:30:00+08:00",
                    "status": "completed",
                    "classification": "candidate",
                    "snapshot_id": "unverified-snapshot",
                    "metrics": {"net_cumulative_return": 8.8},
                },
                {
                    "run_id": "corrupt-run",
                    "created_at": "2026-08-12T11:00:00+08:00",
                    "completed_at": "2026-08-12T11:30:00+08:00",
                    "status": "completed",
                    "classification": "candidate",
                    "snapshot_id": "corrupt-snapshot",
                    "metrics": {"net_cumulative_return": 7.7},
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
                "integrity": {"verified": True},
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
                "factor_catalog": {
                    "catalog_version": "orc_ohlcv_v1",
                    "families": ["trend_crowding"],
                    "factors": [
                        {
                            "name": "ORC_TREND_PATH_CROWD_20",
                            "family": "trend_crowding",
                            "expression": "private expression",
                            "direction": -1,
                            "lookback": 20,
                            "hypothesis": "趋势过度拥挤后更可能回撤。",
                            "secret": "factor-secret",
                        }
                    ],
                    "sha256": "a" * 64,
                    "secret": "catalog-secret",
                },
            },
        )
        self._write_json(
            self.completed_dir / "config.json",
            {"execution": {"account": 20_000}},
        )
        self._write_json(
            self.baseline_dir / "manifest.json",
            {
                "run_id": "baseline-run",
                "status": "completed",
                "integrity": {"verified": True},
            },
        )
        self._write_json(
            self.baseline_dir / "config.json",
            {"execution": {"account": 20_000}},
        )
        self._write_json(
            self.legacy_dir / "manifest.json",
            {
                "run_id": "legacy-large-account",
                "status": "completed",
                "integrity": {"verified": True},
            },
        )
        self._write_json(
            self.legacy_dir / "config.json",
            {"execution": {"account": 10_000_000}},
        )
        self._write_json(
            self.unverified_dir / "manifest.json",
            {
                "run_id": "unverified-run",
                "status": "completed",
                "integrity": {"verified": False},
            },
        )
        self._write_json(
            self.unverified_dir / "config.json",
            {"execution": {"account": 20_000}},
        )
        (self.corrupt_dir / "manifest.json").write_text("{", encoding="utf-8")
        (self.corrupt_dir / "config.json").write_text("not-json", encoding="utf-8")
        self._write_json(
            self.completed_dir / "metrics.json",
            {
                "base_slippage_bps_per_side": 5,
                "ic": 0.01,
                "ic_hac_t_stat": 2.2,
                "rank_ic": 0.02,
                "rank_ic_hac_t_stat": 2.4,
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
        (self.completed_dir / "report.html").write_text(
            f"<html>report at {self.completed_dir.resolve()}</html>", encoding="utf-8"
        )
        self.comparison_id = "baseline-run__vs__completed-run"
        self._write_json(
            self.pipeline_root / "comparisons" / f"{self.comparison_id}.json",
            {
                "schema_version": 1,
                "comparison_status": "completed",
                "status": "not_improved",
                "baseline_run_id": "baseline-run",
                "candidate_run_id": "completed-run",
                "comparable": True,
                "reasons": ["improvement criterion not met: daily_return_hac_significant"],
                "conditions": {
                    "source_fingerprint": "b" * 64,
                    "snapshot_id": "snapshot-1",
                    "base_slippage_bps_per_side": 5,
                    "evaluation_start_date": "2026-01-05",
                    "evaluation_end_date": "2026-03-03",
                    "return_observations": 42,
                    "signal_start_date": "2026-01-05",
                    "signal_end_date": "2026-02-27",
                    "signal_observations": 40,
                    "secret": "condition-secret",
                },
                "thresholds": {
                    "hac_t_stat": 1.96,
                    "hac_max_lag": 5,
                    "fold_win_rate": 0.5,
                    "secret": "threshold-secret",
                },
                "deltas": {
                    "terminal_relative_wealth": {
                        "baseline": 1.01,
                        "candidate": 1.03,
                        "difference": 0.02,
                        "secret": "terminal-secret",
                    },
                    "daily_strategy_return": {
                        "observations": 42,
                        "baseline_mean": 0.0005,
                        "candidate_mean": 0.0009,
                        "mean_difference": 0.0004,
                        "hac_t_stat": 1.5,
                        "secret": "daily-secret",
                    },
                    "ic": {
                        "observations": 40,
                        "baseline_mean": 0.01,
                        "candidate_mean": 0.012,
                        "mean_difference": 0.002,
                        "hac_t_stat": 2.1,
                    },
                    "rank_ic": {
                        "observations": 40,
                        "baseline_mean": 0.02,
                        "candidate_mean": 0.023,
                        "mean_difference": 0.003,
                        "hac_t_stat": 2.3,
                    },
                    "folds": {
                        "folds": 2,
                        "wins": 1,
                        "losses": 1,
                        "ties": 0,
                        "win_rate": 0.5,
                        "records": [
                            {
                                "fold": 1,
                                "start": "2026-01-05",
                                "end": "2026-02-02",
                                "observations": 21,
                                "baseline_terminal_wealth": 1.01,
                                "candidate_terminal_wealth": 1.02,
                                "benchmark_terminal_wealth": 1.0,
                                "baseline_terminal_relative_wealth": 1.01,
                                "candidate_terminal_relative_wealth": 1.02,
                                "terminal_relative_wealth_difference": 0.01,
                                "outcome": "win",
                                "secret": "fold-secret",
                            }
                        ],
                        "secret": "fold-summary-secret",
                    },
                    "secret": "delta-secret",
                },
                "decision": {
                    "claim": "Improvement is not established.",
                    "criteria": {
                        "terminal_relative_wealth_positive": True,
                        "daily_return_difference_positive": True,
                        "daily_return_hac_significant": False,
                        "fold_win_rate_majority": False,
                        "ic_difference_non_negative": True,
                        "rank_ic_difference_non_negative": True,
                        "at_least_one_signal_hac_significant": True,
                        "secret": True,
                    },
                    "secret": "decision-secret",
                },
                "scope": "paired incremental research evidence only",
                "generated_at": "2026-08-12T17:00:00+08:00",
                "private_path": str(self.pipeline_root.resolve()),
                "secret": "comparison-secret",
            },
        )
        self._seal_run(self.baseline_dir)
        self._seal_run(self.completed_dir)
        self.client = TestClient(create_app(self.pipeline_root))

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _seal_run(self, run_dir):
        manifest_path = run_dir / "manifest.json"
        checksum_path = generate_artifact_checksums(run_dir)
        verification = verify_artifact_checksums(run_dir, require_seal=False)
        self.assertTrue(verification["valid"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("artifacts", {}).update(
            {
                "artifact_checksums": checksum_path.name,
                "integrity_seal": "integrity_seal.json",
            }
        )
        manifest["integrity"] = {
            "checksum_manifest": checksum_path.name,
            "checksum_sha256": sha256_file(checksum_path),
            "artifact_count": verification["expected_count"],
            "seal_manifest": "integrity_seal.json",
            "verified": True,
        }
        self._write_json(manifest_path, manifest)
        generate_integrity_seal(run_dir, checksum_path)
        self.assertTrue(verify_artifact_checksums(run_dir)["valid"])

    def test_runs_api_uses_public_allowlist(self):
        response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        runs = response.json()["runs"]
        self.assertEqual([run["run_id"] for run in runs], ["completed-run"])
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
                "relative_wealth_max_drawdown",
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
                "factor_catalog",
            },
        )
        self.assertEqual(set(payload["manifest"]["environment"]), {"python", "platform", "qlib", "lightgbm"})
        self.assertEqual(
            set(payload["manifest"]["factor_catalog"]),
            {"catalog_version", "families", "factors", "sha256"},
        )
        self.assertEqual(
            set(payload["manifest"]["factor_catalog"]["factors"][0]),
            {"name", "family", "direction", "hypothesis", "lookback"},
        )
        self.assertEqual(payload["metrics"]["ic_hac_t_stat"], 2.2)
        self.assertEqual(payload["metrics"]["rank_ic_hac_t_stat"], 2.4)
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

    def test_dashboard_only_displays_trusted_completed_runs(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.text
        self.assertIn("<span class='badge research_only'>仅限研究</span>", page)
        self.assertIn("<td>已完成</td>", page)
        self.assertIn("href='/runs/completed-run'", page)
        self.assertIn(f"href='/comparisons/{self.comparison_id}'", page)
        self.assertIn("比较审计 · 未证明改进", page)
        self.assertNotIn("href='/runs/failed-run'", page)
        self.assertNotIn("failed-run", page)
        self.assertNotIn("legacy-large-account", page)
        self.assertNotIn("unverified-run", page)
        self.assertNotIn("corrupt-run", page)
        self.assertNotIn(">research_only</span>", page)
        self.assertNotIn(">completed</td>", page)

    def test_untrusted_and_corrupt_runs_are_rejected_without_server_errors(self):
        listing = self.client.get("/api/runs")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual([run["run_id"] for run in listing.json()["runs"]], ["completed-run"])

        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        for run_id in ("failed-run", "legacy-large-account", "unverified-run", "corrupt-run"):
            self.assertNotIn(run_id, dashboard.text)
            self.assertEqual(self.client.get(f"/api/runs/{run_id}").status_code, 404)
            self.assertEqual(self.client.get(f"/runs/{run_id}").status_code, 404)

    def test_tampered_artifact_is_hidden_and_direct_access_is_rejected(self):
        (self.completed_dir / "gates.json").write_text(
            '{"status":"candidate","tampered":true}', encoding="utf-8"
        )

        listing = self.client.get("/api/runs")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["runs"], [])
        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertNotIn("completed-run", dashboard.text)
        self.assertEqual(self.client.get("/api/runs/completed-run").status_code, 404)
        self.assertEqual(self.client.get("/runs/completed-run").status_code, 404)
        self.assertEqual(
            self.client.get(f"/api/comparisons/{self.comparison_id}").status_code,
            404,
        )

    def test_comparison_api_uses_nested_allowlist(self):
        response = self.client.get(f"/api/comparisons/{self.comparison_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["comparison_id"], self.comparison_id)
        self.assertEqual(payload["status"], "not_improved")
        self.assertEqual(payload["deltas"]["daily_strategy_return"]["hac_t_stat"], 1.5)
        self.assertEqual(payload["deltas"]["ic"]["hac_t_stat"], 2.1)
        self.assertEqual(payload["deltas"]["rank_ic"]["hac_t_stat"], 2.3)
        self.assertEqual(payload["deltas"]["folds"]["records"][0]["outcome"], "win")
        self.assertEqual(payload["factor_catalog"]["factors"][0]["lookback"], 20)
        self.assertNotIn("expression", payload["factor_catalog"]["factors"][0])
        serialized = response.text
        self.assertNotIn("secret", serialized)
        self.assertNotIn("private_path", serialized)
        self.assertNotIn(str(self.pipeline_root.resolve()), serialized)

        listing = self.client.get("/api/comparisons")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["comparisons"][0], payload)
        by_run = self.client.get("/api/runs/completed-run/comparison")
        self.assertEqual(by_run.status_code, 200)
        self.assertEqual(by_run.json(), payload)

    def test_comparison_page_is_chinese_and_reads_frozen_artifacts(self):
        response = self.client.get(f"/comparisons/{self.comparison_id}")
        self.assertEqual(response.status_code, 200)
        page = response.text
        for expected in (
            "未证明改进",
            "可比较",
            "关键增量与配对 HAC t",
            "日策略收益",
            "Rank IC",
            "预声明判定条件",
            "未通过",
            "未满足改进条件：日策略收益差通过 HAC 显著性门槛",
            "逐折相对财富",
            "候选胜",
            "冻结原创研究候选目录",
            "ORC_TREND_PATH_CROWD_20",
            "趋势过度拥挤后更可能回撤。",
        ):
            self.assertIn(expected, page)
        self.assertNotIn("private expression", page)
        self.assertNotIn("secret", page)
        self.assertNotIn(str(self.pipeline_root.resolve()), page)

        by_run = self.client.get("/runs/completed-run/comparison")
        self.assertEqual(by_run.status_code, 200)
        self.assertIn("未证明改进", by_run.text)

    def test_dashboard_selects_latest_valid_comparison_and_ignores_corrupt_json(self):
        original_path = self.pipeline_root / "comparisons" / f"{self.comparison_id}.json"
        older = json.loads(original_path.read_text(encoding="utf-8"))
        older.update({"status": "improved", "generated_at": "2026-08-12T16:59:00+08:00"})
        self._write_json(self.pipeline_root / "comparisons" / "older.json", older)
        (self.pipeline_root / "comparisons" / "corrupt.json").write_text("{", encoding="utf-8")

        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(f"href='/comparisons/{self.comparison_id}'", dashboard.text)
        self.assertNotIn("href='/comparisons/older'", dashboard.text)
        listing = self.client.get("/api/comparisons")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(
            [item["comparison_id"] for item in listing.json()["comparisons"]],
            [self.comparison_id, "older"],
        )

    def test_report_and_run_paths_stay_inside_runs_directory(self):
        report = self.client.get("/runs/completed-run")
        self.assertEqual(report.status_code, 200)
        self.assertNotIn(str(self.pipeline_root.resolve()), report.text)
        self.assertIn("对应运行目录", report.text)
        self.assertEqual(self.client.get("/api/runs/missing-run").status_code, 404)
        self.assertEqual(self.client.get("/runs/missing-run").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/%2e%2e%5cregistry.json").status_code, 404)
        self.assertEqual(self.client.get("/runs/%2e%2e%5cregistry.json").status_code, 404)
        self.assertEqual(self.client.get("/api/comparisons/missing").status_code, 404)
        self.assertEqual(self.client.get("/comparisons/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/comparisons/%2e%2e%5cregistry").status_code, 404)
        self.assertEqual(self.client.get("/comparisons/%2e%2e%5cregistry").status_code, 404)

    def test_relative_frozen_config_path_does_not_corrupt_report(self):
        manifest_path = self.completed_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config"] = "configs/baseline.yaml"
        self._write_json(manifest_path, manifest)
        generate_integrity_seal(self.completed_dir)
        response = self.client.get("/runs/completed-run")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<html>report at 对应运行目录</html>", response.text)


if __name__ == "__main__":
    unittest.main()
