import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.metrics import evaluation_frame
from quant_pipeline.factors import factor_catalog_manifest
from quant_pipeline.report import (
    _factor_audit,
    _fold_chart,
    _performance_chart,
    _relative_drawdown_chart,
    generate_report,
)


class ReportChartTests(unittest.TestCase):
    def test_charts_use_aligned_returns_and_relative_wealth_drawdown(self):
        report = pd.DataFrame(
            {
                "return": [0.0, 0.10, -0.05],
                "cost": [0.01, 0.002, 0.0],
                "bench": [-0.20, 0.03, 0.10],
            },
            index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        )
        aligned = evaluation_frame(report)
        performance = _performance_chart(aligned)
        self.assertEqual(list(performance.data[0].x), aligned.index.tolist())
        self.assertEqual(len(performance.data), 3)
        expected_strategy = (1.0 + aligned["strategy_net"]).cumprod() - 1.0
        expected_benchmark = (1.0 + aligned["benchmark"]).cumprod() - 1.0
        expected_relative = (1.0 + expected_strategy) / (1.0 + expected_benchmark) - 1.0
        for actual, expected in zip(performance.data[0].y, expected_strategy):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(performance.data[2].y, expected_relative):
            self.assertAlmostEqual(actual, expected)

        drawdown = _relative_drawdown_chart(aligned)
        relative_wealth = (1.0 + aligned["strategy_net"]).cumprod() / (1.0 + aligned["benchmark"]).cumprod()
        expected_drawdown = relative_wealth / relative_wealth.cummax() - 1.0
        for actual, expected in zip(drawdown.data[0].y, expected_drawdown):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(drawdown.layout.title.text, "相对基准财富回撤")

    def test_fold_chart_marks_independent_and_incomplete_folds(self):
        folds = [
            {
                "fold": 1,
                "portfolio": {
                    "excess_cumulative_return": 0.10,
                    "reset_cash": True,
                    "complete_for_gate": True,
                },
            },
            {
                "fold": 2,
                "portfolio": {
                    "excess_cumulative_return": -0.02,
                    "reset_cash": True,
                    "complete_for_gate": False,
                },
            },
        ]
        figure = _fold_chart(folds)
        self.assertEqual(figure.layout.title.text, "各折独立现金重置组合超额")
        self.assertEqual(list(figure.data[0].marker.color), ["#14765a", "#8a959b"])

    def test_plus_original_mode_does_not_hide_a_missing_frozen_catalog(self):
        label, section = _factor_audit({}, {"features": {"mode": "alpha158_plus_original"}})
        self.assertEqual(label, "Alpha158 + 原创研究候选（目录缺失）")
        self.assertEqual(section, "")

    def test_alpha360_mode_is_reported_without_an_original_factor_catalog(self):
        label, section = _factor_audit({}, {"features": {"mode": "alpha360", "families": []}})
        self.assertEqual(label, "Alpha360")
        self.assertEqual(section, "")


class GenerateReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary_directory.name) / "runs" / "test-run"
        backtest_dir = self.run_dir / "backtests" / "slippage_05bps"
        backtest_dir.mkdir(parents=True)
        self.factor_catalog = factor_catalog_manifest(["trend_crowding"])
        self._write_json(
            "manifest.json",
            {
                "run_id": "test-run",
                "snapshot_id": "snapshot-1",
                "status": "reporting",
                "factor_catalog": self.factor_catalog,
            },
        )
        self._write_json(
            "config.json",
            {
                "report": {"title": "ETF 测试报告"},
                "execution": {
                    "commission_bps_per_side": 3,
                    "standard_limit_ratio": 0.10,
                    "wide_limit_ratio": 0.20,
                    "price_tick": 0.001,
                },
                "data": {"benchmark": "SH510300"},
                "rolling": {"purge_bars": 2},
                "features": {
                    "mode": "alpha158_plus_original",
                    "families": ["trend_crowding"],
                },
            },
        )
        self._write_json(
            "metrics.json",
            {
                "base_slippage_bps_per_side": 5,
                "base": {
                    "net_cumulative_return": 0.10,
                    "benchmark_cumulative_return": 0.04,
                    "strategy_max_drawdown": -0.05,
                    "relative_wealth_max_drawdown": -0.03,
                    "fill_rate": 0.98,
                    "submitted_order_fill_rate": 1.0,
                    "market_rejection_rate": 0.0,
                    "policy_rejection_rate": 0.5,
                    "excess_hac_t_stat": 2.1,
                    "information_ratio": 1.2,
                    "beta": 0.9,
                    "beta_adjusted_alpha_annualized": 0.08,
                    "raw_execution_days": 3,
                    "days": 2,
                    "initial_execution_date": "2026-01-05",
                    "evaluation_start_date": "2026-01-06",
                    "evaluation_end_date": "2026-01-07",
                    "alignment_method": "initial_cost_compounded_into_first_realized_return",
                },
                "stress": {
                    "5": {
                        "slippage_bps_per_side": 5,
                        "net_cumulative_return": 0.10,
                        "benchmark_cumulative_return": 0.04,
                    }
                },
                "ic": 0.01,
                "ic_hac_t_stat": 3.25,
                "ic_t_stat": 99.0,
                "rank_ic": 0.02,
                "rank_ic_hac_t_stat": 4.5,
                "rank_ic_t_stat": 98.0,
                "folds": [
                    {
                        "fold": 1,
                        "train_start": "2025-01-01",
                        "train_end": "2025-06-30",
                        "valid_start": "2025-07-03",
                        "valid_end": "2025-08-31",
                        "test_start": "2025-09-03",
                        "test_end": "2025-10-31",
                        "rows": {"train": 1000, "test_features": 100},
                        "ic": 0.01,
                        "rank_ic": 0.02,
                        "best_iterations": [40, 42, 41],
                        "portfolio": {
                            "days": 40,
                            "net_cumulative_return": 0.08,
                            "benchmark_cumulative_return": 0.03,
                            "excess_cumulative_return": 0.05,
                            "reset_cash": True,
                            "complete_for_gate": True,
                        },
                    }
                ],
            },
        )
        self._write_json(
            "gates.json",
            {
                "status": "research_only",
                "passed": 1,
                "total": 2,
                "checks": [
                    {"name": "point_in_time_universe", "passed": False, "value": "current_snapshot"},
                    {"name": "data_valid", "passed": True, "value": True},
                ],
            },
        )
        report = pd.DataFrame(
            {
                "return": [0.0, 0.10, -0.05],
                "cost": [0.01, 0.002, 0.0],
                "bench": [-0.20, 0.03, -0.01],
            },
            index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        )
        report.to_parquet(backtest_dir / "report.parquet")
        prediction_index = pd.MultiIndex.from_product(
            [["ETF1", "ETF2"], pd.to_datetime(["2026-01-06", "2026-01-07"])],
            names=["instrument", "datetime"],
        )
        pd.DataFrame(
            {"score": [0.1, 0.2, 0.3, 0.4], "score_std": [0.01, 0.02, 0.01, 0.02]},
            index=prediction_index,
        ).to_parquet(self.run_dir / "predictions.parquet")
        pd.DataFrame(
            {"ic": [0.01, 0.02], "rank_ic": [0.02, 0.03]},
            index=pd.to_datetime(["2026-01-06", "2026-01-07"]),
        ).to_parquet(self.run_dir / "signal_metrics.parquet")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_json(self, relative_path, value):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_generated_report_uses_new_semantics_and_hides_local_path(self):
        output = generate_report(self.run_dir)
        document = output.read_text(encoding="utf-8")
        self.assertIn("相对财富最大回撤 -3.00%", document)
        self.assertIn("初始交易成本复合计入首个实现收益", document)
        self.assertIn("独立现金重置", document)
        self.assertIn("门禁完整", document)
        self.assertIn("IC / HAC t", document)
        self.assertIn("Rank IC / HAC t", document)
        self.assertIn("HAC t = 3.25", document)
        self.assertIn("HAC t = 4.50", document)
        self.assertIn("计算 10%/20% 方向性涨跌停", document)
        self.assertIn("否则按 10% 失败关闭", document)
        self.assertIn("登记日冻结分红权利", document)
        self.assertIn("发放日才成为可交易现金", document)
        self.assertNotIn("现金分红等价为无摩擦再投资", document)
        self.assertIn("日线回测不是订单簿仿真", document)
        self.assertNotIn("HAC t = 99.00", document)
        self.assertNotIn("HAC t = 98.00", document)
        self.assertIn(f"Alpha158 + {len(self.factor_catalog['factors'])} 个冻结原创研究候选", document)
        self.assertIn("冻结因子目录", document)
        self.assertIn("trend_crowding", document)
        self.assertIn("ORC_TREND_PATH_CROWD_20", document)
        self.assertIn("反向", document)
        self.assertIn("A directionally efficient 20-bar move", document)
        self.assertIn(self.factor_catalog["sha256"], document)
        for marker in ("鍊", "鏀", "绱", "€"):
            with self.subTest(mojibake_marker=marker):
                self.assertNotIn(marker, document)
        self.assertNotIn(str(self.run_dir.resolve()), document)
        self.assertNotIn(str(Path.home()), document)

    def test_generated_report_labels_legacy_signal_t_statistics(self):
        metrics_path = self.run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.pop("ic_hac_t_stat")
        metrics.pop("rank_ic_hac_t_stat")
        metrics["ic_t_stat"] = 2.75
        metrics["rank_ic_t_stat"] = 2.5
        self._write_json("metrics.json", metrics)

        document = generate_report(self.run_dir).read_text(encoding="utf-8")
        self.assertIn("普通 t（旧运行） = 2.75", document)
        self.assertIn("普通 t（旧运行） = 2.50", document)

    def test_completed_report_is_not_overwritten_by_default(self):
        output = generate_report(self.run_dir)
        original = output.read_bytes()
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        self._write_json("manifest.json", manifest)

        with self.assertRaisesRegex(FileExistsError, "immutable"):
            generate_report(self.run_dir)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
