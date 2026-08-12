import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.metrics import (
    compounded_return,
    evaluation_frame,
    hac_t_stat,
    independent_portfolio_performance,
    max_drawdown,
    period_portfolio_performance,
    relative_wealth_drawdown,
)


class MetricTests(unittest.TestCase):
    def test_compounding_and_drawdown(self):
        returns = pd.Series([0.10, -0.10])
        self.assertAlmostEqual(compounded_return(returns), -0.01)
        self.assertAlmostEqual(max_drawdown(returns), -0.10)

    def test_hac_t_stat_is_finite(self):
        values = pd.Series([0.001, 0.002, -0.001, 0.003] * 20)
        self.assertTrue(pd.notna(hac_t_stat(values)))

    def test_period_portfolio_performance_uses_net_compounding(self):
        index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        report = pd.DataFrame(
            {
                "return": [0.10, -0.10, 0.50],
                "cost": [0.01, 0.01, 0.00],
                "bench": [0.02, 0.01, 0.50],
            },
            index=index,
        )
        result = period_portfolio_performance(report, "2026-01-02", "2026-01-05")
        self.assertEqual(result["days"], 2)
        self.assertAlmostEqual(result["net_cumulative_return"], -0.0299)
        self.assertAlmostEqual(result["benchmark_cumulative_return"], 0.0302)
        expected_relative = (1.0 - 0.0299) / (1.0 + 0.0302) - 1.0
        self.assertAlmostEqual(result["excess_cumulative_return"], expected_relative)

    def test_evaluation_frame_keeps_setup_cost_and_drops_preinvestment_benchmark(self):
        report = pd.DataFrame(
            {
                "return": [0.0, 0.10, -0.05],
                "cost": [0.01, 0.002, 0.0],
                "bench": [-0.20, 0.03, -0.01],
            },
            index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        )
        aligned = evaluation_frame(report)
        self.assertEqual(aligned.index.tolist(), report.index[1:].tolist())
        self.assertAlmostEqual(aligned["strategy_net"].iloc[0], (1 - 0.01) * (1 + 0.098) - 1)
        self.assertAlmostEqual(aligned["benchmark"].iloc[0], 0.03)
        raw_terminal = (1 + report["return"] - report["cost"]).prod()
        self.assertAlmostEqual((1 + aligned["strategy_net"]).prod(), raw_terminal)

        summary = independent_portfolio_performance(report)
        self.assertEqual(summary["days"], 2)
        self.assertAlmostEqual(summary["net_cumulative_return"], raw_terminal - 1)
        self.assertEqual(summary["initial_execution_date"], "2026-01-05")
        self.assertEqual(summary["evaluation_start_date"], "2026-01-06")
        self.assertTrue(summary["reset_cash"])

    def test_evaluation_frame_rejects_nonzero_preinvestment_return(self):
        report = pd.DataFrame({"return": [0.01, 0.02], "cost": [0.0, 0.0], "bench": [0.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "initial execution"):
            evaluation_frame(report)

    def test_relative_drawdown_uses_relative_wealth(self):
        strategy = pd.Series([0.10, -0.05, 0.02])
        benchmark = pd.Series([0.00, 0.10, -0.02])
        relative = ((1 + strategy).cumprod() / (1 + benchmark).cumprod())
        expected = (relative / relative.cummax() - 1).min()
        self.assertAlmostEqual(relative_wealth_drawdown(strategy, benchmark), expected)


if __name__ == "__main__":
    unittest.main()
