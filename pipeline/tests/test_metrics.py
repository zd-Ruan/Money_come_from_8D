import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.metrics import compounded_return, hac_t_stat, max_drawdown, period_portfolio_performance


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
        self.assertAlmostEqual(result["excess_cumulative_return"], -0.0601)


if __name__ == "__main__":
    unittest.main()
