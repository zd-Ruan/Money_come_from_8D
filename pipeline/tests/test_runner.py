import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.runner import select_backtest_predictions


class RunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
