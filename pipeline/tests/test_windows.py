import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.windows import build_rolling_folds, shift_session, validate_fold_boundaries


class RollingWindowTests(unittest.TestCase):
    def test_purge_and_contiguous_tests(self):
        calendar = pd.bdate_range("2020-01-01", periods=500)
        folds = build_rolling_folds(
            calendar,
            train_start_date="2020-01-01",
            test_start_date=calendar[300].date().isoformat(),
            validation_days=50,
            test_days=40,
            purge_bars=2,
        )
        validate_fold_boundaries(folds, calendar)
        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0].test_start, calendar[300].date().isoformat())
        self.assertEqual(folds[-1].test_end, calendar[-1].date().isoformat())

    def test_rejects_insufficient_history(self):
        calendar = pd.bdate_range("2020-01-01", periods=100)
        with self.assertRaises(ValueError):
            build_rolling_folds(calendar, "2020-01-01", calendar[20].date().isoformat(), 50, 10, 2)

    def test_shift_session_uses_trading_days(self):
        calendar = pd.bdate_range("2026-08-03", periods=6)
        self.assertEqual(shift_session(calendar, "2026-08-06", 2), "2026-08-10")
        with self.assertRaises(ValueError):
            shift_session(calendar, "2026-08-10", 2)


if __name__ == "__main__":
    unittest.main()
