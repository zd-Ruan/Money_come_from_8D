import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.audit import ensure_future_calendar_boundary, evaluate_upstream_validation


class AuditTests(unittest.TestCase):
    def test_future_calendar_contains_data_calendar_and_one_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            calendar = Path(directory) / "day.txt"
            calendar.write_text("2026-08-07\n2026-08-10\n", encoding="utf-8")
            future = ensure_future_calendar_boundary(calendar)
            values = pd.read_csv(future, header=None).iloc[:, 0].tolist()
            self.assertEqual(values, ["2026-08-07", "2026-08-10", "2026-08-11"])
            self.assertEqual(ensure_future_calendar_boundary(calendar), future)

    def test_pool_external_cache_explains_upstream_count_warnings(self):
        report = {
            "training_ready": False,
            "universe_count": 3,
            "min_latest_date": "2026-08-10",
            "max_latest_date": "2026-08-10",
            "issues": [
                {"error": "raw file count 5 != universe 3"},
                {"error": "normalized file count 5 != universe 3"},
            ],
        }
        blocking, warnings = evaluate_upstream_validation(
            report,
            universe_count=3,
            configured_end=pd.Timestamp("2026-08-10"),
            raw_file_count=5,
            normalized_file_count=5,
            external_raw_count=2,
            external_normalized_count=2,
        )
        self.assertEqual(blocking, [])
        self.assertEqual(len(warnings), 1)

    def test_unexpected_upstream_issue_blocks_training(self):
        report = {
            "training_ready": False,
            "universe_count": 3,
            "min_latest_date": "2026-08-10",
            "max_latest_date": "2026-08-10",
            "issues": [{"error": "non-positive close in sh510300.csv"}],
        }
        blocking, _ = evaluate_upstream_validation(
            report,
            universe_count=3,
            configured_end=pd.Timestamp("2026-08-10"),
            raw_file_count=3,
            normalized_file_count=3,
            external_raw_count=0,
            external_normalized_count=0,
        )
        self.assertIn("upstream validation: non-positive close in sh510300.csv", blocking)


if __name__ == "__main__":
    unittest.main()
