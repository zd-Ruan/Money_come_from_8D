import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.coverage import (
    calculate_prediction_coverage,
    load_qlib_coverage_inputs,
)


DATES = pd.date_range("2026-01-05", periods=4, freq="B")
SPANS = {
    "ETF_A": [(DATES[0], DATES[3])],
    "ETF_B": [(DATES[1], DATES[2])],
    "ETF_C": [(DATES[3], DATES[3])],
}


def eligibility_frame():
    index = pd.MultiIndex.from_tuples(
        [
            (DATES[0], "ETF_A"),
            (DATES[1], "ETF_A"),
            (DATES[2], "ETF_A"),
            (DATES[3], "ETF_A"),
            (DATES[1], "ETF_B"),
            (DATES[2], "ETF_B"),
            (DATES[3], "ETF_C"),
        ],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {
            "liquidity_eligible": [True, True, False, True, True, True, True],
            "$close": [10.0, 10.0, 10.0, 10.0, 20.0, float("nan"), 30.0],
            "$volume": [100.0] * 7,
            "$factor": [1.0] * 7,
        },
        index=index,
    )


def predictions(rows):
    index = pd.MultiIndex.from_tuples(
        [(date, instrument) for date, instrument, _ in rows],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"score": [score for _, _, score in rows]}, index=index)


class PredictionCoverageTests(unittest.TestCase):
    def test_denominator_uses_spans_liquidity_and_available_fields(self):
        pred = predictions(
            [
                (DATES[0], "ETF_A", 0.1),
                (DATES[1], "ETF_A", float("nan")),
                (DATES[1], "ETF_B", 0.2),
                (DATES[2], "ETF_A", 0.3),
                (DATES[3], "ETF_A", 0.4),
                (DATES[3], "ETF_C", 0.5),
            ]
        )
        result = calculate_prediction_coverage(SPANS, DATES, DATES, eligibility_frame(), pred)

        self.assertEqual(result["expected_rows"], 5)
        self.assertEqual(result["scored_rows"], 4)
        self.assertEqual(result["missing_rows"], 1)
        self.assertEqual(result["extra_prediction_rows"], 1)
        self.assertEqual(result["extra_scored_prediction_rows"], 1)
        self.assertEqual(result["coverage"], 0.8)
        self.assertEqual(result["daily_coverage_min"], 0.5)
        self.assertEqual(result["daily_coverage_median"], 1.0)
        self.assertAlmostEqual(result["daily_coverage_mean"], 5 / 6)
        self.assertEqual(result["dates_without_candidates"], [DATES[2].date().isoformat()])
        self.assertEqual(result["date_coverage"], 1.0)
        self.assertEqual(result["instrument_coverage"], 1.0)

    def test_missing_prediction_rows_and_dates_are_reported(self):
        pred = predictions([(DATES[0], "ETF_A", 0.1)])
        result = calculate_prediction_coverage(SPANS, DATES, DATES, eligibility_frame(), pred)
        self.assertEqual(result["expected_rows"], 5)
        self.assertEqual(result["scored_rows"], 1)
        self.assertEqual(result["missing_rows"], 4)
        self.assertEqual(result["date_coverage"], 1 / 3)
        self.assertEqual(result["instrument_coverage"], 1 / 3)
        self.assertEqual(result["dates_without_candidates"], [DATES[2].date().isoformat()])
        self.assertEqual(
            result["dates_without_scores"],
            [DATES[1].date().isoformat(), DATES[3].date().isoformat()],
        )
        self.assertEqual(result["instruments_without_scores"], ["ETF_B", "ETF_C"])

    def test_combined_boolean_mask_is_supported(self):
        frame = eligibility_frame()
        mask = (
            frame["liquidity_eligible"]
            & frame["$close"].notna()
            & frame["$volume"].notna()
            & frame["$factor"].notna()
        )
        pred = predictions([(date, instrument, 1.0) for date, instrument in mask.index[mask]])
        result = calculate_prediction_coverage(SPANS, DATES, DATES, mask, pred)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["extra_prediction_rows"], 0)

    def test_duplicate_prediction_index_is_rejected(self):
        pred = predictions(
            [
                (DATES[0], "ETF_A", 0.1),
                (DATES[0], "ETF_A", 0.2),
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            calculate_prediction_coverage(SPANS, DATES, DATES, eligibility_frame(), pred)

    def test_prediction_date_outside_test_range_is_rejected(self):
        pred = predictions([(DATES[3], "ETF_A", 0.1)])
        with self.assertRaisesRegex(ValueError, "outside test_dates"):
            calculate_prediction_coverage(SPANS, DATES, DATES[:3], eligibility_frame(), pred)

    def test_unknown_prediction_instrument_is_rejected(self):
        frame = eligibility_frame()
        extra = pd.DataFrame(
            {
                "liquidity_eligible": [True],
                "$close": [10.0],
                "$volume": [100.0],
                "$factor": [1.0],
            },
            index=pd.MultiIndex.from_tuples(
                [(DATES[0], "UNKNOWN")], names=["datetime", "instrument"]
            ),
        )
        pred = predictions([(DATES[0], "UNKNOWN", 0.1)])
        with self.assertRaisesRegex(ValueError, "outside active_spans"):
            calculate_prediction_coverage(SPANS, DATES, DATES, pd.concat([frame, extra]), pred)

    def test_missing_active_eligibility_row_is_rejected(self):
        frame = eligibility_frame().drop(index=(DATES[1], "ETF_B"))
        with self.assertRaisesRegex(ValueError, "missing active"):
            calculate_prediction_coverage(SPANS, DATES, DATES, frame, predictions([]))


class FakeQlibProvider:
    def __init__(self):
        self.requested_fields = None

    def instruments(self, market):
        return {"market": market, "filter_pipe": []}

    def list_instruments(self, instruments, **kwargs):
        self.list_kwargs = kwargs
        return {"ETF_A": [(DATES[0], DATES[1])]}

    def calendar(self, **kwargs):
        self.calendar_kwargs = kwargs
        return DATES[:2]

    def features(self, instruments, fields, **kwargs):
        self.requested_fields = fields
        index = pd.MultiIndex.from_product(
            [["ETF_A"], DATES[:2]], names=["instrument", "datetime"]
        )
        return pd.DataFrame(
            [[1.0, 10.0, 100.0, 1.0], [0.0, 10.0, 100.0, 1.0]],
            index=index,
            columns=fields,
        )


class QlibInputAdapterTests(unittest.TestCase):
    def test_qlib_loading_is_separate_and_returns_combined_mask(self):
        provider = FakeQlibProvider()
        result = load_qlib_coverage_inputs(
            "t1_etf",
            DATES[0],
            DATES[1],
            "Mean($close * $volume, 20) > 10000000",
            provider=provider,
        )
        self.assertEqual(result.calendar.tolist(), DATES[:2].tolist())
        self.assertEqual(set(result.active_spans), {"ETF_A"})
        self.assertEqual(result.eligibility.sum(), 1)
        self.assertEqual(
            provider.requested_fields,
            ["Mean($close * $volume, 20) > 10000000", "$close", "$volume", "$factor"],
        )


if __name__ == "__main__":
    unittest.main()
