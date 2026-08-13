import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.action_audit import (
    CorporateActionAuditError,
    audit_corporate_actions,
    detect_material_factor_changes,
)


HASH = "a" * 64
SYMBOL = "SH510300"


def calendar():
    return pd.bdate_range("2026-01-15", "2026-01-27")


def factors(after=10.0 / 9.8):
    return pd.DataFrame(
        {
            "date": calendar(),
            "symbol": SYMBOL,
            "factor": [1.0, 1.0, after, *([after] * (len(calendar()) - 3))],
        }
    )


def action(**updates):
    row = {
        "symbol": SYMBOL,
        "record_date": "2026-01-16",
        "ex_date": "2026-01-19",
        "cash_payment_date": "2026-01-27",
        "cash_dividend_per_old_share": 0.2,
        "share_ratio": 1.0,
        "fractional_share_treatment": "not_applicable_no_share_change",
        "source_sha256": HASH,
    }
    row.update(updates)
    return pd.DataFrame([row])


def raw_prices():
    closes = [10.0, 10.0, 9.9, *([9.9] * (len(calendar()) - 3))]
    return pd.DataFrame(
        {"date": calendar(), "symbol": SYMBOL, "raw_close": closes}
    )


class FactorChangeTests(unittest.TestCase):
    def test_default_tolerance_ignores_float_noise_and_keeps_material_jump(self):
        frame = pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-05", periods=3),
                "symbol": SYMBOL,
                "factor": [1.0, 1.0 + 5e-13, 1.1],
            }
        )
        changes = detect_material_factor_changes(frame)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes.iloc[0]["ex_date"], pd.Timestamp("2026-01-07"))
        self.assertAlmostEqual(changes.iloc[0]["factor_ratio"], 1.1 / (1.0 + 5e-13))


class PresenceAuditTests(unittest.TestCase):
    def test_factor_and_event_only_make_a_presence_claim(self):
        result = audit_corporate_actions(factors(), action(), calendar())
        self.assertTrue(result.passed)
        self.assertEqual(result.summary.iloc[0]["presence_only_count"], 1)
        detail = result.details.iloc[0]
        self.assertEqual(detail["status"], "matched_event_presence_only")
        self.assertIn("not economically verified", detail["economic_claim"])
        self.assertTrue(pd.isna(detail["action_implied_factor_ratio"]))

    def test_normalized_csv_input_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factor_path = root / "factors.csv"
            action_path = root / "actions.csv"
            factors().to_csv(factor_path, index=False)
            action().to_csv(action_path, index=False)
            result = audit_corporate_actions(factor_path, action_path, calendar())
        self.assertTrue(result.passed)

    def test_result_frames_write_to_csv_and_parquet(self):
        result = audit_corporate_actions(factors(), action(), calendar())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in (
                ("summary", result.summary),
                ("details", result.details),
                ("factor_changes", result.factor_changes),
            ):
                frame.to_csv(root / f"{name}.csv", index=False)
                frame.to_parquet(root / f"{name}.parquet", index=False)
                self.assertTrue((root / f"{name}.csv").is_file())
                self.assertTrue((root / f"{name}.parquet").is_file())

    def test_no_jumps_and_no_actions_pass_with_serializable_empty_details(self):
        result = audit_corporate_actions(factors(after=1.0), action().iloc[:0], calendar())
        self.assertTrue(result.passed)
        self.assertTrue(result.details.empty)
        self.assertTrue(result.factor_changes.empty)
        with tempfile.TemporaryDirectory() as directory:
            result.details.to_parquet(Path(directory) / "empty_details.parquet", index=False)
            result.factor_changes.to_parquet(
                Path(directory) / "empty_factor_changes.parquet",
                index=False,
            )

    def test_missing_and_extra_events_fail_with_persistable_diagnostics(self):
        empty_actions = action().iloc[:0]
        with self.assertRaises(CorporateActionAuditError) as missing_context:
            audit_corporate_actions(factors(), empty_actions, calendar())
        missing = missing_context.exception.result
        self.assertEqual(missing.details.iloc[0]["status"], "missing_action")
        self.assertEqual(missing.summary.iloc[0]["missing_action_count"], 1)

        unchanged = factors(after=1.0)
        result = audit_corporate_actions(
            unchanged,
            action(),
            calendar(),
            raise_on_failure=False,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.details.iloc[0]["status"], "extra_action")
        self.assertEqual(result.summary.iloc[0]["extra_action_count"], 1)


class InputIntegrityTests(unittest.TestCase):
    def test_source_hash_and_in_span_event_dates_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            audit_corporate_actions(factors(), action(source_sha256="ABC"), calendar())
        with self.assertRaisesRegex(ValueError, "not a trading session"):
            audit_corporate_actions(
                factors(),
                action(cash_payment_date="2026-01-24"),
                calendar(),
            )

    def test_record_and_payment_dates_may_cross_audit_boundaries(self):
        result = audit_corporate_actions(
            factors(),
            action(record_date="2026-01-14", cash_payment_date="2026-01-28"),
            calendar(),
        )
        self.assertTrue(result.passed)

    def test_cash_event_requires_record_and_payment_dates(self):
        for field in ("record_date", "cash_payment_date"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "cash event dates are incomplete"):
                    audit_corporate_actions(factors(), action(**{field: None}), calendar())

    def test_tiny_positive_cash_still_requires_complete_cash_dates(self):
        split = action(
            cash_dividend_per_old_share=1e-15,
            share_ratio=2.0,
            fractional_share_treatment="unknown_not_provided_by_eastmoney_archive",
            record_date=None,
            cash_payment_date=None,
        )
        with self.assertRaisesRegex(ValueError, "cash event dates are incomplete"):
            audit_corporate_actions(factors(), split, calendar())

    def test_factor_dates_outside_declared_calendar_fail_closed(self):
        outside = factors()
        outside.loc[0, "date"] = pd.Timestamp("2026-01-14")
        with self.assertRaisesRegex(ValueError, "factor date is outside calendar"):
            audit_corporate_actions(outside, action(), calendar())

    def test_unexplained_factor_gap_inside_observed_active_span_fails_closed(self):
        incomplete = factors().loc[lambda frame: frame["date"] != calendar()[1]]
        with self.assertRaisesRegex(
            ValueError,
            "missing a calendar session within the observed active span: "
            f"{SYMBOL} 2026-01-16",
        ):
            audit_corporate_actions(incomplete, action(), calendar())

    def test_factor_coverage_does_not_infer_activity_outside_observed_span(self):
        observed_span = factors().iloc[1:-1].copy()
        result = audit_corporate_actions(observed_span, action(), calendar())
        self.assertTrue(result.passed)


class RawPriceIdentityTests(unittest.TestCase):
    def test_raw_closes_prove_ex_right_identity_and_report_legal_return_separately(self):
        result = audit_corporate_actions(
            factors(),
            action(),
            calendar(),
            raw_prices=raw_prices(),
        )
        self.assertTrue(result.passed)
        detail = result.details.iloc[0]
        self.assertEqual(detail["status"], "matched_event_identity_verified")
        self.assertAlmostEqual(detail["theoretical_ex_close"], 9.8)
        self.assertAlmostEqual(detail["action_implied_factor_ratio"], 10.0 / 9.8)
        self.assertAlmostEqual(
            detail["qlib_adjusted_return_multiplier"],
            detail["action_ex_right_return_multiplier"],
        )
        self.assertNotAlmostEqual(
            detail["legal_holding_return_multiplier"],
            detail["qlib_adjusted_return_multiplier"],
        )

    def test_inconsistent_factor_ratio_fails_closed(self):
        result = audit_corporate_actions(
            factors(after=1.03),
            action(),
            calendar(),
            raw_prices=raw_prices(),
            raise_on_failure=False,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.details.iloc[0]["status"], "factor_identity_mismatch")
        self.assertEqual(result.summary.iloc[0]["identity_mismatch_count"], 1)

    def test_missing_identity_price_is_an_audit_failure(self):
        result = audit_corporate_actions(
            factors(),
            action(),
            calendar(),
            raw_prices=raw_prices().loc[lambda frame: frame["date"] != pd.Timestamp("2026-01-19")],
            raise_on_failure=False,
        )
        self.assertEqual(result.details.iloc[0]["status"], "raw_price_missing")
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
