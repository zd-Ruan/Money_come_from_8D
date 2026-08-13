import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.corporate_actions import (
    EX_DATE_PAYMENT_ASSUMPTION,
    RECEIVABLE_ASSUMPTION,
    RealPosition,
    apply_corporate_action,
    build_corporate_action_events,
    load_corporate_action_events,
    parse_sina_hfq_payload,
    qlib_adjusted_amount_to_raw_shares,
    raw_shares_to_qlib_adjusted_amount,
    reconcile_qlib_adjusted_position,
    save_corporate_action_events,
    validate_corporate_action_events,
)


SOURCE_URL = "https://finance.sina.com.cn/realstock/company/sh510300/hfq.js"


def hfq_payload(*events):
    rows = [{"d": "1900-01-01", "f": "1", "s": "1", "u": "0"}, *events]
    return "var sh510300hfq=" + json.dumps({"total": len(rows), "data": rows}) + "\n/* signature */"


def raw_prices(previous=10.0, observed=9.9):
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["SH510300", "SH510300"],
            "raw_close": [previous, observed],
        }
    )


def build_one(event, previous=10.0, observed=9.9):
    return build_corporate_action_events(
        "sh510300",
        hfq_payload(event),
        raw_prices(previous, observed),
        source_url=SOURCE_URL,
    )


class SinaPayloadTests(unittest.TestCase):
    def test_javascript_wrapper_is_parsed_without_execution(self):
        frame = parse_sina_hfq_payload(
            hfq_payload({"d": "2024-01-03", "f": "1", "s": "1", "u": "0.2"})
        )
        self.assertEqual(frame["date"].dt.date.astype(str).tolist(), ["1900-01-01", "2024-01-03"])
        self.assertEqual(frame.loc[1, "u"], 0.2)

    def test_stock_style_f_only_payload_fails_closed(self):
        payload = 'var sh600519hfq={"total":2,"data":[' \
            '{"d":"1900-01-01","f":"1"},{"d":"2024-01-03","f":"1.1"}]}'
        with self.assertRaisesRegex(ValueError, "lacks d/f/s/u"):
            parse_sina_hfq_payload(payload)

    def test_duplicate_dates_and_incorrect_total_are_rejected(self):
        duplicate = hfq_payload({"d": "1900-01-01", "f": "1", "s": "1", "u": "0"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_sina_hfq_payload(duplicate)
        bad_total = hfq_payload({"d": "2024-01-03", "f": "1", "s": "1", "u": "0"}).replace(
            '"total": 2', '"total": 3'
        )
        with self.assertRaisesRegex(ValueError, "total"):
            parse_sina_hfq_payload(bad_total)


class EventConstructionTests(unittest.TestCase):
    def test_pure_cash_formula_and_unknown_dates_are_explicit(self):
        payload = hfq_payload({"d": "2024-01-03", "f": "1", "s": "1", "u": "0.2"})
        events = build_corporate_action_events(
            "SH510300", payload, raw_prices(), source_url=SOURCE_URL
        )
        event = events.iloc[0]
        self.assertEqual(event["event_type"], "cash_only")
        self.assertAlmostEqual(event["share_ratio"], 1.0)
        self.assertAlmostEqual(event["cash_dividend_per_old_share"], 0.2)
        self.assertAlmostEqual(event["theoretical_ex_price"], 9.8)
        self.assertAlmostEqual(event["qlib_total_return_factor_ratio"], 10.0 / 9.8)
        self.assertAlmostEqual(event["observed_total_return"], 0.01)
        self.assertTrue(pd.isna(event["record_date"]))
        self.assertTrue(pd.isna(event["payment_date"]))
        self.assertFalse(event["cash_settlement_ready"])
        self.assertEqual(event["cash_payment_assumption"], RECEIVABLE_ASSUMPTION)
        self.assertEqual(event["source_sha256"], sha256(payload.encode("utf-8")).hexdigest())

    def test_pure_share_change_formula(self):
        event = build_one(
            {"d": "2024-01-03", "f": "1", "s": "2", "u": "0"},
            observed=5.1,
        ).iloc[0]
        self.assertEqual(event["event_type"], "share_change_only")
        self.assertAlmostEqual(event["share_ratio"], 2.0)
        self.assertEqual(event["cash_dividend_per_old_share"], 0.0)
        self.assertAlmostEqual(event["theoretical_ex_price"], 5.0)
        self.assertAlmostEqual(event["observed_total_return"], 0.02)
        self.assertFalse(event["position_adjustment_ready"])
        self.assertTrue(event["cash_settlement_ready"])
        self.assertIsNone(event["cash_payment_assumption"])

    def test_combined_event_uses_old_affine_scale(self):
        payload = hfq_payload(
            {"d": "2023-01-03", "f": "1", "s": "0.5", "u": "0.1"},
            {"d": "2024-01-03", "f": "1", "s": "1", "u": "0.25"},
        )
        prices = pd.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03", "2024-01-02", "2024-01-03"],
                "raw_close": [5.0, 10.0, 10.0, 4.9],
            }
        )
        events = build_corporate_action_events(
            "SH510300", payload, prices, source_url=SOURCE_URL
        )
        event = events.iloc[1]
        self.assertEqual(event["event_type"], "share_change_and_cash")
        self.assertAlmostEqual(event["share_ratio"], 2.0)
        self.assertAlmostEqual(event["cash_dividend_per_old_share"], 0.3)
        self.assertAlmostEqual(event["theoretical_ex_price"], 4.85)
        old_adjusted = 10.0 * event["old_f"] * event["old_s"] + event["old_u"]
        new_adjusted = event["theoretical_ex_price"] * event["new_f"] * event["new_s"] + event["new_u"]
        self.assertAlmostEqual(old_adjusted, new_adjusted)

    def test_no_op_source_rows_do_not_create_fake_actions(self):
        payload = hfq_payload(
            {"d": "2024-01-03", "f": "2", "s": "0.5", "u": "0"},
        )
        events = build_corporate_action_events(
            "SH510300", payload, raw_prices(), source_url=SOURCE_URL
        )
        self.assertTrue(events.empty)

    def test_missing_ex_date_price_and_negative_inferred_cash_fail_closed(self):
        missing_ex = raw_prices().iloc[:1]
        with self.assertRaisesRegex(ValueError, "ex-date raw close is unavailable"):
            build_corporate_action_events(
                "SH510300",
                hfq_payload({"d": "2024-01-03", "f": "1", "s": "1", "u": "0.2"}),
                missing_ex,
                source_url=SOURCE_URL,
            )
        with self.assertRaisesRegex(ValueError, "negative inferred cash"):
            build_one({"d": "2024-01-03", "f": "1", "s": "1", "u": "-0.1"})


class EventTableIOTests(unittest.TestCase):
    def setUp(self):
        self.events = build_one({"d": "2024-01-03", "f": "1", "s": "1", "u": "0.2"})

    def test_json_and_parquet_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("events.json", "events.parquet"):
                with self.subTest(name=name):
                    path = save_corporate_action_events(self.events, Path(directory) / name)
                    loaded = load_corporate_action_events(path)
                    pd.testing.assert_frame_equal(loaded, self.events, check_dtype=False)

    def test_json_envelope_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_corporate_action_events(self.events, Path(directory) / "events.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["events"][0]["share_ratio"] = 2.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "events_sha256 mismatch"):
                load_corporate_action_events(path)

    def test_algebraic_tampering_is_rejected_even_with_valid_table_shape(self):
        altered = self.events.copy()
        altered.loc[0, "cash_dividend_per_old_share"] = 0.3
        with self.assertRaisesRegex(ValueError, "affine corporate-action identity"):
            validate_corporate_action_events(altered)

    def test_invalid_optional_date_strings_cannot_become_unknown_dates(self):
        for column in ("record_date", "payment_date"):
            with self.subTest(column=column):
                altered = self.events.copy()
                altered[column] = altered[column].astype(object)
                altered.loc[0, column] = "definitely-not-a-date"
                with self.assertRaisesRegex(ValueError, f"{column} contains invalid dates"):
                    validate_corporate_action_events(altered)

    def test_symbol_collisions_after_normalisation_are_rejected(self):
        duplicate = pd.concat([self.events, self.events], ignore_index=True)
        duplicate.loc[0, "symbol"] = " sh510300 "
        with self.assertRaisesRegex(ValueError, "duplicate symbol/ex-date"):
            validate_corporate_action_events(duplicate)

    def test_duplicate_event_ids_are_rejected(self):
        duplicate_id = pd.concat([self.events, self.events], ignore_index=True)
        duplicate_id.loc[1, "ex_date"] = duplicate_id.loc[1, "ex_date"] + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            validate_corporate_action_events(duplicate_id)


class PositionConversionTests(unittest.TestCase):
    def setUp(self):
        self.event = build_one(
            {"d": "2024-01-03", "f": "1", "s": "2", "u": "0.4"},
            observed=4.9,
        ).iloc[0]

    def test_unverified_fractional_treatment_blocks_real_position_adjustment(self):
        position = RealPosition("sh510300", shares=100, cash=1_000)
        with self.assertRaisesRegex(ValueError, "needs a positive lot_size"):
            apply_corporate_action(position, self.event)

    def test_unverified_treatment_allows_a_provably_round_lot_result(self):
        position = RealPosition("sh510300", shares=100, cash=1_000)
        result = apply_corporate_action(position, self.event, lot_size=100)
        self.assertEqual(result.position.shares, 200)
        self.assertEqual(result.position.cash_receivable, 40)

    def test_unverified_treatment_rejects_an_odd_lot_result(self):
        event = build_one(
            {"d": "2024-01-03", "f": "1", "s": "1.25", "u": "0"},
            observed=8.0,
        ).iloc[0]
        with self.assertRaisesRegex(ValueError, "resulting position is not a round lot"):
            apply_corporate_action(
                RealPosition("sh510300", shares=100, cash=1_000),
                event,
                lot_size=100,
            )

    def test_cash_only_event_becomes_non_spendable_receivable_by_default(self):
        event = build_one(
            {"d": "2024-01-03", "f": "1", "s": "1", "u": "0.4"},
            observed=9.6,
        ).iloc[0]
        result = apply_corporate_action(RealPosition("sh510300", shares=100, cash=1_000), event)
        self.assertEqual(result.position.shares, 100)
        self.assertEqual(result.position.cash, 1_000)
        self.assertEqual(result.position.cash_receivable, 40)
        self.assertEqual(result.settlement_basis, RECEIVABLE_ASSUMPTION)

    def test_ex_date_cash_payment_requires_an_explicit_recorded_assumption(self):
        position = RealPosition("SH510300", shares=100, cash=1_000)
        event = build_one(
            {"d": "2024-01-03", "f": "1", "s": "1", "u": "0.4"},
            observed=9.6,
        ).iloc[0]
        result = apply_corporate_action(
            position,
            event,
            assume_cash_paid_on_ex_date=True,
        )
        self.assertEqual(result.position.cash, 1_040)
        self.assertEqual(result.position.cash_receivable, 0)
        self.assertEqual(result.settlement_basis, EX_DATE_PAYMENT_ASSUMPTION)

    def test_qlib_unit_conversion_is_value_conserving_and_invertible(self):
        adjusted_amount = 250.0
        factor = 0.4
        raw_shares = qlib_adjusted_amount_to_raw_shares(adjusted_amount, factor)
        self.assertEqual(raw_shares, 100.0)
        self.assertEqual(raw_shares_to_qlib_adjusted_amount(raw_shares, factor), adjusted_amount)
        adjusted_price = 25.0
        raw_price = adjusted_price / factor
        self.assertEqual(adjusted_amount * adjusted_price, raw_shares * raw_price)

    def test_qlib_reinvestment_is_separated_from_legal_entitlements(self):
        factor_before = 1.0
        factor_after = float(self.event["qlib_total_return_factor_ratio"])
        result = reconcile_qlib_adjusted_position(100, factor_before, factor_after, self.event)
        self.assertAlmostEqual(result.raw_shares_before, 100)
        self.assertAlmostEqual(result.legal_shares_after, 200)
        self.assertAlmostEqual(result.dividend_cash_entitlement, 40)
        self.assertAlmostEqual(result.qlib_implied_raw_shares_after, 100 * 10 / 4.8)
        self.assertAlmostEqual(result.qlib_implicit_reinvestment_shares, 40 / 4.8)
        self.assertAlmostEqual(result.theoretical_reinvestment_price, 4.8)

    def test_mismatched_qlib_factor_jump_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "factor change does not match"):
            reconcile_qlib_adjusted_position(100, 1.0, 1.1, self.event)


if __name__ == "__main__":
    unittest.main()
