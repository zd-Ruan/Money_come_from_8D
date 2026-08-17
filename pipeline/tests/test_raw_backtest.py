import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.raw_backtest import RawBacktestConfig, run_raw_backtest


def calendar(periods=5):
    return pd.bdate_range("2026-01-05", periods=periods)


def bars(symbols, closes, dates=None, *, volumes=1_000_000):
    dates = calendar(len(closes)) if dates is None else pd.DatetimeIndex(dates)
    rows = []
    for symbol in symbols:
        symbol_closes = closes[symbol] if isinstance(closes, dict) else closes
        symbol_volumes = volumes[symbol] if isinstance(volumes, dict) else volumes
        if np.isscalar(symbol_volumes):
            symbol_volumes = [symbol_volumes] * len(dates)
        for date, close, volume in zip(dates, symbol_closes, symbol_volumes):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "raw_open": close,
                    "raw_close": close,
                    "raw_high": close,
                    "raw_low": close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def scores(rows):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), symbol) for date, symbol, _ in rows],
        names=["datetime", "instrument"],
    )
    return pd.Series([score for _, _, score in rows], index=index, name="score")


def no_actions():
    return pd.DataFrame(
        columns=[
            "symbol",
            "record_date",
            "ex_date",
            "cash_payment_date",
            "cash_dividend_per_old_share",
            "share_ratio",
            "fractional_share_treatment",
        ]
    )


def benchmark_frame(dates, close=10.0, symbol="BENCH"):
    values = [close] * len(dates) if np.isscalar(close) else close
    return pd.DataFrame({"date": dates, "symbol": symbol, "raw_close": values})


class FrozenIntentTests(unittest.TestCase):
    def test_signal_executes_next_session_and_limit_rejection_does_not_substitute(self):
        dates = calendar(3)
        raw = bars(["A", "B"], {"A": [10.0, 11.0, 11.0], "B": [10.0, 10.0, 10.0]}, dates)
        # Exactly +10% with no proof of the 20% tier is conservatively limit-up.
        raw.loc[(raw["symbol"] == "A") & (raw["date"] == dates[1]), "raw_low"] = 10.0
        predictions = scores([(dates[0], "A", 2.0), (dates[0], "B", 1.0)])
        result = run_raw_backtest(
            predictions,
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(topk=1, n_drop=1, hold_thresh=1, slippage_bps_per_side=0),
            benchmark_close=benchmark_frame(dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )

        self.assertEqual(len(result.executions), 1)
        order = result.executions.iloc[0]
        self.assertEqual(order["signal_date"], dates[0])
        self.assertEqual(order["execution_date"], dates[1])
        self.assertEqual(order["symbol"], "A")
        self.assertEqual(order["fill_shares"], 0)
        self.assertEqual(order["reason"], "price_limit_buy")
        self.assertNotIn("B", result.executions["symbol"].tolist())
        self.assertEqual(result.summary["zero_fill_intent_count"], 1)
        # price_limit is a MARKET rejection: the order was submitted but the
        # market could not fill it, so it stays in the submitted denominator.
        self.assertEqual(result.summary["market_rejection_count"], 1)
        self.assertEqual(result.summary["policy_rejection_count"], 0)
        self.assertEqual(result.summary["submitted_order_count"], 1)
        self.assertEqual(result.summary["submitted_order_fill_rate"], 0.0)

    def test_wide_tier_must_be_proven_by_range_and_close_controls_rejection(self):
        dates = calendar(2)
        raw = bars(["A"], {"A": [10.0, 11.0]}, dates)
        raw.loc[raw["date"] == dates[1], "raw_high"] = 11.2
        raw.loc[raw["date"] == dates[1], "raw_low"] = 9.9
        result = run_raw_backtest(
            scores([(dates[0], "A", 1.0)]),
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(topk=1, n_drop=1, slippage_bps_per_side=0),
            benchmark_close=benchmark_frame(dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )
        order = result.executions.iloc[0]
        self.assertGreater(order["fill_shares"], 0)
        self.assertTrue(order["wide_limit_tier_proven"])
        self.assertEqual(order["reason"], "filled")


class ExecutionMechanicsTests(unittest.TestCase):
    def test_round_lot_volume_cap_minimum_commission_and_slippage_are_independent(self):
        dates = calendar(2)
        raw = bars(["A"], {"A": [1.0, 1.0]}, dates, volumes=3_000)
        result = run_raw_backtest(
            scores([(dates[0], "A", 1.0)]),
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(
                initial_cash=20_000,
                topk=1,
                n_drop=1,
                risk_degree=0.9,
                commission_bps_per_side=3,
                min_commission=5,
                slippage_bps_per_side=5,
                max_daily_volume_participation=0.05,
            ),
            benchmark_close=benchmark_frame(dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )
        order = result.executions.iloc[0]
        self.assertEqual(order["target_shares"], 18_000)
        self.assertEqual(order["volume_cap_shares"], 100)
        self.assertEqual(order["fill_shares"], 100)
        self.assertEqual(order["commission"], 5.0)
        self.assertEqual(order["slippage"], 0.05)
        self.assertEqual(order["reason"], "partial_volume_limit")
        self.assertEqual(order["fill_shares"] % 100, 0)
        self.assertAlmostEqual(result.report.iloc[-1]["cost"], 5.05 / 20_000)
        self.assertAlmostEqual(result.report.iloc[-1]["account"], 20_000 - 5.05)

    def test_symbol_attribution_reconciles_daily_nav_and_includes_initial_costs(self):
        dates = calendar(4)
        raw = bars(
            ["A", "B"],
            {
                "A": [10.0, 10.0, 10.5, 10.5],
                "B": [20.0, 20.0, 20.0, 20.0],
            },
            dates,
        )
        result = run_raw_backtest(
            scores(
                [
                    (dates[0], "A", 2.0),
                    (dates[0], "B", 1.0),
                    (dates[1], "A", 0.0),
                    (dates[1], "B", 2.0),
                ]
            ),
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(
                initial_cash=20_000,
                topk=1,
                n_drop=1,
                hold_thresh=1,
                risk_degree=0.5,
                commission_bps_per_side=3,
                min_commission=5,
                slippage_bps_per_side=5,
            ),
            benchmark_close=benchmark_frame(dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )

        attribution = result.symbol_attribution
        expected_daily_pnl = result.report["account"].diff()
        expected_daily_pnl.iloc[0] = result.report.iloc[0]["account"] - 20_000
        actual_daily_pnl = (
            attribution.groupby("date")["net_pnl"]
            .sum()
            .reindex(result.report.index, fill_value=0.0)
        )
        np.testing.assert_allclose(actual_daily_pnl, expected_daily_pnl, atol=1e-10)
        np.testing.assert_allclose(
            attribution["net_pnl"],
            attribution[
                [
                    "price_pnl",
                    "commission_pnl",
                    "slippage_pnl",
                    "dividend_receivable_pnl",
                    "dividend_payment_pnl",
                ]
            ].sum(axis=1),
            atol=1e-12,
        )

        initial = attribution.set_index(["date", "symbol"]).loc[(dates[1], "A")]
        self.assertEqual(initial["opening_market_value"], 0.0)
        self.assertEqual(initial["closing_market_value"], initial["buy_notional"])
        self.assertEqual(initial["price_pnl"], 0.0)
        self.assertEqual(initial["commission_pnl"], -5.0)
        self.assertEqual(initial["slippage_pnl"], -5.0)
        self.assertEqual(initial["net_pnl"], -10.0)
        sale = attribution.set_index(["date", "symbol"]).loc[(dates[2], "A")]
        self.assertEqual(sale["buy_commission"], 0.0)
        self.assertEqual(sale["sell_commission"], 5.0)
        self.assertEqual(sale["commission_pnl"], -5.0)
        self.assertEqual(sale["sell_slippage"], 5.25)
        self.assertEqual(sale["slippage_pnl"], -5.25)
        self.assertAlmostEqual(attribution["abs_contribution_share"].sum(), 1.0)
        self.assertEqual(
            result.summary["max_single_etf_abs_contribution_share"],
            result.summary["max_single_etf_gross_abs_contribution_share"],
        )
        self.assertGreater(
            result.summary["single_etf_gross_abs_contribution_denominator_cny"],
            0.0,
        )
        self.assertTrue(result.summary["symbol_attribution_reconciled"])

    def test_hold_threshold_blocks_frozen_sell_and_records_zero_fill(self):
        dates = calendar(4)
        raw = bars(["A", "B"], {"A": [10.0] * 4, "B": [10.0] * 4}, dates)
        predictions = scores(
            [
                (dates[0], "A", 2.0),
                (dates[0], "B", 1.0),
                (dates[1], "A", 0.0),
                (dates[1], "B", 2.0),
            ]
        )
        result = run_raw_backtest(
            predictions,
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(
                initial_cash=2_000,
                topk=1,
                n_drop=1,
                hold_thresh=2,
                risk_degree=0.5,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )
        sell = result.executions[
            (result.executions["execution_date"] == dates[2])
            & (result.executions["symbol"] == "A")
            & (result.executions["direction"] == "sell")
        ].iloc[0]
        self.assertEqual(sell["fill_shares"], 0)
        self.assertEqual(sell["reason"], "hold_threshold_t_plus_one")
        # Layered execution accounting: hold_threshold and below_round_lot are
        # POLICY rejections (strategy/account declined to submit), NOT market
        # execution failures, and must not drag down the submitted fill rate.
        self.assertEqual(result.summary["policy_rejection_count"], 2)
        self.assertEqual(result.summary["market_rejection_count"], 0)
        self.assertEqual(result.summary["submitted_order_count"], 1)
        self.assertEqual(result.summary["submitted_order_fill_rate"], 1.0)

    def test_repeated_rotation_keeps_risky_value_at_configured_nav_fraction(self):
        dates = calendar(5)
        raw = bars(["A", "B", "C", "D"], [1.0] * len(dates), dates)
        predictions = scores(
            [
                (dates[0], "A", 4.0),
                (dates[0], "B", 3.0),
                (dates[0], "C", 2.0),
                (dates[1], "B", 4.0),
                (dates[1], "C", 3.0),
                (dates[1], "A", 1.0),
                (dates[2], "C", 4.0),
                (dates[2], "D", 3.0),
                (dates[2], "B", 1.0),
            ]
        )
        result = run_raw_backtest(
            predictions,
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(
                initial_cash=20_000,
                topk=2,
                n_drop=1,
                hold_thresh=1,
                risk_degree=0.9,
                commission_bps_per_side=0,
                min_commission=0,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(dates, close=1.0),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )

        for execution_date in dates[1:4]:
            row = result.report.loc[execution_date]
            self.assertAlmostEqual(row["value"] / row["account"], 0.9, places=12)

    def test_joint_buy_budget_removes_order_dependent_minimum_commission_shortfall(self):
        dates = calendar(2)
        raw = bars(["A", "B"], [1.0, 1.0], dates)
        result = run_raw_backtest(
            scores([(dates[0], "A", 2.0), (dates[0], "B", 1.0)]),
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(
                initial_cash=20_000,
                topk=2,
                n_drop=1,
                risk_degree=1.0,
                commission_bps_per_side=0,
                min_commission=5,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(dates, close=1.0),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )

        buys = result.executions[result.executions["direction"] == "buy"].set_index("symbol")
        self.assertEqual(buys.loc["A", "target_shares"], buys.loc["B", "target_shares"])
        self.assertEqual(buys.loc["A", "fill_shares"], buys.loc["B", "fill_shares"])
        self.assertEqual(buys.loc["A", "reason"], "partial_joint_cash_budget")
        self.assertEqual(buys.loc["B", "reason"], "partial_joint_cash_budget")

    def test_missing_score_for_held_symbol_is_kept_and_audited(self):
        dates = calendar(4)
        raw = bars(["A", "B"], [10.0] * len(dates), dates)
        result = run_raw_backtest(
            scores(
                [
                    (dates[0], "A", 1.0),
                    (dates[1], "B", 2.0),
                ]
            ),
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(
                initial_cash=2_000,
                topk=1,
                n_drop=1,
                hold_thresh=1,
                risk_degree=0.5,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )

        self.assertEqual(result.executions["symbol"].tolist(), ["A"])
        self.assertEqual(result.positions.loc[result.positions["date"] == dates[2], "symbol"].tolist(), ["A"])
        self.assertEqual(result.summary["held_missing_prediction_event_count"], 1)
        self.assertEqual(
            result.summary["held_missing_prediction_events"],
            [{"signal_date": dates[1].date().isoformat(), "symbol": "A", "action": "held"}],
        )


class CorporateActionLedgerTests(unittest.TestCase):
    def test_dividend_is_receivable_at_record_cash_at_payment_and_split_changes_real_shares(self):
        dates = calendar(6)
        raw = bars(["A"], {"A": [10.0, 10.0, 10.0, 4.9, 4.9, 4.9]}, dates)
        actions = pd.DataFrame(
            {
                "symbol": ["A"],
                "record_date": [dates[2]],
                "ex_date": [dates[3]],
                "cash_payment_date": [dates[4]],
                "cash_dividend_per_old_share": [0.2],
                "share_ratio": [2.0],
                "fractional_share_treatment": ["unknown_not_provided_by_eastmoney_archive"],
            }
        )
        result = run_raw_backtest(
            scores([(dates[0], "A", 1.0)]),
            raw,
            actions,
            dates,
            RawBacktestConfig(
                initial_cash=3_000,
                topk=1,
                n_drop=1,
                risk_degree=0.5,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(dates, [10.0, 10.0, 10.0, 4.9, 4.9, 4.9], "A"),
            benchmark_symbol="A",
            factor_jumps_pre_audited=True,
        )

        record = result.report.loc[dates[2]]
        self.assertEqual(record["receivable"], 0.0)
        self.assertAlmostEqual(record["cash"], 1_995.0)
        self.assertAlmostEqual(record["account"], result.report.loc[dates[1], "account"])
        ex_date = result.report.loc[dates[3]]
        self.assertEqual(ex_date["receivable"], 20.0)
        self.assertAlmostEqual(ex_date["account"], record["account"])
        self.assertAlmostEqual(ex_date["return"] - ex_date["cost"], 0.0)
        payment = result.report.loc[dates[4]]
        self.assertEqual(payment["receivable"], 0.0)
        self.assertAlmostEqual(payment["cash"], 2_015.0)
        latest = result.positions[result.positions["date"] == dates[-1]].iloc[0]
        self.assertEqual(latest["shares"], 200.0)
        self.assertAlmostEqual(payment["account"], result.report.loc[dates[3], "account"])

        ledger = result.corporate_action_ledger.set_index("action")
        self.assertEqual(ledger.loc["dividend_entitlement", "amount"], 20.0)
        self.assertEqual(ledger.loc["dividend_entitlement", "receivable_after"], 0.0)
        self.assertEqual(ledger.loc["dividend_receivable", "amount"], 20.0)
        self.assertEqual(ledger.loc["dividend_receivable", "receivable_after"], 20.0)
        self.assertEqual(ledger.loc["share_adjustment", "shares_before"], 100.0)
        self.assertEqual(ledger.loc["share_adjustment", "shares_after"], 200.0)
        self.assertEqual(ledger.loc["cash_payment", "amount"], 20.0)
        self.assertTrue((result.corporate_action_ledger["commission"] == 0).all())
        self.assertTrue((result.corporate_action_ledger["turnover"] == 0).all())
        self.assertAlmostEqual(result.report.loc[dates[3], "bench"], 0.0)
        self.assertEqual(result.summary["pending_dividend_entitlement_count"], 0)
        self.assertEqual(
            result.summary["corporate_action_policy"],
            "record_date_entitlement_off_balance_sheet_"
            "ex_date_receivable_and_share_adjustment_payment_date_cash",
        )

        attribution = result.symbol_attribution.set_index(["date", "symbol"])
        record_attribution = attribution.loc[(dates[2], "A")]
        self.assertEqual(record_attribution["dividend_entitlement"], 20.0)
        self.assertEqual(record_attribution["dividend_receivable_pnl"], 0.0)
        ex_attribution = attribution.loc[(dates[3], "A")]
        self.assertAlmostEqual(ex_attribution["price_pnl"], -20.0)
        self.assertEqual(ex_attribution["dividend_receivable_pnl"], 20.0)
        self.assertAlmostEqual(ex_attribution["net_pnl"], 0.0)
        payment_attribution = attribution.loc[(dates[4], "A")]
        self.assertEqual(payment_attribution["dividend_receivable_pnl"], -20.0)
        self.assertEqual(payment_attribution["dividend_payment_pnl"], 20.0)
        self.assertEqual(payment_attribution["net_pnl"], 0.0)

    def test_pre_start_record_date_freezes_known_zero_entitlement(self):
        dates = calendar(4)
        simulation_dates = dates[1:]
        raw = bars(["A"], {"A": [10.0, 9.8, 9.8]}, simulation_dates)
        actions = pd.DataFrame(
            {
                "symbol": ["A"],
                "record_date": [dates[0]],
                "ex_date": [dates[2]],
                "cash_payment_date": [dates[3]],
                "cash_dividend_per_old_share": [0.2],
                "share_ratio": [1.0],
                "fractional_share_treatment": ["not_applicable_no_share_change"],
            }
        )
        result = run_raw_backtest(
            scores([(dates[1], "A", 1.0)]),
            raw,
            actions,
            simulation_dates,
            RawBacktestConfig(
                initial_cash=3_000,
                topk=1,
                n_drop=1,
                risk_degree=0.5,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(simulation_dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )

        ledger = result.corporate_action_ledger.set_index("action")
        self.assertEqual(ledger.loc["dividend_receivable", "entitlement_shares"], 0.0)
        self.assertEqual(ledger.loc["dividend_receivable", "amount"], 0.0)
        self.assertEqual(ledger.loc["cash_payment", "amount"], 0.0)
        self.assertEqual(result.summary["pending_dividend_entitlement_count"], 0)
        self.assertTrue((result.report["receivable"] == 0).all())

    def test_same_day_record_and_ex_uses_old_shares_without_double_counting_nav(self):
        dates = calendar(4)
        raw = bars(["A"], {"A": [10.0, 10.0, 4.9, 4.9]}, dates)
        actions = pd.DataFrame(
            {
                "symbol": ["A"],
                "record_date": [dates[2]],
                "ex_date": [dates[2]],
                "cash_payment_date": [dates[3]],
                "cash_dividend_per_old_share": [0.2],
                "share_ratio": [2.0],
                "fractional_share_treatment": ["unknown_not_provided_by_eastmoney_archive"],
            }
        )
        result = run_raw_backtest(
            scores([(dates[0], "A", 1.0)]),
            raw,
            actions,
            dates,
            RawBacktestConfig(
                initial_cash=3_000,
                topk=1,
                n_drop=1,
                risk_degree=0.5,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(dates, [10.0, 10.0, 4.9, 4.9], "A"),
            benchmark_symbol="A",
            factor_jumps_pre_audited=True,
        )
        ledger = result.corporate_action_ledger.set_index("action")
        self.assertEqual(ledger.loc["dividend_entitlement", "entitlement_shares"], 100.0)
        self.assertEqual(ledger.loc["dividend_receivable", "amount"], 20.0)
        self.assertEqual(ledger.loc["share_adjustment", "shares_after"], 200.0)
        self.assertAlmostEqual(result.report.loc[dates[2], "account"], result.report.loc[dates[1], "account"])

    def test_unknown_record_or_payment_date_fails_before_backtest(self):
        dates = calendar(2)
        raw = bars(["A"], {"A": [10.0, 10.0]}, dates)
        base = {
            "symbol": "A",
            "record_date": dates[0],
            "ex_date": dates[1],
            "cash_payment_date": dates[1] + pd.Timedelta(days=1),
            "cash_dividend_per_old_share": 0.2,
            "share_ratio": 1.0,
            "fractional_share_treatment": "not_applicable_no_share_change",
        }
        for missing in ("record_date", "cash_payment_date"):
            with self.subTest(missing=missing):
                action = dict(base)
                action[missing] = pd.NaT
                with self.assertRaisesRegex(ValueError, missing):
                    run_raw_backtest(
                        scores([(dates[0], "A", 1.0)]),
                        raw,
                        pd.DataFrame([action]),
                        dates,
                        RawBacktestConfig(topk=1, n_drop=1),
                        benchmark_close=benchmark_frame(dates),
                        benchmark_symbol="BENCH",
                        factor_jumps_pre_audited=True,
                    )

    def test_pure_share_split_allows_null_cash_dates_and_only_adjusts_shares(self):
        dates = calendar(4)
        raw = bars(["A"], {"A": [10.0, 10.0, 5.0, 5.0]}, dates)
        actions = pd.DataFrame(
            {
                "symbol": ["A"],
                "record_date": [pd.NaT],
                "ex_date": [dates[2]],
                "cash_payment_date": [pd.NaT],
                "cash_dividend_per_old_share": [0.0],
                "share_ratio": [2.0],
                "fractional_share_treatment": ["unknown_not_provided_by_eastmoney_archive"],
            }
        )
        result = run_raw_backtest(
            scores([(dates[0], "A", 1.0)]),
            raw,
            actions,
            dates,
            RawBacktestConfig(
                initial_cash=3_000,
                topk=1,
                n_drop=1,
                risk_degree=0.5,
                slippage_bps_per_side=0,
            ),
            benchmark_close=benchmark_frame(dates, [10.0, 10.0, 5.0, 5.0], "A"),
            benchmark_symbol="A",
            factor_jumps_pre_audited=True,
        )
        self.assertEqual(result.corporate_action_ledger["action"].tolist(), ["share_adjustment"])
        self.assertEqual(result.corporate_action_ledger.iloc[0]["shares_after"], 200.0)
        self.assertTrue((result.report["receivable"] == 0).all())
        self.assertAlmostEqual(result.report.loc[dates[2], "account"], result.report.loc[dates[1], "account"])

    def test_share_adjustment_that_creates_odd_lot_fails_before_mutating_position(self):
        dates = calendar(4)
        raw = bars(["A"], {"A": [10.0, 10.0, 8.0, 8.0]}, dates)
        actions = pd.DataFrame(
            {
                "symbol": ["A"],
                "record_date": [pd.NaT],
                "ex_date": [dates[2]],
                "cash_payment_date": [pd.NaT],
                "cash_dividend_per_old_share": [0.0],
                "share_ratio": [1.25],
                "fractional_share_treatment": ["unknown_not_provided_by_eastmoney_archive"],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "creates a non-round-lot position"):
            run_raw_backtest(
                scores([(dates[0], "A", 1.0)]),
                raw,
                actions,
                dates,
                RawBacktestConfig(
                    initial_cash=3_000,
                    topk=1,
                    n_drop=1,
                    risk_degree=0.5,
                    slippage_bps_per_side=0,
                ),
                benchmark_close=benchmark_frame(dates, [10.0, 10.0, 8.0, 8.0], "A"),
                benchmark_symbol="A",
                factor_jumps_pre_audited=True,
            )

    def test_unheld_share_adjustment_does_not_require_a_fractional_assumption(self):
        dates = calendar(3)
        raw = bars(["A"], {"A": [10.0, 8.0, 8.0]}, dates)
        actions = pd.DataFrame(
            {
                "symbol": ["A"],
                "record_date": [pd.NaT],
                "ex_date": [dates[1]],
                "cash_payment_date": [pd.NaT],
                "cash_dividend_per_old_share": [0.0],
                "share_ratio": [1.25],
                "fractional_share_treatment": ["unknown_not_provided_by_eastmoney_archive"],
            }
        )
        result = run_raw_backtest(
            scores([(dates[1], "A", 1.0)]),
            raw,
            actions,
            dates,
            RawBacktestConfig(initial_cash=3_000, topk=1, n_drop=1, slippage_bps_per_side=0),
            benchmark_close=benchmark_frame(dates, [10.0, 8.0, 8.0], "A"),
            benchmark_symbol="A",
            factor_jumps_pre_audited=True,
        )
        self.assertEqual(result.corporate_action_ledger.iloc[0]["shares_after"], 0.0)


class IntegrityTests(unittest.TestCase):
    def test_report_reconciles_to_nav_and_benchmark_total_return(self):
        dates = calendar(3)
        raw = bars(["A"], {"A": [10.0, 10.0, 11.0]}, dates)
        benchmark_actions = pd.DataFrame(
            {
                "symbol": ["BENCH"],
                "record_date": [dates[0]],
                "ex_date": [dates[1]],
                "cash_payment_date": [dates[2]],
                "cash_dividend_per_old_share": [1.0],
                "share_ratio": [1.0],
                "fractional_share_treatment": ["not_applicable_no_share_change"],
            }
        )
        result = run_raw_backtest(
            scores([(dates[0], "A", 1.0)]),
            raw,
            benchmark_actions,
            dates,
            RawBacktestConfig(topk=1, n_drop=1, slippage_bps_per_side=0),
            benchmark_close=benchmark_frame(dates, [10.0, 9.0, 9.9]),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )
        ratios = result.report["account"] / result.report["account"].shift(1)
        expected = 1.0 + result.report["return"] - result.report["cost"]
        np.testing.assert_allclose(ratios.iloc[1:], expected.iloc[1:], atol=1e-12)
        self.assertAlmostEqual(result.report.loc[dates[1], "bench"], 0.0)
        self.assertAlmostEqual(result.report.loc[dates[2], "bench"], 0.1)
        self.assertTrue(result.summary["nav_reconciled"])

    def test_prediction_without_next_execution_session_fails_closed(self):
        dates = calendar(2)
        raw = bars(["A"], {"A": [10.0, 10.0]}, dates)
        with self.assertRaisesRegex(ValueError, "next execution session"):
            run_raw_backtest(
                scores([(dates[-1], "A", 1.0)]),
                raw,
                no_actions(),
                dates,
                RawBacktestConfig(topk=1, n_drop=1),
                benchmark_close=benchmark_frame(dates),
                benchmark_symbol="BENCH",
                factor_jumps_pre_audited=True,
            )

    def test_declared_factor_jump_without_matching_event_fails_closed(self):
        dates = calendar(2)
        raw = bars(["A"], {"A": [10.0, 5.0]}, dates)
        raw["adjustment_factor"] = [1.0, 2.0]
        with self.assertRaisesRegex(ValueError, "factor jump.*corporate-action"):
            run_raw_backtest(
                scores([(dates[0], "A", 1.0)]),
                raw,
                no_actions(),
                dates,
                RawBacktestConfig(topk=1, n_drop=1),
                benchmark_close=benchmark_frame(dates).assign(adjustment_factor=1.0),
                benchmark_symbol="BENCH",
            )

    def test_factor_audit_cannot_be_silently_skipped(self):
        dates = calendar(2)
        raw = bars(["A"], {"A": [10.0, 10.0]}, dates)
        with self.assertRaisesRegex(ValueError, "adjustment_factor.*pre-audited"):
            run_raw_backtest(
                scores([(dates[0], "A", 1.0)]),
                raw,
                no_actions(),
                dates,
                RawBacktestConfig(topk=1, n_drop=1),
                benchmark_close=benchmark_frame(dates),
                benchmark_symbol="BENCH",
            )


class SuspensionCarryForwardTests(unittest.TestCase):
    def test_held_position_without_a_bar_carries_forward_and_freezes_trading(self):
        dates = calendar(4)
        raw = bars(["A", "B"], {"A": [10.0, 10.0, 10.0, 10.0], "B": [8.0, 8.0, 8.0, 8.0]}, dates)
        held_missing_date = dates[2]
        raw = raw[~((raw["symbol"] == "A") & (raw["date"] == held_missing_date))]
        predictions = scores(
            [(dates[0], "A", 2.0), (dates[0], "B", 1.0), (dates[1], "A", -2.0)]
        )
        result = run_raw_backtest(
            predictions,
            raw,
            no_actions(),
            dates,
            RawBacktestConfig(topk=1, n_drop=1, hold_thresh=5, slippage_bps_per_side=0),
            benchmark_close=benchmark_frame(dates),
            benchmark_symbol="BENCH",
            factor_jumps_pre_audited=True,
        )
        events = result.summary["missing_market_data_carry_forward_events"]
        self.assertEqual(result.summary["missing_market_data_carry_forward_count"], 1)
        self.assertEqual(events[0]["symbol"], "A")
        self.assertEqual(events[0]["date"], held_missing_date)
        self.assertEqual(events[0]["price"], 10.0)
        sell = result.executions[
            (result.executions["symbol"] == "A")
            & (result.executions["execution_date"] == held_missing_date)
        ]
        self.assertEqual(len(sell), 1)
        self.assertEqual(sell.iloc[0]["reason"], "missing_market_data")
        self.assertEqual(sell.iloc[0]["fill_shares"], 0)
        self.assertTrue(np.isfinite(result.report["account"].to_numpy()).all())
        self.assertTrue(result.summary["nav_reconciled"])


if __name__ == "__main__":
    unittest.main()
