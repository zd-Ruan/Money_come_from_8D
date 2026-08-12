import logging
import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlib.backtest.decision import Order
from qlib.backtest.report import Indicator
from quant_pipeline.small_account import (
    QLIB_BACKTEST_API,
    LinearCostModel,
    SmallAccountExchange,
    records_from_qlib_execute_results,
    records_from_qlib_order_indicators,
    summarize_execution_records,
    validate_qlib_backtest_api,
)


class FakePosition:
    def __init__(self, cash, holdings=None):
        self.cash = float(cash)
        self.holdings = dict(holdings or {})

    def get_cash(self):
        return self.cash

    def check_stock(self, stock_id):
        return stock_id in self.holdings

    def get_stock_amount(self, stock_id):
        return self.holdings[stock_id]

    def update_order(self, order, trade_val, cost, trade_price):
        del order, trade_val, cost, trade_price


class FakeAccount:
    def __init__(self, cash, holdings=None):
        self.current_position = FakePosition(cash, holdings)

    def update_order(self, order, trade_val, cost, trade_price):
        self.current_position.update_order(order, trade_val, cost, trade_price)


class StubSmallAccountExchange(SmallAccountExchange):
    """Exercise the installed Qlib algorithm without loading market data."""

    def __init__(self, commission_rate=0.0003, min_commission=5.0, slippage_bps=5.0):
        self.cost_model = LinearCostModel(commission_rate, min_commission, slippage_bps)
        self._execution_records = []
        self.open_cost = self.close_cost = self.cost_model.commission_rate
        self.min_cost = self.cost_model.min_commission
        self.impact_cost = 0.0
        self.trade_unit = 100
        self.trade_w_adj_price = False
        self.buy_vol_limit = self.sell_vol_limit = None
        self.logger = logging.getLogger("small-account-test")
        self.tradable = True
        self.price = 10.0
        self.factor = 1.0
        self.volume = 1_000_000.0

    def check_order(self, order):
        del order
        return self.tradable

    def get_deal_price(self, stock_id, start_time, end_time, direction, method="ts_data_last"):
        del stock_id, start_time, end_time, direction, method
        return self.price

    def get_factor(self, stock_id, start_time, end_time):
        del stock_id, start_time, end_time
        return self.factor

    def get_volume(self, stock_id, start_time, end_time, method="sum"):
        del stock_id, start_time, end_time, method
        return self.volume

    def _clip_amount_by_volume(self, order, dealt_order_amount):
        del order, dealt_order_amount


def make_order(amount=100.0, stock_id="SH510300", direction=Order.BUY):
    timestamp = pd.Timestamp("2026-08-10")
    return Order(stock_id, amount, direction, timestamp, timestamp)


class QlibCompatibilityTests(unittest.TestCase):
    def test_installed_api_contract_is_supported(self):
        result = validate_qlib_backtest_api()
        self.assertTrue(result["compatible"])
        self.assertEqual(result["qlib_version"], QLIB_BACKTEST_API["qlib_version"])
        self.assertIn("trade_account", result["signatures"]["Exchange.deal_order"])


class CostModelTests(unittest.TestCase):
    def test_minimum_commission_does_not_absorb_linear_slippage(self):
        model = LinearCostModel(commission_rate=0.0003, min_commission=5.0, slippage_bps=5.0)
        small = model.calculate(1_000.0)
        self.assertEqual(small.commission, 5.0)
        self.assertEqual(small.slippage, 0.5)
        self.assertEqual(small.total, 5.5)
        self.assertTrue(small.minimum_commission_applied)

        large = model.calculate(100_000.0)
        self.assertAlmostEqual(large.commission, 30.0)
        self.assertAlmostEqual(large.slippage, 50.0)
        self.assertAlmostEqual(large.total, 80.0)
        self.assertFalse(large.minimum_commission_applied)

    def test_affordable_notional_reserves_both_cost_components(self):
        model = LinearCostModel(commission_rate=0.0003, min_commission=5.0, slippage_bps=5.0)
        maximum = model.max_affordable_notional(2_005.50)
        self.assertLess(maximum, 2_000.0)
        costs = model.calculate(maximum)
        self.assertAlmostEqual(maximum + costs.total, 2_005.50)


class SmallAccountExchangeTests(unittest.TestCase):
    def test_round_lot_and_independent_cost_are_applied(self):
        exchange = StubSmallAccountExchange()
        order = make_order(amount=250.0)
        trade_value, trade_cost, trade_price = exchange.deal_order(
            order,
            trade_account=FakeAccount(10_000.0),
        )
        self.assertEqual(order.deal_amount, 200.0)
        self.assertEqual(trade_price, 10.0)
        self.assertEqual(trade_value, 2_000.0)
        self.assertEqual(trade_cost, 6.0)
        self.assertEqual(exchange.execution_records[0].commission, 5.0)
        self.assertEqual(exchange.execution_records[0].slippage, 1.0)

    def test_cash_boundary_cannot_spend_the_slippage_reserve(self):
        exchange = StubSmallAccountExchange()
        order = make_order(amount=200.0)
        trade_value, trade_cost, _ = exchange.deal_order(
            order,
            position=FakePosition(2_005.50),
        )
        self.assertEqual(order.deal_amount, 100.0)
        self.assertEqual(trade_value, 1_000.0)
        self.assertEqual(trade_cost, 5.5)

        exchange = StubSmallAccountExchange()
        order = make_order(amount=200.0)
        trade_value, trade_cost, _ = exchange.deal_order(
            order,
            position=FakePosition(2_006.00),
        )
        self.assertEqual(order.deal_amount, 200.0)
        self.assertEqual(trade_value + trade_cost, 2_006.0)

    def test_round_lot_fill_never_exceeds_available_cash(self):
        for cash in [0.0, 5.0, 1_005.49, 1_005.50, 2_005.99, 2_006.00, 50_000.0]:
            with self.subTest(cash=cash):
                exchange = StubSmallAccountExchange()
                order = make_order(amount=10_000.0)
                trade_value, trade_cost, _ = exchange.deal_order(
                    order,
                    position=FakePosition(cash),
                )
                self.assertLessEqual(trade_value + trade_cost, cash + 1e-8)
                self.assertEqual(order.deal_amount % 100, 0.0)

    def test_zero_fill_is_recorded_with_target_notional(self):
        exchange = StubSmallAccountExchange()
        exchange.tradable = False
        order = make_order(amount=100.0)
        exchange.deal_order(order, trade_account=FakeAccount(10_000.0))
        record = exchange.execution_records[0]
        self.assertEqual(record.target_notional, 1_000.0)
        self.assertEqual(record.fill_notional, 0.0)
        self.assertEqual(record.total_cost, 0.0)


class ExecutionStatisticsTests(unittest.TestCase):
    def test_executor_tuples_are_normalized_and_old_cost_semantics_are_rejected(self):
        model = LinearCostModel(0.0003, 5.0, 5.0)
        order = make_order()
        order.deal_amount = 100.0
        records = records_from_qlib_execute_results([(order, 1_000.0, 5.5, 10.0)], model)
        self.assertEqual(records[0].commission, 5.0)
        self.assertEqual(records[0].slippage, 0.5)

        with self.assertRaisesRegex(ValueError, "does not match"):
            records_from_qlib_execute_results([(order, 1_000.0, 5.0, 10.0)], model)

    def test_indicator_history_and_summary_include_fill_and_cost_quality(self):
        model = LinearCostModel(0.0003, 5.0, 5.0)
        history = {
            pd.Timestamp("2026-08-10"): {
                "amount": pd.Series({"SH510300": 100.0, "SH510500": 100.0}),
                "deal_amount": pd.Series({"SH510300": 100.0, "SH510500": 0.0}),
                "trade_price": pd.Series({"SH510300": 10.0, "SH510500": float("nan")}),
                "trade_value": pd.Series({"SH510300": 1_000.0, "SH510500": 0.0}),
                "trade_cost": pd.Series({"SH510300": 5.5, "SH510500": 0.0}),
                "trade_dir": pd.Series({"SH510300": Order.BUY, "SH510500": Order.BUY}),
            }
        }
        records = records_from_qlib_order_indicators(
            history,
            model,
            target_price_resolver=lambda timestamp, stock_id, direction, observed: observed or 10.0,
        )
        summary = summarize_execution_records(records)
        self.assertEqual(summary["target_notional"], 2_000.0)
        self.assertEqual(summary["fill_notional"], 1_000.0)
        self.assertEqual(summary["fill_rate"], 0.5)
        self.assertEqual(summary["zero_fill_order_rate"], 0.5)
        self.assertEqual(summary["commission_total"], 5.0)
        self.assertEqual(summary["slippage_total"], 0.5)
        self.assertEqual(summary["slippage_effective_bps"], 5.0)
        self.assertEqual(summary["minimum_commission_order_rate"], 1.0)

    def test_native_qlib_indicator_history_is_supported(self):
        model = LinearCostModel(0.0003, 5.0, 5.0)
        filled = make_order(amount=100.0, stock_id="SH510300")
        filled.deal_amount = 100.0
        rejected = make_order(amount=100.0, stock_id="SH510500")
        indicator = Indicator()
        indicator.update_order_indicators(
            [
                (filled, 1_000.0, 5.5, 10.0),
                (rejected, 0.0, 0.0, float("nan")),
            ]
        )
        indicator.record(pd.Timestamp("2026-08-10"))

        records = records_from_qlib_order_indicators(
            indicator,
            model,
            target_price_resolver=lambda timestamp, stock_id, direction, observed: observed or 10.0,
        )
        summary = summarize_execution_records(records)
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["fill_rate"], 0.5)
        self.assertEqual(summary["zero_fill_order_count"], 1)


if __name__ == "__main__":
    unittest.main()
