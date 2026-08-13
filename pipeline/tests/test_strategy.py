import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.signal import Signal
from quant_pipeline.strategy import AuditedTopkDropoutStrategy, OrderIntent


class StubSignal(Signal):
    def __init__(self, scores):
        self.scores = pd.Series(scores, dtype=float)

    def get_signal(self, start_time, end_time):
        del start_time, end_time
        return self.scores


class StubCalendar:
    timestamp = pd.Timestamp("2026-08-10")

    def get_trade_step(self):
        return 0

    def get_step_time(self, trade_step=None, shift=0):
        del trade_step
        timestamp = self.timestamp - pd.Timedelta(days=shift)
        return timestamp, timestamp

    def get_freq(self):
        return "day"


class StubPosition:
    def __init__(self, cash=20_000.0, holdings=None, counts=None):
        self.cash = float(cash)
        self.holdings = dict(holdings or {})
        self.counts = dict(counts or {})

    def __deepcopy__(self, memo):
        del memo
        return StubPosition(self.cash, copy.deepcopy(self.holdings), copy.deepcopy(self.counts))

    def get_cash(self):
        return self.cash

    def get_stock_list(self):
        return list(self.holdings)

    def get_stock_count(self, code, bar):
        del bar
        return self.counts.get(code, 99)

    def get_stock_amount(self, code):
        return self.holdings.get(code, 0.0)

    def check_stock(self, stock_id):
        return stock_id in self.holdings

    def update_order(self, order, trade_val, cost, trade_price):
        del trade_price
        if order.direction == Order.SELL:
            self.holdings[order.stock_id] -= order.deal_amount
            self.cash += trade_val - cost


class StubAccount:
    def __init__(self, position):
        self.current_position = position


class StubExchange:
    def __init__(self, *, blocked=None, suspended=None, prices=None, factors=None):
        self.blocked = set(blocked or ())
        self.suspended = set(suspended or ())
        self.prices = dict(prices or {})
        self.factors = dict(factors or {})
        self.tradability_calls = []
        self.executed_orders = []

    def is_stock_tradable(self, stock_id, start_time, end_time, direction=None):
        del start_time, end_time
        self.tradability_calls.append((stock_id, direction))
        return stock_id not in self.blocked and stock_id not in self.suspended

    def check_stock_suspended(self, stock_id, start_time, end_time):
        del start_time, end_time
        return stock_id in self.suspended

    def get_deal_price(self, stock_id, start_time, end_time, direction):
        del start_time, end_time
        self.last_price_direction = direction
        return self.prices.get(stock_id, float("nan"))

    def get_factor(self, stock_id, start_time, end_time):
        del start_time, end_time
        return self.factors.get(stock_id, 1.0)

    def round_amount_by_trade_unit(self, amount, factor):
        return (amount * factor // 100) * 100 / factor

    def deal_order(self, order, position=None, trade_account=None):
        self.executed_orders.append(order)
        if order.stock_id in self.blocked or order.stock_id in self.suspended:
            order.deal_amount = 0.0
            return 0.0, 0.0, float("nan")
        price = self.prices[order.stock_id]
        order.deal_amount = order.amount
        trade_value = order.amount * price
        if position is not None:
            position.update_order(order, trade_value, 0.0, price)
        return trade_value, 0.0, price


def make_strategy(scores, exchange, position=None, **kwargs):
    position = position or StubPosition()
    return AuditedTopkDropoutStrategy(
        signal=StubSignal(scores),
        topk=kwargs.pop("topk", 1),
        n_drop=kwargs.pop("n_drop", 1),
        risk_degree=kwargs.pop("risk_degree", 0.9),
        trade_exchange=exchange,
        level_infra={"trade_calendar": StubCalendar()},
        common_infra={"trade_account": StubAccount(position)},
        **kwargs,
    )


class AuditedTopkDropoutStrategyTests(unittest.TestCase):
    def test_price_limited_top_rank_is_not_replaced_and_reaches_executor(self):
        exchange = StubExchange(
            blocked={"LIMITED"},
            prices={"LIMITED": 10.0, "NEXT": 10.0},
        )
        strategy = make_strategy({"LIMITED": 2.0, "NEXT": 1.0}, exchange)

        orders = strategy.generate_trade_decision().get_decision()

        self.assertEqual([order.stock_id for order in orders], ["LIMITED"])
        self.assertEqual(orders[0].direction, OrderDir.BUY)
        self.assertGreater(orders[0].amount, 0.0)
        self.assertEqual(exchange.tradability_calls, [("LIMITED", OrderDir.BUY)])
        self.assertEqual(strategy.pretrade_rejections, ())

        exchange.deal_order(orders[0], trade_account=StubAccount(StubPosition()))
        self.assertEqual(orders[0].deal_amount, 0.0)
        self.assertEqual(exchange.executed_orders[-1].stock_id, "LIMITED")

    def test_suspended_top_rank_is_audited_without_fallback(self):
        sink = []
        exchange = StubExchange(
            suspended={"SUSPENDED"},
            prices={"NEXT": 10.0},
        )
        strategy = make_strategy(
            {"SUSPENDED": 2.0, "NEXT": 1.0},
            exchange,
            pretrade_rejection_sink=sink,
        )

        orders = strategy.generate_trade_decision().get_decision()

        self.assertEqual(orders, [])
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0]["stock_id"], "SUSPENDED")
        self.assertEqual(sink[0]["direction"], "buy")
        self.assertEqual(sink[0]["reason"], "suspended")
        self.assertIsNone(sink[0]["intended_amount"])
        self.assertNotIn("NEXT", [stock_id for stock_id, _ in exchange.tradability_calls])

    def test_missing_price_is_audited_without_fallback(self):
        exchange = StubExchange(prices={"NEXT": 10.0})
        strategy = make_strategy({"NO_PRICE": 2.0, "NEXT": 1.0}, exchange)

        orders = strategy.generate_trade_decision().get_decision()

        self.assertEqual(orders, [])
        rejection = strategy.pretrade_rejections[0]
        self.assertEqual(rejection["stock_id"], "NO_PRICE")
        self.assertEqual(rejection["reason"], "missing_or_invalid_price")
        self.assertIsNone(rejection["intended_amount"])
        self.assertNotIn("NEXT", [stock_id for stock_id, _ in exchange.tradability_calls])

    def test_sell_and_buy_checks_are_directional_and_locked_sell_is_submitted(self):
        position = StubPosition(cash=2_000.0, holdings={"HELD": 100.0})
        exchange = StubExchange(
            blocked={"HELD"},
            prices={"HELD": 8.0, "NEW": 10.0},
        )
        strategy = make_strategy({"NEW": 2.0, "HELD": 1.0}, exchange, position=position)

        orders = strategy.generate_trade_decision().get_decision()

        self.assertEqual(
            [(order.stock_id, order.direction) for order in orders],
            [("HELD", OrderDir.SELL), ("NEW", OrderDir.BUY)],
        )
        self.assertEqual(orders[0].amount, 100.0)
        self.assertEqual(
            exchange.tradability_calls,
            [("HELD", OrderDir.SELL), ("NEW", OrderDir.BUY)],
        )
        self.assertNotIn(orders[0], exchange.executed_orders)

    def test_optional_corporate_action_hook_can_adjust_amount(self):
        observed = []

        def halve_amount(intent: OrderIntent):
            observed.append(intent)
            return replace(intent, amount=intent.amount / 2.0)

        exchange = StubExchange(prices={"TOP": 10.0})
        strategy = make_strategy(
            {"TOP": 1.0},
            exchange,
            risk_degree=1.0,
            corporate_action_hook=halve_amount,
        )

        order = strategy.generate_trade_decision().get_decision()[0]

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].direction, OrderDir.BUY)
        self.assertEqual(order.amount, 1_000.0)

    def test_execution_day_filtering_flags_are_rejected(self):
        exchange = StubExchange(prices={"TOP": 10.0})
        with self.assertRaisesRegex(ValueError, "only_tradable=False"):
            make_strategy({"TOP": 1.0}, exchange, only_tradable=True)
        with self.assertRaisesRegex(ValueError, "forbid_all_trade_at_limit=False"):
            make_strategy({"TOP": 1.0}, exchange, forbid_all_trade_at_limit=True)

    def test_zero_dropout_does_not_sell_holdings(self):
        position = StubPosition(cash=2_000.0, holdings={"HELD": 100.0})
        exchange = StubExchange(prices={"HELD": 8.0, "NEW": 10.0})
        strategy = make_strategy(
            {"NEW": 2.0, "HELD": 1.0},
            exchange,
            position=position,
            n_drop=0,
        )

        self.assertEqual(strategy.generate_trade_decision().get_decision(), [])
        self.assertEqual(exchange.tradability_calls, [])


if __name__ == "__main__":
    unittest.main()
