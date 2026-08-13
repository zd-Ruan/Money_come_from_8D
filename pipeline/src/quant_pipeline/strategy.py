"""Auditable signal strategies for the ETF research pipeline."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, MutableSequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


@dataclass(frozen=True)
class OrderIntent:
    """A sized order selected by the signal before exchange execution.

    A corporate-action hook may adjust ``amount`` by returning a replacement
    instance.  Instrument, direction, and execution window are immutable
    identity fields and may not be changed by the hook.
    """

    stock_id: str
    direction: OrderDir
    amount: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    signal_score: float | None
    target_value: float | None = None


CorporateActionHook = Callable[[OrderIntent], OrderIntent | None]


def _positive_float(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class AuditedTopkDropoutStrategy(TopkDropoutStrategy):
    """Top-k dropout without execution-day substitution.

    Selection uses only the lagged signal and current holdings.  Tradability is
    inspected only after the buy and sell lists have been frozen.  A selected
    price-limited instrument therefore remains an order and reaches the
    executor, where the exchange records a zero fill.  A selected buy whose
    amount cannot be sized (for example, suspension or missing price/factor) is
    retained as a pre-trade rejection and is never replaced with a lower-ranked
    instrument.

    ``pretrade_rejection_sink`` is useful when Qlib constructs the strategy
    from a configuration dictionary: the caller keeps the same list and can
    read the accumulated JSON-compatible audit records after backtesting.
    """

    def __init__(
        self,
        *,
        topk: int,
        n_drop: int,
        method_sell: str = "bottom",
        method_buy: str = "top",
        hold_thresh: int = 1,
        only_tradable: bool = False,
        forbid_all_trade_at_limit: bool = False,
        corporate_action_hook: CorporateActionHook | None = None,
        pretrade_rejection_sink: MutableSequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if only_tradable:
            raise ValueError(
                "AuditedTopkDropoutStrategy requires only_tradable=False; "
                "execution-day tradability must not alter signal ranking"
            )
        if forbid_all_trade_at_limit:
            raise ValueError(
                "AuditedTopkDropoutStrategy requires forbid_all_trade_at_limit=False "
                "so buy and sell restrictions remain directional"
            )
        if corporate_action_hook is not None and not callable(corporate_action_hook):
            raise TypeError("corporate_action_hook must be callable or None")
        if pretrade_rejection_sink is not None and not hasattr(pretrade_rejection_sink, "append"):
            raise TypeError("pretrade_rejection_sink must support append")

        super().__init__(
            topk=topk,
            n_drop=n_drop,
            method_sell=method_sell,
            method_buy=method_buy,
            hold_thresh=hold_thresh,
            only_tradable=False,
            forbid_all_trade_at_limit=False,
            **kwargs,
        )
        self.corporate_action_hook = corporate_action_hook
        self._pretrade_rejections = (
            pretrade_rejection_sink if pretrade_rejection_sink is not None else []
        )

    @property
    def pretrade_rejections(self) -> tuple[dict[str, Any], ...]:
        """Return a snapshot of rejection records accumulated during this run."""

        return tuple(dict(record) for record in self._pretrade_rejections)

    def clear_pretrade_rejections(self) -> None:
        """Clear both the strategy view and a caller-provided audit sink."""

        self._pretrade_rejections.clear()

    @staticmethod
    def _direction_name(direction: OrderDir) -> str:
        return "buy" if direction == OrderDir.BUY else "sell"

    def _record_rejection(
        self,
        *,
        stock_id: str,
        direction: OrderDir,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        reason: str,
        signal_score: float | None,
        target_value: float | None = None,
        intended_amount: float | None = None,
        detail: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "stock_id": str(stock_id),
            "direction": self._direction_name(direction),
            "trade_start_time": pd.Timestamp(start_time).isoformat(),
            "trade_end_time": pd.Timestamp(end_time).isoformat(),
            "reason": reason,
            "signal_score": signal_score,
            "target_value": target_value,
            "intended_amount": intended_amount,
        }
        if detail is not None:
            record["detail"] = detail
        self._pretrade_rejections.append(record)

    def _directional_tradability(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        direction: OrderDir,
    ) -> tuple[bool, bool]:
        """Return directional tradability and suspension state.

        The first value is deliberately not used to filter a selected order.
        It controls only copied-position sell simulation.  This is what lets a
        price-limit rejection reach the real executor as a zero fill.
        """

        tradable = bool(
            self.trade_exchange.is_stock_tradable(
                stock_id=stock_id,
                start_time=start_time,
                end_time=end_time,
                direction=direction,
            )
        )
        suspended = False
        if not tradable:
            suspended = bool(
                self.trade_exchange.check_stock_suspended(
                    stock_id=stock_id,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        return tradable, suspended

    def _apply_corporate_action_hook(self, intent: OrderIntent) -> OrderIntent | None:
        if self.corporate_action_hook is None:
            return intent
        adjusted = self.corporate_action_hook(intent)
        if adjusted is None:
            return None
        if not isinstance(adjusted, OrderIntent):
            raise TypeError("corporate_action_hook must return OrderIntent or None")
        identity = (intent.stock_id, intent.direction, intent.start_time, intent.end_time)
        adjusted_identity = (
            adjusted.stock_id,
            adjusted.direction,
            adjusted.start_time,
            adjusted.end_time,
        )
        if adjusted_identity != identity:
            raise ValueError(
                "corporate_action_hook may adjust amount metadata but not order identity"
            )
        if _positive_float(adjusted.amount) is None:
            raise ValueError("corporate_action_hook must return a positive finite amount")
        return adjusted

    def _intent_to_order(self, intent: OrderIntent) -> Order:
        return Order(
            stock_id=intent.stock_id,
            amount=float(intent.amount),
            start_time=intent.start_time,
            end_time=intent.end_time,
            direction=intent.direction,
        )

    @staticmethod
    def _score_for(pred_score: pd.Series, stock_id: str) -> float | None:
        try:
            return _optional_float(pred_score.loc[stock_id])
        except KeyError:
            return None

    def _select(self, pred_score: pd.Series, current_stock_list: list[str]) -> tuple[list[str], list[str]]:
        """Freeze sell and buy candidates using signal data only."""

        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        candidate_count = max(0, self.n_drop + self.topk - len(last))

        if self.method_buy == "top":
            available = pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index
            today = list(available[:candidate_count])
        elif self.method_buy == "random":
            topk_candidates = list(pred_score.sort_values(ascending=False).index[: self.topk])
            candidates = [stock_id for stock_id in topk_candidates if stock_id not in last]
            count = min(candidate_count, len(candidates))
            today = list(np.random.choice(candidates, count, replace=False)) if count else []
        else:
            raise NotImplementedError(f"unsupported buy method: {self.method_buy}")

        combined = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        if self.method_sell == "bottom":
            bottom = set(combined[-self.n_drop :]) if self.n_drop > 0 else set()
            sell = [stock_id for stock_id in last if stock_id in bottom]
        elif self.method_sell == "random":
            count = min(self.n_drop, len(last))
            sell = list(np.random.choice(list(last), count, replace=False)) if count else []
        else:
            raise NotImplementedError(f"unsupported sell method: {self.method_sell}")

        buy_count = max(0, len(sell) + self.topk - len(last))
        return sell, today[:buy_count]

    def generate_trade_decision(self, execute_result: list | None = None) -> TradeDecisionWO:
        del execute_result
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None or pred_score.empty:
            return TradeDecisionWO([], self)

        current_temp: Position = copy.deepcopy(self.trade_position)
        current_stock_list = list(current_temp.get_stock_list())
        sell, buy = self._select(pred_score, current_stock_list)
        cash = float(current_temp.get_cash())
        sell_orders: list[Order] = []
        buy_orders: list[Order] = []

        for stock_id in sell:
            stock_id = str(stock_id)
            if current_temp.get_stock_count(stock_id, bar=self.trade_calendar.get_freq()) < self.hold_thresh:
                continue
            amount = _positive_float(current_temp.get_stock_amount(code=stock_id))
            score = self._score_for(pred_score, stock_id)
            if amount is None:
                self._record_rejection(
                    stock_id=stock_id,
                    direction=OrderDir.SELL,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    reason="missing_or_invalid_position_amount",
                    signal_score=score,
                )
                continue

            tradable, _ = self._directional_tradability(
                stock_id, trade_start_time, trade_end_time, OrderDir.SELL
            )
            intent = OrderIntent(
                stock_id=stock_id,
                direction=OrderDir.SELL,
                amount=amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                signal_score=score,
            )
            adjusted = self._apply_corporate_action_hook(intent)
            if adjusted is None:
                self._record_rejection(
                    stock_id=stock_id,
                    direction=OrderDir.SELL,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    reason="corporate_action_hook_rejected",
                    signal_score=score,
                    intended_amount=amount,
                )
                continue

            sell_order = self._intent_to_order(adjusted)
            sell_orders.append(sell_order)
            if tradable:
                trade_value, trade_cost, _ = self.trade_exchange.deal_order(
                    sell_order, position=current_temp
                )
                cash += float(trade_value) - float(trade_cost)

        target_value = cash * self.get_risk_degree(trade_step) / len(buy) if buy else 0.0
        for stock_id in buy:
            stock_id = str(stock_id)
            score = self._score_for(pred_score, stock_id)
            _, suspended = self._directional_tradability(
                stock_id, trade_start_time, trade_end_time, OrderDir.BUY
            )
            if suspended:
                self._record_rejection(
                    stock_id=stock_id,
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    reason="suspended",
                    signal_score=score,
                    target_value=target_value,
                )
                continue

            try:
                price_value = self.trade_exchange.get_deal_price(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )
            except (KeyError, TypeError, ValueError, IndexError):
                price_value = None
            price = _positive_float(price_value)
            if price is None:
                self._record_rejection(
                    stock_id=stock_id,
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    reason="missing_or_invalid_price",
                    signal_score=score,
                    target_value=target_value,
                )
                continue

            try:
                factor_value = self.trade_exchange.get_factor(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except (KeyError, TypeError, ValueError, IndexError):
                factor_value = None
            factor = _positive_float(factor_value)
            if factor is None:
                self._record_rejection(
                    stock_id=stock_id,
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    reason="missing_or_invalid_factor",
                    signal_score=score,
                    target_value=target_value,
                )
                continue

            raw_amount = target_value / price
            amount = _positive_float(
                self.trade_exchange.round_amount_by_trade_unit(raw_amount, factor)
            )
            if amount is None:
                self._record_rejection(
                    stock_id=stock_id,
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    reason="below_trade_unit_or_non_positive_amount",
                    signal_score=score,
                    target_value=target_value,
                    intended_amount=_positive_float(raw_amount),
                )
                continue

            intent = OrderIntent(
                stock_id=stock_id,
                direction=OrderDir.BUY,
                amount=amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                signal_score=score,
                target_value=target_value,
            )
            adjusted = self._apply_corporate_action_hook(intent)
            if adjusted is None:
                self._record_rejection(
                    stock_id=stock_id,
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    reason="corporate_action_hook_rejected",
                    signal_score=score,
                    target_value=target_value,
                    intended_amount=amount,
                )
                continue
            buy_orders.append(self._intent_to_order(adjusted))

        return TradeDecisionWO(sell_orders + buy_orders, self)


__all__ = [
    "AuditedTopkDropoutStrategy",
    "CorporateActionHook",
    "OrderIntent",
]
