from __future__ import annotations

import inspect
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import qlib
from qlib.backtest.decision import Order
from qlib.backtest.exchange import Exchange


ROUND_LOT = 100
_COST_TOLERANCE = 1e-8


def validate_qlib_backtest_api() -> dict[str, Any]:
    """Fail closed if the installed Qlib exchange contract is incompatible."""
    required_signatures = {
        "Exchange.__init__": (
            Exchange.__init__,
            {"open_cost", "close_cost", "min_cost", "impact_cost"},
        ),
        "Exchange.deal_order": (
            Exchange.deal_order,
            {"order", "trade_account", "position", "dealt_order_amount"},
        ),
        "Exchange._calc_trade_info_by_order": (
            Exchange._calc_trade_info_by_order,
            {"order", "position", "dealt_order_amount"},
        ),
    }
    signatures: dict[str, str] = {}
    problems: list[str] = []
    for name, (callable_object, required_parameters) in required_signatures.items():
        signature = inspect.signature(callable_object)
        signatures[name] = str(signature)
        missing = required_parameters - set(signature.parameters)
        if missing:
            problems.append(f"{name} is missing parameters: {sorted(missing)}")

    required_methods = {
        "check_order",
        "get_deal_price",
        "get_factor",
        "get_volume",
        "round_amount_by_trade_unit",
    }
    missing_methods = sorted(name for name in required_methods if not callable(getattr(Exchange, name, None)))
    if missing_methods:
        problems.append(f"Exchange is missing methods: {missing_methods}")

    order_fields = set(getattr(Order, "__dataclass_fields__", {}))
    missing_order_fields = {"stock_id", "amount", "direction", "start_time", "end_time"} - order_fields
    if missing_order_fields:
        problems.append(f"Order is missing fields: {sorted(missing_order_fields)}")
    if not hasattr(Order, "deal_amount") or not hasattr(Order, "deal_amount_delta"):
        problems.append("Order does not expose deal_amount and deal_amount_delta")

    if problems:
        raise RuntimeError("incompatible Qlib backtest API: " + "; ".join(problems))
    return {
        "compatible": True,
        "qlib_version": str(getattr(qlib, "__version__", "unknown")),
        "signatures": signatures,
    }


QLIB_BACKTEST_API = validate_qlib_backtest_api()


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _optional_positive_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _finite_absolute(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        number = abs(float(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    return _finite_nonnegative(name, number)


@dataclass(frozen=True)
class CostBreakdown:
    notional: float
    commission: float
    slippage: float
    total: float
    minimum_commission_applied: bool


@dataclass(frozen=True)
class LinearCostModel:
    """Commission with an absolute floor plus independent linear slippage."""

    commission_rate: float
    min_commission: float = 5.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "commission_rate",
            _finite_nonnegative("commission_rate", self.commission_rate),
        )
        object.__setattr__(
            self,
            "min_commission",
            _finite_nonnegative("min_commission", self.min_commission),
        )
        object.__setattr__(
            self,
            "slippage_bps",
            _finite_nonnegative("slippage_bps", self.slippage_bps),
        )

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    def calculate(self, notional: float) -> CostBreakdown:
        notional = _finite_nonnegative("notional", notional)
        if notional == 0:
            return CostBreakdown(0.0, 0.0, 0.0, 0.0, False)
        proportional_commission = self.commission_rate * notional
        commission = max(proportional_commission, self.min_commission)
        slippage = self.slippage_rate * notional
        return CostBreakdown(
            notional=notional,
            commission=commission,
            slippage=slippage,
            total=commission + slippage,
            minimum_commission_applied=proportional_commission <= self.min_commission,
        )

    def max_affordable_notional(self, cash: float) -> float:
        """Solve notional + commission + slippage <= cash exactly."""
        cash = _finite_nonnegative("cash", cash)
        fixed_floor_bound = (cash - self.min_commission) / (1.0 + self.slippage_rate)
        proportional_bound = cash / (1.0 + self.commission_rate + self.slippage_rate)
        return max(0.0, min(fixed_floor_bound, proportional_bound))


@dataclass(frozen=True)
class ExecutionRecord:
    timestamp: Any
    stock_id: str
    direction: str
    target_amount: float
    fill_amount: float
    target_notional: float | None
    fill_notional: float
    commission: float
    slippage: float
    total_cost: float
    minimum_commission_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stock_id, str) or not self.stock_id:
            raise ValueError("stock_id must be a non-empty string")
        if self.direction not in {"buy", "sell"}:
            raise ValueError("direction must be 'buy' or 'sell'")
        for name in (
            "target_amount",
            "fill_amount",
            "fill_notional",
            "commission",
            "slippage",
            "total_cost",
        ):
            object.__setattr__(self, name, _finite_nonnegative(name, getattr(self, name)))
        if self.target_notional is not None:
            object.__setattr__(
                self,
                "target_notional",
                _finite_nonnegative("target_notional", self.target_notional),
            )
        if self.fill_amount > self.target_amount + _COST_TOLERANCE:
            raise ValueError("fill_amount cannot exceed target_amount")
        if self.fill_amount <= _COST_TOLERANCE and (
            self.fill_notional > _COST_TOLERANCE or self.total_cost > _COST_TOLERANCE
        ):
            raise ValueError("a zero-fill order cannot have fill notional or costs")
        if not math.isclose(
            self.total_cost,
            self.commission + self.slippage,
            rel_tol=_COST_TOLERANCE,
            abs_tol=_COST_TOLERANCE,
        ):
            raise ValueError("total_cost must equal commission plus slippage")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _direction_name(direction: Any) -> str:
    try:
        numeric = int(direction)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported Qlib order direction: {direction!r}") from exc
    if numeric == int(Order.BUY):
        return "buy"
    if numeric == int(Order.SELL):
        return "sell"
    raise ValueError(f"unsupported Qlib order direction: {direction!r}")


def _record_from_values(
    *,
    timestamp: Any,
    stock_id: str,
    direction: Any,
    target_amount: Any,
    fill_amount: Any,
    target_price: Any,
    fill_notional: Any,
    reported_total_cost: Any,
    cost_model: LinearCostModel,
    validate_reported_cost: bool,
) -> ExecutionRecord:
    target_amount = _finite_absolute("target_amount", target_amount)
    fill_amount = _finite_absolute("fill_amount", fill_amount)
    fill_notional = _finite_absolute("fill_notional", fill_notional)
    target_price = _optional_positive_price(target_price)
    target_notional = None if target_price is None else target_amount * target_price
    costs = cost_model.calculate(fill_notional)
    reported_total_cost = _finite_nonnegative("reported_total_cost", reported_total_cost)
    if validate_reported_cost and not math.isclose(
        reported_total_cost,
        costs.total,
        rel_tol=_COST_TOLERANCE,
        abs_tol=_COST_TOLERANCE,
    ):
        raise ValueError(
            f"reported Qlib cost for {stock_id} ({reported_total_cost}) does not match "
            f"independent commission plus slippage ({costs.total})"
        )
    return ExecutionRecord(
        timestamp=timestamp,
        stock_id=stock_id,
        direction=_direction_name(direction),
        target_amount=target_amount,
        fill_amount=fill_amount,
        target_notional=target_notional,
        fill_notional=fill_notional,
        commission=costs.commission,
        slippage=costs.slippage,
        total_cost=costs.total,
        minimum_commission_applied=costs.minimum_commission_applied,
    )


def records_from_qlib_execute_results(
    execute_results: Iterable[tuple[Any, float, float, float]],
    cost_model: LinearCostModel,
    *,
    target_price_resolver: Callable[[Any, float], float | None] | None = None,
    validate_reported_cost: bool = True,
) -> tuple[ExecutionRecord, ...]:
    """Normalize SimulatorExecutor ``(Order, value, cost, price)`` tuples."""
    records: list[ExecutionRecord] = []
    for result in execute_results:
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError("each Qlib execute result must contain order, value, cost, and price")
        order, trade_value, trade_cost, trade_price = result
        for attribute in ("stock_id", "amount", "deal_amount", "direction", "start_time"):
            if not hasattr(order, attribute):
                raise TypeError(f"Qlib order is missing {attribute!r}")
        target_price = (
            target_price_resolver(order, trade_price)
            if target_price_resolver is not None
            else trade_price
        )
        records.append(
            _record_from_values(
                timestamp=order.start_time,
                stock_id=str(order.stock_id),
                direction=order.direction,
                target_amount=order.amount,
                fill_amount=order.deal_amount,
                target_price=target_price,
                fill_notional=trade_value,
                reported_total_cost=trade_cost,
                cost_model=cost_model,
                validate_reported_cost=validate_reported_cost,
            )
        )
    return tuple(records)


def _indicator_series(indicator: Any) -> Mapping[str, Any]:
    if isinstance(indicator, Mapping):
        return indicator
    data = getattr(indicator, "data", None)
    if isinstance(data, Mapping):
        converted: dict[str, Any] = {}
        for name, metric in data.items():
            to_dict = getattr(metric, "to_dict", None)
            converted[name] = to_dict() if callable(to_dict) else metric
        return converted
    if callable(getattr(indicator, "to_series", None)):
        result = indicator.to_series()
        if isinstance(result, Mapping):
            return result
    raise TypeError("order indicator must be a metric mapping or expose to_series()")


def _metric_value(metric: Any, stock_id: Any, default: Any = None) -> Any:
    if metric is None:
        return default
    if isinstance(metric, Mapping):
        return metric.get(stock_id, default)
    getter = getattr(metric, "get", None)
    return getter(stock_id, default) if callable(getter) else default


def records_from_qlib_order_indicators(
    indicator_or_history: Any,
    cost_model: LinearCostModel,
    *,
    target_price_resolver: Callable[[Any, str, str, float | None], float | None] | None = None,
    validate_reported_cost: bool = True,
) -> tuple[ExecutionRecord, ...]:
    """Normalize Qlib ``Indicator.order_indicator_his`` records."""
    history = getattr(indicator_or_history, "order_indicator_his", indicator_or_history)
    if not isinstance(history, Mapping):
        raise TypeError("indicator history must be a timestamp-to-indicator mapping")

    records: list[ExecutionRecord] = []
    for timestamp, raw_indicator in history.items():
        metrics = _indicator_series(raw_indicator)
        missing = {"amount", "deal_amount", "trade_value", "trade_cost"} - set(metrics)
        if missing:
            raise ValueError(f"Qlib order indicator is missing metrics: {sorted(missing)}")
        amount_metric = metrics["amount"]
        index = getattr(amount_metric, "index", None)
        stock_ids = list(index if index is not None else amount_metric.keys())
        for stock_id in stock_ids:
            signed_target = float(_metric_value(amount_metric, stock_id, 0.0))
            signed_fill = float(_metric_value(metrics["deal_amount"], stock_id, 0.0))
            direction_value = _metric_value(metrics.get("trade_dir"), stock_id)
            if direction_value is None:
                direction_value = Order.BUY if signed_target > 0 else Order.SELL
            direction_name = _direction_name(direction_value)
            observed_price = _optional_positive_price(
                _metric_value(metrics.get("trade_price"), stock_id)
            )
            target_price = (
                target_price_resolver(timestamp, str(stock_id), direction_name, observed_price)
                if target_price_resolver is not None
                else observed_price
            )
            records.append(
                _record_from_values(
                    timestamp=timestamp,
                    stock_id=str(stock_id),
                    direction=direction_value,
                    target_amount=abs(signed_target),
                    fill_amount=abs(signed_fill),
                    target_price=target_price,
                    fill_notional=abs(float(_metric_value(metrics["trade_value"], stock_id, 0.0))),
                    reported_total_cost=_metric_value(metrics["trade_cost"], stock_id, 0.0),
                    cost_model=cost_model,
                    validate_reported_cost=validate_reported_cost,
                )
            )
    return tuple(records)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def summarize_execution_records(records: Iterable[ExecutionRecord]) -> dict[str, Any]:
    """Aggregate execution quality; ``fill_rate`` is notional-weighted when fully priced."""
    records = tuple(records)
    if any(not isinstance(record, ExecutionRecord) for record in records):
        raise TypeError("records must contain ExecutionRecord instances")
    active = tuple(record for record in records if record.target_amount > _COST_TOLERANCE)
    filled = tuple(record for record in active if record.fill_amount > _COST_TOLERANCE)
    zero_fills = tuple(record for record in active if record.fill_amount <= _COST_TOLERANCE)
    partial_fills = tuple(
        record
        for record in active
        if _COST_TOLERANCE < record.fill_amount < record.target_amount - _COST_TOLERANCE
    )
    unpriced = tuple(record for record in active if record.target_notional is None)

    target_amount = sum(record.target_amount for record in active)
    fill_amount = sum(record.fill_amount for record in active)
    priced_target_notional = sum(record.target_notional or 0.0 for record in active)
    target_notional = None if unpriced else priced_target_notional
    fill_notional = sum(record.fill_notional for record in active)
    commission = sum(record.commission for record in active)
    slippage = sum(record.slippage for record in active)
    total_cost = sum(record.total_cost for record in active)
    minimum_orders = sum(record.minimum_commission_applied for record in filled)
    notional_fill_rate = None if target_notional is None else _ratio(fill_notional, target_notional)

    return {
        "order_count": len(active),
        "filled_order_count": len(filled),
        "partial_fill_order_count": len(partial_fills),
        "zero_fill_order_count": len(zero_fills),
        "zero_fill_order_rate": _ratio(len(zero_fills), len(active)),
        "target_amount": target_amount,
        "fill_amount": fill_amount,
        "amount_fill_rate": _ratio(fill_amount, target_amount),
        "target_notional": target_notional,
        "priced_target_notional": priced_target_notional,
        "unpriced_target_order_count": len(unpriced),
        "fill_notional": fill_notional,
        "notional_fill_rate": notional_fill_rate,
        "fill_rate": notional_fill_rate,
        "commission_total": commission,
        "commission_mean_per_filled_order": _ratio(commission, len(filled)),
        "commission_effective_bps": (
            None if fill_notional <= 0 else commission / fill_notional * 10_000.0
        ),
        "minimum_commission_order_count": minimum_orders,
        "minimum_commission_order_rate": _ratio(minimum_orders, len(filled)),
        "slippage_total": slippage,
        "slippage_mean_per_filled_order": _ratio(slippage, len(filled)),
        "slippage_effective_bps": (
            None if fill_notional <= 0 else slippage / fill_notional * 10_000.0
        ),
        "total_cost": total_cost,
        "total_cost_effective_bps": (
            None if fill_notional <= 0 else total_cost / fill_notional * 10_000.0
        ),
    }


class SmallAccountExchange(Exchange):
    """Qlib Exchange with 100-share lots and correctly separated linear costs."""

    def __init__(
        self,
        *,
        commission_rate: float,
        min_commission: float = 5.0,
        slippage_bps: float = 0.0,
        trade_unit: int = ROUND_LOT,
        **exchange_kwargs: Any,
    ) -> None:
        if isinstance(trade_unit, bool) or trade_unit != ROUND_LOT:
            raise ValueError(f"trade_unit must be exactly {ROUND_LOT}")
        conflicting = {"open_cost", "close_cost", "min_cost", "impact_cost"} & set(exchange_kwargs)
        if conflicting:
            raise TypeError(
                "use commission_rate, min_commission, and slippage_bps instead of "
                f"Qlib cost arguments: {sorted(conflicting)}"
            )
        self.cost_model = LinearCostModel(commission_rate, min_commission, slippage_bps)
        self._execution_records: list[ExecutionRecord] = []
        super().__init__(
            open_cost=self.cost_model.commission_rate,
            close_cost=self.cost_model.commission_rate,
            min_cost=self.cost_model.min_commission,
            impact_cost=0.0,
            trade_unit=trade_unit,
            **exchange_kwargs,
        )
        if self.trade_w_adj_price:
            raise RuntimeError(
                "100-share lot enforcement requires complete $factor data; "
                "Qlib switched to adjusted-price mode"
            )

    @property
    def execution_records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._execution_records)

    def clear_execution_records(self) -> None:
        self._execution_records.clear()

    def _get_buy_amount_by_cash_limit(self, trade_price: float, cash: float, cost_ratio: float) -> float:
        del cost_ratio
        if not math.isfinite(float(trade_price)) or trade_price <= 0:
            return 0.0
        return self.cost_model.max_affordable_notional(cash) / trade_price

    def _round_amount_with_cash_limit(
        self,
        amount: float,
        factor: float,
        trade_price: float,
        cash: float,
    ) -> float:
        rounded = self.round_amount_by_trade_unit(amount, factor)
        lot = self.get_amount_of_trade_unit(factor=factor)
        if lot is None:
            raise RuntimeError("100-share lot enforcement became unavailable")
        while rounded > _COST_TOLERANCE:
            notional = rounded * trade_price
            if notional + self.cost_model.calculate(notional).total <= cash + _COST_TOLERANCE:
                break
            rounded = max(0.0, rounded - lot)
        return rounded

    def _calc_trade_info_by_order(
        self,
        order: Order,
        position: Any,
        dealt_order_amount: dict,
    ) -> tuple[float, float, float]:
        trade_price, trade_value, _ = super()._calc_trade_info_by_order(
            order,
            position,
            dealt_order_amount,
        )
        price = _optional_positive_price(trade_price)
        if order.deal_amount > _COST_TOLERANCE and price is None:
            raise RuntimeError("Qlib returned a positive fill without a valid trade price")

        if price is not None and order.direction == Order.BUY and position is not None:
            cash = _finite_nonnegative("position cash", position.get_cash())
            costs = self.cost_model.calculate(trade_value)
            if trade_value + costs.total > cash + _COST_TOLERANCE:
                affordable_amount = self.cost_model.max_affordable_notional(cash) / price
                order.deal_amount = self._round_amount_with_cash_limit(
                    min(float(order.deal_amount), affordable_amount),
                    order.factor,
                    price,
                    cash,
                )
                trade_value = float(order.deal_amount) * price

        costs = self.cost_model.calculate(trade_value)
        if (
            order.direction == Order.SELL
            and position is not None
            and position.get_cash() + trade_value < costs.total - _COST_TOLERANCE
        ):
            order.deal_amount = 0.0
            trade_value = 0.0
            costs = self.cost_model.calculate(0.0)
        return trade_price, trade_value, costs.total

    def deal_order(
        self,
        order: Order,
        trade_account: Any = None,
        position: Any = None,
        dealt_order_amount: dict | None = None,
    ) -> tuple[float, float, float]:
        if dealt_order_amount is None:
            dealt_order_amount = defaultdict(float)
        trade_value, trade_cost, trade_price = super().deal_order(
            order,
            trade_account=trade_account,
            position=position,
            dealt_order_amount=dealt_order_amount,
        )
        target_price = _optional_positive_price(trade_price)
        if target_price is None:
            try:
                target_price = _optional_positive_price(
                    self.get_deal_price(
                        order.stock_id,
                        order.start_time,
                        order.end_time,
                        direction=order.direction,
                    )
                )
            except (KeyError, TypeError, ValueError):
                target_price = None
        record = _record_from_values(
            timestamp=order.start_time,
            stock_id=str(order.stock_id),
            direction=order.direction,
            target_amount=order.amount,
            fill_amount=order.deal_amount,
            target_price=target_price,
            fill_notional=trade_value,
            reported_total_cost=trade_cost,
            cost_model=self.cost_model,
            validate_reported_cost=True,
        )
        # TopkDropoutStrategy simulates sells against a copied Position before
        # submitting orders. Only executor calls carry a trade_account.
        if trade_account is not None:
            self._execution_records.append(record)
        return trade_value, trade_cost, trade_price
