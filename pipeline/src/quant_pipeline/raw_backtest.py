"""Deterministic ETF backtesting in raw prices and legal fund shares.

This module intentionally does not use Qlib's adjusted-position account.  A
signal observed after session ``t`` selects immutable symbols for the close of
the next trading session.  Execution-day prices may size or reject that frozen
intent, but can never replace it with a lower-ranked instrument.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


_EPSILON = 1e-9
_TICK_RECONSTRUCTION_TOLERANCE = 1e-3
_NO_SHARE_CHANGE_TREATMENT = "not_applicable_no_share_change"
_UNKNOWN_FRACTIONAL_TREATMENTS = frozenset(
    {
        "unknown_not_provided_by_eastmoney_archive",
        "unknown_not_provided_by_sina_hfq",
    }
)
_ACTION_COLUMNS = (
    "symbol",
    "record_date",
    "ex_date",
    "cash_payment_date",
    "cash_dividend_per_old_share",
    "share_ratio",
    "fractional_share_treatment",
)
_EXECUTION_COLUMNS = (
    "intent_id",
    "signal_date",
    "execution_date",
    "symbol",
    "direction",
    "signal_score",
    "target_shares",
    "fill_shares",
    "target_notional",
    "fill_notional",
    "execution_price",
    "commission",
    "slippage",
    "total_cost",
    "minimum_commission_applied",
    "raw_volume",
    "volume_cap_shares",
    "cash_before",
    "cash_after",
    "shares_before",
    "shares_after",
    "wide_limit_tier_proven",
    "reason",
)
_POSITION_COLUMNS = (
    "date",
    "symbol",
    "shares",
    "eligible_shares",
    "locked_shares",
    "raw_close",
    "market_value",
)
_ACTION_LEDGER_COLUMNS = (
    "event_id",
    "date",
    "symbol",
    "action",
    "shares_before",
    "shares_after",
    "entitlement_shares",
    "amount",
    "cash_before",
    "cash_after",
    "receivable_before",
    "receivable_after",
    "commission",
    "turnover",
)
_ATTRIBUTION_COLUMNS = (
    "date",
    "symbol",
    "opening_shares",
    "closing_shares",
    "opening_market_value",
    "closing_market_value",
    "market_value_change",
    "buy_shares",
    "sell_shares",
    "buy_notional",
    "sell_notional",
    "price_pnl",
    "buy_commission",
    "sell_commission",
    "commission_pnl",
    "buy_slippage",
    "sell_slippage",
    "slippage_pnl",
    "dividend_entitlement",
    "dividend_receivable_increase",
    "dividend_receivable_release",
    "dividend_receivable_pnl",
    "dividend_payment",
    "dividend_payment_pnl",
    "net_pnl",
    "abs_contribution_share",
    "daily_abs_contribution_share",
    "account_net_pnl",
    "attributed_net_pnl",
    "reconciliation_error",
)


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class RawBacktestConfig:
    """Execution and top-k policy for a CNY cash account."""

    initial_cash: float = 20_000.0
    topk: int = 5
    n_drop: int = 1
    hold_thresh: int = 5
    risk_degree: float = 0.90
    commission_bps_per_side: float = 3.0
    min_commission: float = 5.0
    slippage_bps_per_side: float = 5.0
    lot_size: int = 100
    max_daily_volume_participation: float = 0.05
    standard_limit_ratio: float = 0.10
    wide_limit_ratio: float = 0.20
    price_tick: float = 0.001

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_cash", _finite("initial_cash", self.initial_cash, minimum=_EPSILON))
        object.__setattr__(self, "topk", _integer("topk", self.topk, minimum=1))
        object.__setattr__(self, "n_drop", _integer("n_drop", self.n_drop, minimum=0))
        object.__setattr__(self, "hold_thresh", _integer("hold_thresh", self.hold_thresh, minimum=1))
        object.__setattr__(self, "lot_size", _integer("lot_size", self.lot_size, minimum=1))
        for name in (
            "risk_degree",
            "commission_bps_per_side",
            "min_commission",
            "slippage_bps_per_side",
            "max_daily_volume_participation",
            "standard_limit_ratio",
            "wide_limit_ratio",
            "price_tick",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.0))
        if self.n_drop > self.topk:
            raise ValueError("n_drop cannot exceed topk")
        if not 0.0 < self.risk_degree <= 1.0:
            raise ValueError("risk_degree must be in (0, 1]")
        if not 0.0 < self.max_daily_volume_participation <= 0.25:
            raise ValueError("max_daily_volume_participation must be in (0, 0.25]")
        if not 0.0 < self.standard_limit_ratio < self.wide_limit_ratio < 1.0:
            raise ValueError("price limit ratios must satisfy 0 < standard < wide < 1")
        if self.price_tick <= 0.0:
            raise ValueError("price_tick must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RawBacktestConfig":
        """Accept either flat values or the pipeline's strategy/execution sections."""

        if not isinstance(value, Mapping):
            raise TypeError("config must be RawBacktestConfig or a mapping")
        strategy = value.get("strategy", {})
        execution = value.get("execution", {})
        if strategy and not isinstance(strategy, Mapping):
            raise TypeError("config.strategy must be a mapping")
        if execution and not isinstance(execution, Mapping):
            raise TypeError("config.execution must be a mapping")
        merged = dict(value)
        merged.update(strategy)
        aliases = {
            "account": "initial_cash",
            "min_cost": "min_commission",
            "base_slippage_bps_per_side": "slippage_bps_per_side",
            "trade_unit": "lot_size",
        }
        for key, item in execution.items():
            merged[aliases.get(key, key)] = item
        fields = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in merged.items() if key in fields})


@dataclass(frozen=True)
class RawBacktestResult:
    report: pd.DataFrame
    positions: pd.DataFrame
    executions: pd.DataFrame
    corporate_action_ledger: pd.DataFrame
    summary: dict[str, Any]
    symbol_attribution: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_ATTRIBUTION_COLUMNS)
    )


@dataclass
class _Lot:
    shares: float
    acquired_session: int


def _calendar(values: Iterable[Any]) -> pd.DatetimeIndex:
    try:
        raw = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise"))
    except Exception as exc:
        raise ValueError("trading_calendar contains invalid dates") from exc
    if raw.empty:
        raise ValueError("trading_calendar must not be empty")
    if raw.tz is not None:
        raise ValueError("trading_calendar must be timezone-naive")
    result = raw.normalize()
    if result.has_duplicates:
        raise ValueError("trading_calendar contains duplicate sessions")
    if not result.is_monotonic_increasing:
        raise ValueError("trading_calendar must be strictly increasing")
    return result


def _normalise_predictions(value: pd.Series | pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if "score" in value.columns:
            series = value["score"].copy()
        elif value.shape[1] == 1:
            series = value.iloc[:, 0].copy()
        else:
            raise ValueError("predictions DataFrame must have one column or a score column")
    elif isinstance(value, pd.Series):
        series = value.copy()
    else:
        raise TypeError("predictions must be a pandas Series or DataFrame")
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("predictions must use a MultiIndex(datetime, instrument)")
    if list(series.index.names) != ["datetime", "instrument"]:
        raise ValueError("prediction index levels must be named datetime and instrument")
    dates = pd.to_datetime(series.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any() or dates.tz is not None:
        raise ValueError("prediction dates must be valid and timezone-naive")
    instruments = series.index.get_level_values("instrument").map(str)
    if (instruments.str.strip() == "").any():
        raise ValueError("prediction instruments must be non-empty")
    series.index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(dates).normalize(), instruments], names=["datetime", "instrument"]
    )
    if series.index.has_duplicates:
        raise ValueError("predictions contain duplicate datetime/instrument rows")
    series = pd.to_numeric(series, errors="coerce")
    if series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("prediction scores must be finite")
    outside = pd.DatetimeIndex(series.index.get_level_values("datetime").unique()).difference(calendar)
    if not outside.empty:
        raise ValueError(f"prediction dates are outside the trading calendar: {outside[0].date()}")
    final = calendar[-1]
    if final in series.index.get_level_values("datetime"):
        raise ValueError(f"prediction on {final.date()} has no next execution session")
    return series.sort_index()


def _normalise_bars(value: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("raw_bars must be a pandas DataFrame")
    required = {"date", "symbol", "raw_open", "raw_close", "raw_high", "raw_low", "volume"}
    missing = required - set(value.columns)
    if missing:
        raise ValueError(f"raw_bars lacks columns: {sorted(missing)}")
    optional_factor = "adjustment_factor" in value.columns
    columns = list(required) + (["adjustment_factor"] if optional_factor else [])
    frame = value.loc[:, columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].dt.tz is not None:
        raise ValueError("raw bar dates must be valid and timezone-naive")
    frame["date"] = frame["date"].dt.normalize()
    frame["symbol"] = frame["symbol"].map(str).str.strip()
    if (frame["symbol"] == "").any():
        raise ValueError("raw bar symbols must be non-empty")
    numeric = ["raw_open", "raw_close", "raw_high", "raw_low", "volume"]
    if optional_factor:
        numeric.append("adjustment_factor")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric].isna().any().any() or not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("raw bars contain missing or non-finite numeric values")
    if (frame[["raw_open", "raw_close", "raw_high", "raw_low"]] <= 0).any().any():
        raise ValueError("raw OHLC prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("raw volume must be non-negative")
    if optional_factor and (frame["adjustment_factor"] <= 0).any():
        raise ValueError("adjustment_factor must be positive")
    if (frame["raw_low"] > frame[["raw_open", "raw_close"]].min(axis=1)).any() or (
        frame["raw_high"] < frame[["raw_open", "raw_close"]].max(axis=1)
    ).any() or (frame["raw_low"] > frame["raw_high"]).any():
        raise ValueError("raw bars violate OHLC range invariants")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("raw_bars contains duplicate date/symbol rows")
    return frame.set_index(["date", "symbol"]).sort_index()


def _normalise_actions(value: pd.DataFrame) -> pd.DataFrame:
    if value is None:
        raise ValueError("corporate_actions is required; pass an explicit empty table when verified empty")
    if not isinstance(value, pd.DataFrame):
        raise TypeError("corporate_actions must be a pandas DataFrame")
    missing = set(_ACTION_COLUMNS) - set(value.columns)
    if missing:
        raise ValueError(f"corporate_actions lacks columns: {sorted(missing)}")
    frame = value.loc[:, _ACTION_COLUMNS].copy()
    if frame.empty:
        frame["event_id"] = pd.Series(dtype=str)
        return frame
    frame["symbol"] = frame["symbol"].map(str).str.strip()
    if (frame["symbol"] == "").any():
        raise ValueError("corporate-action symbols must be non-empty")
    for column in ("cash_dividend_per_old_share", "share_ratio"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numbers = frame[["cash_dividend_per_old_share", "share_ratio"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("corporate-action economics must be finite")
    if (frame["cash_dividend_per_old_share"] < 0).any() or (frame["share_ratio"] <= 0).any():
        raise ValueError("corporate-action cash must be non-negative and share_ratio positive")
    treatment = frame["fractional_share_treatment"]
    if treatment.isna().any() or treatment.astype(str).str.strip().eq("").any():
        raise ValueError("corporate-action fractional_share_treatment must be explicit")
    frame["fractional_share_treatment"] = treatment.astype(str).str.strip()
    share_change = ~np.isclose(frame["share_ratio"], 1.0, rtol=0.0, atol=_EPSILON)
    if (
        ~share_change & frame["fractional_share_treatment"].ne(_NO_SHARE_CHANGE_TREATMENT)
    ).any():
        raise ValueError("cash-only events must mark fractional-share treatment not applicable")
    if (
        share_change
        & ~frame["fractional_share_treatment"].isin(_UNKNOWN_FRACTIONAL_TREATMENTS)
    ).any():
        raise ValueError("unsupported corporate-action fractional-share treatment declaration")
    for column in ("record_date", "ex_date", "cash_payment_date"):
        original = frame[column]
        parsed = pd.to_datetime(original, errors="coerce")
        if (parsed.isna() & original.notna()).any():
            raise ValueError(f"corporate-action {column} contains an invalid date")
        non_null = parsed.dropna()
        if not non_null.empty and non_null.dt.tz is not None:
            raise ValueError(f"corporate-action {column} must be timezone-naive")
        frame[column] = parsed.dt.normalize()
    if frame["ex_date"].isna().any():
        raise ValueError("corporate-action ex_date is required and may not be unknown")
    cash_events = frame["cash_dividend_per_old_share"] > _EPSILON
    for column in ("record_date", "cash_payment_date"):
        if frame.loc[cash_events, column].isna().any():
            raise ValueError(
                f"corporate-action {column} is required for a cash distribution and may not be unknown"
            )
    if ((frame["record_date"] > frame["ex_date"]) & frame["record_date"].notna()).any():
        raise ValueError("corporate-action record_date must not follow ex_date")
    if ((frame["cash_payment_date"] < frame["ex_date"]) & frame["cash_payment_date"].notna()).any():
        raise ValueError("corporate-action cash_payment_date must not precede ex_date")
    if frame.duplicated(["symbol", "ex_date"]).any():
        raise ValueError("duplicate symbol/ex_date corporate actions are ambiguous")
    frame = frame.sort_values(["symbol", "ex_date"], kind="stable").reset_index(drop=True)
    identities = []
    for row in frame.itertuples(index=False):
        identity = "|".join(
            [
                row.symbol,
                "null" if pd.isna(row.record_date) else row.record_date.date().isoformat(),
                row.ex_date.date().isoformat(),
                "null" if pd.isna(row.cash_payment_date) else row.cash_payment_date.date().isoformat(),
                format(float(row.cash_dividend_per_old_share), ".17g"),
                format(float(row.share_ratio), ".17g"),
                row.fractional_share_treatment,
            ]
        )
        identities.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    frame["event_id"] = identities
    return frame


def _normalise_benchmark(
    value: pd.Series | pd.DataFrame,
    symbol: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        frame = pd.DataFrame({"date": value.index, "raw_close": value.to_numpy()})
    elif isinstance(value, pd.DataFrame):
        if not {"date", "raw_close"}.issubset(value.columns):
            raise ValueError("benchmark_close must contain date and raw_close")
        frame = value.copy()
        if "symbol" in frame.columns:
            frame = frame.loc[frame["symbol"].map(str) == symbol]
    else:
        raise TypeError("benchmark_close must be a pandas Series or DataFrame")
    columns = ["date", "raw_close"] + (["adjustment_factor"] if "adjustment_factor" in frame else [])
    frame = frame.loc[:, columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].dt.tz is not None:
        raise ValueError("benchmark dates must be valid and timezone-naive")
    frame["date"] = frame["date"].dt.normalize()
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[columns[1:]].isna().any().any() or not np.isfinite(frame[columns[1:]].to_numpy(dtype=float)).all():
        raise ValueError("benchmark values must be finite")
    if (frame[columns[1:]] <= 0).any().any():
        raise ValueError("benchmark close and adjustment factor must be positive")
    if frame["date"].duplicated().any():
        raise ValueError("benchmark_close contains duplicate dates")
    frame = frame.set_index("date").sort_index()
    missing = calendar.difference(frame.index)
    extra = frame.index.difference(calendar)
    if not missing.empty or not extra.empty:
        raise ValueError("benchmark_close must cover the trading calendar exactly")
    return frame.reindex(calendar)


def _event_for(actions: pd.DataFrame, symbol: str, date: pd.Timestamp) -> pd.Series | None:
    selected = actions[(actions["symbol"] == symbol) & (actions["ex_date"] == date)]
    return None if selected.empty else selected.iloc[0]


def _validate_factor_jumps(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    factor_column: str,
    close_column: str,
    symbol: str | None = None,
) -> None:
    if factor_column not in frame.columns:
        return
    if isinstance(frame.index, pd.MultiIndex):
        groups = frame.reset_index().groupby("symbol", sort=False)
    else:
        if symbol is None:
            raise ValueError("benchmark factor validation requires benchmark_symbol")
        temporary = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        temporary["symbol"] = symbol
        groups = temporary.groupby("symbol", sort=False)
    for current_symbol, group in groups:
        group = group.sort_values("date", kind="stable").reset_index(drop=True)
        for position in range(1, len(group)):
            before = float(group.loc[position - 1, factor_column])
            after = float(group.loc[position, factor_column])
            ratio = after / before
            date = pd.Timestamp(group.loc[position, "date"])
            event = _event_for(actions, str(current_symbol), date)
            if math.isclose(ratio, 1.0, rel_tol=1e-10, abs_tol=1e-12):
                if event is not None:
                    raise ValueError(
                        f"{current_symbol} {date.date()}: corporate-action event has no matching factor jump"
                    )
                continue
            if event is None:
                raise ValueError(
                    f"{current_symbol} {date.date()}: factor jump lacks a matching corporate-action event"
                )
            previous_close = float(group.loc[position - 1, close_column])
            cash = float(event["cash_dividend_per_old_share"])
            share_ratio = float(event["share_ratio"])
            theoretical_ex = (previous_close - cash) / share_ratio
            if theoretical_ex <= 0:
                raise ValueError(f"{current_symbol} {date.date()}: invalid corporate-action reference price")
            expected = previous_close / theoretical_ex
            if not math.isclose(ratio, expected, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError(
                    f"{current_symbol} {date.date()}: factor jump does not match corporate-action economics"
                )


def _price_ticks(price: float, tick: float) -> tuple[int, bool]:
    scaled = price / tick
    rounded = int(np.rint(scaled))
    valid = price > 0 and math.isfinite(price) and math.isclose(
        scaled, rounded, rel_tol=0.0, abs_tol=_TICK_RECONSTRUCTION_TOLERANCE
    )
    return (rounded if valid else 0), valid


def _rounded_limit(previous_ticks: int, ratio: float, *, upper: bool) -> int:
    bps = round(ratio * 10_000)
    if not math.isclose(ratio, bps / 10_000.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("price limit ratios must resolve to whole basis points")
    multiplier = 10_000 + bps if upper else 10_000 - bps
    return (previous_ticks * multiplier + 5_000) // 10_000


def _price_limit_flags(
    previous: pd.Series,
    current: pd.Series,
    config: RawBacktestConfig,
    *,
    corporate_action: bool,
) -> tuple[bool, bool, bool, str | None]:
    names = ("raw_close", "raw_high", "raw_low")
    current_ticks = {}
    for name in names:
        current_ticks[name], valid = _price_ticks(float(current[name]), config.price_tick)
        if not valid:
            return True, True, False, "invalid_price_reference"
    previous_ticks, previous_valid = _price_ticks(float(previous["raw_close"]), config.price_tick)
    if not previous_valid:
        return True, True, False, "invalid_price_reference"
    if corporate_action:
        return True, True, False, "corporate_action_price_reference"
    standard_upper = _rounded_limit(previous_ticks, config.standard_limit_ratio, upper=True)
    standard_lower = _rounded_limit(previous_ticks, config.standard_limit_ratio, upper=False)
    wide = current_ticks["raw_high"] > standard_upper or current_ticks["raw_low"] < standard_lower
    ratio = config.wide_limit_ratio if wide else config.standard_limit_ratio
    upper = _rounded_limit(previous_ticks, ratio, upper=True)
    lower = _rounded_limit(previous_ticks, ratio, upper=False)
    return current_ticks["raw_close"] >= upper, current_ticks["raw_close"] <= lower, wide, None


def _shares(lots: list[_Lot]) -> float:
    return float(sum(lot.shares for lot in lots))


def _eligible_shares(lots: list[_Lot], session: int, hold_thresh: int) -> float:
    return float(sum(lot.shares for lot in lots if session - lot.acquired_session >= hold_thresh))


def _consume_eligible(lots: list[_Lot], amount: float, session: int, hold_thresh: int) -> None:
    remaining = amount
    for lot in lots:
        if session - lot.acquired_session < hold_thresh:
            continue
        taken = min(lot.shares, remaining)
        lot.shares -= taken
        remaining -= taken
        if remaining <= _EPSILON:
            break
    lots[:] = [lot for lot in lots if lot.shares > _EPSILON]
    if remaining > _EPSILON:
        raise RuntimeError("eligible-share ledger failed to consume a completed sell")


def _costs(notional: float, config: RawBacktestConfig) -> tuple[float, float, float, bool]:
    if notional <= _EPSILON:
        return 0.0, 0.0, 0.0, False
    proportional = notional * config.commission_bps_per_side / 10_000.0
    commission = max(proportional, config.min_commission)
    slippage = notional * config.slippage_bps_per_side / 10_000.0
    return commission, slippage, commission + slippage, proportional <= config.min_commission


def _affordable_shares(cash: float, price: float, config: RawBacktestConfig) -> int:
    commission_rate = config.commission_bps_per_side / 10_000.0
    slippage_rate = config.slippage_bps_per_side / 10_000.0
    fixed = (cash - config.min_commission) / (1.0 + slippage_rate)
    proportional = cash / (1.0 + commission_rate + slippage_rate)
    notional = max(0.0, min(fixed, proportional))
    return int(math.floor((notional / price + _EPSILON) / config.lot_size) * config.lot_size)


def _joint_buy_cash_caps(
    cash: float,
    orders: Mapping[str, tuple[int, float]],
    config: RawBacktestConfig,
    *,
    nav_before_buys: float,
    retained_market_value: float,
) -> dict[str, int]:
    """Apply one common scale to all executable buys before any cash is spent."""

    targets = {str(symbol): max(0, int(shares)) for symbol, (shares, _) in orders.items()}
    if not targets:
        return {}

    def proposal(scale: float) -> tuple[dict[str, int], float, float]:
        shares_by_symbol = {
            symbol: int(
                math.floor((targets[symbol] * scale + _EPSILON) / config.lot_size)
                * config.lot_size
            )
            for symbol in targets
        }
        total_notional = 0.0
        total_cost = 0.0
        for symbol, shares in shares_by_symbol.items():
            notional = shares * float(orders[symbol][1])
            total_notional += notional
            total_cost += _costs(notional, config)[2]
        return shares_by_symbol, total_notional, total_cost

    def feasible(scale: float) -> bool:
        _, total_notional, total_cost = proposal(scale)
        cash_ok = total_notional + total_cost <= cash + _EPSILON
        final_nav = nav_before_buys - total_cost
        risk_ok = retained_market_value + total_notional <= (
            config.risk_degree * nav_before_buys + _EPSILON
        )
        return cash_ok and final_nav > _EPSILON and risk_ok

    if feasible(1.0):
        return targets
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if feasible(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return proposal(lower)[0]


def _rank(scores: pd.Series) -> list[str]:
    return [item[0] for item in sorted(((str(symbol), float(score)) for symbol, score in scores.items()), key=lambda item: (-item[1], item[0]))]


def _select_intents(scores: pd.Series, held: list[str], config: RawBacktestConfig) -> tuple[list[str], list[str]]:
    ranked_all = _rank(scores)
    protected = {symbol for symbol in held if symbol not in scores.index}
    held_ranked = sorted(
        (symbol for symbol in held if symbol not in protected),
        key=lambda symbol: (-float(scores[symbol]), symbol),
    )
    remaining_slots = max(0, config.topk - len(protected))
    candidate_count = max(0, config.n_drop + remaining_slots - len(held_ranked))
    held_set = set(held)
    candidates = [symbol for symbol in ranked_all if symbol not in held_set][:candidate_count]
    combined = held_ranked + candidates
    combined.sort(key=lambda symbol: (-float(scores[symbol]), symbol))
    bottom = set(combined[-config.n_drop :]) if config.n_drop else set()
    sell = [symbol for symbol in held_ranked if symbol in bottom]
    buy_count = max(0, len(sell) + remaining_slots - len(held_ranked))
    return sell, candidates[:buy_count]


def _benchmark_returns(
    benchmark: pd.DataFrame,
    actions: pd.DataFrame,
    symbol: str,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    result = pd.Series(0.0, index=calendar, name="bench")
    for position in range(1, len(calendar)):
        date = calendar[position]
        previous = float(benchmark.iloc[position - 1]["raw_close"])
        current = float(benchmark.iloc[position]["raw_close"])
        event = _event_for(actions, symbol, date)
        if event is None:
            result.iloc[position] = current / previous - 1.0
        else:
            result.iloc[position] = (
                float(event["share_ratio"]) * current
                + float(event["cash_dividend_per_old_share"])
            ) / previous - 1.0
    if (result <= -1.0).any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError("benchmark total-return series is invalid")
    return result


def _symbol_attribution(
    report: pd.DataFrame,
    positions: pd.DataFrame,
    executions: pd.DataFrame,
    action_ledger: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    initial_cash: float,
) -> tuple[pd.DataFrame, float]:
    """Build a daily ETF P&L bridge and reconcile it to account NAV."""

    value_columns = _ATTRIBUTION_COLUMNS[2:25]
    records: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}

    def record(date: Any, symbol: Any) -> dict[str, float]:
        key = (pd.Timestamp(date), str(symbol))
        if key not in records:
            records[key] = {column: 0.0 for column in value_columns}
        return records[key]

    next_session = dict(zip(sessions[:-1], sessions[1:]))
    for row in positions.itertuples(index=False):
        current = record(row.date, row.symbol)
        current["closing_shares"] += float(row.shares)
        current["closing_market_value"] += float(row.market_value)
        following = next_session.get(pd.Timestamp(row.date))
        if following is not None:
            opening = record(following, row.symbol)
            opening["opening_shares"] += float(row.shares)
            opening["opening_market_value"] += float(row.market_value)

    for row in executions.itertuples(index=False):
        item = record(row.execution_date, row.symbol)
        prefix = str(row.direction)
        if prefix not in {"buy", "sell"}:
            raise RuntimeError(f"unsupported execution direction in attribution: {prefix!r}")
        item[f"{prefix}_shares"] += float(row.fill_shares)
        item[f"{prefix}_notional"] += float(row.fill_notional)
        item[f"{prefix}_commission"] += float(row.commission)
        item[f"{prefix}_slippage"] += float(row.slippage)

    for row in action_ledger.itertuples(index=False):
        if row.action == "dividend_entitlement":
            record(row.date, row.symbol)["dividend_entitlement"] += float(row.amount)
        elif row.action == "dividend_receivable":
            record(row.date, row.symbol)["dividend_receivable_increase"] += float(row.amount)
        elif row.action == "cash_payment":
            item = record(row.date, row.symbol)
            item["dividend_receivable_release"] += float(row.amount)
            item["dividend_payment"] += float(row.amount)

    rows: list[dict[str, Any]] = []
    for (date, symbol), item in sorted(records.items()):
        item["market_value_change"] = (
            item["closing_market_value"] - item["opening_market_value"]
        )
        item["price_pnl"] = (
            item["market_value_change"]
            - item["buy_notional"]
            + item["sell_notional"]
        )
        item["commission_pnl"] = -item["buy_commission"] - item["sell_commission"]
        item["slippage_pnl"] = -item["buy_slippage"] - item["sell_slippage"]
        item["dividend_receivable_pnl"] = (
            item["dividend_receivable_increase"]
            - item["dividend_receivable_release"]
        )
        item["dividend_payment_pnl"] = item["dividend_payment"]
        item["net_pnl"] = sum(
            item[column]
            for column in (
                "price_pnl",
                "commission_pnl",
                "slippage_pnl",
                "dividend_receivable_pnl",
                "dividend_payment_pnl",
            )
        )
        rows.append({"date": date, "symbol": symbol, **item})

    attribution = pd.DataFrame(rows, columns=_ATTRIBUTION_COLUMNS[:25])
    account_net_pnl = report["account"].diff()
    account_net_pnl.iloc[0] = float(report.iloc[0]["account"]) - initial_cash
    if attribution.empty:
        attributed_net_pnl = pd.Series(0.0, index=report.index)
    else:
        attributed_net_pnl = (
            attribution.groupby("date", sort=True)["net_pnl"]
            .sum()
            .reindex(report.index, fill_value=0.0)
        )
    reconciliation_error = attributed_net_pnl - account_net_pnl
    maximum_error = float(reconciliation_error.abs().max())
    if not np.allclose(
        attributed_net_pnl.to_numpy(dtype=float),
        account_net_pnl.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-8,
    ):
        failed_date = reconciliation_error.abs().idxmax()
        raise RuntimeError(
            "symbol attribution failed account NAV reconciliation on "
            f"{failed_date.date()}: error={reconciliation_error.loc[failed_date]:.12g}"
        )

    if attribution.empty:
        return pd.DataFrame(columns=_ATTRIBUTION_COLUMNS), maximum_error

    absolute_pnl = attribution["net_pnl"].abs()
    total_absolute_pnl = float(absolute_pnl.sum())
    attribution["abs_contribution_share"] = (
        absolute_pnl / total_absolute_pnl if total_absolute_pnl > _EPSILON else np.nan
    )
    daily_absolute_pnl = absolute_pnl.groupby(attribution["date"]).transform("sum")
    attribution["daily_abs_contribution_share"] = np.where(
        daily_absolute_pnl > _EPSILON,
        absolute_pnl / daily_absolute_pnl,
        np.nan,
    )
    attribution["account_net_pnl"] = attribution["date"].map(account_net_pnl)
    attribution["attributed_net_pnl"] = attribution["date"].map(attributed_net_pnl)
    attribution["reconciliation_error"] = attribution["date"].map(reconciliation_error)
    return attribution.loc[:, _ATTRIBUTION_COLUMNS], maximum_error


def _symbol_concentration(attribution: pd.DataFrame) -> dict[str, Any]:
    zero_policy = "concentration_null_fail_closed"
    if attribution.empty:
        gross = pd.Series(dtype=float)
        net = pd.Series(dtype=float)
    else:
        grouped = attribution.groupby("symbol", sort=True)["net_pnl"]
        gross = grouped.apply(lambda values: float(values.abs().sum()))
        net = grouped.sum().abs()

    def concentration(values: pd.Series) -> dict[str, Any]:
        denominator = float(values.sum())
        if denominator <= _EPSILON:
            return {
                "symbol": None,
                "numerator_cny": 0.0,
                "denominator_cny": denominator,
                "share": None,
            }
        symbol = str(values.idxmax())
        numerator = float(values.loc[symbol])
        return {
            "symbol": symbol,
            "numerator_cny": numerator,
            "denominator_cny": denominator,
            "share": numerator / denominator,
        }

    return {
        "zero_denominator_policy": zero_policy,
        "gross_abs": concentration(gross),
        "net_abs": concentration(net),
    }


def run_raw_backtest(
    predictions: pd.Series | pd.DataFrame,
    raw_bars: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    trading_calendar: Iterable[Any],
    config: RawBacktestConfig | Mapping[str, Any],
    benchmark_close: pd.Series | pd.DataFrame,
    benchmark_symbol: str,
    *,
    factor_jumps_pre_audited: bool = False,
) -> RawBacktestResult:
    """Run a close-to-close, raw-share ETF simulation with auditable ledgers."""

    policy = config if isinstance(config, RawBacktestConfig) else RawBacktestConfig.from_mapping(config)
    if not isinstance(factor_jumps_pre_audited, bool):
        raise TypeError("factor_jumps_pre_audited must be bool")
    sessions = _calendar(trading_calendar)
    scores = _normalise_predictions(predictions, sessions)
    bars = _normalise_bars(raw_bars)
    actions = _normalise_actions(corporate_actions)
    benchmark_symbol = str(benchmark_symbol).strip()
    if not benchmark_symbol:
        raise ValueError("benchmark_symbol must be non-empty")
    benchmark = _normalise_benchmark(benchmark_close, benchmark_symbol, sessions)

    bar_dates = pd.DatetimeIndex(bars.index.get_level_values("date").unique())
    if not bar_dates.difference(sessions).empty:
        raise ValueError("raw_bars contains dates outside the trading calendar")
    prediction_symbols = set(scores.index.get_level_values("instrument"))
    unknown_symbols = prediction_symbols - set(bars.index.get_level_values("symbol"))
    if unknown_symbols:
        raise ValueError(f"predictions reference symbols absent from raw_bars: {sorted(unknown_symbols)}")
    for column in ("record_date", "ex_date", "cash_payment_date"):
        in_span = actions[column].notna() & actions[column].between(sessions[0], sessions[-1])
        outside_calendar = pd.DatetimeIndex(actions.loc[in_span, column]).difference(sessions)
        if not outside_calendar.empty:
            raise ValueError(f"corporate-action {column} is not a trading session: {outside_calendar[0].date()}")
    if not factor_jumps_pre_audited:
        if "adjustment_factor" not in bars.columns:
            raise ValueError(
                "raw_bars must include adjustment_factor unless factor jumps were pre-audited"
            )
        if "adjustment_factor" not in benchmark.columns:
            raise ValueError(
                "benchmark_close must include adjustment_factor unless factor jumps were pre-audited"
            )
        _validate_factor_jumps(
            bars,
            actions,
            factor_column="adjustment_factor",
            close_column="raw_close",
        )
        _validate_factor_jumps(
            benchmark,
            actions,
            factor_column="adjustment_factor",
            close_column="raw_close",
            symbol=benchmark_symbol,
        )
    bench = _benchmark_returns(benchmark, actions, benchmark_symbol, sessions)

    signal_by_execution: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}
    date_position = {date: position for position, date in enumerate(sessions)}
    for signal_date in pd.DatetimeIndex(scores.index.get_level_values("datetime").unique()).sort_values():
        execution_date = sessions[date_position[signal_date] + 1]
        daily_scores = scores.xs(signal_date, level="datetime")
        signal_by_execution[execution_date] = (signal_date, daily_scores)

    lots: dict[str, list[_Lot]] = {}
    frozen_entitlements: dict[str, tuple[str, float, float]] = {}
    receivables: dict[str, float] = {}
    cash = policy.initial_cash
    execution_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    held_missing_prediction_events: list[dict[str, str]] = []
    last_known_close: dict[str, float] = {}
    mark_carry_forward_events: list[dict[str, Any]] = []
    mark_carry_forward_seen: set[tuple[pd.Timestamp, str]] = set()

    def record_carry_forward(
        symbol: str, date: pd.Timestamp, shares: float, price: float
    ) -> None:
        key = (date, symbol)
        if key in mark_carry_forward_seen:
            return
        mark_carry_forward_seen.add(key)
        mark_carry_forward_events.append(
            {
                "date": date,
                "symbol": symbol,
                "shares": shares,
                "price": price,
                "source": "previous_raw_close",
            }
        )

    previous_nav = policy.initial_cash

    # This simulator always starts from cash with no opening positions.  A cash
    # event whose record date predates that initial state therefore has a known
    # zero entitlement, even when its ex-date falls inside the simulation.
    pre_start_records = actions[
        (actions["cash_dividend_per_old_share"] > _EPSILON)
        & (actions["record_date"] < sessions[0])
        & (actions["ex_date"] >= sessions[0])
    ]
    for _, event in pre_start_records.iterrows():
        frozen_entitlements[str(event["event_id"])] = (str(event["symbol"]), 0.0, 0.0)

    def total_receivable() -> float:
        return float(sum(receivables.values()))

    def post_payment(row: pd.Series, date: pd.Timestamp) -> None:
        nonlocal cash
        event_id = str(row["event_id"])
        if event_id not in receivables:
            return
        amount = receivables.pop(event_id)
        entitlement = frozen_entitlements.pop(event_id, (str(row["symbol"]), 0.0, 0.0))
        cash_before = cash
        before = total_receivable() + amount
        cash += amount
        action_rows.append(
            {
                "event_id": event_id,
                "date": date,
                "symbol": row["symbol"],
                "action": "cash_payment",
                "shares_before": _shares(lots.get(row["symbol"], [])),
                "shares_after": _shares(lots.get(row["symbol"], [])),
                "entitlement_shares": entitlement[1],
                "amount": amount,
                "cash_before": cash_before,
                "cash_after": cash,
                "receivable_before": before,
                "receivable_after": total_receivable(),
                "commission": 0.0,
                "turnover": 0.0,
            }
        )

    for session_number, date in enumerate(sessions):
        daily_cost = 0.0
        daily_notional = 0.0

        # A same-day record/ex event freezes entitlement from old shares before
        # the legal share adjustment.  Close buyers on an ex-date are not
        # entitled to a distribution already detached from the fund price.
        same_day_events = actions[
            (actions["record_date"] == date) & (actions["ex_date"] == date)
        ]
        for _, event in same_day_events.iterrows():
            if float(event["cash_dividend_per_old_share"]) <= _EPSILON:
                continue
            symbol = str(event["symbol"])
            event_id = str(event["event_id"])
            entitled_shares = _shares(lots.get(symbol, []))
            entitlement = entitled_shares * float(event["cash_dividend_per_old_share"])
            frozen_entitlements[event_id] = (symbol, entitled_shares, entitlement)
            action_rows.append(
                {
                    "event_id": event_id,
                    "date": date,
                    "symbol": symbol,
                    "action": "dividend_entitlement",
                    "shares_before": entitled_shares,
                    "shares_after": entitled_shares,
                    "entitlement_shares": entitled_shares,
                    "amount": entitlement,
                    "cash_before": cash,
                    "cash_after": cash,
                    "receivable_before": total_receivable(),
                    "receivable_after": total_receivable(),
                    "commission": 0.0,
                    "turnover": 0.0,
                }
            )

        # The detached cash right becomes an on-balance-sheet receivable on the
        # ex-date.  Before then its value is still embedded in the cum-dividend
        # raw close and adding it to NAV would double count the distribution.
        for _, event in actions[actions["ex_date"] == date].iterrows():
            if float(event["cash_dividend_per_old_share"]) <= _EPSILON:
                continue
            event_id = str(event["event_id"])
            if event_id not in frozen_entitlements:
                raise RuntimeError(
                    f"{event['symbol']} {date.date()}: dividend entitlement was not frozen on record_date"
                )
            symbol, entitled_shares, entitlement = frozen_entitlements[event_id]
            before = total_receivable()
            receivables[event_id] = entitlement
            action_rows.append(
                {
                    "event_id": event_id,
                    "date": date,
                    "symbol": symbol,
                    "action": "dividend_receivable",
                    "shares_before": _shares(lots.get(symbol, [])),
                    "shares_after": _shares(lots.get(symbol, [])),
                    "entitlement_shares": entitled_shares,
                    "amount": entitlement,
                    "cash_before": cash,
                    "cash_after": cash,
                    "receivable_before": before,
                    "receivable_after": total_receivable(),
                    "commission": 0.0,
                    "turnover": 0.0,
                }
            )

        # Legal share changes occur before the ex-date trading session.
        for _, event in actions[actions["ex_date"] == date].iterrows():
            symbol = str(event["symbol"])
            symbol_lots = lots.get(symbol, [])
            before = _shares(symbol_lots)
            after = before * float(event["share_ratio"])
            nearest_round_lot = round(after / policy.lot_size) * policy.lot_size
            if before > _EPSILON and not math.isclose(
                after,
                nearest_round_lot,
                rel_tol=0.0,
                abs_tol=_EPSILON,
            ):
                raise RuntimeError(
                    f"{symbol} {date.date()}: share adjustment {before:.12g} -> {after:.12g} "
                    f"creates a non-round-lot position while fractional_share_treatment="
                    f"{event['fractional_share_treatment']!r}; settlement or odd-lot execution "
                    "must be source-verified before backtesting"
                )
            for lot in symbol_lots:
                lot.shares *= float(event["share_ratio"])
            after = _shares(symbol_lots)
            action_rows.append(
                {
                    "event_id": event["event_id"],
                    "date": date,
                    "symbol": symbol,
                    "action": "share_adjustment",
                    "shares_before": before,
                    "shares_after": after,
                    "entitlement_shares": 0.0,
                    "amount": 0.0,
                    "cash_before": cash,
                    "cash_after": cash,
                    "receivable_before": total_receivable(),
                    "receivable_after": total_receivable(),
                    "commission": 0.0,
                    "turnover": 0.0,
                }
            )

        # Verified payments are available to the close executor.  This order
        # also handles ex-date/payment-date equality without delaying cash.
        for _, event in actions[actions["cash_payment_date"] == date].iterrows():
            post_payment(event, date)

        scheduled = signal_by_execution.get(date)
        if scheduled is not None:
            signal_date, daily_scores = scheduled
            held = sorted(symbol for symbol, symbol_lots in lots.items() if _shares(symbol_lots) > _EPSILON)
            for symbol in sorted(set(held) - set(daily_scores.index)):
                held_missing_prediction_events.append(
                    {
                        "signal_date": signal_date.date().isoformat(),
                        "symbol": symbol,
                        "action": "held",
                    }
                )
            sell_symbols, buy_symbols = _select_intents(daily_scores, held, policy)

            def market_row(symbol: str) -> pd.Series | None:
                try:
                    row = bars.loc[(date, symbol)]
                except KeyError:
                    return None
                last_known_close[symbol] = float(row["raw_close"])
                return row

            def limit_state(symbol: str, row: pd.Series) -> tuple[bool, bool, bool, str | None]:
                previous_date = sessions[session_number - 1]
                try:
                    previous = bars.loc[(previous_date, symbol)]
                except KeyError:
                    return True, True, False, "missing_previous_price_reference"
                action_today = _event_for(actions, symbol, date) is not None
                return _price_limit_flags(previous, row, policy, corporate_action=action_today)

            volume_used: dict[str, int] = {}

            def append_execution(
                symbol: str,
                direction: str,
                score: float,
                target: int,
                fill: int,
                row: pd.Series | None,
                commission: float,
                slippage: float,
                minimum: bool,
                cash_before: float,
                shares_before: float,
                wide: bool,
                reason: str,
            ) -> None:
                price = None if row is None else float(row["raw_close"])
                raw_volume = None if row is None else float(row["volume"])
                cap = 0 if row is None else int(
                    math.floor(raw_volume * policy.max_daily_volume_participation / policy.lot_size)
                    * policy.lot_size
                )
                notional = 0.0 if price is None else fill * price
                execution_rows.append(
                    {
                        "intent_id": hashlib.sha256(
                            f"{signal_date.isoformat()}|{date.isoformat()}|{symbol}|{direction}".encode("utf-8")
                        ).hexdigest(),
                        "signal_date": signal_date,
                        "execution_date": date,
                        "symbol": symbol,
                        "direction": direction,
                        "signal_score": score,
                        "target_shares": target,
                        "fill_shares": fill,
                        "target_notional": None if price is None else target * price,
                        "fill_notional": notional,
                        "execution_price": price,
                        "commission": commission,
                        "slippage": slippage,
                        "total_cost": commission + slippage,
                        "minimum_commission_applied": minimum,
                        "raw_volume": raw_volume,
                        "volume_cap_shares": cap,
                        "cash_before": cash_before,
                        "cash_after": cash,
                        "shares_before": shares_before,
                        "shares_after": _shares(lots.get(symbol, [])),
                        "wide_limit_tier_proven": wide,
                        "reason": reason,
                    }
                )

            for symbol in sell_symbols:
                symbol_lots = lots.get(symbol, [])
                shares_before = _shares(symbol_lots)
                target = int(math.floor((shares_before + _EPSILON) / policy.lot_size) * policy.lot_size)
                row = market_row(symbol)
                before_cash = cash
                fill = 0
                commission = slippage = 0.0
                minimum = wide = False
                reason = "missing_market_data" if row is None else "filled"
                if row is not None:
                    limit_buy, limit_sell, wide, reference_reason = limit_state(symbol, row)
                    if reference_reason is not None:
                        reason = reference_reason
                    elif limit_sell:
                        reason = "price_limit_sell"
                    elif target <= 0:
                        reason = "below_round_lot"
                    else:
                        eligible = int(
                            math.floor(
                                (_eligible_shares(symbol_lots, session_number, policy.hold_thresh) + _EPSILON)
                                / policy.lot_size
                            )
                            * policy.lot_size
                        )
                        cap = int(
                            math.floor(
                                float(row["volume"])
                                * policy.max_daily_volume_participation
                                / policy.lot_size
                            )
                            * policy.lot_size
                        )
                        available_cap = max(0, cap - volume_used.get(symbol, 0))
                        fill = min(target, eligible, available_cap)
                        if fill <= 0:
                            if eligible <= 0:
                                reason = "hold_threshold_t_plus_one"
                            else:
                                reason = "volume_limit_below_round_lot"
                        else:
                            notional = fill * float(row["raw_close"])
                            commission, slippage, cost, minimum = _costs(notional, policy)
                            cash += notional - cost
                            _consume_eligible(symbol_lots, fill, session_number, policy.hold_thresh)
                            volume_used[symbol] = volume_used.get(symbol, 0) + fill
                            daily_cost += cost
                            daily_notional += notional
                            if fill < target:
                                reason = "partial_hold_threshold" if eligible < target else "partial_volume_limit"
                append_execution(
                    symbol,
                    "sell",
                    float(daily_scores.get(symbol, -np.inf)),
                    target,
                    fill,
                    row,
                    commission,
                    slippage,
                    minimum,
                    before_cash,
                    shares_before,
                    wide,
                    reason,
                )

            retained_market_value = 0.0
            for symbol, symbol_lots in lots.items():
                held_shares = _shares(symbol_lots)
                if held_shares <= _EPSILON:
                    continue
                row = market_row(symbol)
                if row is None:
                    price = last_known_close.get(symbol)
                    if price is None:
                        raise ValueError(
                            f"{symbol} {date.date()}: held position lacks a raw close "
                            "and no prior close is available"
                        )
                    record_carry_forward(symbol, date, held_shares, price)
                else:
                    price = float(row["raw_close"])
                retained_market_value += held_shares * price
            nav_before_buys = cash + total_receivable() + retained_market_value
            target_risky_value = nav_before_buys * policy.risk_degree
            aggregate_buy_value = max(0.0, target_risky_value - retained_market_value)
            target_value = aggregate_buy_value / len(buy_symbols) if buy_symbols else 0.0

            buy_intents: dict[str, dict[str, Any]] = {}
            executable_orders: dict[str, tuple[int, float]] = {}
            for symbol in buy_symbols:
                row = market_row(symbol)
                target = 0
                reason = "missing_market_data" if row is None else "filled"
                wide = False
                available_cap = 0
                if row is not None:
                    price = float(row["raw_close"])
                    target = int(
                        math.floor((target_value / price + _EPSILON) / policy.lot_size)
                        * policy.lot_size
                    )
                    limit_buy, _, wide, reference_reason = limit_state(symbol, row)
                    if reference_reason is not None:
                        reason = reference_reason
                    elif limit_buy:
                        reason = "price_limit_buy"
                    elif target <= 0:
                        reason = "below_round_lot"
                    else:
                        cap = int(
                            math.floor(
                                float(row["volume"])
                                * policy.max_daily_volume_participation
                                / policy.lot_size
                            )
                            * policy.lot_size
                        )
                        available_cap = max(0, cap - volume_used.get(symbol, 0))
                        executable = min(target, available_cap)
                        if executable <= 0:
                            reason = "volume_limit_below_round_lot"
                        else:
                            executable_orders[symbol] = (executable, price)
                buy_intents[symbol] = {
                    "row": row,
                    "target": target,
                    "reason": reason,
                    "wide": wide,
                    "available_cap": available_cap,
                }

            joint_cash_caps = _joint_buy_cash_caps(
                cash,
                executable_orders,
                policy,
                nav_before_buys=nav_before_buys,
                retained_market_value=retained_market_value,
            )
            for symbol in buy_symbols:
                intent = buy_intents[symbol]
                row = intent["row"]
                before_cash = cash
                shares_before = _shares(lots.get(symbol, []))
                target = int(intent["target"])
                fill = 0
                commission = slippage = 0.0
                minimum = False
                wide = bool(intent["wide"])
                reason = str(intent["reason"])
                if symbol in executable_orders:
                    price = float(row["raw_close"])
                    volume_limited = int(executable_orders[symbol][0])
                    fill = min(volume_limited, int(joint_cash_caps.get(symbol, 0)))
                    if fill <= 0:
                        reason = "insufficient_cash"
                    else:
                        notional = fill * price
                        commission, slippage, cost, minimum = _costs(notional, policy)
                        cash -= notional + cost
                        if cash < -_EPSILON:
                            raise RuntimeError("joint buy plan exceeded available cash")
                        cash = max(0.0, cash)
                        lots.setdefault(symbol, []).append(_Lot(float(fill), session_number))
                        volume_used[symbol] = volume_used.get(symbol, 0) + fill
                        daily_cost += cost
                        daily_notional += notional
                        if fill < volume_limited:
                            reason = "partial_joint_cash_budget"
                        elif volume_limited < target:
                            reason = "partial_volume_limit"
                append_execution(
                    symbol,
                    "buy",
                    float(daily_scores[symbol]),
                    target,
                    fill,
                    row,
                    commission,
                    slippage,
                    minimum,
                    before_cash,
                    shares_before,
                    wide,
                    reason,
                )

        # On an ordinary record date, close ownership freezes a future cash
        # right off balance sheet.  It enters NAV only when detached ex-date.
        ordinary_records = actions[
            (actions["record_date"] == date) & (actions["ex_date"] != date)
        ]
        for _, event in ordinary_records.iterrows():
            if float(event["cash_dividend_per_old_share"]) <= _EPSILON:
                continue
            symbol = str(event["symbol"])
            held_shares = _shares(lots.get(symbol, []))
            entitlement = held_shares * float(event["cash_dividend_per_old_share"])
            event_id = str(event["event_id"])
            frozen_entitlements[event_id] = (symbol, held_shares, entitlement)
            action_rows.append(
                {
                    "event_id": event_id,
                    "date": date,
                    "symbol": symbol,
                    "action": "dividend_entitlement",
                    "shares_before": held_shares,
                    "shares_after": held_shares,
                    "entitlement_shares": held_shares,
                    "amount": entitlement,
                    "cash_before": cash,
                    "cash_after": cash,
                    "receivable_before": total_receivable(),
                    "receivable_after": total_receivable(),
                    "commission": 0.0,
                    "turnover": 0.0,
                }
            )

        market_value = 0.0
        for symbol in sorted(lots):
            held_shares = _shares(lots[symbol])
            if held_shares <= _EPSILON:
                continue
            try:
                row = bars.loc[(date, symbol)]
                price = float(row["raw_close"])
                last_known_close[symbol] = price
            except KeyError:
                price = last_known_close.get(symbol)
                if price is None:
                    raise ValueError(
                        f"{symbol} {date.date()}: held position lacks a raw close "
                        "and no prior close is available"
                    )
                record_carry_forward(symbol, date, held_shares, price)
            value = held_shares * price
            eligible = _eligible_shares(lots[symbol], session_number, policy.hold_thresh)
            position_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "shares": held_shares,
                    "eligible_shares": eligible,
                    "locked_shares": held_shares - eligible,
                    "raw_close": price,
                    "market_value": value,
                }
            )
            market_value += value
        receivable = total_receivable()
        nav = cash + receivable + market_value
        if not math.isfinite(nav) or nav <= 0:
            raise RuntimeError(f"non-positive or invalid account NAV on {date.date()}")
        cost_ratio = daily_cost / previous_nav
        turnover = daily_notional / previous_nav
        gross_return = nav / previous_nav - 1.0 + cost_ratio
        report_rows.append(
            {
                "date": date,
                "return": gross_return,
                "cost": cost_ratio,
                "turnover": turnover,
                "account": nav,
                "cash": cash,
                "bench": float(bench.loc[date]),
                "value": market_value,
                "receivable": receivable,
            }
        )
        previous_nav = nav

    report = pd.DataFrame(report_rows).set_index("date")
    positions = pd.DataFrame(position_rows, columns=_POSITION_COLUMNS)
    executions = pd.DataFrame(execution_rows, columns=_EXECUTION_COLUMNS)
    action_ledger = pd.DataFrame(action_rows, columns=_ACTION_LEDGER_COLUMNS)
    if not actions.empty:
        in_scope_records = actions["record_date"].between(sessions[0], sessions[-1]) & (
            actions["cash_dividend_per_old_share"] > _EPSILON
        )
        expected = set(actions.loc[in_scope_records, "event_id"])
        recorded = set(
            action_ledger.loc[action_ledger["action"] == "dividend_entitlement", "event_id"]
        )
        if expected != recorded:
            raise RuntimeError("corporate-action record ledger is incomplete")
    ratios = report["account"] / report["account"].shift(1)
    reconciled = np.isclose(
        ratios.iloc[1:].to_numpy(),
        (1.0 + report["return"] - report["cost"]).iloc[1:].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    ).all()
    symbol_attribution, attribution_max_error = _symbol_attribution(
        report,
        positions,
        executions,
        action_ledger,
        sessions,
        policy.initial_cash,
    )
    symbol_concentration = _symbol_concentration(symbol_attribution)
    target_notional = float(executions["target_notional"].fillna(0.0).sum()) if not executions.empty else 0.0
    fill_notional = float(executions["fill_notional"].sum()) if not executions.empty else 0.0
    filled_count = int((executions["fill_shares"] > 0).sum()) if not executions.empty else 0
    zero_count = int((executions["fill_shares"] <= 0).sum()) if not executions.empty else 0
    summary = {
        "engine": "raw_share_daily_v1",
        "research_only": True,
        "signal_timing": "signal_after_t_executes_next_session_close",
        "selection_policy": "frozen_topk_dropout_no_execution_day_substitution",
        "price_limit_policy": "ohlc_proven_wide_tier_otherwise_10_percent_fail_closed",
        "corporate_action_policy": (
            "record_date_entitlement_off_balance_sheet_"
            "ex_date_receivable_and_share_adjustment_payment_date_cash"
        ),
        "initial_account": policy.initial_cash,
        "final_account": float(report.iloc[-1]["account"]),
        "net_cumulative_return": float(report.iloc[-1]["account"] / policy.initial_cash - 1.0),
        "benchmark_cumulative_return": float((1.0 + report["bench"]).prod() - 1.0),
        "intent_count": int(len(executions)),
        "filled_intent_count": filled_count,
        "zero_fill_intent_count": zero_count,
        "intent_fill_rate": filled_count / len(executions) if len(executions) else 0.0,
        "zero_fill_intent_rate": zero_count / len(executions) if len(executions) else 0.0,
        "target_notional": target_notional,
        "fill_notional": fill_notional,
        "fill_rate": fill_notional / target_notional if target_notional > 0 else 0.0,
        "target_notional_coverage": (
            float(executions["target_notional"].notna().mean()) if not executions.empty else 0.0
        ),
        "commission_total": float(executions["commission"].sum()) if not executions.empty else 0.0,
        "commission_effective_bps": (
            float(executions["commission"].sum() / executions["fill_notional"].sum() * 10000.0)
            if not executions.empty and float(executions["fill_notional"].sum()) > _EPSILON
            else 0.0
        ),
        "slippage_total": float(executions["slippage"].sum()) if not executions.empty else 0.0,
        "cost_total": float(executions["total_cost"].sum()) if not executions.empty else 0.0,
        "corporate_action_count": int(len(actions)),
        "pending_dividend_entitlement_count": int(
            sum(
                amount > _EPSILON and event_id not in receivables
                for event_id, (_, _, amount) in frozen_entitlements.items()
            )
        ),
        "unpaid_receivable": float(report.iloc[-1]["receivable"]),
        "held_missing_prediction_event_count": len(held_missing_prediction_events),
        "held_missing_prediction_events": held_missing_prediction_events,
        "missing_market_data_carry_forward_count": len(mark_carry_forward_events),
        "missing_market_data_carry_forward_events": mark_carry_forward_events,
        "nav_reconciled": bool(reconciled),
        "symbol_attribution_reconciled": True,
        "symbol_attribution_max_abs_reconciliation_error": attribution_max_error,
        "symbol_attribution_total_net_pnl": float(symbol_attribution["net_pnl"].sum()),
        "max_single_etf_abs_contribution_share": symbol_concentration["gross_abs"]["share"],
        "max_single_etf_gross_abs_contribution_share": symbol_concentration["gross_abs"]["share"],
        "max_single_etf_gross_abs_contribution_symbol": symbol_concentration["gross_abs"]["symbol"],
        "max_single_etf_gross_abs_contribution_numerator_cny": (
            symbol_concentration["gross_abs"]["numerator_cny"]
        ),
        "single_etf_gross_abs_contribution_denominator_cny": (
            symbol_concentration["gross_abs"]["denominator_cny"]
        ),
        "max_single_etf_net_abs_contribution_share": symbol_concentration["net_abs"]["share"],
        "max_single_etf_net_abs_contribution_symbol": symbol_concentration["net_abs"]["symbol"],
        "max_single_etf_net_abs_contribution_numerator_cny": (
            symbol_concentration["net_abs"]["numerator_cny"]
        ),
        "single_etf_net_abs_contribution_denominator_cny": (
            symbol_concentration["net_abs"]["denominator_cny"]
        ),
        "symbol_attribution_concentration": symbol_concentration,
        "config": asdict(policy),
    }
    if not reconciled:
        raise RuntimeError("daily report failed NAV reconciliation")
    return RawBacktestResult(
        report,
        positions,
        executions,
        action_ledger,
        summary,
        symbol_attribution,
    )


backtest_raw_shares = run_raw_backtest


__all__ = [
    "RawBacktestConfig",
    "RawBacktestResult",
    "backtest_raw_shares",
    "run_raw_backtest",
]
