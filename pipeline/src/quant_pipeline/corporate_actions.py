"""Auditable corporate-action reconstruction for Sina ETF adjustment data.

Sina's ETF ``hfq.js`` records define an affine adjusted-price mapping::

    adjusted_price = raw_price * f * s + u

If the mapping changes from ``(f0, s0, u0)`` to ``(f1, s1, u1)``, continuity
of the adjusted unit and the ex-right identity

``old_price = share_ratio * ex_price + cash_per_old_share`` imply::

    share_ratio = (f1 * s1) / (f0 * s0)
    cash_per_old_share = (u1 - u0) / (f0 * s0)

The Sina source supplies only ``d``.  This module treats it as the reported
ex-date, never invents record or payment dates, and keeps cash as a receivable
until a verified payment date (or an explicit caller assumption) is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

import pandas as pd


EVENT_SCHEMA_VERSION = "1.0"
SINA_UNKNOWN_DATE_STATUS = "unknown_not_provided_by_sina_hfq"
SINA_UNKNOWN_FRACTIONAL_TREATMENT = "unknown_not_provided_by_sina_hfq"
SINA_EX_DATE_BASIS = "sina_hfq_d_reported_ex_date"
RECEIVABLE_ASSUMPTION = "cash_remains_receivable_until_payment_date_is_verified"
EX_DATE_PAYMENT_ASSUMPTION = "caller_explicitly_assumed_cash_paid_on_ex_date"

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_SYMBOL_PATTERN = re.compile(r"(?:SH|SZ)\d{6}")
_NUMERIC_TOLERANCE = 1e-10

EVENT_COLUMNS = (
    "schema_version",
    "event_id",
    "symbol",
    "ex_date",
    "record_date",
    "payment_date",
    "record_date_status",
    "payment_date_status",
    "ex_date_basis",
    "share_ratio",
    "cash_dividend_per_old_share",
    "event_type",
    "fractional_share_treatment",
    "position_adjustment_ready",
    "cash_settlement_ready",
    "cash_payment_assumption",
    "previous_price_date",
    "previous_raw_close",
    "theoretical_ex_price",
    "observed_ex_close",
    "observed_total_return",
    "qlib_total_return_factor_ratio",
    "old_f",
    "old_s",
    "old_u",
    "new_f",
    "new_s",
    "new_u",
    "source_url",
    "source_sha256",
)

_DATE_COLUMNS = ("ex_date", "record_date", "payment_date", "previous_price_date")
_NUMERIC_COLUMNS = (
    "share_ratio",
    "cash_dividend_per_old_share",
    "previous_raw_close",
    "theoretical_ex_price",
    "observed_ex_close",
    "observed_total_return",
    "qlib_total_return_factor_ratio",
    "old_f",
    "old_s",
    "old_u",
    "new_f",
    "new_s",
    "new_u",
)


@dataclass(frozen=True)
class RealPosition:
    """A cash-account position in real fund shares and settled CNY cash."""

    symbol: str
    shares: float
    cash: float
    cash_receivable: float = 0.0

    def __post_init__(self) -> None:
        symbol = _normalise_symbol(self.symbol)
        shares = _finite_nonnegative("shares", self.shares)
        cash = _finite_nonnegative("cash", self.cash)
        receivable = _finite_nonnegative("cash_receivable", self.cash_receivable)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "shares", shares)
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "cash_receivable", receivable)


@dataclass(frozen=True)
class PositionTransition:
    """Result of applying one event to a real position."""

    position: RealPosition
    shares_before: float
    shares_after: float
    dividend_entitlement: float
    cash_settled: float
    receivable_added: float
    settlement_basis: str


@dataclass(frozen=True)
class QlibPositionReconciliation:
    """Decomposition of Qlib's adjusted units across one action."""

    raw_shares_before: float
    legal_shares_after: float
    qlib_implied_raw_shares_after: float
    dividend_cash_entitlement: float
    qlib_implicit_reinvestment_shares: float
    theoretical_reinvestment_price: float | None


def _normalise_symbol(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("symbol must be a string")
    symbol = value.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"invalid ETF symbol: {value!r}")
    return symbol


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(name: str, value: Any) -> float:
    result = _finite_number(name, value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_source_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source_url must be a non-empty HTTP(S) URL")
    source_url = value.strip()
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be a non-empty HTTP(S) URL")
    return source_url


def _payload_bytes(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    raise TypeError("Sina hfq payload must be str or bytes so its source hash is reproducible")


def parse_sina_hfq_payload(payload: str | bytes) -> pd.DataFrame:
    """Parse an exact Sina ``hfq.js`` response without executing JavaScript.

    All four source fields (``d``, ``f``, ``s`` and ``u``) are required.  A
    payload containing only the stock-style cumulative ``f`` field is not
    sufficient to decompose share changes from cash and is rejected.
    """

    raw = _payload_bytes(payload)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    start = text.find("{")
    if start < 0:
        raise ValueError("invalid Sina hfq payload: JSON object not found")
    try:
        decoded, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid Sina hfq payload JSON") from exc
    if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), list):
        raise ValueError("invalid Sina hfq payload: data must be a list")
    records = decoded["data"]
    if decoded.get("total") is not None:
        try:
            total = int(decoded["total"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Sina hfq payload total") from exc
        if total != len(records):
            raise ValueError("Sina hfq payload total does not match data length")
    if not records:
        raise ValueError("empty Sina hfq payload")

    required = {"d", "f", "s", "u"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"Sina hfq row {index} lacks d/f/s/u; economic decomposition is unavailable")
    frame = pd.DataFrame(records)[["d", "f", "s", "u"]].rename(columns={"d": "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("f", "s", "u"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", "f", "s", "u"]].isna().any().any():
        raise ValueError("Sina hfq payload contains invalid dates or parameters")
    if frame["date"].dt.tz is not None:
        raise ValueError("Sina hfq dates must be timezone-naive")
    frame["date"] = frame["date"].dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("Sina hfq payload contains duplicate dates")
    values = frame[["f", "s", "u"]].to_numpy(dtype=float)
    if not all(math.isfinite(value) for value in values.ravel()):
        raise ValueError("Sina hfq parameters must be finite")
    if (frame[["f", "s"]] <= 0).any().any():
        raise ValueError("Sina hfq f and s parameters must be positive")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def _normalise_raw_prices(
    raw_prices: pd.DataFrame,
    symbol: str,
    *,
    date_column: str,
    close_column: str,
) -> pd.DataFrame:
    if not isinstance(raw_prices, pd.DataFrame):
        raise TypeError("raw_prices must be a pandas DataFrame")
    missing = {date_column, close_column} - set(raw_prices.columns)
    if missing:
        raise ValueError(f"raw price table lacks columns: {sorted(missing)}")
    if "symbol" in raw_prices.columns:
        symbols = {_normalise_symbol(value) for value in raw_prices["symbol"].dropna().unique()}
        if symbols != {symbol}:
            raise ValueError(f"raw price table symbols do not match {symbol}: {sorted(symbols)}")
    frame = raw_prices[[date_column, close_column]].copy()
    frame.columns = ["date", "raw_close"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["raw_close"] = pd.to_numeric(frame["raw_close"], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("raw price table contains missing or invalid dates/prices")
    if frame["date"].dt.tz is not None:
        raise ValueError("raw price dates must be timezone-naive")
    frame["date"] = frame["date"].dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("raw price table contains duplicate dates")
    values = frame["raw_close"].to_numpy(dtype=float)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("raw closing prices must be finite and positive")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def _event_type(share_ratio: float, cash_dividend: float) -> str:
    has_share_change = not math.isclose(share_ratio, 1.0, rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE)
    has_cash = cash_dividend > _NUMERIC_TOLERANCE
    if has_share_change and has_cash:
        return "share_change_and_cash"
    if has_share_change:
        return "share_change_only"
    return "cash_only"


def _event_id(symbol: str, ex_date: pd.Timestamp, source_hash: str) -> str:
    identity = f"{EVENT_SCHEMA_VERSION}|{symbol}|{ex_date.date().isoformat()}|{source_hash}"
    return sha256(identity.encode("ascii")).hexdigest()


def _empty_event_table() -> pd.DataFrame:
    frame = pd.DataFrame(columns=EVENT_COLUMNS)
    for column in _DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def build_corporate_action_events(
    symbol: str,
    hfq_payload: str | bytes,
    raw_prices: pd.DataFrame,
    *,
    source_url: str,
    date_column: str = "date",
    close_column: str = "raw_close",
) -> pd.DataFrame:
    """Build a validated event table from exact Sina bytes and raw closes.

    The previous trading close is used only to derive the theoretical ex-price
    and the total-return factor.  The observed ex-date close is retained as an
    audit value; it is not forced to equal the theoretical price because normal
    market return occurs on the ex-date.
    """

    canonical_symbol = _normalise_symbol(symbol)
    canonical_url = _validate_source_url(source_url)
    source_bytes = _payload_bytes(hfq_payload)
    source_hash = sha256(source_bytes).hexdigest()
    factors = parse_sina_hfq_payload(source_bytes)
    prices = _normalise_raw_prices(
        raw_prices,
        canonical_symbol,
        date_column=date_column,
        close_column=close_column,
    )

    rows: list[dict[str, Any]] = []
    for index in range(1, len(factors)):
        old = factors.iloc[index - 1]
        new = factors.iloc[index]
        old_scale = float(old["f"] * old["s"])
        new_scale = float(new["f"] * new["s"])
        if math.isclose(old_scale, new_scale, rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE) and math.isclose(
            float(old["u"]), float(new["u"]), rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE
        ):
            continue

        ex_date = pd.Timestamp(new["date"])
        prior = prices.loc[prices["date"] < ex_date]
        observed = prices.loc[prices["date"] == ex_date, "raw_close"]
        if prior.empty:
            raise ValueError(f"{canonical_symbol} {ex_date.date()}: previous raw close is unavailable")
        if len(observed) != 1:
            raise ValueError(f"{canonical_symbol} {ex_date.date()}: ex-date raw close is unavailable")
        previous = prior.iloc[-1]
        previous_close = float(previous["raw_close"])
        observed_close = float(observed.iloc[0])

        share_ratio = new_scale / old_scale
        cash_dividend = (float(new["u"]) - float(old["u"])) / old_scale
        if cash_dividend < -_NUMERIC_TOLERANCE:
            raise ValueError(
                f"{canonical_symbol} {ex_date.date()}: negative inferred cash is not a supported dividend"
            )
        if abs(cash_dividend) <= _NUMERIC_TOLERANCE:
            cash_dividend = 0.0
        theoretical_ex_price = (previous_close - cash_dividend) / share_ratio
        if not math.isfinite(theoretical_ex_price) or theoretical_ex_price <= 0:
            raise ValueError(f"{canonical_symbol} {ex_date.date()}: non-positive theoretical ex-price")
        qlib_factor_ratio = previous_close / theoretical_ex_price
        observed_total_return = (share_ratio * observed_close + cash_dividend) / previous_close - 1.0
        cash_event = cash_dividend > _NUMERIC_TOLERANCE

        rows.append(
            {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_id": _event_id(canonical_symbol, ex_date, source_hash),
                "symbol": canonical_symbol,
                "ex_date": ex_date,
                "record_date": pd.NaT,
                "payment_date": pd.NaT,
                "record_date_status": SINA_UNKNOWN_DATE_STATUS,
                "payment_date_status": SINA_UNKNOWN_DATE_STATUS,
                "ex_date_basis": SINA_EX_DATE_BASIS,
                "share_ratio": share_ratio,
                "cash_dividend_per_old_share": cash_dividend,
                "event_type": _event_type(share_ratio, cash_dividend),
                "fractional_share_treatment": SINA_UNKNOWN_FRACTIONAL_TREATMENT,
                "position_adjustment_ready": math.isclose(
                    share_ratio, 1.0, rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE
                ),
                "cash_settlement_ready": not cash_event,
                "cash_payment_assumption": RECEIVABLE_ASSUMPTION if cash_event else None,
                "previous_price_date": pd.Timestamp(previous["date"]),
                "previous_raw_close": previous_close,
                "theoretical_ex_price": theoretical_ex_price,
                "observed_ex_close": observed_close,
                "observed_total_return": observed_total_return,
                "qlib_total_return_factor_ratio": qlib_factor_ratio,
                "old_f": float(old["f"]),
                "old_s": float(old["s"]),
                "old_u": float(old["u"]),
                "new_f": float(new["f"]),
                "new_s": float(new["s"]),
                "new_u": float(new["u"]),
                "source_url": canonical_url,
                "source_sha256": source_hash,
            }
        )
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS) if rows else _empty_event_table()
    return validate_corporate_action_events(frame)


def _is_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=_NUMERIC_TOLERANCE)


def validate_corporate_action_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate event schema, provenance, algebra, and fail-closed date flags."""

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")
    missing = set(EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"corporate-action table lacks columns: {sorted(missing)}")
    frame = events.loc[:, EVENT_COLUMNS].copy()
    for column in _DATE_COLUMNS:
        original_missing = frame[column].isna()
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if (parsed.isna() & ~original_missing).any():
            raise ValueError(f"{column} contains invalid dates")
        non_null = parsed.dropna()
        if not non_null.empty and non_null.dt.tz is not None:
            raise ValueError(f"{column} must be timezone-naive")
        frame[column] = parsed.dt.normalize()
    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty:
        return frame
    if frame["ex_date"].isna().any() or frame["previous_price_date"].isna().any():
        raise ValueError("ex_date and previous_price_date are required")
    if frame[list(_NUMERIC_COLUMNS)].isna().any().any():
        raise ValueError("corporate-action table contains missing numeric values")
    frame["symbol"] = frame["symbol"].map(_normalise_symbol)
    if frame.duplicated(["symbol", "ex_date"]).any():
        raise ValueError("corporate-action table contains duplicate symbol/ex-date events")
    if frame["event_id"].duplicated().any():
        raise ValueError("corporate-action table contains duplicate event_id values")

    validated_rows: list[pd.Series] = []
    for position, row in frame.iterrows():
        context = f"event row {position}"
        symbol = row["symbol"]
        source_url = _validate_source_url(row["source_url"])
        source_hash = row["source_sha256"]
        if not isinstance(source_hash, str) or not _HASH_PATTERN.fullmatch(source_hash):
            raise ValueError(f"{context}: source_sha256 must be a lowercase SHA-256")
        if row["schema_version"] != EVENT_SCHEMA_VERSION:
            raise ValueError(f"{context}: unsupported schema_version")
        expected_id = _event_id(symbol, row["ex_date"], source_hash)
        if row["event_id"] != expected_id:
            raise ValueError(f"{context}: event_id does not match provenance")
        if row["ex_date_basis"] != SINA_EX_DATE_BASIS:
            raise ValueError(f"{context}: invalid ex_date_basis")
        if row["previous_price_date"] >= row["ex_date"]:
            raise ValueError(f"{context}: previous_price_date must precede ex_date")
        if pd.notna(row["record_date"]) or row["record_date_status"] != SINA_UNKNOWN_DATE_STATUS:
            raise ValueError(f"{context}: Sina hfq does not verify a record date")
        if pd.notna(row["payment_date"]) or row["payment_date_status"] != SINA_UNKNOWN_DATE_STATUS:
            raise ValueError(f"{context}: Sina hfq does not verify a payment date")
        if row["fractional_share_treatment"] != SINA_UNKNOWN_FRACTIONAL_TREATMENT:
            raise ValueError(f"{context}: Sina hfq does not verify fractional-share treatment")

        numbers = {column: float(row[column]) for column in _NUMERIC_COLUMNS}
        if not all(math.isfinite(value) for value in numbers.values()):
            raise ValueError(f"{context}: numeric values must be finite")
        if min(numbers["old_f"], numbers["old_s"], numbers["new_f"], numbers["new_s"]) <= 0:
            raise ValueError(f"{context}: f and s parameters must be positive")
        if min(
            numbers["share_ratio"],
            numbers["previous_raw_close"],
            numbers["theoretical_ex_price"],
            numbers["observed_ex_close"],
            numbers["qlib_total_return_factor_ratio"],
        ) <= 0:
            raise ValueError(f"{context}: ratios and raw prices must be positive")
        if numbers["cash_dividend_per_old_share"] < 0:
            raise ValueError(f"{context}: cash dividend must be non-negative")

        old_scale = numbers["old_f"] * numbers["old_s"]
        new_scale = numbers["new_f"] * numbers["new_s"]
        expected_ratio = new_scale / old_scale
        expected_cash = (numbers["new_u"] - numbers["old_u"]) / old_scale
        expected_ex_price = (
            numbers["previous_raw_close"] - numbers["cash_dividend_per_old_share"]
        ) / numbers["share_ratio"]
        expected_qlib_ratio = numbers["previous_raw_close"] / numbers["theoretical_ex_price"]
        expected_return = (
            numbers["share_ratio"] * numbers["observed_ex_close"]
            + numbers["cash_dividend_per_old_share"]
        ) / numbers["previous_raw_close"] - 1.0
        checks = {
            "share_ratio": (numbers["share_ratio"], expected_ratio),
            "cash_dividend_per_old_share": (numbers["cash_dividend_per_old_share"], expected_cash),
            "theoretical_ex_price": (numbers["theoretical_ex_price"], expected_ex_price),
            "qlib_total_return_factor_ratio": (
                numbers["qlib_total_return_factor_ratio"],
                expected_qlib_ratio,
            ),
            "observed_total_return": (numbers["observed_total_return"], expected_return),
        }
        for name, (actual, expected) in checks.items():
            if not _is_close(actual, expected):
                raise ValueError(f"{context}: {name} violates the affine corporate-action identity")
        expected_type = _event_type(numbers["share_ratio"], numbers["cash_dividend_per_old_share"])
        if row["event_type"] != expected_type:
            raise ValueError(f"{context}: event_type does not match event economics")
        expected_position_ready = math.isclose(
            numbers["share_ratio"], 1.0, rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE
        )
        if bool(row["position_adjustment_ready"]) is not expected_position_ready:
            raise ValueError(
                f"{context}: position adjustment readiness does not match unknown "
                "fractional-share treatment"
            )
        cash_event = numbers["cash_dividend_per_old_share"] > _NUMERIC_TOLERANCE
        if bool(row["cash_settlement_ready"]) is cash_event:
            raise ValueError(f"{context}: unknown payment date must fail closed for cash settlement")
        expected_assumption = RECEIVABLE_ASSUMPTION if cash_event else None
        assumption = row["cash_payment_assumption"]
        if pd.isna(assumption):
            assumption = None
        if assumption != expected_assumption:
            raise ValueError(f"{context}: invalid cash payment assumption")
        row["symbol"] = symbol
        row["source_url"] = source_url
        validated_rows.append(row)
    return pd.DataFrame(validated_rows, columns=EVENT_COLUMNS).sort_values(
        ["symbol", "ex_date"], kind="stable"
    ).reset_index(drop=True)


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_records(events: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_scalar(value) for key, value in record.items()} for record in events.to_dict("records")]


def save_corporate_action_events(events: pd.DataFrame, path: str | Path) -> Path:
    """Write a validated JSON or Parquet table based on the file suffix."""

    destination = Path(path)
    validated = validate_corporate_action_events(events)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".json":
        records = _json_records(validated)
        canonical = json.dumps(records, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        envelope = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "events_sha256": sha256(canonical.encode("utf-8")).hexdigest(),
            "events": records,
        }
        destination.write_text(json.dumps(envelope, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    elif suffix in {".parquet", ".pq"}:
        validated.to_parquet(destination, index=False)
    else:
        raise ValueError("corporate-action table path must end in .json, .parquet, or .pq")
    return destination


def load_corporate_action_events(path: str | Path) -> pd.DataFrame:
    """Read and validate a JSON or Parquet corporate-action table."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        decoded = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict) or decoded.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise ValueError("invalid corporate-action JSON envelope")
        records = decoded.get("events")
        if not isinstance(records, list):
            raise ValueError("corporate-action JSON events must be a list")
        canonical = json.dumps(records, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        expected_hash = sha256(canonical.encode("utf-8")).hexdigest()
        if decoded.get("events_sha256") != expected_hash:
            raise ValueError("corporate-action JSON events_sha256 mismatch")
        frame = pd.DataFrame(records, columns=EVENT_COLUMNS) if records else _empty_event_table()
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise ValueError("corporate-action table path must end in .json, .parquet, or .pq")
    return validate_corporate_action_events(frame)


def _validated_event(event: Mapping[str, Any] | pd.Series) -> pd.Series:
    if isinstance(event, pd.Series):
        record = event.to_dict()
    elif isinstance(event, Mapping):
        record = dict(event)
    else:
        raise TypeError("event must be a mapping or pandas Series")
    return validate_corporate_action_events(pd.DataFrame([record])).iloc[0]


def apply_corporate_action(
    position: RealPosition,
    event: Mapping[str, Any] | pd.Series,
    *,
    assume_cash_paid_on_ex_date: bool = False,
    lot_size: int | None = None,
) -> PositionTransition:
    """Apply legal share/cash entitlements without silently timing cash.

    By default a dividend becomes non-spendable ``cash_receivable`` because
    Sina does not provide a payment date.  The opt-in flag is deliberately
    verbose: using it settles cash immediately and records that assumption in
    the returned transition.
    """

    if not isinstance(position, RealPosition):
        raise TypeError("position must be a RealPosition")
    if not isinstance(assume_cash_paid_on_ex_date, bool):
        raise TypeError("assume_cash_paid_on_ex_date must be bool")
    row = _validated_event(event)
    if position.symbol != row["symbol"]:
        raise ValueError("position and corporate-action symbols do not match")
    shares_before = position.shares
    shares_after = shares_before * float(row["share_ratio"])
    if not bool(row["position_adjustment_ready"]):
        if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size < 1:
            raise ValueError(
                "corporate-action position adjustment needs a positive lot_size because "
                "fractional-share treatment is unverified"
            )
        nearest_round_lot = round(shares_after / lot_size) * lot_size
        if not math.isclose(
            shares_after,
            nearest_round_lot,
            rel_tol=0.0,
            abs_tol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError(
                "corporate-action position adjustment is not ready because fractional-share "
                "treatment is unverified and the resulting position is not a round lot"
            )
    entitlement = shares_before * float(row["cash_dividend_per_old_share"])
    if assume_cash_paid_on_ex_date:
        cash_settled = entitlement
        receivable_added = 0.0
        basis = EX_DATE_PAYMENT_ASSUMPTION if entitlement else "not_applicable"
    else:
        cash_settled = 0.0
        receivable_added = entitlement
        basis = RECEIVABLE_ASSUMPTION if entitlement else "not_applicable"
    updated = RealPosition(
        symbol=position.symbol,
        shares=shares_after,
        cash=position.cash + cash_settled,
        cash_receivable=position.cash_receivable + receivable_added,
    )
    return PositionTransition(
        position=updated,
        shares_before=shares_before,
        shares_after=shares_after,
        dividend_entitlement=entitlement,
        cash_settled=cash_settled,
        receivable_added=receivable_added,
        settlement_basis=basis,
    )


def qlib_adjusted_amount_to_raw_shares(adjusted_amount: Real, qlib_factor: Real) -> float:
    """Convert Qlib amount units to its raw-share equivalent at one instant.

    With this repository's convention, ``raw_price = adjusted_price/factor``;
    value conservation therefore gives ``raw_shares = amount*factor``.  A
    total-return factor includes Qlib's implicit dividend reinvestment and is
    not, by itself, a legal no-reinvestment share count across cash events.
    """

    amount = _finite_nonnegative("adjusted_amount", adjusted_amount)
    factor = _finite_number("qlib_factor", qlib_factor)
    if factor <= 0:
        raise ValueError("qlib_factor must be positive")
    return amount * factor


def raw_shares_to_qlib_adjusted_amount(raw_shares: Real, qlib_factor: Real) -> float:
    """Inverse of :func:`qlib_adjusted_amount_to_raw_shares` at one instant."""

    shares = _finite_nonnegative("raw_shares", raw_shares)
    factor = _finite_number("qlib_factor", qlib_factor)
    if factor <= 0:
        raise ValueError("qlib_factor must be positive")
    return shares / factor


def reconcile_qlib_adjusted_position(
    adjusted_amount: Real,
    factor_before: Real,
    factor_after: Real,
    event: Mapping[str, Any] | pd.Series,
) -> QlibPositionReconciliation:
    """Separate legal entitlements from Qlib's implicit reinvestment.

    ``adjusted_amount`` is assumed unchanged across the event.  The function
    verifies that the supplied factor jump matches the event table before
    reporting the extra shares Qlib creates by reinvesting the cash dividend at
    the theoretical ex-price.
    """

    amount = _finite_nonnegative("adjusted_amount", adjusted_amount)
    before = _finite_number("factor_before", factor_before)
    after = _finite_number("factor_after", factor_after)
    if before <= 0 or after <= 0:
        raise ValueError("Qlib factors must be positive")
    row = _validated_event(event)
    expected_factor_ratio = float(row["qlib_total_return_factor_ratio"])
    if not _is_close(after / before, expected_factor_ratio):
        raise ValueError("Qlib factor change does not match the corporate-action event")

    raw_before = amount * before
    legal_after = raw_before * float(row["share_ratio"])
    qlib_after = amount * after
    cash_entitlement = raw_before * float(row["cash_dividend_per_old_share"])
    implicit_shares = qlib_after - legal_after
    if abs(implicit_shares) <= _NUMERIC_TOLERANCE:
        implicit_shares = 0.0
    if implicit_shares < 0:
        raise ValueError("event implies negative Qlib dividend reinvestment shares")
    reinvestment_price = cash_entitlement / implicit_shares if implicit_shares else None
    if reinvestment_price is not None and not _is_close(
        reinvestment_price, float(row["theoretical_ex_price"])
    ):
        raise ValueError("Qlib implicit reinvestment price does not match the theoretical ex-price")
    return QlibPositionReconciliation(
        raw_shares_before=raw_before,
        legal_shares_after=legal_after,
        qlib_implied_raw_shares_after=qlib_after,
        dividend_cash_entitlement=cash_entitlement,
        qlib_implicit_reinvestment_shares=implicit_shares,
        theoretical_reinvestment_price=reinvestment_price,
    )


__all__ = [
    "EVENT_COLUMNS",
    "EVENT_SCHEMA_VERSION",
    "EX_DATE_PAYMENT_ASSUMPTION",
    "PositionTransition",
    "QlibPositionReconciliation",
    "RECEIVABLE_ASSUMPTION",
    "RealPosition",
    "apply_corporate_action",
    "build_corporate_action_events",
    "load_corporate_action_events",
    "parse_sina_hfq_payload",
    "qlib_adjusted_amount_to_raw_shares",
    "raw_shares_to_qlib_adjusted_amount",
    "reconcile_qlib_adjusted_position",
    "save_corporate_action_events",
    "validate_corporate_action_events",
]
