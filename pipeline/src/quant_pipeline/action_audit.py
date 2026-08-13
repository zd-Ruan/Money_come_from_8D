"""Fail-closed reconciliation of Qlib factors and Eastmoney fund actions.

The factor table alone does not identify the economics of a factor jump.  A
same-symbol, same-ex-date Eastmoney event therefore establishes only event
coverage.  When raw closes are supplied, this module additionally verifies the
standard ex-right adjustment identity::

    F1 / F0 = C0 * R / (C0 - D)

where ``R`` is new shares per old share and ``D`` is cash per old share.  The
equivalent return identity uses both closes::

    (C1 * F1) / (C0 * F0) = C1 / ((C0 - D) / R)

This is an adjustment-factor identity, not a claim that it exactly equals the
legal holding return ``(R * C1 + D) / C0``.  Both quantities are reported so
the distinction remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FACTOR_COLUMNS = ("date", "symbol", "factor")
ACTION_COLUMNS = (
    "record_date",
    "ex_date",
    "cash_payment_date",
    "cash_dividend_per_old_share",
    "share_ratio",
    "fractional_share_treatment",
    "source_sha256",
)
DETAIL_COLUMNS = (
    "symbol",
    "ex_date",
    "audit_passed",
    "status",
    "issue_code",
    "verification_level",
    "economic_claim",
    "factor_before_date",
    "factor_before",
    "factor_after",
    "factor_ratio",
    "record_date",
    "cash_payment_date",
    "cash_dividend_per_old_share",
    "share_ratio",
    "source_sha256",
    "raw_close_before",
    "raw_close_ex_date",
    "theoretical_ex_close",
    "action_implied_factor_ratio",
    "factor_ratio_absolute_error",
    "qlib_adjusted_return_multiplier",
    "action_ex_right_return_multiplier",
    "legal_holding_return_multiplier",
    "legal_vs_qlib_return_gap",
)
SUMMARY_COLUMNS = (
    "audit_passed",
    "material_factor_jump_count",
    "corporate_action_count",
    "matched_event_count",
    "missing_action_count",
    "extra_action_count",
    "identity_mismatch_count",
    "presence_only_count",
    "identity_verified_count",
    "raw_price_identity_requested",
    "factor_start_date",
    "factor_end_date",
    "rtol",
    "atol",
    "economic_scope",
)

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_SYMBOL_PATTERN = re.compile(r"(?:SH|SZ)\d{6}")


@dataclass(frozen=True)
class CorporateActionAuditResult:
    """CSV/Parquet-friendly reconciliation output."""

    summary: pd.DataFrame
    details: pd.DataFrame
    factor_changes: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(self.summary.iloc[0]["audit_passed"])


class CorporateActionAuditError(ValueError):
    """Raised on a coverage or economic mismatch while preserving diagnostics."""

    def __init__(self, result: CorporateActionAuditResult):
        self.result = result
        summary = result.summary.iloc[0]
        message = (
            "corporate-action audit failed: "
            f"missing={int(summary['missing_action_count'])}, "
            f"extra={int(summary['extra_action_count'])}, "
            f"identity_mismatch={int(summary['identity_mismatch_count'])}"
        )
        super().__init__(message)


def _validate_tolerances(rtol: float, atol: float) -> tuple[float, float]:
    values = []
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a finite non-negative number")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        values.append(number)
    return values[0], values[1]


def _read_csv_or_frame(value: pd.DataFrame | str | Path, *, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() != ".csv":
            raise ValueError(f"{name} path must be a normalized CSV")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} CSV is missing or unsafe: {path}")
        return pd.read_csv(path)
    raise TypeError(f"{name} must be a pandas DataFrame or CSV path")


def _parse_dates(series: pd.Series, *, name: str, allow_missing: bool = False) -> pd.Series:
    missing_input = series.isna()
    try:
        parsed = pd.to_datetime(series, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid dates") from exc
    invalid = parsed.isna() & ~missing_input if allow_missing else parsed.isna()
    if invalid.any():
        raise ValueError(f"{name} contains missing or invalid dates")
    non_null = parsed.dropna()
    if not non_null.empty and non_null.dt.tz is not None:
        raise ValueError(f"{name} must be timezone-naive")
    if not non_null.empty and (non_null != non_null.dt.normalize()).any():
        raise ValueError(f"{name} must contain normalized dates without times")
    return parsed.dt.normalize()


def _normalise_symbols(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} contains missing symbols")
    symbols = series.astype(str).str.strip().str.upper()
    invalid = ~symbols.str.fullmatch(_SYMBOL_PATTERN)
    if invalid.any():
        examples = sorted(symbols[invalid].unique())[:3]
        raise ValueError(f"{name} contains invalid mainland ETF symbols: {examples}")
    return symbols


def _normalise_calendar(
    calendar: pd.DataFrame | pd.Series | pd.Index | Sequence[Any] | str | Path,
) -> pd.DatetimeIndex:
    if isinstance(calendar, (str, Path)):
        frame = _read_csv_or_frame(calendar, name="calendar")
        if "date" not in frame:
            raise ValueError("calendar CSV must contain date")
        values = frame["date"]
    elif isinstance(calendar, pd.DataFrame):
        if "date" not in calendar:
            raise ValueError("calendar DataFrame must contain date")
        values = calendar["date"]
    elif isinstance(calendar, (pd.Series, pd.Index)):
        values = pd.Series(calendar)
    elif isinstance(calendar, Sequence) and not isinstance(calendar, (str, bytes)):
        values = pd.Series(list(calendar))
    else:
        raise TypeError("calendar must be a date sequence, DataFrame, or CSV path")
    parsed = _parse_dates(values, name="calendar")
    if parsed.empty:
        raise ValueError("calendar must not be empty")
    if parsed.duplicated().any():
        raise ValueError("calendar contains duplicate sessions")
    if not parsed.is_monotonic_increasing:
        raise ValueError("calendar must be strictly increasing")
    return pd.DatetimeIndex(parsed)


def _normalise_factors(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    frame = _read_csv_or_frame(value, name="factor table")
    missing = set(FACTOR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"factor table lacks columns: {sorted(missing)}")
    frame = frame.loc[:, FACTOR_COLUMNS].copy()
    if frame.empty:
        raise ValueError("factor table must not be empty")
    frame["date"] = _parse_dates(frame["date"], name="factor date")
    frame["symbol"] = _normalise_symbols(frame["symbol"], name="factor table")
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    values = frame["factor"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("factor values must be finite and positive")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("factor table contains duplicate symbol/date rows")
    return frame.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def _assert_factor_calendar_coverage(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    raw_sessions: Mapping[str, set[pd.Timestamp]] | None = None,
) -> None:
    """Require complete sessions inside each symbol's observed active span.

    The factor schema has no listing, delisting, or suspension evidence, so the
    first and last observed rows define the only defensible active interval.
    When the raw-price table is supplied, sessions missing from it are treated
    as suspension evidence and exempted from the factor-coverage requirement;
    every other missing session fails closed.  Dates outside the interval are
    not inferred to be active.
    """

    outside = frame.loc[~frame["date"].isin(sessions), ["symbol", "date"]]
    if not outside.empty:
        row = outside.iloc[0]
        raise ValueError(f"factor date is outside calendar: {row['symbol']} {row['date'].date()}")

    for symbol, symbol_frame in frame.groupby("symbol", sort=False):
        first_date = symbol_frame["date"].iloc[0]
        last_date = symbol_frame["date"].iloc[-1]
        expected = sessions[(sessions >= first_date) & (sessions <= last_date)]
        missing = expected.difference(pd.DatetimeIndex(symbol_frame["date"]))
        if raw_sessions is not None:
            present = raw_sessions.get(str(symbol), set())
            exempt = expected.difference(pd.DatetimeIndex(sorted(present)))
            missing = missing.difference(exempt)
        if not missing.empty:
            raise ValueError(
                "factor table is missing a calendar session within the observed "
                f"active span: {symbol} {missing[0].date()}"
            )


def detect_material_factor_changes(
    factors: pd.DataFrame | str | Path,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Return changes whose consecutive factor values are not numerically close."""

    relative, absolute = _validate_tolerances(rtol, atol)
    frame = _normalise_factors(factors)
    previous_date = frame.groupby("symbol", sort=False)["date"].shift(1)
    previous_factor = frame.groupby("symbol", sort=False)["factor"].shift(1)
    material = previous_factor.notna() & ~np.isclose(
        frame["factor"].to_numpy(dtype=float),
        previous_factor.fillna(frame["factor"]).to_numpy(dtype=float),
        rtol=relative,
        atol=absolute,
    )
    result = pd.DataFrame(
        {
            "symbol": frame.loc[material, "symbol"],
            "ex_date": frame.loc[material, "date"],
            "factor_before_date": previous_date.loc[material],
            "factor_before": previous_factor.loc[material],
            "factor_after": frame.loc[material, "factor"],
        }
    )
    result["factor_ratio"] = result["factor_after"] / result["factor_before"]
    return result.reset_index(drop=True)


def _normalise_actions(
    value: pd.DataFrame | str | Path,
    sessions: pd.DatetimeIndex,
    *,
    rtol: float,
    atol: float,
) -> pd.DataFrame:
    frame = _read_csv_or_frame(value, name="corporate-action table")
    required = {"symbol", *ACTION_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"corporate-action table lacks columns: {sorted(missing)}")
    frame = frame.loc[:, ["symbol", *ACTION_COLUMNS]].copy()
    frame["symbol"] = _normalise_symbols(frame["symbol"], name="corporate-action table")
    frame["ex_date"] = _parse_dates(frame["ex_date"], name="ex_date")
    for column in ("record_date", "cash_payment_date"):
        frame[column] = _parse_dates(frame[column], name=column, allow_missing=True)
    for column in ("cash_dividend_per_old_share", "share_ratio"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = frame[["cash_dividend_per_old_share", "share_ratio"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("corporate-action economics must be finite")
    if (frame["cash_dividend_per_old_share"] < 0).any():
        raise ValueError("cash dividends must be non-negative")
    if (frame["share_ratio"] <= 0).any():
        raise ValueError("share ratios must be positive")
    no_cash = np.isclose(frame["cash_dividend_per_old_share"], 0.0, rtol=rtol, atol=atol)
    no_share_change = np.isclose(frame["share_ratio"], 1.0, rtol=rtol, atol=atol)
    treatments = frame["fractional_share_treatment"]
    if treatments.isna().any() or treatments.astype(str).str.strip().eq("").any():
        raise ValueError("fractional_share_treatment must be explicit")
    frame["fractional_share_treatment"] = treatments.astype(str).str.strip()
    if (
        no_share_change
        & frame["fractional_share_treatment"].ne("not_applicable_no_share_change")
    ).any():
        raise ValueError("cash-only events must mark fractional-share treatment not applicable")
    unknown_treatments = {
        "unknown_not_provided_by_eastmoney_archive",
        "unknown_not_provided_by_sina_hfq",
    }
    if (
        ~no_share_change
        & ~frame["fractional_share_treatment"].isin(unknown_treatments)
    ).any():
        raise ValueError("unsupported fractional-share treatment declaration")
    if (no_cash & no_share_change).any():
        raise ValueError("corporate-action table contains an economically empty event")
    cash_event = frame["cash_dividend_per_old_share"] > 0.0
    incomplete_cash_dates = cash_event & (
        frame["record_date"].isna() | frame["cash_payment_date"].isna()
    )
    if incomplete_cash_dates.any():
        row = frame.loc[incomplete_cash_dates].iloc[0]
        raise ValueError(
            f"cash event dates are incomplete for {row['symbol']} {row['ex_date'].date()}"
        )
    if (frame["record_date"].notna() & (frame["record_date"] > frame["ex_date"])).any():
        raise ValueError("record_date must not follow ex_date")
    if (
        frame["cash_payment_date"].notna()
        & (frame["cash_payment_date"] < frame["ex_date"])
    ).any():
        raise ValueError("cash_payment_date must not precede ex_date")
    invalid_hash = ~frame["source_sha256"].map(
        lambda value: isinstance(value, str) and _HASH_PATTERN.fullmatch(value) is not None
    )
    if invalid_hash.any():
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    if frame.duplicated(["symbol", "ex_date"]).any():
        raise ValueError("corporate-action table contains duplicate symbol/ex_date events")
    if (~frame["ex_date"].isin(sessions)).any():
        row = frame.loc[~frame["ex_date"].isin(sessions)].iloc[0]
        raise ValueError(
            f"ex_date is outside calendar for {row['symbol']}: {row['ex_date'].date()}"
        )
    for column in ("record_date", "cash_payment_date"):
        in_audit_span = frame[column].notna() & frame[column].between(sessions[0], sessions[-1])
        non_session = in_audit_span & ~frame[column].isin(sessions)
        if non_session.any():
            row = frame.loc[non_session].iloc[0]
            raise ValueError(
                f"{column} is not a trading session for {row['symbol']}: {row[column].date()}"
            )
    return frame.sort_values(["symbol", "ex_date"], kind="stable").reset_index(drop=True)


def _normalise_raw_prices(
    value: pd.DataFrame | str | Path,
    sessions: pd.DatetimeIndex,
    *,
    close_column: str,
) -> pd.DataFrame:
    frame = _read_csv_or_frame(value, name="raw-price table")
    required = {"date", "symbol", close_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"raw-price table lacks columns: {sorted(missing)}")
    frame = frame.loc[:, ["date", "symbol", close_column]].copy()
    frame.columns = ["date", "symbol", "raw_close"]
    frame["date"] = _parse_dates(frame["date"], name="raw-price date")
    frame["symbol"] = _normalise_symbols(frame["symbol"], name="raw-price table")
    frame["raw_close"] = pd.to_numeric(frame["raw_close"], errors="coerce")
    values = frame["raw_close"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("raw closes must be finite and positive")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("raw-price table contains duplicate symbol/date rows")
    if (~frame["date"].isin(sessions)).any():
        raise ValueError("raw-price table contains dates outside calendar")
    return frame.set_index(["symbol", "date"]).sort_index()


def _empty_details() -> pd.DataFrame:
    frame = pd.DataFrame(columns=DETAIL_COLUMNS)
    for column in ("ex_date", "factor_before_date", "record_date", "cash_payment_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def audit_corporate_actions(
    factors: pd.DataFrame | str | Path,
    corporate_actions: pd.DataFrame | str | Path,
    calendar: pd.DataFrame | pd.Series | pd.Index | Sequence[Any] | str | Path,
    *,
    raw_prices: pd.DataFrame | str | Path | None = None,
    raw_close_column: str = "raw_close",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    identity_rtol: float = 1e-3,
    identity_atol: float = 1e-4,
    raise_on_failure: bool = True,
) -> CorporateActionAuditResult:
    """Reconcile all material factor jumps one-to-one with dated actions.

    Missing actions, extra actions, and raw-close identity mismatches fail the
    audit.  With no raw prices, matched rows deliberately claim only event
    presence because a factor and event row alone cannot prove the multiplier.
    The factor table and the corporate-action archive come from different
    vendors, so their identity comparison uses a wider, cross-vendor tolerance
    (``identity_rtol``/``identity_atol``) that absorbs published dividend
    rounding while still failing on economically material disagreement.
    Set ``raise_on_failure=False`` only when diagnostics must be persisted.
    """

    relative, absolute = _validate_tolerances(rtol, atol)
    identity_relative, identity_absolute = _validate_tolerances(identity_rtol, identity_atol)
    if not isinstance(raise_on_failure, bool):
        raise TypeError("raise_on_failure must be bool")
    if not isinstance(raw_close_column, str) or not raw_close_column:
        raise ValueError("raw_close_column must be a non-empty column name")
    sessions = _normalise_calendar(calendar)
    factor_frame = _normalise_factors(factors)
    raw = (
        _normalise_raw_prices(raw_prices, sessions, close_column=raw_close_column)
        if raw_prices is not None
        else None
    )
    raw_sessions = None
    if raw is not None:
        raw_sessions = {
            str(symbol): set(group.index.get_level_values("date"))
            for symbol, group in raw.groupby(level="symbol", sort=False)
        }
    _assert_factor_calendar_coverage(factor_frame, sessions, raw_sessions)
    changes = detect_material_factor_changes(factor_frame, rtol=relative, atol=absolute)
    actions = _normalise_actions(
        corporate_actions,
        sessions,
        rtol=relative,
        atol=absolute,
    )

    joined = changes.merge(
        actions,
        how="outer",
        on=["symbol", "ex_date"],
        indicator=True,
        validate="one_to_one",
    ).sort_values(["symbol", "ex_date"], kind="stable")
    rows: list[dict[str, Any]] = []
    for _, joined_row in joined.iterrows():
        row = {column: np.nan for column in DETAIL_COLUMNS}
        for column in (
            "symbol",
            "ex_date",
            "factor_before_date",
            "factor_before",
            "factor_after",
            "factor_ratio",
            "record_date",
            "cash_payment_date",
            "cash_dividend_per_old_share",
            "share_ratio",
            "source_sha256",
        ):
            row[column] = joined_row.get(column, np.nan)
        origin = joined_row["_merge"]
        if origin == "left_only":
            row.update(
                {
                    "audit_passed": False,
                    "status": "missing_action",
                    "issue_code": "material_factor_jump_without_event",
                    "verification_level": "none",
                    "economic_claim": "none",
                }
            )
        elif origin == "right_only":
            row.update(
                {
                    "audit_passed": False,
                    "status": "extra_action",
                    "issue_code": "event_without_material_factor_jump",
                    "verification_level": "none",
                    "economic_claim": "none",
                }
            )
        elif raw is None:
            row.update(
                {
                    "audit_passed": True,
                    "status": "matched_event_presence_only",
                    "issue_code": "",
                    "verification_level": "event_presence_only",
                    "economic_claim": (
                        "same-date event exists; factor multiplier is not economically verified"
                    ),
                }
            )
        else:
            before_key = (joined_row["symbol"], joined_row["factor_before_date"])
            ex_key = (joined_row["symbol"], joined_row["ex_date"])
            if before_key not in raw.index or ex_key not in raw.index:
                row.update(
                    {
                        "audit_passed": False,
                        "status": "raw_price_missing",
                        "issue_code": "identity_price_missing",
                        "verification_level": "raw_close_ex_right_identity",
                        "economic_claim": "none",
                    }
                )
            else:
                close_before = float(raw.loc[before_key, "raw_close"])
                close_ex = float(raw.loc[ex_key, "raw_close"])
                cash = float(joined_row["cash_dividend_per_old_share"])
                ratio = float(joined_row["share_ratio"])
                denominator = close_before - cash
                row["raw_close_before"] = close_before
                row["raw_close_ex_date"] = close_ex
                if denominator <= 0:
                    row.update(
                        {
                            "audit_passed": False,
                            "status": "factor_identity_mismatch",
                            "issue_code": "non_positive_theoretical_ex_price",
                            "verification_level": "raw_close_ex_right_identity",
                            "economic_claim": "none",
                        }
                    )
                else:
                    theoretical_ex = denominator / ratio
                    implied_ratio = close_before / theoretical_ex
                    factor_ratio = float(joined_row["factor_ratio"])
                    qlib_return = close_ex * factor_ratio / close_before
                    action_return = close_ex / theoretical_ex
                    legal_return = (ratio * close_ex + cash) / close_before
                    identity_matches = math.isclose(
                        factor_ratio,
                        implied_ratio,
                        rel_tol=identity_relative,
                        abs_tol=identity_absolute,
                    ) and math.isclose(
                        qlib_return,
                        action_return,
                        rel_tol=identity_relative,
                        abs_tol=identity_absolute,
                    )
                    row.update(
                        {
                            "raw_close_before": close_before,
                            "raw_close_ex_date": close_ex,
                            "theoretical_ex_close": theoretical_ex,
                            "action_implied_factor_ratio": implied_ratio,
                            "factor_ratio_absolute_error": abs(factor_ratio - implied_ratio),
                            "qlib_adjusted_return_multiplier": qlib_return,
                            "action_ex_right_return_multiplier": action_return,
                            "legal_holding_return_multiplier": legal_return,
                            "legal_vs_qlib_return_gap": legal_return - qlib_return,
                            "audit_passed": identity_matches,
                            "status": (
                                "matched_event_identity_verified"
                                if identity_matches
                                else "factor_identity_mismatch"
                            ),
                            "issue_code": "" if identity_matches else "factor_ratio_disagrees_with_action",
                            "verification_level": "raw_close_ex_right_identity",
                            "economic_claim": (
                                "standard ex-right factor identity verified; "
                                "exact legal holding return is reported separately"
                                if identity_matches
                                else "none"
                            ),
                        }
                    )
        rows.append(row)

    details = pd.DataFrame(rows, columns=DETAIL_COLUMNS) if rows else _empty_details()
    missing_count = int((details["status"] == "missing_action").sum())
    extra_count = int((details["status"] == "extra_action").sum())
    mismatch_count = int(
        details["status"].isin(["factor_identity_mismatch", "raw_price_missing"]).sum()
    )
    presence_count = int((details["status"] == "matched_event_presence_only").sum())
    identity_count = int((details["status"] == "matched_event_identity_verified").sum())
    matched_count = presence_count + identity_count + mismatch_count
    passed = bool(details["audit_passed"].all()) if not details.empty else True
    scope = (
        "standard ex-right adjustment identity checked; legal holding return is a separate diagnostic"
        if raw is not None
        else "event presence only; factor/event tables alone cannot prove an economic multiplier"
    )
    summary = pd.DataFrame(
        [
            {
                "audit_passed": passed,
                "material_factor_jump_count": len(changes),
                "corporate_action_count": len(actions),
                "matched_event_count": matched_count,
                "missing_action_count": missing_count,
                "extra_action_count": extra_count,
                "identity_mismatch_count": mismatch_count,
                "presence_only_count": presence_count,
                "identity_verified_count": identity_count,
                "raw_price_identity_requested": raw is not None,
                "factor_start_date": factor_frame["date"].min(),
                "factor_end_date": factor_frame["date"].max(),
                "rtol": relative,
                "atol": absolute,
                "economic_scope": scope,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    result = CorporateActionAuditResult(summary=summary, details=details, factor_changes=changes)
    if not passed and raise_on_failure:
        raise CorporateActionAuditError(result)
    return result


__all__ = [
    "ACTION_COLUMNS",
    "DETAIL_COLUMNS",
    "FACTOR_COLUMNS",
    "SUMMARY_COLUMNS",
    "CorporateActionAuditError",
    "CorporateActionAuditResult",
    "audit_corporate_actions",
    "detect_material_factor_changes",
]
