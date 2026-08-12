from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DATE_LEVEL = "datetime"
INSTRUMENT_LEVEL = "instrument"
ELIGIBILITY_COLUMN = "eligible"
LIQUIDITY_COLUMN = "liquidity_eligible"
DEFAULT_TRADABLE_FIELDS = ("$close", "$volume", "$factor")


@dataclass(frozen=True)
class QlibCoverageInputs:
    active_spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]
    calendar: pd.DatetimeIndex
    eligibility: pd.Series


def _datetime_index(values: Iterable[Any], name: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(list(values)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid datetimes") from exc
    if index.hasnans:
        raise ValueError(f"{name} contains missing datetimes")
    if index.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate dates")
    return index


def _normalize_calendar(calendar: Iterable[Any]) -> pd.DatetimeIndex:
    index = _datetime_index(calendar, "calendar")
    if not index.is_monotonic_increasing:
        raise ValueError("calendar must be strictly increasing")
    return index


def _normalize_test_dates(test_dates: Iterable[Any], calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = _datetime_index(test_dates, "test_dates")
    if index.empty:
        raise ValueError("test_dates must not be empty")
    outside = index.difference(calendar)
    if not outside.empty:
        raise ValueError(f"test_dates are outside the calendar: {_format_values(outside)}")
    return calendar[calendar.isin(index)]


def _format_values(values: Iterable[Any], limit: int = 5) -> str:
    items = [str(value) for value in list(values)[:limit]]
    return ", ".join(items)


def _is_span_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and not (
            isinstance(value[0], Sequence)
            and not isinstance(value[0], (str, bytes, pd.Timestamp))
        )
    )


def _normalize_active_spans(
    active_spans: Mapping[str, Any] | pd.DataFrame,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    normalized: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if isinstance(active_spans, pd.DataFrame):
        columns = set(active_spans.columns)
        instrument_column = "instrument" if "instrument" in columns else "symbol" if "symbol" in columns else None
        start_column = "start_date" if "start_date" in columns else "start" if "start" in columns else None
        end_column = "end_date" if "end_date" in columns else "end" if "end" in columns else None
        if instrument_column is None or start_column is None or end_column is None:
            raise ValueError(
                "active span DataFrame requires instrument/symbol, start/start_date, and end/end_date columns"
            )
        records = (
            (row[instrument_column], (row[start_column], row[end_column]))
            for _, row in active_spans.iterrows()
        )
    elif isinstance(active_spans, Mapping):
        expanded: list[tuple[Any, Any]] = []
        for instrument, raw_spans in active_spans.items():
            spans = [raw_spans] if _is_span_pair(raw_spans) else raw_spans
            if not isinstance(spans, Iterable) or isinstance(spans, (str, bytes)):
                raise ValueError(f"active spans for {instrument!r} must be a span or iterable of spans")
            for span in spans:
                expanded.append((instrument, span))
        records = iter(expanded)
    else:
        raise TypeError("active_spans must be a mapping or DataFrame")

    for raw_instrument, raw_span in records:
        instrument = str(raw_instrument)
        if not instrument or instrument.lower() == "nan":
            raise ValueError("active spans contain an invalid instrument")
        if not _is_span_pair(raw_span):
            raise ValueError(f"invalid active span for {instrument!r}: {raw_span!r}")
        try:
            start, end = pd.Timestamp(raw_span[0]), pd.Timestamp(raw_span[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid active span for {instrument!r}: {raw_span!r}") from exc
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"invalid active span for {instrument!r}: {raw_span!r}")
        normalized.setdefault(instrument, []).append((start, end))

    if not normalized:
        raise ValueError("active_spans must contain at least one instrument")
    for instrument, spans in normalized.items():
        normalized[instrument] = sorted(set(spans))
    return dict(sorted(normalized.items()))


def _normalize_multiindex(
    index: pd.Index,
    name: str,
    *,
    reject_duplicates: bool,
) -> pd.MultiIndex:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError(f"{name} index must be a two-level MultiIndex")
    if reject_duplicates and index.has_duplicates:
        duplicates = index[index.duplicated(keep=False)].unique()
        raise ValueError(f"{name} index contains duplicate rows: {_format_values(duplicates)}")
    if DATE_LEVEL not in index.names or INSTRUMENT_LEVEL not in index.names:
        raise ValueError(
            f"{name} index levels must be named {DATE_LEVEL!r} and {INSTRUMENT_LEVEL!r}"
        )
    dates = pd.to_datetime(index.get_level_values(DATE_LEVEL), errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{name} index contains invalid datetimes")
    instruments = index.get_level_values(INSTRUMENT_LEVEL)
    if pd.isna(instruments).any():
        raise ValueError(f"{name} index contains missing instruments")
    normalized = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(dates), instruments.map(str)],
        names=[DATE_LEVEL, INSTRUMENT_LEVEL],
    )
    if reject_duplicates and normalized.has_duplicates:
        duplicates = normalized[normalized.duplicated(keep=False)].unique()
        raise ValueError(f"{name} index contains duplicate normalized rows: {_format_values(duplicates)}")
    return normalized


def _finite_available(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"eligibility DataFrame is missing tradable fields: {missing_columns}")
    available = pd.Series(True, index=frame.index, dtype=bool)
    for column in columns:
        values = frame[column]
        available &= values.notna()
        if pd.api.types.is_numeric_dtype(values.dtype):
            available &= np.isfinite(values.astype(float))
    return available


def build_eligibility_mask(
    eligibility: pd.Series | pd.DataFrame,
    *,
    eligibility_column: str = ELIGIBILITY_COLUMN,
    liquidity_column: str = LIQUIDITY_COLUMN,
    required_tradable_fields: Sequence[str] = DEFAULT_TRADABLE_FIELDS,
) -> pd.Series:
    """Return a strict combined liquidity and base-field availability mask."""
    if not isinstance(eligibility, (pd.Series, pd.DataFrame)):
        raise TypeError("eligibility must be a Series or DataFrame")
    normalized_index = _normalize_multiindex(
        eligibility.index,
        "eligibility",
        reject_duplicates=True,
    )

    if isinstance(eligibility, pd.Series):
        values = eligibility.copy()
    elif eligibility_column in eligibility.columns:
        values = eligibility[eligibility_column].copy()
    else:
        if liquidity_column not in eligibility.columns:
            raise ValueError(
                f"eligibility DataFrame requires either {eligibility_column!r} or {liquidity_column!r}"
            )
        liquidity = eligibility[liquidity_column]
        if not (pd.api.types.is_bool_dtype(liquidity.dtype) or pd.api.types.is_numeric_dtype(liquidity.dtype)):
            raise TypeError("liquidity eligibility values must be boolean or numeric")
        values = liquidity.notna() & liquidity.fillna(False).astype(bool)
        values &= _finite_available(eligibility, tuple(required_tradable_fields))

    if not (pd.api.types.is_bool_dtype(values.dtype) or pd.api.types.is_numeric_dtype(values.dtype)):
        raise TypeError("eligibility values must be boolean or numeric")
    mask = values.notna() & values.fillna(False).astype(bool)
    mask.index = normalized_index
    mask.name = ELIGIBILITY_COLUMN
    return mask.sort_index()


def _active_grid(
    spans: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    test_dates: pd.DatetimeIndex,
) -> pd.MultiIndex:
    rows: list[tuple[pd.Timestamp, str]] = []
    for instrument, instrument_spans in spans.items():
        active = np.zeros(len(test_dates), dtype=bool)
        for start, end in instrument_spans:
            active |= (test_dates >= start) & (test_dates <= end)
        rows.extend((date, instrument) for date in test_dates[active])
    return pd.MultiIndex.from_tuples(rows, names=[DATE_LEVEL, INSTRUMENT_LEVEL]).sort_values()


def _prediction_scores(predictions: pd.Series | pd.DataFrame, score_column: str) -> pd.Series:
    if isinstance(predictions, pd.Series):
        scores = predictions.copy()
    elif isinstance(predictions, pd.DataFrame):
        if score_column not in predictions.columns:
            raise ValueError(f"predictions are missing score column {score_column!r}")
        scores = predictions[score_column].copy()
    else:
        raise TypeError("predictions must be a Series or DataFrame")
    scores.index = _normalize_multiindex(predictions.index, "predictions", reject_duplicates=True)
    return scores


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def calculate_prediction_coverage(
    active_spans: Mapping[str, Any] | pd.DataFrame,
    calendar: Iterable[Any],
    test_dates: Iterable[Any],
    eligibility: pd.Series | pd.DataFrame,
    predictions: pd.Series | pd.DataFrame,
    *,
    score_column: str = "score",
    eligibility_column: str = ELIGIBILITY_COLUMN,
    liquidity_column: str = LIQUIDITY_COLUMN,
    required_tradable_fields: Sequence[str] = DEFAULT_TRADABLE_FIELDS,
) -> dict[str, Any]:
    """Calculate score coverage against the actual daily eligible candidate grid."""
    normalized_calendar = _normalize_calendar(calendar)
    normalized_test_dates = _normalize_test_dates(test_dates, normalized_calendar)
    spans = _normalize_active_spans(active_spans)
    mask = build_eligibility_mask(
        eligibility,
        eligibility_column=eligibility_column,
        liquidity_column=liquidity_column,
        required_tradable_fields=required_tradable_fields,
    )
    scores = _prediction_scores(predictions, score_column)

    prediction_dates = pd.DatetimeIndex(scores.index.get_level_values(DATE_LEVEL).unique())
    outside_dates = prediction_dates.difference(normalized_test_dates)
    if not outside_dates.empty:
        raise ValueError(f"predictions contain dates outside test_dates: {_format_values(outside_dates)}")
    prediction_instruments = set(scores.index.get_level_values(INSTRUMENT_LEVEL))
    unknown_instruments = sorted(prediction_instruments - set(spans))
    if unknown_instruments:
        raise ValueError(
            "predictions contain instruments outside active_spans: "
            f"{_format_values(unknown_instruments)}"
        )

    active_grid = _active_grid(spans, normalized_test_dates)
    missing_eligibility = active_grid.difference(mask.index)
    if not missing_eligibility.empty:
        raise ValueError(
            "eligibility is missing active instrument-date rows: "
            f"{_format_values(missing_eligibility)}"
        )
    relevant_mask = mask.reindex(active_grid)
    if relevant_mask.isna().any():
        raise ValueError("eligibility contains unresolved active instrument-date rows")
    expected_index = relevant_mask.index[relevant_mask.to_numpy(dtype=bool)]

    extra_index = scores.index.difference(expected_index)
    expected_scores = scores.reindex(expected_index)
    scored_mask = expected_scores.notna()
    scored_index = expected_scores.index[scored_mask]
    missing_index = expected_scores.index[~scored_mask]

    expected_rows = len(expected_index)
    scored_rows = len(scored_index)
    daily_expected = pd.Series(1, index=expected_index, dtype=int).groupby(level=DATE_LEVEL).sum()
    daily_scored = pd.Series(1, index=scored_index, dtype=int).groupby(level=DATE_LEVEL).sum()
    daily = pd.DataFrame(index=normalized_test_dates)
    daily.index.name = DATE_LEVEL
    daily["expected_rows"] = daily_expected.reindex(daily.index, fill_value=0).astype(int)
    daily["scored_rows"] = daily_scored.reindex(daily.index, fill_value=0).astype(int)
    candidate_daily = daily.loc[daily["expected_rows"] > 0].copy()
    candidate_daily["coverage"] = candidate_daily["scored_rows"] / candidate_daily["expected_rows"]

    expected_instruments = set(expected_index.get_level_values(INSTRUMENT_LEVEL))
    scored_instruments = set(scored_index.get_level_values(INSTRUMENT_LEVEL))
    expected_dates = set(expected_index.get_level_values(DATE_LEVEL))
    scored_dates = set(scored_index.get_level_values(DATE_LEVEL))
    fully_scored_dates = int(
        (candidate_daily["expected_rows"] == candidate_daily["scored_rows"]).sum()
    )

    return {
        "coverage": _safe_ratio(scored_rows, expected_rows),
        "expected_rows": expected_rows,
        "scored_rows": scored_rows,
        "missing_rows": len(missing_index),
        "extra_prediction_rows": len(extra_index),
        "extra_scored_prediction_rows": int(scores.reindex(extra_index).notna().sum()),
        "daily_coverage_min": (
            _finite_or_none(candidate_daily["coverage"].min()) if not candidate_daily.empty else None
        ),
        "daily_coverage_median": (
            _finite_or_none(candidate_daily["coverage"].median()) if not candidate_daily.empty else None
        ),
        "daily_coverage_mean": (
            _finite_or_none(candidate_daily["coverage"].mean()) if not candidate_daily.empty else None
        ),
        "test_date_count": len(normalized_test_dates),
        "candidate_date_count": len(expected_dates),
        "scored_date_count": len(scored_dates),
        "fully_scored_date_count": fully_scored_dates,
        "date_coverage": _safe_ratio(len(scored_dates), len(expected_dates)),
        "expected_instrument_count": len(expected_instruments),
        "scored_instrument_count": len(scored_instruments),
        "instrument_coverage": _safe_ratio(len(scored_instruments), len(expected_instruments)),
        "dates_without_candidates": [
            date.date().isoformat() for date in daily.index[daily["expected_rows"] == 0]
        ],
        "dates_without_scores": sorted(
            pd.Timestamp(date).date().isoformat() for date in expected_dates - scored_dates
        ),
        "instruments_without_scores": sorted(expected_instruments - scored_instruments),
    }


def load_qlib_coverage_inputs(
    market: str,
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    liquidity_expression: str,
    *,
    required_tradable_fields: Sequence[str] = DEFAULT_TRADABLE_FIELDS,
    freq: str = "day",
    provider: Any = None,
) -> QlibCoverageInputs:
    """Read coverage inputs from an already initialized Qlib provider."""
    if provider is None:
        from qlib.data import D

        provider = D
    instrument_config = provider.instruments(market)
    raw_spans = provider.list_instruments(
        instrument_config,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        as_list=False,
    )
    spans = _normalize_active_spans(raw_spans)
    calendar = _normalize_calendar(
        provider.calendar(start_time=start_time, end_time=end_time, freq=freq)
    )
    fields = [liquidity_expression, *required_tradable_fields]
    frame = provider.features(
        instrument_config,
        fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
    ).copy()
    if len(frame.columns) != len(fields):
        raise ValueError("Qlib coverage feature query returned an unexpected number of columns")
    frame.columns = [LIQUIDITY_COLUMN, *required_tradable_fields]
    eligibility = build_eligibility_mask(
        frame,
        liquidity_column=LIQUIDITY_COLUMN,
        required_tradable_fields=required_tradable_fields,
    )
    return QlibCoverageInputs(spans, calendar, eligibility)
