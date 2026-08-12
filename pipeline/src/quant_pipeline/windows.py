from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class RollingFold:
    fold: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    purge_bars: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_calendar(path, start_date: str, end_date: str) -> pd.DatetimeIndex:
    calendar = pd.to_datetime(pd.read_csv(path, header=None).iloc[:, 0])
    return pd.DatetimeIndex(calendar[(calendar >= pd.Timestamp(start_date)) & (calendar <= pd.Timestamp(end_date))])


def build_rolling_folds(
    calendar: pd.DatetimeIndex,
    train_start_date: str,
    test_start_date: str,
    validation_days: int,
    test_days: int,
    purge_bars: int,
) -> list[RollingFold]:
    if not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise ValueError("calendar must be unique and increasing")
    test_positions = calendar.searchsorted(pd.Timestamp(test_start_date), side="left")
    if test_positions >= len(calendar):
        raise ValueError("test_start_date is outside the calendar")
    train_start_idx = calendar.searchsorted(pd.Timestamp(train_start_date), side="left")
    folds: list[RollingFold] = []
    test_start_idx = int(test_positions)
    fold_no = 1
    while test_start_idx < len(calendar):
        test_end_idx = min(test_start_idx + test_days - 1, len(calendar) - 1)
        valid_end_idx = test_start_idx - purge_bars - 1
        valid_start_idx = valid_end_idx - validation_days + 1
        train_end_idx = valid_start_idx - purge_bars - 1
        if train_end_idx < train_start_idx:
            raise ValueError("not enough history for the requested rolling windows")
        folds.append(
            RollingFold(
                fold=fold_no,
                train_start=calendar[train_start_idx].date().isoformat(),
                train_end=calendar[train_end_idx].date().isoformat(),
                valid_start=calendar[valid_start_idx].date().isoformat(),
                valid_end=calendar[valid_end_idx].date().isoformat(),
                test_start=calendar[test_start_idx].date().isoformat(),
                test_end=calendar[test_end_idx].date().isoformat(),
                purge_bars=purge_bars,
            )
        )
        fold_no += 1
        test_start_idx = test_end_idx + 1
    return folds


def validate_fold_boundaries(folds: list[RollingFold], calendar: pd.DatetimeIndex) -> None:
    positions = {timestamp.date().isoformat(): index for index, timestamp in enumerate(calendar)}
    for fold in folds:
        train_gap = positions[fold.valid_start] - positions[fold.train_end] - 1
        valid_gap = positions[fold.test_start] - positions[fold.valid_end] - 1
        if train_gap != fold.purge_bars or valid_gap != fold.purge_bars:
            raise ValueError(f"fold {fold.fold} has an invalid purge boundary")
    for previous, current in zip(folds, folds[1:]):
        if positions[current.test_start] != positions[previous.test_end] + 1:
            raise ValueError("test windows are not contiguous")


def shift_session(calendar: pd.DatetimeIndex, date: str, bars: int) -> str:
    location = calendar.get_indexer([pd.Timestamp(date)])[0]
    if location < 0:
        raise ValueError(f"date is not in trading calendar: {date}")
    shifted = location + int(bars)
    if shifted < 0 or shifted >= len(calendar):
        raise ValueError(f"cannot shift {date} by {bars} sessions within available calendar")
    return calendar[shifted].date().isoformat()
