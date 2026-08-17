"""Unit tests for the Alpha191 operator layer and factor registry.

Run from the repository root with the ``quant`` environment:

.. code-block:: powershell

    $env:PYTHONPATH = (Resolve-Path .\\qlib\\pipeline\\src).Path
    C:\\Exception\\quant\\python.exe -m pytest .\\qlib\\pipeline\\tests\\test_alpha191_ops.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.alpha191 import (
    Alpha191Data,
    CORR,
    COUNT,
    DECAYLINEAR,
    DELAY,
    DELTA,
    HIGHDAY,
    IFELSE,
    LOWDAY,
    MAX,
    MEAN,
    MIN,
    RANK,
    REGBETA,
    SMA,
    STD,
    SUM,
    TS_MAX,
    TS_MIN,
    TS_RANK,
    WMA,
    alpha191_registry,
    build_all_factors,
    rolling_slope,
)


def _wide(values: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-01", periods=values.shape[0])
    columns = [f"SYM{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=index, columns=columns)


def _random_frame(t: int = 120, n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _wide(rng.normal(0, 1, (t, n)))


def test_registry_complete() -> None:
    registry = alpha191_registry()
    numbers = {f.number for f in registry}
    assert len(registry) == 191
    assert numbers == set(range(1, 192))
    names = [f.name for f in registry]
    assert names == [f"alpha{i:03d}" for i in range(1, 192)]


def test_rank_matches_manual() -> None:
    frame = _random_frame()
    got = RANK(frame)
    for row in range(frame.shape[0]):
        values = frame.iloc[row].to_numpy(dtype=float)
        order = np.argsort(np.argsort(values)) + 1
        expected = order / len(values)
        np.testing.assert_allclose(got.iloc[row].to_numpy(), expected, atol=1e-12)


def test_sma_matches_recursive_formula() -> None:
    values = _random_frame(n=1).iloc[:, 0].to_numpy(dtype=float)
    n, m = 7, 2
    got = SMA(_wide(values.reshape(-1, 1)), n, m).iloc[:, 0].to_numpy()
    expected = np.full_like(values, np.nan)
    expected[0] = values[0]
    for i in range(1, len(values)):
        expected[i] = (m * values[i] + (n - m) * expected[i - 1]) / n
    np.testing.assert_allclose(got, expected, atol=1e-12, equal_nan=True)


def test_wma_matches_manual() -> None:
    values = np.arange(1.0, 21.0).reshape(-1, 1)
    frame = _wide(values)
    n = 5
    got = WMA(frame, n).iloc[:, 0].to_numpy()
    weights = np.arange(1, n + 1) / (n * (n + 1) / 2)
    expected = np.full(20, np.nan)
    for i in range(n - 1, 20):
        expected[i] = float(np.dot(values[i - n + 1 : i + 1, 0], weights))
    np.testing.assert_allclose(got, expected, atol=1e-12, equal_nan=True)


def test_decaylinear_equals_wma() -> None:
    frame = _random_frame()
    np.testing.assert_allclose(
        DECAYLINEAR(frame, 9).to_numpy(),
        WMA(frame, 9).to_numpy(),
        atol=1e-12,
        equal_nan=True,
    )


def test_ts_rank_matches_pandas() -> None:
    frame = _random_frame()
    got = TS_RANK(frame, 10)
    expected = frame.rolling(10).rank(pct=True)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), atol=1e-12, equal_nan=True)


def test_corr_covariance_match_pandas() -> None:
    x = _random_frame(seed=1)
    y = _random_frame(seed=2)
    np.testing.assert_allclose(
        CORR(x, y, 6).to_numpy(), x.rolling(6).corr(y).to_numpy(), atol=1e-10, equal_nan=True
    )
    np.testing.assert_allclose(
        CORR(x, y, 6).to_numpy(),
        np.nan_to_num(CORR(x, y, 6).to_numpy(), nan=np.nan),
        equal_nan=True,
    )


def test_regbeta_is_slope_of_x_on_y() -> None:
    x = _random_frame(seed=3)
    y = _random_frame(seed=4)
    got = REGBETA(x, y, 10)
    # slope of x on y per window: compare against numpy polyfit for one column
    xv = x.iloc[:, 0].to_numpy(dtype=float)
    yv = y.iloc[:, 0].to_numpy(dtype=float)
    for i in range(9, len(xv), 7):
        window_x = xv[i - 9 : i + 1]
        window_y = yv[i - 9 : i + 1]
        mask = np.isfinite(window_x) & np.isfinite(window_y)
        slope, _ = np.polyfit(window_y[mask], window_x[mask], 1)
        np.testing.assert_allclose(got.iloc[i, 0], slope, atol=1e-8)


def test_rolling_slope_matches_polyfit() -> None:
    frame = _random_frame(seed=5)
    got = rolling_slope(frame, 10)
    xv = frame.iloc[:, 3].to_numpy(dtype=float)
    for i in range(9, len(xv), 5):
        window = xv[i - 9 : i + 1]
        slope, _ = np.polyfit(np.arange(1, 11), window, 1)
        np.testing.assert_allclose(got.iloc[i, 3], slope, atol=1e-8)


def test_highday_lowday_semantics() -> None:
    values = np.array(
        [[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [2.5, 2.0], [4.0, 1.0], [3.5, 2.5]], dtype=float
    )
    frame = _wide(values)
    high = HIGHDAY(frame, 3)
    low = LOWDAY(frame, 3)
    # column 0: window [2,3,2.5] at row 3 -> max at row 2 (1 day ago), min at row 1 (2 days ago)
    assert high.iloc[3, 0] == 1.0
    assert low.iloc[3, 0] == 2.0
    # column 1 row 5: window [2,1,2.5] -> max today (2.5) -> 0; min at row 4 -> 1
    assert high.iloc[5, 1] == 0.0
    assert low.iloc[5, 1] == 1.0
    # warm-up rows are NaN
    assert math.isnan(high.iloc[0, 0])
    assert math.isnan(low.iloc[1, 1])


def test_ifelse_both_operand_orders() -> None:
    frame = _random_frame()
    cond = frame > 0
    a = frame * 2
    b = frame - 1
    got = IFELSE(cond, a, b)
    np.testing.assert_allclose(got.to_numpy(), np.where(cond.to_numpy(), a.to_numpy(), b.to_numpy()))
    got_rev = IFELSE(cond, b, a)
    np.testing.assert_allclose(got_rev.to_numpy(), np.where(cond.to_numpy(), b.to_numpy(), a.to_numpy()))
    got_scalar = IFELSE(cond, 0.0, a)
    np.testing.assert_allclose(got_scalar.to_numpy(), np.where(cond.to_numpy(), 0.0, a.to_numpy()))


def test_count_matches_manual() -> None:
    frame = _random_frame()
    cond = frame > 0
    got = COUNT(cond, 5)
    expected = cond.astype(float).rolling(5).sum()
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), atol=1e-12, equal_nan=True)


def test_build_all_factors_on_small_data() -> None:
    rng = np.random.default_rng(11)
    t, n = 260, 6
    dates = pd.bdate_range("2025-01-01", periods=t)
    cols = [f"SYM{i}" for i in range(n)]
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (t, n)), axis=0), index=dates, columns=cols)
    opn = close * (1 + rng.normal(0, 0.002, (t, n)))
    high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.003, (t, n))))
    low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.003, (t, n))))
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (t, n)).astype(float), index=dates, columns=cols)
    amount = volume * close
    data = Alpha191Data(open=opn, high=high, low=low, close=close, volume=volume, amount=amount)
    factors, errors = build_all_factors(data, on_error="warn")
    assert len(factors) == 191
    assert errors == []
    for name, frame in factors.items():
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (t, n)
        assert np.isinf(frame.to_numpy()).sum() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
