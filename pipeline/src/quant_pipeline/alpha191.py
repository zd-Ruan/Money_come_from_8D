"""Alpha191 factor library for the ETF research pipeline.

Scope
-----
Implements the full Alpha191 formula set: WorldQuant/Kakushadze Alpha001-101
("101 Formulaic Alphas", arXiv:1601.00991) plus the extended Alpha102-191 set
that circulates in the Chinese quant community (JoinQuant/BigQuant syntax).
All 191 formulas are reproduced in the docstring of each factor function and
in ``Alpha191_FACTORS`` metadata.

Operator semantics follow the JoinQuant-style conventions used by the
canonical alpha191 implementations, with these deliberate choices:

- ``RANK`` is cross-sectional: per-date percentile rank across instruments.
- ``SMA(X, N, M)`` is the recursive smoothed moving average
  ``Y_t = (M * X_t + (N - M) * Y_{t-1}) / N`` (``ewm(alpha=M/N, adjust=False)``).
- ``WMA``/``DECAYLINEAR`` use linearly decaying weights ``i / (N*(N+1)/2)``
  for ``i = 1..N`` (``i = 1`` oldest) and require a full window (NaN warm-up).
- ``TS_RANK(X, N)`` is the rolling percentile rank of the latest value.
- ``HIGHDAY``/``LOWDAY`` return the number of sessions since the most recent
  (last) window extreme, ``0`` when the extreme is the current bar.
- ``REGBETA(X, Y, N)`` is the slope of ``X`` on ``Y``: ``cov(X, Y) / var(Y)``.

Data layout
-----------
Factors operate on *wide* DataFrames indexed by ``DatetimeIndex`` with one
column per instrument (the layout used by the canonical implementations).
``Alpha191Data`` bundles the OHLCV/amount/vwap fields plus optional benchmark
series for the benchmark-relative alphas (30, 75, 149, 181, 182).

Documented deviations / fixes versus the reference implementation
-----------------------------------------------------------------
- Alpha030: the reference marks it unfinished (it needs MKT/SMB/HML). We
  regress returns on the benchmark ETF return over 60 sessions (SMB/HML = 0)
  and square the residual, then apply WMA(20).
- Alpha029/144/150: the reference multiplies by ``log(volume)``; the formulas
  say ``volume``/``amount`` and we follow the formulas.
- Alpha033/062: the reference substitutes turnover; the formulas say
  ``TSRANK(VOLUME, 5)`` / ``RANK(VOLUME)`` and we follow the formulas.
- Alpha073: the reference expands the leading ``-1 *`` wrongly; we follow the
  formula ``-1 * (A - B)``.
- Alpha164: the reference drops parentheses; we follow the formula
  ``(A - TS_MIN(A, 12)) / (HIGH - LOW) * 100``.
- Alpha166: the reference replaces the constant ``-20 * 19^1.5 / (19 * 18)``
  with an approximation ``5``; we keep the exact constant.
- Alpha165/183: the reference reads ``SUMAC`` as a rolling sum over the same
  window; we keep that convention (documented; ``SUMAC`` is path dependent).
- A small ``+1e-12`` epsilon guards a handful of denominators (alphas 40, 50,
  51, 52, 55, 76, 110, 112, 118, 128, 145, 146, 159, 162, 168, 172, 183, 186,
  188).  In degenerate all-flat windows this turns an ``inf``/``NaN`` into a
  large finite value instead of dividing by zero; this is an accepted
  engineering trade-off (never flips a discrete branch).
- Alpha149 compresses the series to benchmark down days before the rolling
  regression (the NaN-mask approach would never complete a 252-window on a
  ~50% down-day sample).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Primitive operators (JoinQuant-style semantics on wide DataFrames)
# ---------------------------------------------------------------------------


def RANK(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date (0..1, average ties)."""
    return x.rank(axis=1, pct=True)


def DELAY(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(n)


def DELTA(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.diff(n)


def SUM(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).sum()


def MEAN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).mean()


def STD(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).std()


def CORR(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling Pearson correlation; undefined (constant input) -> NaN."""
    out = x.rolling(n).corr(y)
    return out.replace([np.inf, -np.inf], np.nan)


def COVARIANCE(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling covariance; undefined (constant input) -> NaN."""
    out = x.rolling(n).cov(y)
    return out.replace([np.inf, -np.inf], np.nan)


def TS_MAX(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).max()


def TS_MIN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).min()


def TS_RANK(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling percentile rank of the latest value inside the window."""
    return x.rolling(n).rank(pct=True)


def COUNT(cond: pd.DataFrame, n: int, na_map: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rolling count of a boolean condition; ``na_map`` propagates NaN."""
    c = cond.astype(float)
    if na_map is not None:
        c = c + na_map
    return c.rolling(n).sum()


def SMA(x: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    """Recursive smoothed moving average ``Y_t = (M*X_t + (N-M)*Y_{t-1}) / N``."""
    return x.ewm(alpha=m / n, adjust=False, min_periods=1).mean()


def _rolling_weighted_mean(x: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    arr = x.to_numpy(dtype=float)
    t, n_cols = arr.shape
    w = len(weights)
    out = np.full_like(arr, np.nan)
    if t >= w:
        views = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)  # (t-w+1, n_cols, w)
        with np.errstate(invalid="ignore"):
            out[w - 1 :] = views @ weights  # NaN in window propagates
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def WMA(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Weighted moving average with weights 1..n (most recent heaviest)."""
    weights = (np.arange(1, n + 1, dtype=float)) / (n * (n + 1) / 2.0)
    return _rolling_weighted_mean(x, weights)


def DECAYLINEAR(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Linear decay weighted average; identical weights to WMA."""
    weights = np.array([2.0 * i / (n * (n + 1)) for i in range(1, n + 1)])
    return _rolling_weighted_mean(x, weights)


def MAX(a: pd.DataFrame, b) -> pd.DataFrame:
    return np.maximum(a, b)


def MIN(a: pd.DataFrame, b) -> pd.DataFrame:
    return np.minimum(a, b)


def IFELSE(cond: pd.DataFrame, a, b) -> pd.DataFrame:
    """``a`` where ``cond`` else ``b``.

    At least one of ``a``/``b`` must be a DataFrame; the DataFrame's own NaN
    mask is preserved in the branch it supplies.
    """
    if isinstance(a, pd.DataFrame):
        out = a.where(cond, b)
        return out.mask(a.isna())
    if isinstance(b, pd.DataFrame):
        out = b.where(~cond, a)
        return out.mask(b.isna())
    raise ValueError("IFELSE requires at least one DataFrame operand")


def SUMIF(x: pd.DataFrame, n: int, cond: pd.DataFrame) -> pd.DataFrame:
    return x.where(cond, 0).rolling(n).sum()


def REGBETA(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Slope of ``x`` on ``y`` over a trailing ``n`` window: cov(x, y) / var(y)."""
    return x.rolling(n).cov(y) / y.rolling(n).var()


def REGRESI(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Intercept of the regression of ``x`` on ``y`` over a trailing window."""
    return x.rolling(n).mean() - REGBETA(x, y, n) * y.rolling(n).mean()


def rolling_slope(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Slope of ``x`` regressed on the sequence 1..n over a trailing window."""
    arr = x.to_numpy(dtype=float)
    t, n_cols = arr.shape
    seq = np.arange(1, t + 1, dtype=float)  # 1-based absolute time index
    mean_x = pd.DataFrame(arr, index=x.index, columns=x.columns).rolling(n).mean()
    mean_xt = pd.DataFrame(arr * seq[:, None], index=x.index, columns=x.columns).rolling(n).mean()
    mean_seq = seq - (n - 1) / 2.0  # mean of the window sequence ending at each row
    var_seq = (n * n - 1) / 12.0  # population variance of 1..n
    slope = (mean_xt - mean_x.mul(mean_seq, axis=0)) / var_seq
    return slope


def HIGHDAY(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sessions since the most recent window maximum (0 = today)."""
    return _days_since_extreme(x, n, which="max")


def LOWDAY(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sessions since the most recent window minimum (0 = today)."""
    return _days_since_extreme(x, n, which="min")


def _days_since_extreme(x: pd.DataFrame, n: int, *, which: str) -> pd.DataFrame:
    arr = x.to_numpy(dtype=float)
    t, n_cols = arr.shape
    out = np.full((t, n_cols), np.nan)
    if t >= n:
        fill = -np.inf if which == "max" else np.inf
        safe = np.where(np.isnan(arr), fill, arr)
        for i in range(n - 1, t):
            win = safe[i - n + 1 : i + 1]  # (n, n_cols)
            if which == "max":
                first_pos = np.argmax(win[::-1], axis=0)  # last occurrence (most recent)
            else:
                first_pos = np.argmin(win[::-1], axis=0)
            last_pos = n - 1 - first_pos
            out[i] = (n - 1) - last_pos
            out[i, np.isnan(arr[i - n + 1 : i + 1]).any(axis=0)] = np.nan
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def LOG(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x)


def ABS(x: pd.DataFrame) -> pd.DataFrame:
    return np.abs(x)


def SIGN(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class Alpha191Data:
    """Wide OHLCV+amount inputs plus optional benchmark series.

    All DataFrames share one ``DatetimeIndex`` (rows) and one set of symbol
    columns. ``vwap`` defaults to ``amount / volume``; ``returns`` is derived.
    ``benchmark_close``/``benchmark_open`` are per-date Series (used by
    alphas 30, 75, 149, 181, 182) and broadcast over the symbol columns.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    amount: Optional[pd.DataFrame] = None
    vwap: Optional[pd.DataFrame] = None
    benchmark_close: Optional[pd.Series] = None
    benchmark_open: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        if self.amount is None:
            self.amount = self.close * 0.0
        if self.vwap is None:
            self.vwap = self.amount / self.volume.replace(0.0, np.nan)

    @property
    def returns(self) -> pd.DataFrame:
        return self.close / self.close.shift(1) - 1.0


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

FactorFn = Callable[[Alpha191Data], pd.DataFrame]

FAMILY_TREND = "trend_momentum"
FAMILY_VOLATILITY = "volatility"
FAMILY_VOLUME = "volume_liquidity"
FAMILY_PRICE_VOLUME = "price_volume"
FAMILY_SESSION = "session_structure"
FAMILY_BENCHMARK = "benchmark_relative"


@dataclass(frozen=True)
class AlphaFactor:
    number: int
    name: str
    max_depend: int
    family: str
    formula: str
    fn: FactorFn


_FACTORS: Dict[int, AlphaFactor] = {}


def alpha_factor(number: int, max_depend: int, family: str, formula: str):
    def decorator(fn: FactorFn) -> FactorFn:
        _FACTORS[number] = AlphaFactor(
            number=number,
            name=f"alpha{number:03d}",
            max_depend=max_depend,
            family=family,
            formula=formula,
            fn=fn,
        )
        return fn

    return decorator


def alpha191_registry() -> Tuple[AlphaFactor, ...]:
    """All registered factors in ascending number order."""
    return tuple(_FACTORS[i] for i in sorted(_FACTORS))


def _nan_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.full(df.shape, np.nan), index=df.index, columns=df.columns)


# ---------------------------------------------------------------------------
# Alpha001 - Alpha101 (Kakushadze, "101 Formulaic Alphas")
# ---------------------------------------------------------------------------


@alpha_factor(1, 10, FAMILY_PRICE_VOLUME, "(-1*CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6))")
def alpha_001(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * CORR(RANK(DELTA(LOG(d.volume), 1)), RANK((d.close - d.open) / d.open), 6)


@alpha_factor(2, 10, FAMILY_VOLATILITY, "(-1*DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1))")
def alpha_002(d: Alpha191Data) -> pd.DataFrame:
    part = ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)
    return -1.0 * DELTA(part, 1)


@alpha_factor(3, 10, FAMILY_PRICE_VOLUME, "SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)")
def alpha_003(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    cond_up = d.close > delay1
    cond_down = d.close < delay1
    inner = np.where(cond_up, d.close - MIN(d.low, delay1), d.close - MAX(d.high, delay1))
    inner = pd.DataFrame(np.where(d.close == delay1, 0.0, inner), index=d.close.index, columns=d.close.columns)
    return SUM(inner, 6)


@alpha_factor(4, 20, FAMILY_TREND, "((SUM(CLOSE,8)/8+STD(CLOSE,8))<(SUM(CLOSE,2)/2)?-1:((SUM(CLOSE,2)/2)<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:((1<VOLUME/MEAN(VOLUME,20))||(VOLUME/MEAN(VOLUME,20)==1)?1:-1)))")
def alpha_004(d: Alpha191Data) -> pd.DataFrame:
    ma8 = MEAN(d.close, 8)
    std8 = STD(d.close, 8)
    ma2 = MEAN(d.close, 2)
    vol_times = d.volume / MEAN(d.volume, 20)
    cond1 = (ma8 + std8) < ma2
    cond2 = ma2 < (ma8 - std8)
    cond3 = vol_times >= 1.0
    out = np.select([cond1, cond2, cond3], [-1.0, 1.0, 1.0], default=-1.0)
    return pd.DataFrame(out, index=d.close.index, columns=d.close.columns)


@alpha_factor(5, 20, FAMILY_PRICE_VOLUME, "(-1*TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3))")
def alpha_005(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * TS_MAX(CORR(TS_RANK(d.volume, 5), TS_RANK(d.high, 5), 5), 3)


@alpha_factor(6, 10, FAMILY_PRICE_VOLUME, "(RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15, 4)))*-1)")
def alpha_006(d: Alpha191Data) -> pd.DataFrame:
    val = d.open * 0.85 + d.high * 0.15
    return -1.0 * RANK(SIGN(DELTA(val, 4)))


@alpha_factor(7, 10, FAMILY_PRICE_VOLUME, "((RANK(TSMAX(VWAP-CLOSE,3))+RANK(TSMIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3)))")
def alpha_007(d: Alpha191Data) -> pd.DataFrame:
    rkmax = RANK(TS_MAX(d.vwap - d.close, 3))
    rkmin = RANK(TS_MIN(d.vwap - d.close, 3))
    rkdelta = RANK(DELTA(d.volume, 3))
    return (rkmax + rkmin) * rkdelta


@alpha_factor(8, 10, FAMILY_PRICE_VOLUME, "RANK(DELTA(((HIGH+LOW)/2)*0.2+VWAP*0.8, 4)*-1)")
def alpha_008(d: Alpha191Data) -> pd.DataFrame:
    val = (d.high + d.low) / 2.0 * 0.2 + d.vwap * 0.8
    return RANK(-1.0 * DELTA(val, 4))


@alpha_factor(9, 10, FAMILY_PRICE_VOLUME, "SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)")
def alpha_009(d: Alpha191Data) -> pd.DataFrame:
    part1 = (d.high + d.low) / 2.0 - (DELAY(d.high, 1) + DELAY(d.low, 1)) / 2.0
    part2 = (d.high - d.low) / d.volume.replace(0.0, np.nan)
    return SMA(part1 * part2, 7, 2)


@alpha_factor(10, 30, FAMILY_VOLATILITY, "(RANK(TSMAX(((RET<0)?STD(RET,20):CLOSE)^2,5)))")
def alpha_010(d: Alpha191Data) -> pd.DataFrame:
    ret = d.returns
    std20 = STD(ret, 20)
    inner = np.where(ret < 0, std20, d.close) ** 2
    inner = pd.DataFrame(inner, index=d.close.index, columns=d.close.columns)
    return RANK(TS_MAX(inner, 5))


@alpha_factor(11, 10, FAMILY_PRICE_VOLUME, "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)")
def alpha_011(d: Alpha191Data) -> pd.DataFrame:
    part = (2.0 * d.close - d.low - d.high) / (d.high - d.low)
    return SUM(part * d.volume, 6)


@alpha_factor(12, 20, FAMILY_PRICE_VOLUME, "(RANK(OPEN-SUM(VWAP,10)/10))*(-1*RANK(ABS(CLOSE-VWAP)))")
def alpha_012(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(d.open - MEAN(d.vwap, 10))
    part2 = -1.0 * RANK(ABS(d.close - d.vwap))
    return part1 * part2


@alpha_factor(13, 10, FAMILY_PRICE_VOLUME, "((HIGH*LOW)^0.5)-VWAP")
def alpha_013(d: Alpha191Data) -> pd.DataFrame:
    return (d.high * d.low) ** 0.5 - d.vwap


@alpha_factor(14, 10, FAMILY_TREND, "CLOSE-DELAY(CLOSE,5)")
def alpha_014(d: Alpha191Data) -> pd.DataFrame:
    return d.close - DELAY(d.close, 5)


@alpha_factor(15, 10, FAMILY_TREND, "OPEN/DELAY(CLOSE,1)-1")
def alpha_015(d: Alpha191Data) -> pd.DataFrame:
    return d.open / DELAY(d.close, 1) - 1.0


@alpha_factor(16, 20, FAMILY_PRICE_VOLUME, "(-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))")
def alpha_016(d: Alpha191Data) -> pd.DataFrame:
    corr = CORR(RANK(d.volume), RANK(d.vwap), 5)
    return -1.0 * TS_MAX(RANK(corr), 5)


@alpha_factor(17, 20, FAMILY_PRICE_VOLUME, "RANK(VWAP-TSMAX(VWAP,15))^DELTA(CLOSE,5)")
def alpha_017(d: Alpha191Data) -> pd.DataFrame:
    part = RANK(d.vwap - TS_MAX(d.vwap, 15))
    return part ** DELTA(d.close, 5)


@alpha_factor(18, 10, FAMILY_TREND, "CLOSE/DELAY(CLOSE,5)")
def alpha_018(d: Alpha191Data) -> pd.DataFrame:
    return d.close / DELAY(d.close, 5)


@alpha_factor(19, 10, FAMILY_TREND, "(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))")
def alpha_019(d: Alpha191Data) -> pd.DataFrame:
    delay5 = DELAY(d.close, 5)
    cond_up = d.close < delay5
    part2 = (d.close - delay5) / d.close
    out = np.where(cond_up, (d.close - delay5) / delay5, part2)
    out = pd.DataFrame(np.where(d.close == delay5, 0.0, out), index=d.close.index, columns=d.close.columns)
    return out


@alpha_factor(20, 10, FAMILY_TREND, "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100")
def alpha_020(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - DELAY(d.close, 6)) / DELAY(d.close, 6) * 100.0


@alpha_factor(21, 10, FAMILY_TREND, "REGBETA(MEAN(CLOSE,6), SEQUENCE(6))")
def alpha_021(d: Alpha191Data) -> pd.DataFrame:
    return rolling_slope(MEAN(d.close, 6), 6)


@alpha_factor(22, 30, FAMILY_TREND, "SMA((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3),12,1)")
def alpha_022(d: Alpha191Data) -> pd.DataFrame:
    val = (d.close - MEAN(d.close, 6)) / MEAN(d.close, 6)
    return SMA(val - DELAY(val, 3), 12, 1)


@alpha_factor(23, 50, FAMILY_TREND, "SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100")
def alpha_023(d: Alpha191Data) -> pd.DataFrame:
    cond = d.close > DELAY(d.close, 1)
    std20 = STD(d.close, 20)
    part1 = IFELSE(cond, std20, 0.0)
    part2 = IFELSE(~cond, std20, 0.0)
    sma1 = SMA(part1, 20, 1)
    sma2 = SMA(part2, 20, 1)
    return sma1 / (sma1 + sma2) * 100.0


@alpha_factor(24, 10, FAMILY_TREND, "SMA(CLOSE-DELAY(CLOSE,5),5,1)")
def alpha_024(d: Alpha191Data) -> pd.DataFrame:
    return SMA(d.close - DELAY(d.close, 5), 5, 1)


@alpha_factor(25, 300, FAMILY_TREND, "((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR(VOLUME/MEAN(VOLUME,20),9))))))*(1+RANK(SUM(RET,250))))")
def alpha_025(d: Alpha191Data) -> pd.DataFrame:
    part1 = -1.0 * RANK(DELTA(d.close, 7) * (1.0 - RANK(DECAYLINEAR(d.volume / (MEAN(d.volume, 20) + 1e-12), 9))))
    part2 = 1.0 + RANK(SUM(d.returns, 250))
    return part1 * part2


@alpha_factor(26, 300, FAMILY_TREND, "(SUM(CLOSE,7)/7-CLOSE)+CORR(VWAP,DELAY(CLOSE,5),230)")
def alpha_026(d: Alpha191Data) -> pd.DataFrame:
    return (MEAN(d.close, 7) - d.close) + CORR(d.vwap, DELAY(d.close, 5), 230)


@alpha_factor(27, 30, FAMILY_TREND, "WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)")
def alpha_027(d: Alpha191Data) -> pd.DataFrame:
    short = (d.close - DELAY(d.close, 3)) / DELAY(d.close, 3) * 100.0
    long = (d.close - DELAY(d.close, 6)) / DELAY(d.close, 6) * 100.0
    return WMA(short + long, 12)


@alpha_factor(28, 30, FAMILY_TREND, "3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)")
def alpha_028(d: Alpha191Data) -> pd.DataFrame:
    low_min = TS_MIN(d.low, 9)
    high_max = TS_MAX(d.high, 9)
    base = (d.close - low_min) / (high_max - low_min) * 100.0
    sma1 = SMA(base, 3, 1)
    return 3.0 * sma1 - 2.0 * SMA(sma1, 3, 1)


@alpha_factor(29, 10, FAMILY_TREND, "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME")
def alpha_029(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - DELAY(d.close, 6)) / DELAY(d.close, 6) * d.volume


@alpha_factor(30, 100, FAMILY_BENCHMARK, "WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)  # MKT=benchmark ret, SMB=HML=0")
def alpha_030(d: Alpha191Data) -> pd.DataFrame:
    if d.benchmark_close is None:
        return _nan_like(d.close)
    ret = d.returns
    mkt = d.benchmark_close.pct_change()  # Series over dates
    r60 = ret.rolling(60).mean()
    m60 = mkt.rolling(60).mean()
    rm60 = ret.mul(mkt, axis=0).rolling(60).mean()
    m2_60 = (mkt ** 2).rolling(60).mean()
    var_m = m2_60 - m60 ** 2
    cov_rm = rm60 - r60.mul(m60, axis=0)
    beta = cov_rm.div(var_m, axis=0)
    alpha_ = r60 - beta.mul(m60, axis=0)
    resid = ret - alpha_ - beta.mul(mkt, axis=0)
    return WMA(resid ** 2, 20)


@alpha_factor(31, 30, FAMILY_TREND, "(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100")
def alpha_031(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - MEAN(d.close, 12)) / MEAN(d.close, 12) * 100.0


@alpha_factor(32, 30, FAMILY_PRICE_VOLUME, "(-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))")
def alpha_032(d: Alpha191Data) -> pd.DataFrame:
    corr = CORR(RANK(d.high), RANK(d.volume), 3)
    return -1.0 * SUM(RANK(corr), 3)


@alpha_factor(33, 300, FAMILY_PRICE_VOLUME, "((-1*TSMIN(LOW,5))+DELAY(TSMIN(LOW,5),5))*RANK(((SUM(RET,240)-SUM(RET,20))/220))*TSRANK(VOLUME,5)")
def alpha_033(d: Alpha191Data) -> pd.DataFrame:
    low_min = TS_MIN(d.low, 5)
    ret_ratio = (SUM(d.returns, 240) - SUM(d.returns, 20)) / 220.0
    return (-1.0 * low_min + DELAY(low_min, 5)) * RANK(ret_ratio) * TS_RANK(d.volume, 5)


@alpha_factor(34, 30, FAMILY_TREND, "MEAN(CLOSE,12)/CLOSE")
def alpha_034(d: Alpha191Data) -> pd.DataFrame:
    return MEAN(d.close, 12) / d.close


@alpha_factor(35, 40, FAMILY_PRICE_VOLUME, "(MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)), RANK(DECAYLINEAR(CORR(VOLUME, OPEN, 17), 7)))*-1)")
def alpha_035(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(DELTA(d.open, 1), 15))
    part2 = RANK(DECAYLINEAR(CORR(d.volume, d.open, 17), 7))
    return MIN(part1, part2) * -1.0


@alpha_factor(36, 30, FAMILY_PRICE_VOLUME, "RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP),6),2))")
def alpha_036(d: Alpha191Data) -> pd.DataFrame:
    corr = CORR(RANK(d.volume), RANK(d.vwap), 6)
    return RANK(SUM(corr, 2))


@alpha_factor(37, 30, FAMILY_PRICE_VOLUME, "(-1*RANK(SUM(OPEN,5)*SUM(RET,5)-DELAY(SUM(OPEN,5)*SUM(RET,5),10)))")
def alpha_037(d: Alpha191Data) -> pd.DataFrame:
    part = SUM(d.open, 5) * SUM(d.returns, 5)
    return -1.0 * RANK(part - DELAY(part, 10))


@alpha_factor(38, 30, FAMILY_TREND, "((MEAN(HIGH,20)<HIGH)?(-1*DELTA(HIGH,2)):0)")
def alpha_038(d: Alpha191Data) -> pd.DataFrame:
    cond = MEAN(d.high, 20) < d.high
    return IFELSE(cond, -1.0 * DELTA(d.high, 2), 0.0)


@alpha_factor(39, 300, FAMILY_PRICE_VOLUME, "(RANK(DECAYLINEAR(DELTA(CLOSE,2),8))-RANK(DECAYLINEAR(CORR(VWAP*0.3+OPEN*0.7, SUM(MEAN(VOLUME,180),37),14),12)))*-1")
def alpha_039(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(DELTA(d.close, 2), 8))
    part2 = RANK(DECAYLINEAR(CORR(d.vwap * 0.3 + d.open * 0.7, SUM(MEAN(d.volume, 180), 37), 14), 12))
    return -1.0 * (part1 - part2)


@alpha_factor(40, 50, FAMILY_PRICE_VOLUME, "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100")
def alpha_040(d: Alpha191Data) -> pd.DataFrame:
    cond = d.close > DELAY(d.close, 1)
    part1 = SUM(IFELSE(cond, d.volume, 0.0), 26)
    part2 = SUM(IFELSE(~cond, d.volume, 0.0), 26)
    return part1 / (part2 + 1e-12) * 100.0


@alpha_factor(41, 30, FAMILY_PRICE_VOLUME, "(RANK(TSMAX(DELTA(VWAP,3),5))*-1)")
def alpha_041(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * RANK(TS_MAX(DELTA(d.vwap, 3), 5))


@alpha_factor(42, 30, FAMILY_PRICE_VOLUME, "(-1*RANK(STD(HIGH,10)))*CORR(HIGH,VOLUME,10)")
def alpha_042(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * RANK(STD(d.high, 10)) * CORR(d.high, d.volume, 10)


@alpha_factor(43, 10, FAMILY_PRICE_VOLUME, "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)")
def alpha_043(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part = IFELSE(d.close < delay1, -d.volume, 0.0)
    part = IFELSE(d.close > delay1, d.volume, part)
    return SUM(part, 6)


@alpha_factor(44, 40, FAMILY_PRICE_VOLUME, "TSRANK(DECAYLINEAR(CORR(LOW,MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA(VWAP,3),10),15)")
def alpha_044(d: Alpha191Data) -> pd.DataFrame:
    part1 = TS_RANK(DECAYLINEAR(CORR(d.low, MEAN(d.volume, 10), 7), 6), 4)
    part2 = TS_RANK(DECAYLINEAR(DELTA(d.vwap, 3), 10), 15)
    return part1 + part2


@alpha_factor(45, 300, FAMILY_PRICE_VOLUME, "RANK(DELTA(CLOSE*0.6+OPEN*0.4,1))*RANK(CORR(VWAP,MEAN(VOLUME,150),15))")
def alpha_045(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DELTA(d.close * 0.6 + d.open * 0.4, 1))
    part2 = RANK(CORR(d.vwap, MEAN(d.volume, 150), 15))
    return part1 * part2


@alpha_factor(46, 30, FAMILY_TREND, "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)")
def alpha_046(d: Alpha191Data) -> pd.DataFrame:
    return (MEAN(d.close, 3) + MEAN(d.close, 6) + MEAN(d.close, 12) + MEAN(d.close, 24)) / (4.0 * d.close)


@alpha_factor(47, 30, FAMILY_PRICE_VOLUME, "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)")
def alpha_047(d: Alpha191Data) -> pd.DataFrame:
    high_max = TS_MAX(d.high, 6)
    base = (high_max - d.close) / (high_max - TS_MIN(d.low, 6)) * 100.0
    return SMA(base, 9, 1)


@alpha_factor(48, 30, FAMILY_PRICE_VOLUME, "-1*(RANK(SIGN(CLOSE-DELAY(CLOSE,1))+SIGN(DELAY(CLOSE,1)-DELAY(CLOSE,2))+SIGN(DELAY(CLOSE,2)-DELAY(CLOSE,3)))*SUM(VOLUME,5))/SUM(VOLUME,20)")
def alpha_048(d: Alpha191Data) -> pd.DataFrame:
    sign_sum = SIGN(DELTA(d.close, 1)) + SIGN(DELTA(DELAY(d.close, 1), 1)) + SIGN(DELTA(DELAY(d.close, 2), 1))
    return -1.0 * RANK(sign_sum) * SUM(d.volume, 5) / SUM(d.volume, 20)


@alpha_factor(49, 30, FAMILY_PRICE_VOLUME, "SUM((HIGH+LOW>=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(...)+SUM(...))")
def alpha_049(d: Alpha191Data) -> pd.DataFrame:
    prev = DELAY(d.high, 1) + DELAY(d.low, 1)
    part = np.maximum(ABS(d.high - DELAY(d.high, 1)), ABS(d.low - DELAY(d.low, 1)))
    sum_down = SUM(IFELSE((d.high + d.low) < prev, part, 0.0), 12)
    sum_up = SUM(IFELSE((d.high + d.low) > prev, part, 0.0), 12)
    return sum_down / (sum_down + sum_up + 1e-12)


@alpha_factor(50, 30, FAMILY_PRICE_VOLUME, "SUM((HIGH+LOW<=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(...)-SUM((HIGH+LOW>=...?0:...),12)/(...)")
def alpha_050(d: Alpha191Data) -> pd.DataFrame:
    cond1 = (d.high + d.low) <= (DELAY(d.high, 1) + DELAY(d.low, 1))
    cond2 = (d.high + d.low) >= (DELAY(d.high, 1) + DELAY(d.low, 1))
    part = np.maximum(ABS(d.high - DELAY(d.high, 1)), ABS(d.low - DELAY(d.low, 1)))
    sum1 = SUM(IFELSE(~cond1, part, 0.0), 12)
    sum2 = SUM(IFELSE(~cond2, part, 0.0), 12)
    return (sum1 - sum2) / (sum1 + sum2 + 1e-12)


@alpha_factor(51, 30, FAMILY_PRICE_VOLUME, "SUM((HIGH+LOW<=DELAY(HIGH,1)+DELAY(LOW,1)?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(...)+SUM(...))")
def alpha_051(d: Alpha191Data) -> pd.DataFrame:
    cond1 = (d.high + d.low) <= (DELAY(d.high, 1) + DELAY(d.low, 1))
    cond2 = (d.high + d.low) >= (DELAY(d.high, 1) + DELAY(d.low, 1))
    part = np.maximum(ABS(d.high - DELAY(d.high, 1)), ABS(d.low - DELAY(d.low, 1)))
    sum1 = SUM(IFELSE(~cond1, part, 0.0), 12)
    sum2 = SUM(IFELSE(~cond2, part, 0.0), 12)
    return sum1 / (sum1 + sum2 + 1e-12)


@alpha_factor(52, 50, FAMILY_PRICE_VOLUME, "SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100")
def alpha_052(d: Alpha191Data) -> pd.DataFrame:
    typ = (d.high + d.low + d.close) / 3.0
    delay_typ = DELAY(typ, 1)
    part1 = SUM(MAX(0.0, d.high - delay_typ), 26)
    part2 = SUM(MAX(0.0, delay_typ - d.low), 26)
    return part1 / (part2 + 1e-12) * 100.0


@alpha_factor(53, 30, FAMILY_TREND, "COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100")
def alpha_053(d: Alpha191Data) -> pd.DataFrame:
    na_map = d.close.isna().astype(float).replace(1.0, np.nan)
    alpha = COUNT(d.close > DELAY(d.close, 1), 12, na_map=na_map) / 12.0 * 100.0
    return alpha + na_map


@alpha_factor(54, 30, FAMILY_VOLATILITY, "(-1*RANK(STD(ABS(CLOSE-OPEN),10)+(CLOSE-OPEN)+CORR(CLOSE,OPEN,10)))")
def alpha_054(d: Alpha191Data) -> pd.DataFrame:
    part = STD(ABS(d.close - d.open), 10) + (d.close - d.open) + CORR(d.close, d.open, 10)
    return -1.0 * RANK(part)


@alpha_factor(55, 50, FAMILY_PRICE_VOLUME, "SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((cond)?...)*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)")
def alpha_055(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part1 = ABS(d.high - delay1)
    part2 = ABS(d.low - delay1)
    part3 = ABS(d.high - DELAY(d.low, 1))
    part4 = ABS(delay1 - DELAY(d.open, 1))
    var1 = part1 + part2 / 2.0 + part4 / 4.0
    var2 = part2 + part1 / 2.0 + part4 / 4.0
    var3 = part3 + part4 / 4.0
    denom = IFELSE((part1 > part2) & (part1 > part3), var1, IFELSE((part2 > part3) & (part2 > part1), var2, var3))
    numerator = 16.0 * (d.close - delay1 + (d.close - d.open) / 2.0 + delay1 - DELAY(d.open, 1))
    alpha = numerator / (denom + 1e-12) * np.maximum(part1, part2)
    return SUM(alpha, 20)


@alpha_factor(56, 100, FAMILY_TREND, "(RANK(OPEN-TSMIN(OPEN,12)) < RANK(RANK(CORR(SUM((HIGH+LOW)/2,19), SUM(MEAN(VOLUME,40),19),13))^5))")
def alpha_056(d: Alpha191Data) -> pd.DataFrame:
    na_map = d.open.isna().astype(float).replace(1.0, np.nan)
    left = RANK(d.open - TS_MIN(d.open, 12))
    right = RANK(RANK(CORR(SUM((d.high + d.low) / 2.0, 19), SUM(MEAN(d.volume, 40), 19), 13)) ** 5)
    alpha = (left < right).astype(float)
    return alpha + na_map


@alpha_factor(57, 30, FAMILY_TREND, "SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)")
def alpha_057(d: Alpha191Data) -> pd.DataFrame:
    base = (d.close - TS_MIN(d.low, 9)) / (TS_MAX(d.high, 9) - TS_MIN(d.low, 9)) * 100.0
    return SMA(base, 3, 1)


@alpha_factor(58, 30, FAMILY_TREND, "COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100")
def alpha_058(d: Alpha191Data) -> pd.DataFrame:
    na_map = d.open.isna().astype(float).replace(1.0, np.nan)
    alpha = COUNT(d.close > DELAY(d.close, 1), 20, na_map=na_map) / 20.0 * 100.0
    return alpha + na_map


@alpha_factor(59, 50, FAMILY_PRICE_VOLUME, "SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)")
def alpha_059(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    cond_up = d.close > delay1
    cond_down = d.close < delay1
    inner = np.where(cond_up, d.close - MIN(d.low, delay1), d.close - MAX(d.high, delay1))
    inner = pd.DataFrame(np.where(d.close == delay1, 0.0, inner), index=d.close.index, columns=d.close.columns)
    return SUM(inner, 20)


@alpha_factor(60, 50, FAMILY_PRICE_VOLUME, "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)")
def alpha_060(d: Alpha191Data) -> pd.DataFrame:
    part = (2.0 * d.close - d.low - d.high) / (d.high - d.low)
    return SUM(part * d.volume, 20)


@alpha_factor(61, 100, FAMILY_PRICE_VOLUME, "(MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)), RANK(DECAYLINEAR(RANK(CORR(LOW,MEAN(VOLUME,80),8)),17)))*-1)")
def alpha_061(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(DELTA(d.vwap, 1), 12))
    part2 = RANK(DECAYLINEAR(RANK(CORR(d.low, MEAN(d.volume, 80), 8)), 17))
    return MAX(part1, part2) * -1.0


@alpha_factor(62, 30, FAMILY_PRICE_VOLUME, "(-1*CORR(HIGH,RANK(VOLUME),5))")
def alpha_062(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * CORR(d.high, RANK(d.volume), 5)


@alpha_factor(63, 30, FAMILY_PRICE_VOLUME, "SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100")
def alpha_063(d: Alpha191Data) -> pd.DataFrame:
    part = d.close - DELAY(d.close, 1)
    return SMA(MAX(part, 0.0), 6, 1) / SMA(ABS(part), 6, 1) * 100.0


@alpha_factor(64, 100, FAMILY_PRICE_VOLUME, "(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)), RANK(DECAYLINEAR(TSMAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14)))*-1)")
def alpha_064(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(CORR(RANK(d.vwap), RANK(d.volume), 4), 4))
    part2 = RANK(DECAYLINEAR(TS_MAX(CORR(RANK(d.close), RANK(MEAN(d.volume, 60)), 4), 13), 14))
    return MAX(part1, part2) * -1.0


@alpha_factor(65, 30, FAMILY_TREND, "MEAN(CLOSE,6)/CLOSE")
def alpha_065(d: Alpha191Data) -> pd.DataFrame:
    return MEAN(d.close, 6) / d.close


@alpha_factor(66, 30, FAMILY_TREND, "(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100")
def alpha_066(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - MEAN(d.close, 6)) / MEAN(d.close, 6) * 100.0


@alpha_factor(67, 50, FAMILY_PRICE_VOLUME, "SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100")
def alpha_067(d: Alpha191Data) -> pd.DataFrame:
    part = d.close - DELAY(d.close, 1)
    return SMA(MAX(part, 0.0), 24, 1) / SMA(ABS(part), 24, 1) * 100.0


@alpha_factor(68, 30, FAMILY_PRICE_VOLUME, "SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)")
def alpha_068(d: Alpha191Data) -> pd.DataFrame:
    part1 = (d.high + d.low) / 2.0 - (DELAY(d.high, 1) + DELAY(d.low, 1)) / 2.0
    part2 = (d.high - d.low) / d.volume.replace(0.0, np.nan)
    return SMA(part1 * part2, 15, 2)


@alpha_factor(69, 30, FAMILY_SESSION, "(SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)?0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))")
def alpha_069(d: Alpha191Data) -> pd.DataFrame:
    delay_open = DELAY(d.open, 1)
    dtm = IFELSE(d.open <= delay_open, 0.0, MAX(d.high - d.open, d.open - delay_open))
    dbm = IFELSE(d.open >= delay_open, 0.0, MAX(d.open - d.low, d.open - delay_open))
    sum_dtm = SUM(dtm, 20)
    sum_dbm = SUM(dbm, 20)
    diff = sum_dtm - sum_dbm
    out = np.where(sum_dtm > sum_dbm, diff / (sum_dtm + 1e-12), np.where(sum_dtm == sum_dbm, 0.0, diff / (sum_dbm + 1e-12)))
    return pd.DataFrame(out, index=d.close.index, columns=d.close.columns)


@alpha_factor(70, 30, FAMILY_VOLUME, "STD(AMOUNT,6)")
def alpha_070(d: Alpha191Data) -> pd.DataFrame:
    return STD(d.amount, 6)


@alpha_factor(71, 50, FAMILY_TREND, "(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100")
def alpha_071(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - MEAN(d.close, 24)) / MEAN(d.close, 24) * 100.0


@alpha_factor(72, 50, FAMILY_TREND, "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)")
def alpha_072(d: Alpha191Data) -> pd.DataFrame:
    high_max = TS_MAX(d.high, 6)
    base = (high_max - d.close) / (high_max - TS_MIN(d.low, 6)) * 100.0
    return SMA(base, 15, 1)


@alpha_factor(73, 50, FAMILY_PRICE_VOLUME, "((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE,VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))*-1)")
def alpha_073(d: Alpha191Data) -> pd.DataFrame:
    part1 = TS_RANK(DECAYLINEAR(DECAYLINEAR(CORR(d.close, d.volume, 10), 16), 4), 5)
    part2 = RANK(DECAYLINEAR(CORR(d.vwap, MEAN(d.volume, 30), 4), 3))
    return -1.0 * (part1 - part2)


@alpha_factor(74, 100, FAMILY_PRICE_VOLUME, "RANK(CORR(SUM(LOW*0.35+VWAP*0.65,20), SUM(MEAN(VOLUME,40),20),7))+RANK(CORR(RANK(VWAP),RANK(VOLUME),6))")
def alpha_074(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(CORR(SUM(d.low * 0.35 + d.vwap * 0.65, 20), SUM(MEAN(d.volume, 40), 20), 7))
    part2 = RANK(CORR(RANK(d.vwap), RANK(d.volume), 6))
    return part1 + part2


@alpha_factor(75, 100, FAMILY_BENCHMARK, "COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)")
def alpha_075(d: Alpha191Data) -> pd.DataFrame:
    if d.benchmark_close is None or d.benchmark_open is None:
        return _nan_like(d.close)
    na_map = d.close.isna().astype(float).replace(1.0, np.nan)
    bench_down = (d.benchmark_close < d.benchmark_open).astype(float)
    both = (d.close > d.open).astype(float).mul(bench_down, axis=0)
    denominator = (
        bench_down.reindex(d.close.index)
        .to_frame(d.close.columns[0])
        .reindex(columns=d.close.columns)
    )
    numerator = COUNT(both, 50, na_map=na_map)
    return numerator.div(COUNT(denominator, 50).replace(0.0, np.nan))


@alpha_factor(76, 50, FAMILY_VOLATILITY, "STD(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)/MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/VOLUME,20)")
def alpha_076(d: Alpha191Data) -> pd.DataFrame:
    part = ABS(d.close / DELAY(d.close, 1) - 1.0) / d.volume
    return STD(part, 20) / (MEAN(part, 20) + 1e-12)


@alpha_factor(77, 60, FAMILY_PRICE_VOLUME, "MIN(RANK(DECAYLINEAR(((HIGH+LOW)/2+HIGH)-(VWAP+HIGH),20)), RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),3),6)))")
def alpha_077(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR((((d.high + d.low) / 2.0 + d.high) - (d.vwap + d.high)), 20))
    part2 = RANK(DECAYLINEAR(CORR((d.high + d.low) / 2.0, MEAN(d.volume, 40), 3), 6))
    return MIN(part1, part2)


@alpha_factor(78, 50, FAMILY_VOLATILITY, "((HIGH+LOW+CLOSE)/3-MEAN((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))")
def alpha_078(d: Alpha191Data) -> pd.DataFrame:
    typ = (d.high + d.low + d.close) / 3.0
    ma = MEAN(typ, 12)
    denom = 0.015 * MEAN(ABS(d.close - ma), 12)
    return (typ - ma) / (denom + 1e-12)


@alpha_factor(79, 50, FAMILY_PRICE_VOLUME, "SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100")
def alpha_079(d: Alpha191Data) -> pd.DataFrame:
    part = d.close - DELAY(d.close, 1)
    return SMA(MAX(part, 0.0), 12, 1) / SMA(ABS(part), 12, 1) * 100.0


@alpha_factor(80, 30, FAMILY_VOLUME, "(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100")
def alpha_080(d: Alpha191Data) -> pd.DataFrame:
    return (d.volume - DELAY(d.volume, 5)) / DELAY(d.volume, 5) * 100.0


@alpha_factor(81, 50, FAMILY_VOLUME, "SMA(VOLUME,21,2)")
def alpha_081(d: Alpha191Data) -> pd.DataFrame:
    return SMA(d.volume, 21, 2)


@alpha_factor(82, 50, FAMILY_TREND, "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)")
def alpha_082(d: Alpha191Data) -> pd.DataFrame:
    high_max = TS_MAX(d.high, 6)
    base = (high_max - d.close) / (high_max - TS_MIN(d.low, 6)) * 100.0
    return SMA(base, 20, 1)


@alpha_factor(83, 50, FAMILY_PRICE_VOLUME, "(-1*RANK(COVARIANCE(RANK(HIGH),RANK(VOLUME),5)))")
def alpha_083(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * RANK(COVARIANCE(RANK(d.high), RANK(d.volume), 5))


@alpha_factor(84, 50, FAMILY_PRICE_VOLUME, "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)")
def alpha_084(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part = IFELSE(d.close < delay1, -d.volume, 0.0)
    part = IFELSE(d.close > delay1, d.volume, part)
    return SUM(part, 20)


@alpha_factor(85, 50, FAMILY_PRICE_VOLUME, "TSRANK(VOLUME/MEAN(VOLUME,20),20)*TSRANK(-1*DELTA(CLOSE,7),8)")
def alpha_085(d: Alpha191Data) -> pd.DataFrame:
    part1 = TS_RANK(d.volume / (MEAN(d.volume, 20) + 1e-12), 20)
    part2 = TS_RANK(-1.0 * DELTA(d.close, 7), 8)
    return part1 * part2


@alpha_factor(86, 50, FAMILY_TREND, "((0.25<(part1-part2))?-1:((part1-part2)<0?1:(-1*(CLOSE-DELAY(CLOSE,1)))))")
def alpha_086(d: Alpha191Data) -> pd.DataFrame:
    part1 = (DELAY(d.close, 20) - DELAY(d.close, 10)) / 10.0
    part2 = (DELAY(d.close, 10) - d.close) / 10.0
    diff = part1 - part2
    cond1 = diff > 0.25
    cond2 = diff < 0.0
    default = -1.0 * (d.close - DELAY(d.close, 1))
    out = np.select([cond1, cond2], [-1.0, 1.0], default=default)
    return pd.DataFrame(out, index=d.close.index, columns=d.close.columns)


@alpha_factor(87, 30, FAMILY_PRICE_VOLUME, "(RANK(DECAYLINEAR(DELTA(VWAP,4),7))+TSRANK(DECAYLINEAR(((LOW*0.9+LOW*0.1)-VWAP)/(OPEN-((HIGH+LOW)/2)),11),7))*-1")
def alpha_087(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(DELTA(d.vwap, 4), 7))
    denom = d.open - (d.high + d.low) / 2.0
    part2 = TS_RANK(DECAYLINEAR((d.low - d.vwap) / (denom + 1e-12), 11), 7)
    return -1.0 * (part1 + part2)


@alpha_factor(88, 50, FAMILY_TREND, "(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100")
def alpha_088(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - DELAY(d.close, 20)) / DELAY(d.close, 20) * 100.0


@alpha_factor(89, 50, FAMILY_TREND, "2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))")
def alpha_089(d: Alpha191Data) -> pd.DataFrame:
    ma_short = SMA(d.close, 13, 2)
    ma_long = SMA(d.close, 27, 2)
    return 2.0 * (ma_short - ma_long - SMA(ma_short - ma_long, 10, 2))


@alpha_factor(90, 50, FAMILY_PRICE_VOLUME, "(RANK(CORR(RANK(VWAP),RANK(VOLUME),5))*-1)")
def alpha_090(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * RANK(CORR(RANK(d.vwap), RANK(d.volume), 5))


@alpha_factor(91, 50, FAMILY_PRICE_VOLUME, "((RANK(CLOSE-TSMAX(CLOSE,5))*RANK(CORR(MEAN(VOLUME,40),LOW,5)))*-1)")
def alpha_091(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(d.close - TS_MAX(d.close, 5))
    part2 = RANK(CORR(MEAN(d.volume, 40), d.low, 5))
    return -1.0 * part1 * part2


@alpha_factor(92, 300, FAMILY_PRICE_VOLUME, "(MAX(RANK(DECAYLINEAR(DELTA(CLOSE*0.35+VWAP*0.65,2),3)), TSRANK(DECAYLINEAR(ABS(CORR(MEAN(VOLUME,180),CLOSE,13)),5),15))*-1)")
def alpha_092(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(DELTA(d.close * 0.35 + d.vwap * 0.65, 2), 3))
    part2 = TS_RANK(DECAYLINEAR(ABS(CORR(MEAN(d.volume, 180), d.close, 13)), 5), 15)
    return MAX(part1, part2) * -1.0


@alpha_factor(93, 50, FAMILY_SESSION, "SUM((OPEN>=DELAY(OPEN,1)?0:MAX(OPEN-LOW,OPEN-DELAY(OPEN,1))),20)")
def alpha_093(d: Alpha191Data) -> pd.DataFrame:
    delay_open = DELAY(d.open, 1)
    part = IFELSE(d.open >= delay_open, 0.0, MAX(d.open - d.low, d.open - delay_open))
    return SUM(part, 20)


@alpha_factor(94, 50, FAMILY_PRICE_VOLUME, "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)")
def alpha_094(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part = IFELSE(d.close < delay1, -d.volume, 0.0)
    part = IFELSE(d.close > delay1, d.volume, part)
    return SUM(part, 30)


@alpha_factor(95, 50, FAMILY_VOLUME, "STD(AMOUNT,20)")
def alpha_095(d: Alpha191Data) -> pd.DataFrame:
    return STD(d.amount, 20)


@alpha_factor(96, 50, FAMILY_TREND, "SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)")
def alpha_096(d: Alpha191Data) -> pd.DataFrame:
    base = (d.close - TS_MIN(d.low, 9)) / (TS_MAX(d.high, 9) - TS_MIN(d.low, 9)) * 100.0
    return SMA(SMA(base, 3, 1), 3, 1)


@alpha_factor(97, 30, FAMILY_VOLUME, "STD(VOLUME,10)")
def alpha_097(d: Alpha191Data) -> pd.DataFrame:
    return STD(d.volume, 10)


@alpha_factor(98, 200, FAMILY_TREND, "(((DELTA(SUM(CLOSE,100)/100,100)/DELAY(CLOSE,100))<=0.05)?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3)))")
def alpha_098(d: Alpha191Data) -> pd.DataFrame:
    condition = DELTA(MEAN(d.close, 100), 100) / DELAY(d.close, 100) <= 0.05
    return IFELSE(condition, -1.0 * (d.close - TS_MIN(d.close, 100)), -1.0 * DELTA(d.close, 3))


@alpha_factor(99, 50, FAMILY_PRICE_VOLUME, "(-1*RANK(COVARIANCE(RANK(CLOSE),RANK(VOLUME),5)))")
def alpha_099(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * RANK(COVARIANCE(RANK(d.close), RANK(d.volume), 5))


@alpha_factor(100, 50, FAMILY_VOLUME, "STD(VOLUME,20)")
def alpha_100(d: Alpha191Data) -> pd.DataFrame:
    return STD(d.volume, 20)


@alpha_factor(101, 100, FAMILY_PRICE_VOLUME, "((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(HIGH*0.1+VWAP*0.9),RANK(VOLUME),11)))*-1)")
def alpha_101(d: Alpha191Data) -> pd.DataFrame:
    left = RANK(CORR(d.close, SUM(MEAN(d.volume, 30), 37), 15))
    right = RANK(CORR(RANK(d.high * 0.1 + d.vwap * 0.9), RANK(d.volume), 11))
    alpha = (left < right).astype(float) * -1.0
    return alpha


# ---------------------------------------------------------------------------
# Alpha102 - Alpha191 (extended set)
# ---------------------------------------------------------------------------


@alpha_factor(102, 30, FAMILY_VOLUME, "SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100")
def alpha_102(d: Alpha191Data) -> pd.DataFrame:
    part = d.volume - DELAY(d.volume, 1)
    return SMA(MAX(part, 0.0), 6, 1) / SMA(ABS(part), 6, 1) * 100.0


@alpha_factor(103, 50, FAMILY_PRICE_VOLUME, "((20-LOWDAY(LOW,20))/20)*100")
def alpha_103(d: Alpha191Data) -> pd.DataFrame:
    return (20.0 - LOWDAY(d.low, 20)) / 20.0 * 100.0


@alpha_factor(104, 50, FAMILY_VOLATILITY, "(-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))")
def alpha_104(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * (DELTA(CORR(d.high, d.volume, 5), 5) * RANK(STD(d.close, 20)))


@alpha_factor(105, 30, FAMILY_PRICE_VOLUME, "(-1*CORR(RANK(OPEN),RANK(VOLUME),10))")
def alpha_105(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * CORR(RANK(d.open), RANK(d.volume), 10)


@alpha_factor(106, 50, FAMILY_TREND, "CLOSE-DELAY(CLOSE,20)")
def alpha_106(d: Alpha191Data) -> pd.DataFrame:
    return d.close - DELAY(d.close, 20)


@alpha_factor(107, 50, FAMILY_PRICE_VOLUME, "((-1*RANK(OPEN-DELAY(HIGH,1)))*RANK(OPEN-DELAY(CLOSE,1)))*RANK(OPEN-DELAY(LOW,1))")
def alpha_107(d: Alpha191Data) -> pd.DataFrame:
    part1 = -1.0 * RANK(d.open - DELAY(d.high, 1))
    part2 = RANK(d.open - DELAY(d.close, 1))
    part3 = RANK(d.open - DELAY(d.low, 1))
    return part1 * part2 * part3


@alpha_factor(108, 200, FAMILY_PRICE_VOLUME, "((RANK(HIGH-TSMIN(HIGH,2))^RANK(CORR(VWAP,MEAN(VOLUME,120),6)))*-1)")
def alpha_108(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(d.high - TS_MIN(d.high, 2))
    part2 = RANK(CORR(d.vwap, MEAN(d.volume, 120), 6))
    return -1.0 * (part1 ** part2)


@alpha_factor(109, 50, FAMILY_PRICE_VOLUME, "SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)")
def alpha_109(d: Alpha191Data) -> pd.DataFrame:
    rng = d.high - d.low
    return SMA(rng, 10, 2) / SMA(SMA(rng, 10, 2), 10, 2)


@alpha_factor(110, 50, FAMILY_PRICE_VOLUME, "SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100")
def alpha_110(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part1 = SUM(MAX(0.0, d.high - delay1), 20)
    part2 = SUM(MAX(0.0, delay1 - d.low), 20)
    return part1 / (part2 + 1e-12) * 100.0


@alpha_factor(111, 50, FAMILY_VOLUME, "SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)")
def alpha_111(d: Alpha191Data) -> pd.DataFrame:
    part = d.volume * ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)
    return SMA(part, 11, 2) - SMA(part, 4, 2)


@alpha_factor(112, 50, FAMILY_PRICE_VOLUME, "(SUM(MAX(CLOSE-DELAY(CLOSE,1),0),12)-SUM(MAX(DELAY(CLOSE,1)-CLOSE,0),12))/(SUM(MAX(CLOSE-DELAY(CLOSE,1),0),12)+SUM(MAX(DELAY(CLOSE,1)-CLOSE,0),12))*100")
def alpha_112(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part1 = SUM(MAX(d.close - delay1, 0.0), 12)
    part2 = SUM(MAX(delay1 - d.close, 0.0), 12)
    return (part1 - part2) / (part1 + part2 + 1e-12) * 100.0


@alpha_factor(113, 50, FAMILY_PRICE_VOLUME, "(-1*((RANK(SUM(DELAY(CLOSE,5),20)/20)*CORR(CLOSE,VOLUME,2))*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))")
def alpha_113(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(MEAN(DELAY(d.close, 5), 20))
    part2 = CORR(d.close, d.volume, 2)
    part3 = RANK(CORR(SUM(d.close, 5), SUM(d.close, 20), 2))
    return -1.0 * part1 * part2 * part3


@alpha_factor(114, 30, FAMILY_PRICE_VOLUME, "(RANK(DELAY((HIGH-LOW)/(SUM(CLOSE,5)/5),2))*RANK(RANK(VOLUME)))/(((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE))")
def alpha_114(d: Alpha191Data) -> pd.DataFrame:
    part = (d.high - d.low) / (MEAN(d.close, 5) + 1e-12)
    numerator = RANK(DELAY(part, 2)) * RANK(RANK(d.volume))
    return numerator / (part / (d.vwap - d.close + 1e-12))


@alpha_factor(115, 50, FAMILY_PRICE_VOLUME, "RANK(CORR(HIGH*0.9+CLOSE*0.1,MEAN(VOLUME,30),10))^RANK(CORR(TSRANK((HIGH+LOW)/2,4),TSRANK(VOLUME,10),7))")
def alpha_115(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(CORR(d.high * 0.9 + d.close * 0.1, MEAN(d.volume, 30), 10))
    part2 = RANK(CORR(TS_RANK((d.high + d.low) / 2.0, 4), TS_RANK(d.volume, 10), 7))
    return part1 ** part2


@alpha_factor(116, 50, FAMILY_TREND, "REGBETA(CLOSE,SEQUENCE(20))")
def alpha_116(d: Alpha191Data) -> pd.DataFrame:
    return rolling_slope(d.close, 20)


@alpha_factor(117, 50, FAMILY_TREND, "(TSRANK(VOLUME,32)*(1-TSRANK(CLOSE+HIGH-LOW,16)))*(1-TSRANK(RET,32))")
def alpha_117(d: Alpha191Data) -> pd.DataFrame:
    part1 = TS_RANK(d.volume, 32)
    part2 = 1.0 - TS_RANK(d.close + d.high - d.low, 16)
    part3 = 1.0 - TS_RANK(d.returns, 32)
    return part1 * part2 * part3


@alpha_factor(118, 50, FAMILY_PRICE_VOLUME, "SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100")
def alpha_118(d: Alpha191Data) -> pd.DataFrame:
    part1 = SUM(d.high - d.open, 20)
    part2 = SUM(d.open - d.low, 20)
    return part1 / (part2 + 1e-12) * 100.0


@alpha_factor(119, 100, FAMILY_PRICE_VOLUME, "RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(TSMIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8))")
def alpha_119(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(CORR(d.vwap, SUM(MEAN(d.volume, 5), 26), 5), 7))
    corr21 = CORR(RANK(d.open), RANK(MEAN(d.volume, 15)), 21)
    part2 = RANK(DECAYLINEAR(TS_RANK(TS_MIN(corr21, 9), 7), 8))
    return part1 - part2


@alpha_factor(120, 30, FAMILY_PRICE_VOLUME, "(RANK(VWAP-CLOSE)/RANK(VWAP+CLOSE))")
def alpha_120(d: Alpha191Data) -> pd.DataFrame:
    return RANK(d.vwap - d.close) / RANK(d.vwap + d.close)


@alpha_factor(121, 100, FAMILY_PRICE_VOLUME, "((RANK(VWAP-TSMIN(VWAP,12))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))*-1)")
def alpha_121(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(d.vwap - TS_MIN(d.vwap, 12))
    part2 = TS_RANK(CORR(TS_RANK(d.vwap, 20), TS_RANK(MEAN(d.volume, 60), 2), 18), 3)
    return -1.0 * (part1 ** part2)


@alpha_factor(122, 100, FAMILY_TREND, "(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)")
def alpha_122(d: Alpha191Data) -> pd.DataFrame:
    part = SMA(SMA(SMA(LOG(d.close), 13, 2), 13, 2), 13, 2)
    return (part - DELAY(part, 1)) / DELAY(part, 1)


@alpha_factor(123, 100, FAMILY_PRICE_VOLUME, "((RANK(CORR(SUM((HIGH+LOW)/2,20),SUM(MEAN(VOLUME,60),20),9))<RANK(CORR(LOW,VOLUME,6)))*-1)")
def alpha_123(d: Alpha191Data) -> pd.DataFrame:
    left = RANK(CORR(SUM((d.high + d.low) / 2.0, 20), SUM(MEAN(d.volume, 60), 20), 9))
    right = RANK(CORR(d.low, d.volume, 6))
    return (left < right).astype(float) * -1.0


@alpha_factor(124, 50, FAMILY_PRICE_VOLUME, "(CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)")
def alpha_124(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - d.vwap) / DECAYLINEAR(RANK(TS_MAX(d.close, 30)), 2)


@alpha_factor(125, 200, FAMILY_PRICE_VOLUME, "RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(CLOSE*0.5+VWAP*0.5,3),16))")
def alpha_125(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(CORR(d.vwap, MEAN(d.volume, 80), 17), 20))
    part2 = RANK(DECAYLINEAR(DELTA(d.close * 0.5 + d.vwap * 0.5, 3), 16))
    return part1 / part2


@alpha_factor(126, 10, FAMILY_TREND, "(CLOSE+HIGH+LOW)/3")
def alpha_126(d: Alpha191Data) -> pd.DataFrame:
    return (d.close + d.high + d.low) / 3.0


@alpha_factor(127, 50, FAMILY_TREND, "(MEAN((100*(CLOSE-TSMAX(CLOSE,12))/TSMAX(CLOSE,12))^2,12))^0.5")
def alpha_127(d: Alpha191Data) -> pd.DataFrame:
    high_max = TS_MAX(d.close, 12)
    part = 100.0 * (d.close - high_max) / high_max
    return (MEAN(part ** 2, 12)) ** 0.5


@alpha_factor(128, 50, FAMILY_TREND, "100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1))?(HIGH+LOW+CLOSE)/3*VOLUME:0,14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1))?(HIGH+LOW+CLOSE)/3*VOLUME:0,14)))")
def alpha_128(d: Alpha191Data) -> pd.DataFrame:
    typ = (d.high + d.low + d.close) / 3.0
    cond = typ > DELAY(typ, 1)
    part1 = SUM(IFELSE(cond, typ * d.volume, 0.0), 14)
    part2 = SUM(IFELSE(typ < DELAY(typ, 1), typ * d.volume, 0.0), 14)
    return 100.0 - (100.0 / (1.0 + part1 / (part2 + 1e-12)))


@alpha_factor(129, 50, FAMILY_TREND, "SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)")
def alpha_129(d: Alpha191Data) -> pd.DataFrame:
    part = d.close - DELAY(d.close, 1)
    return SUM(IFELSE(part < 0, ABS(part), 0.0), 12)


@alpha_factor(130, 100, FAMILY_PRICE_VOLUME, "RANK(DECAYLINEAR(CORR((HIGH+LOW)/2,MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3))")
def alpha_130(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(CORR((d.high + d.low) / 2.0, MEAN(d.volume, 40), 9), 10))
    part2 = RANK(DECAYLINEAR(CORR(RANK(d.vwap), RANK(d.volume), 7), 3))
    return part1 / part2


@alpha_factor(131, 100, FAMILY_PRICE_VOLUME, "(RANK(DELTA(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))")
def alpha_131(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DELTA(d.vwap, 1))
    part2 = TS_RANK(CORR(d.close, MEAN(d.volume, 50), 18), 18)
    return part1 ** part2


@alpha_factor(132, 50, FAMILY_VOLUME, "MEAN(AMOUNT,20)")
def alpha_132(d: Alpha191Data) -> pd.DataFrame:
    return MEAN(d.amount, 20)


@alpha_factor(133, 50, FAMILY_PRICE_VOLUME, "((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100")
def alpha_133(d: Alpha191Data) -> pd.DataFrame:
    return ((20.0 - HIGHDAY(d.high, 20)) / 20.0) * 100.0 - ((20.0 - LOWDAY(d.low, 20)) / 20.0) * 100.0


@alpha_factor(134, 50, FAMILY_VOLUME, "(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME")
def alpha_134(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - DELAY(d.close, 12)) / DELAY(d.close, 12) * d.volume


@alpha_factor(135, 50, FAMILY_TREND, "SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)")
def alpha_135(d: Alpha191Data) -> pd.DataFrame:
    return SMA(DELAY(d.close / DELAY(d.close, 20), 1), 20, 1)


@alpha_factor(136, 50, FAMILY_TREND, "((-1*RANK(DELTA(RET,3)))*CORR(OPEN,VOLUME,10))")
def alpha_136(d: Alpha191Data) -> pd.DataFrame:
    return (-1.0 * RANK(DELTA(d.returns, 3))) * CORR(d.open, d.volume, 10)


@alpha_factor(137, 50, FAMILY_SESSION, "16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/(cond?A:B)*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))")
def alpha_137(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    abshc = ABS(d.high - delay1)
    abslc = ABS(d.low - delay1)
    absco = ABS(delay1 - DELAY(d.open, 1))
    abshl = ABS(d.high - DELAY(d.low, 1))
    denom = IFELSE(
        (abshc > abslc) & (abshc > abshl),
        abshc + abslc / 2.0 + absco / 4.0,
        IFELSE((abslc > abshl) & (abslc > abshc), abslc + abshc / 2.0 + absco / 4.0, abshl + absco / 4.0),
    )
    numerator = 16.0 * (d.close - delay1 + (d.close - d.open) / 2.0 + delay1 - DELAY(d.open, 1))
    return numerator / (denom + 1e-12) * np.maximum(abshc, abslc)


@alpha_factor(138, 200, FAMILY_PRICE_VOLUME, "((RANK(DECAYLINEAR(DELTA(LOW*0.7+VWAP*0.3,3),20))-TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7))*-1)")
def alpha_138(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(DELTA(d.low * 0.7 + d.vwap * 0.3, 3), 20))
    corr = CORR(TS_RANK(d.low, 8), TS_RANK(MEAN(d.volume, 60), 17), 5)
    part2 = TS_RANK(DECAYLINEAR(TS_RANK(corr, 19), 16), 7)
    return -1.0 * (part1 - part2)


@alpha_factor(139, 50, FAMILY_PRICE_VOLUME, "(-1*CORR(OPEN,VOLUME,10))")
def alpha_139(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * CORR(d.open, d.volume, 10)


@alpha_factor(140, 150, FAMILY_PRICE_VOLUME, "MIN(RANK(DECAYLINEAR(RANK(OPEN)+RANK(LOW)-RANK(HIGH)-RANK(CLOSE),8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))")
def alpha_140(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(RANK(d.open) + RANK(d.low) - RANK(d.high) - RANK(d.close), 8))
    part2 = TS_RANK(DECAYLINEAR(CORR(TS_RANK(d.close, 8), TS_RANK(MEAN(d.volume, 60), 20), 8), 7), 3)
    return MIN(part1, part2)


@alpha_factor(141, 50, FAMILY_PRICE_VOLUME, "(RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))*-1)")
def alpha_141(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * RANK(CORR(RANK(d.high), RANK(MEAN(d.volume, 15)), 9))


@alpha_factor(142, 50, FAMILY_TREND, "((-1*RANK(TSRANK(CLOSE,10)))*RANK(DELTA(DELTA(CLOSE,1),1)))*RANK(TSRANK(VOLUME/MEAN(VOLUME,20),5))")
def alpha_142(d: Alpha191Data) -> pd.DataFrame:
    part1 = -1.0 * RANK(TS_RANK(d.close, 10))
    part2 = RANK(DELTA(DELTA(d.close, 1), 1))
    part3 = RANK(TS_RANK(d.volume / (MEAN(d.volume, 20) + 1e-12), 5))
    return part1 * part2 * part3


@alpha_factor(143, 50, FAMILY_TREND, "CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF")
def alpha_143(d: Alpha191Data) -> pd.DataFrame:
    close = d.close.to_numpy(dtype=float)
    delay1 = DELAY(d.close, 1).to_numpy(dtype=float)
    up = close > delay1
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(up, close / delay1 - 1.0, 1.0)
    m = np.where(np.isnan(m) & up, 0.0, m)  # degenerate divide by zero on up day
    log_m = np.log(np.where(m > 0, m, 1.0))
    log_m = np.where(up, log_m, 0.0)
    self_val = np.exp(np.cumsum(log_m, axis=0))
    return pd.DataFrame(self_val, index=d.close.index, columns=d.close.columns)


@alpha_factor(144, 50, FAMILY_PRICE_VOLUME, "SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)")
def alpha_144(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part = ABS(d.close / delay1 - 1.0) / (d.amount.replace(0.0, np.nan))
    cond = d.close < delay1
    na_map = part.isna().astype(float).replace(1.0, np.nan)
    return SUMIF(part, 20, cond) / (COUNT(cond, 20, na_map=na_map) + 1e-12)


@alpha_factor(145, 50, FAMILY_VOLUME, "(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100")
def alpha_145(d: Alpha191Data) -> pd.DataFrame:
    return (MEAN(d.volume, 9) - MEAN(d.volume, 26)) / (MEAN(d.volume, 12) + 1e-12) * 100.0


@alpha_factor(146, 100, FAMILY_TREND, "MEAN(part-sma(part,61,2),20)*(part-sma(part,61,2))/SMA((part-(part-sma(part,61,2)))^2,61,2)  # part=(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)")
def alpha_146(d: Alpha191Data) -> pd.DataFrame:
    part = (d.close - DELAY(d.close, 1)) / DELAY(d.close, 1)
    part_ma = part - SMA(part, 61, 2)
    denom = SMA((part - part_ma) ** 2, 61, 2)
    return MEAN(part_ma, 20) * part_ma / (denom + 1e-12)


@alpha_factor(147, 50, FAMILY_TREND, "REGBETA(MEAN(CLOSE,12),SEQUENCE(12))")
def alpha_147(d: Alpha191Data) -> pd.DataFrame:
    return rolling_slope(MEAN(d.close, 12), 12)


@alpha_factor(148, 100, FAMILY_PRICE_VOLUME, "((RANK(CORR(OPEN,SUM(MEAN(VOLUME,60),9),6))<RANK(OPEN-TSMIN(OPEN,14)))*-1)")
def alpha_148(d: Alpha191Data) -> pd.DataFrame:
    left = RANK(CORR(d.open, SUM(MEAN(d.volume, 60), 9), 6))
    right = RANK(d.open - TS_MIN(d.open, 14))
    return (left < right).astype(float) * -1.0


@alpha_factor(149, 300, FAMILY_BENCHMARK, "REGBETA(FILTER(RET,bench down), FILTER(BENCH_RET,bench down), 252)")
def alpha_149(d: Alpha191Data) -> pd.DataFrame:
    """Beta of the stock return on the benchmark return, computed only on
    benchmark down days (FILTER semantics): rows are compressed to down days,
    a 252-window regression runs on the compressed series, and the result is
    mapped back onto the original calendar.
    """
    if d.benchmark_close is None:
        return _nan_like(d.close)
    bench_ret = d.benchmark_close.pct_change()
    down = bench_ret < 0.0
    down_index = d.close.index[down.reindex(d.close.index).fillna(False).to_numpy()]
    if len(down_index) < 30:
        return _nan_like(d.close)
    x = d.returns.loc[down_index]
    bench_df = pd.DataFrame({c: bench_ret for c in d.close.columns}, index=d.close.index).loc[down_index]
    beta = x.rolling(252).cov(bench_df) / bench_df.rolling(252).var()
    return beta.reindex(d.close.index)


@alpha_factor(150, 10, FAMILY_TREND, "(CLOSE+HIGH+LOW)/3*VOLUME")
def alpha_150(d: Alpha191Data) -> pd.DataFrame:
    return (d.close + d.high + d.low) / 3.0 * d.volume


@alpha_factor(151, 50, FAMILY_TREND, "SMA(CLOSE-DELAY(CLOSE,20),20,1)")
def alpha_151(d: Alpha191Data) -> pd.DataFrame:
    return SMA(d.close - DELAY(d.close, 20), 20, 1)


@alpha_factor(152, 50, FAMILY_TREND, "SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)")
def alpha_152(d: Alpha191Data) -> pd.DataFrame:
    part = DELAY(SMA(DELAY(d.close / DELAY(d.close, 9), 1), 9, 1), 1)
    return SMA(MEAN(part, 12) - MEAN(part, 26), 9, 1)


@alpha_factor(153, 50, FAMILY_TREND, "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4")
def alpha_153(d: Alpha191Data) -> pd.DataFrame:
    return (MEAN(d.close, 3) + MEAN(d.close, 6) + MEAN(d.close, 12) + MEAN(d.close, 24)) / 4.0


@alpha_factor(154, 300, FAMILY_PRICE_VOLUME, "((VWAP-TSMIN(VWAP,16))<(CORR(VWAP,MEAN(VOLUME,180),18)))")
def alpha_154(d: Alpha191Data) -> pd.DataFrame:
    left = d.vwap - TS_MIN(d.vwap, 16)
    right = CORR(d.vwap, MEAN(d.volume, 180), 18)
    return (left < right).astype(float)


@alpha_factor(155, 50, FAMILY_VOLUME, "SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)")
def alpha_155(d: Alpha191Data) -> pd.DataFrame:
    ma_short = SMA(d.volume, 13, 2)
    ma_long = SMA(d.volume, 27, 2)
    return ma_short - ma_long - SMA(ma_short - ma_long, 10, 2)


@alpha_factor(156, 50, FAMILY_PRICE_VOLUME, "(MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)), RANK(DECAYLINEAR((DELTA(OPEN*0.15+LOW*0.85,2)/(OPEN*0.15+LOW*0.85))*-1,3)))*-1)")
def alpha_156(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(DECAYLINEAR(DELTA(d.vwap, 5), 3))
    val = d.open * 0.15 + d.low * 0.85
    part2 = RANK(DECAYLINEAR((DELTA(val, 2) / val) * -1.0, 3))
    return MAX(part1, part2) * -1.0


@alpha_factor(157, 50, FAMILY_PRICE_VOLUME, "MIN(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-1*RANK(DELTA(CLOSE-1,5)))),2),1)))),5)+TSRANK(DELAY(-1*RET,6),5)")
def alpha_157(d: Alpha191Data) -> pd.DataFrame:
    inner = RANK(RANK(-1.0 * RANK(DELTA(d.close - 1.0, 5))))
    ts_min = TS_MIN(inner, 2)
    log_sum = np.log(SUM(ts_min, 1))
    part1 = TS_MIN(RANK(RANK(log_sum)), 5)
    part2 = TS_RANK(DELAY(-1.0 * d.returns, 6), 5)
    return part1 + part2


@alpha_factor(158, 50, FAMILY_TREND, "((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE")
def alpha_158(d: Alpha191Data) -> pd.DataFrame:
    sma = SMA(d.close, 15, 2)
    return ((d.high - sma) - (d.low - sma)) / d.close


@alpha_factor(159, 50, FAMILY_SESSION, "((CLOSE-SUM(part2,6))/SUM(part1,6)*288+(CLOSE-SUM(part2,12))/SUM(part3-part2,12)*144+(CLOSE-SUM(part2,24))/SUM(part1,24)*144)*100/504")
def alpha_159(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part2 = MIN(d.low, delay1)
    part3 = MAX(d.high, delay1)
    part1 = part3 - part2
    term1 = (d.close - SUM(part2, 6)) / (SUM(part1, 6) + 1e-12) * 288.0
    term2 = (d.close - SUM(part2, 12)) / (SUM(part3 - part2, 12) + 1e-12) * 144.0
    term3 = (d.close - SUM(part2, 24)) / (SUM(part1, 24) + 1e-12) * 144.0
    return (term1 + term2 + term3) * 100.0 / 504.0


@alpha_factor(160, 50, FAMILY_TREND, "SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)")
def alpha_160(d: Alpha191Data) -> pd.DataFrame:
    cond = d.close <= DELAY(d.close, 1)
    part = IFELSE(cond, STD(d.close, 20), 0.0)
    return SMA(part, 20, 1)


@alpha_factor(161, 50, FAMILY_VOLATILITY, "MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)")
def alpha_161(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part = MAX(MAX(d.high - d.low, ABS(delay1 - d.high)), ABS(delay1 - d.low))
    return MEAN(part, 12)


@alpha_factor(162, 50, FAMILY_TREND, "(SMA(MAX(part1,0),12,1)/SMA(ABS(part1),12,1)*100-TSMIN(ratio,12))/(TSMAX(ratio,12)-TSMIN(ratio,12))")
def alpha_162(d: Alpha191Data) -> pd.DataFrame:
    part1 = d.close - DELAY(d.close, 1)
    ratio = SMA(MAX(part1, 0.0), 12, 1) / SMA(ABS(part1), 12, 1) * 100.0
    return (ratio - TS_MIN(ratio, 12)) / (TS_MAX(ratio, 12) - TS_MIN(ratio, 12) + 1e-12)


@alpha_factor(163, 50, FAMILY_PRICE_VOLUME, "RANK((-1*RET)*MEAN(VOLUME,20)*VWAP*(HIGH-CLOSE))")
def alpha_163(d: Alpha191Data) -> pd.DataFrame:
    return RANK((-1.0 * d.returns) * MEAN(d.volume, 20) * d.vwap * (d.high - d.close))


@alpha_factor(164, 50, FAMILY_TREND, "SMA(((cond?1/(CLOSE-DELAY(CLOSE,1)):1)-TSMIN(cond?1/(CLOSE-DELAY(CLOSE,1)):1,12))/(HIGH-LOW)*100,13,2)")
def alpha_164(d: Alpha191Data) -> pd.DataFrame:
    diff = d.close - DELAY(d.close, 1)
    cond = d.close > DELAY(d.close, 1)
    a = IFELSE(cond, 1.0 / diff, 1.0)
    alpha = (a - TS_MIN(a, 12)) / (d.high - d.low + 1e-12) * 100.0
    return SMA(alpha, 13, 2)


@alpha_factor(165, 200, FAMILY_PRICE_VOLUME, "TSMAX(SUM(CLOSE-MEAN(CLOSE,48),48),48)-TSMIN(SUM(CLOSE-MEAN(CLOSE,48),48),48)/STD(CLOSE,48)")
def alpha_165(d: Alpha191Data) -> pd.DataFrame:
    diff = d.close - MEAN(d.close, 48)
    sumac = SUM(diff, 48)
    return TS_MAX(sumac, 48) - TS_MIN(sumac, 48) / (STD(d.close, 48) + 1e-12)


@alpha_factor(166, 100, FAMILY_TREND, "-20*19^1.5*SUM(RET-MEAN(RET,20),20)/((19*18)*(SUM(MEAN(CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)")
def alpha_166(d: Alpha191Data) -> pd.DataFrame:
    part = d.close / DELAY(d.close, 1)
    ret = part - 1.0
    numerator = SUM(ret - MEAN(ret, 20), 20)
    denominator = (SUM(MEAN(part, 20) ** 2, 20)) ** 1.5
    constant = -20.0 * (19.0 ** 1.5) / (19.0 * 18.0)
    return constant * numerator / (denominator + 1e-12)


@alpha_factor(167, 50, FAMILY_TREND, "SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)")
def alpha_167(d: Alpha191Data) -> pd.DataFrame:
    part = d.close - DELAY(d.close, 1)
    return SUM(IFELSE(part > 0, part, 0.0), 12)


@alpha_factor(168, 50, FAMILY_VOLUME, "(-1*VOLUME/MEAN(VOLUME,20))")
def alpha_168(d: Alpha191Data) -> pd.DataFrame:
    return -1.0 * d.volume / (MEAN(d.volume, 20) + 1e-12)


@alpha_factor(169, 50, FAMILY_TREND, "SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)")
def alpha_169(d: Alpha191Data) -> pd.DataFrame:
    part = SMA(d.close - DELAY(d.close, 1), 9, 1)
    return SMA(MEAN(DELAY(part, 1), 12) - MEAN(DELAY(part, 1), 26), 10, 1)


@alpha_factor(170, 50, FAMILY_PRICE_VOLUME, "((RANK(1/CLOSE)*VOLUME)/MEAN(VOLUME,20))*((HIGH*RANK(HIGH-CLOSE))/(SUM(HIGH,5)/5))-RANK(VWAP-DELAY(VWAP,5))")
def alpha_170(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(1.0 / d.close) * d.volume / (MEAN(d.volume, 20) + 1e-12)
    part2 = (d.high * RANK(d.high - d.close)) / (MEAN(d.high, 5) + 1e-12)
    part3 = RANK(d.vwap - DELAY(d.vwap, 5))
    return part1 * part2 - part3


@alpha_factor(171, 50, FAMILY_TREND, "((-1*((LOW-CLOSE)*(OPEN^5)))/((CLOSE-HIGH)*(CLOSE^5)))")
def alpha_171(d: Alpha191Data) -> pd.DataFrame:
    numerator = -1.0 * (d.low - d.close) * (d.open ** 5)
    denominator = (d.close - d.high + 1e-12) * (d.close ** 5)
    return numerator / denominator


@alpha_factor(172, 50, FAMILY_PRICE_VOLUME, "MEAN(ABS(p1-p2)/(p1+p2)*100,6)")
def alpha_172(d: Alpha191Data) -> pd.DataFrame:
    tr = MAX(MAX(d.high - d.low, ABS(d.high - DELAY(d.close, 1))), ABS(d.low - DELAY(d.close, 1)))
    hd = d.high - DELAY(d.high, 1)
    ld = DELAY(d.low, 1) - d.low
    sum_tr = SUM(tr, 14)
    part1 = SUM(IFELSE((ld > 0) & (ld > hd), ld, 0.0), 14) * 100.0 / (sum_tr + 1e-12)
    part2 = SUM(IFELSE((hd > 0) & (hd > ld), hd, 0.0), 14) * 100.0 / (sum_tr + 1e-12)
    return MEAN(ABS(part1 - part2) / (part1 + part2 + 1e-12) * 100.0, 6)


@alpha_factor(173, 50, FAMILY_TREND, "3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)")
def alpha_173(d: Alpha191Data) -> pd.DataFrame:
    ma = SMA(d.close, 13, 2)
    log_ma = SMA(SMA(SMA(LOG(d.close), 13, 2), 13, 2), 13, 2)
    return 3.0 * ma - 2.0 * SMA(ma, 13, 2) + log_ma


@alpha_factor(174, 50, FAMILY_TREND, "SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)")
def alpha_174(d: Alpha191Data) -> pd.DataFrame:
    cond = d.close > DELAY(d.close, 1)
    part = IFELSE(cond, STD(d.close, 20), 0.0)
    return SMA(part, 20, 1)


@alpha_factor(175, 50, FAMILY_VOLATILITY, "MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)")
def alpha_175(d: Alpha191Data) -> pd.DataFrame:
    delay1 = DELAY(d.close, 1)
    part = MAX(MAX(d.high - d.low, ABS(delay1 - d.high)), ABS(delay1 - d.low))
    return MEAN(part, 6)


@alpha_factor(176, 50, FAMILY_PRICE_VOLUME, "CORR(RANK((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12))),RANK(VOLUME),6)")
def alpha_176(d: Alpha191Data) -> pd.DataFrame:
    part = (d.close - TS_MIN(d.low, 12)) / (TS_MAX(d.high, 12) - TS_MIN(d.low, 12))
    return CORR(RANK(part), RANK(d.volume), 6)


@alpha_factor(177, 50, FAMILY_PRICE_VOLUME, "((20-HIGHDAY(HIGH,20))/20)*100")
def alpha_177(d: Alpha191Data) -> pd.DataFrame:
    return (20.0 - HIGHDAY(d.high, 20)) / 20.0 * 100.0


@alpha_factor(178, 50, FAMILY_TREND, "(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME")
def alpha_178(d: Alpha191Data) -> pd.DataFrame:
    return (d.close - DELAY(d.close, 1)) / DELAY(d.close, 1) * d.volume


@alpha_factor(179, 100, FAMILY_PRICE_VOLUME, "(RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))")
def alpha_179(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(CORR(d.vwap, d.volume, 4))
    part2 = RANK(CORR(RANK(d.low), RANK(MEAN(d.volume, 50)), 12))
    return part1 * part2


@alpha_factor(180, 100, FAMILY_PRICE_VOLUME, "(MEAN(VOLUME,20)<VOLUME?(-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):-VOLUME)")
def alpha_180(d: Alpha191Data) -> pd.DataFrame:
    cond = MEAN(d.volume, 20) < d.volume
    part = (-1.0 * TS_RANK(ABS(DELTA(d.close, 7)), 60)) * SIGN(DELTA(d.close, 7))
    return IFELSE(cond, part, -d.volume)


@alpha_factor(181, 100, FAMILY_BENCHMARK, "SUM((RET-MEAN(RET,20))-(BENCH_CLOSE-MEAN(BENCH_CLOSE,20))^2,20)/SUM((BENCH_CLOSE-MEAN(BENCH_CLOSE,20))^3)")
def alpha_181(d: Alpha191Data) -> pd.DataFrame:
    if d.benchmark_close is None:
        return _nan_like(d.close)
    bench = d.benchmark_close
    bench_dev = bench - bench.rolling(20).mean()
    ret_dev = d.returns - MEAN(d.returns, 20)
    numerator = SUM(ret_dev.sub(bench_dev ** 2, axis=0), 20)
    denominator = SUM(bench_dev ** 3, 20)
    return numerator.div(denominator, axis=0)


@alpha_factor(182, 50, FAMILY_BENCHMARK, "COUNT((CLOSE>OPEN & BENCH_CLOSE>BENCH_OPEN)|(CLOSE<OPEN & BENCH_CLOSE<BENCH_OPEN),20)/20")
def alpha_182(d: Alpha191Data) -> pd.DataFrame:
    if d.benchmark_close is None or d.benchmark_open is None:
        return _nan_like(d.close)
    bench_up = (d.benchmark_close > d.benchmark_open).astype(float)
    bench_down = (d.benchmark_close < d.benchmark_open).astype(float)
    both_up = (d.close > d.open).astype(float).mul(bench_up, axis=0)
    both_down = (d.close < d.open).astype(float).mul(bench_down, axis=0)
    match = (both_up + both_down) > 0.0
    na_map = d.close.isna().astype(float).replace(1.0, np.nan)
    return COUNT(match, 20, na_map=na_map) / 20.0


@alpha_factor(183, 100, FAMILY_PRICE_VOLUME, "TSMAX(SUM(CLOSE-MEAN(CLOSE,24),24),24)-TSMIN(SUM(CLOSE-MEAN(CLOSE,24),24),24)/STD(CLOSE,24)")
def alpha_183(d: Alpha191Data) -> pd.DataFrame:
    diff = d.close - MEAN(d.close, 24)
    sumac = SUM(diff, 24)
    return TS_MAX(sumac, 24) - TS_MIN(sumac, 24) / (STD(d.close, 24) + 1e-12)


@alpha_factor(184, 300, FAMILY_PRICE_VOLUME, "(RANK(CORR(DELAY(OPEN-CLOSE,1),CLOSE,200))+RANK(OPEN-CLOSE))")
def alpha_184(d: Alpha191Data) -> pd.DataFrame:
    part1 = RANK(CORR(DELAY(d.open - d.close, 1), d.close, 200))
    part2 = RANK(d.open - d.close)
    return part1 + part2


@alpha_factor(185, 50, FAMILY_TREND, "RANK((-1*((1-(OPEN/CLOSE))^2)))")
def alpha_185(d: Alpha191Data) -> pd.DataFrame:
    return RANK(-1.0 * ((1.0 - d.open / d.close) ** 2))


@alpha_factor(186, 50, FAMILY_TREND, "(MEAN(ABS(p1-p2)/(p1+p2)*100,6)+DELAY(MEAN(ABS(p1-p2)/(p1+p2)*100,6),6))/2")
def alpha_186(d: Alpha191Data) -> pd.DataFrame:
    tr = MAX(MAX(d.high - d.low, ABS(d.high - DELAY(d.close, 1))), ABS(d.low - DELAY(d.close, 1)))
    hd = d.high - DELAY(d.high, 1)
    ld = DELAY(d.low, 1) - d.low
    sum_tr = SUM(tr, 14)
    part1 = SUM(IFELSE((ld > 0) & (ld > hd), ld, 0.0), 14) * 100.0 / (sum_tr + 1e-12)
    part2 = SUM(IFELSE((hd > 0) & (hd > ld), hd, 0.0), 14) * 100.0 / (sum_tr + 1e-12)
    part3 = ABS(part1 - part2) / (part1 + part2 + 1e-12) * 100.0
    part4 = MEAN(part3, 6)
    return (part4 + DELAY(part4, 6)) / 2.0


@alpha_factor(187, 50, FAMILY_SESSION, "SUM((OPEN<=DELAY(OPEN,1)?0:MAX(HIGH-OPEN,OPEN-DELAY(OPEN,1))),20)")
def alpha_187(d: Alpha191Data) -> pd.DataFrame:
    delay_open = DELAY(d.open, 1)
    part = IFELSE(d.open <= delay_open, 0.0, MAX(d.high - d.open, d.open - delay_open))
    return SUM(part, 20)


@alpha_factor(188, 50, FAMILY_TREND, "((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100")
def alpha_188(d: Alpha191Data) -> pd.DataFrame:
    rng = d.high - d.low
    sma = SMA(rng, 11, 2)
    return (rng - sma) / (sma + 1e-12) * 100.0


@alpha_factor(189, 50, FAMILY_VOLATILITY, "MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)")
def alpha_189(d: Alpha191Data) -> pd.DataFrame:
    return MEAN(ABS(d.close - MEAN(d.close, 6)), 6)


@alpha_factor(190, 50, FAMILY_TREND, "LOG((COUNT(p1>p2,20)-1)*SUMIF((p1-p2)^2,20,p1<p2)/((COUNT(p1<p2,20))*(SUMIF((p1-p2)^2,20,p1>p2))))")
def alpha_190(d: Alpha191Data) -> pd.DataFrame:
    part1 = d.close / DELAY(d.close, 1) - 1.0
    part2 = (d.close / DELAY(d.close, 19)) ** (1.0 / 20.0) - 1.0
    part3 = (part1 - part2) ** 2
    na_map = part3.isna().astype(float).replace(1.0, np.nan)
    count_up = COUNT(part1 > part2, 20, na_map=na_map)
    count_down = COUNT(part1 < part2, 20, na_map=na_map)
    sum_down = SUMIF(part3, 20, part1 < part2)
    sum_up = SUMIF(part3, 20, part1 > part2)
    inner = (count_up - 1.0) * sum_down / (count_down * sum_up + 1e-12)
    return pd.DataFrame(np.log(np.where(inner > 0, inner, np.nan)), index=d.close.index, columns=d.close.columns)


@alpha_factor(191, 50, FAMILY_PRICE_VOLUME, "((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)")
def alpha_191(d: Alpha191Data) -> pd.DataFrame:
    return (CORR(MEAN(d.volume, 20), d.low, 5) + (d.high + d.low) / 2.0) - d.close


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------


def build_all_factors(
    data: Alpha191Data,
    numbers: Optional[Iterable[int]] = None,
    on_error: str = "raise",
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Compute the requested factors as wide DataFrames.

    Returns ``(factors, errors)`` where ``factors`` maps factor names
    (``alpha001``..) to wide DataFrames and ``errors`` lists factor names that
    failed (empty when ``on_error == "raise"`` and nothing failed).
    """

    registry = alpha191_registry()
    if numbers is None:
        selected = registry
    else:
        wanted = set(numbers)
        selected = tuple(f for f in registry if f.number in wanted)
        unknown = sorted(wanted - {f.number for f in registry})
        if unknown:
            raise ValueError(f"unknown alpha numbers: {unknown}")

    factors: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []
    for factor in selected:
        if progress is not None:
            progress(factor.name)
        try:
            factors[factor.name] = factor.fn(data)
        except Exception as exc:  # noqa: BLE001 - reported per factor
            if on_error == "raise":
                raise
            errors.append(f"{factor.name}: {type(exc).__name__}: {exc}")
            factors[factor.name] = _nan_like(data.close)
    return factors, errors


def stack_factors(wide: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack wide factor DataFrames into one long table.

    Returns a DataFrame with a ``(date, symbol)`` MultiIndex and one column
    per factor. Missing values are preserved (not dropped).
    """

    longs: List[pd.DataFrame] = []
    for name, df in wide.items():
        melted = df.reset_index().melt(id_vars=df.index.name or "index", var_name="symbol", value_name=name)
        melted = melted.rename(columns={df.index.name or "index": "date"})
        longs.append(melted)
    if not longs:
        return pd.DataFrame()
    combined = longs[0]
    for other in longs[1:]:
        combined = combined.merge(other, on=["date", "symbol"], how="outer")
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "symbol"]).set_index(["date", "symbol"])
    return combined


def factor_metadata_table() -> pd.DataFrame:
    """Registry metadata as a DataFrame (name, number, family, max_depend, formula)."""

    rows = []
    for factor in alpha191_registry():
        rows.append(
            {
                "name": factor.name,
                "number": factor.number,
                "family": factor.family,
                "max_depend": factor.max_depend,
                "formula": factor.formula,
            }
        )
    return pd.DataFrame(rows)
