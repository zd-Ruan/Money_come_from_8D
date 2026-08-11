#!/usr/bin/env python
"""Build a point-in-time daily Qlib dataset for tradable mainland equity ETFs.

The universe is the intersection of Eastmoney's exchange-traded ETF snapshot
and funds classified as ``指数型-股票`` in Eastmoney's fund master. Mainland
equity ETFs settle T+1. Overseas equity, fixed-income, commodity/gold and money
market ETFs are deliberately excluded because they use different settlement or
trading rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("cn_etf_collector")
CUR_DIR = Path(__file__).resolve().parent
QLIB_ROOT = CUR_DIR.parents[2]
DEFAULT_DATA_DIR = QLIB_ROOT / "data" / "cn_etf"

SPOT_URL = "https://push2delay.eastmoney.com/api/qt/clist/get"
FUND_MASTER_URL = "https://fund.eastmoney.com/js/fundcode_search.js"
HISTORY_URLS = (
    "https://7.push2his.eastmoney.com/api/qt/stock/kline/get",
)
TENCENT_HISTORY_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
EASTMONEY_TOKEN = "7eea3edcaed734bea9cbfc24409ed989"
T1_FUND_TYPE = "指数型-股票"

RAW_COLUMNS = [
    "date",
    "symbol",
    "raw_open",
    "raw_close",
    "raw_high",
    "raw_low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "price_change",
    "turnover_rate",
    "qfq_open",
    "qfq_close",
    "qfq_high",
    "qfq_low",
    "data_source",
    "amount_quality",
]
QLIB_FIELDS = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "factor",
    "change",
    "amount",
    "vwap",
    "turnover_rate",
    "amount_estimated",
    "paused",
]


@dataclass(frozen=True)
class DownloadResult:
    symbol: str
    rows: int = 0
    start_date: str | None = None
    end_date: str | None = None
    refreshed: bool = False
    error: str | None = None


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def make_session(insecure: bool = False) -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    session.mount("http://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/",
        }
    )
    session.verify = not insecure
    if insecure:
        requests.packages.urllib3.disable_warnings(category=requests.packages.urllib3.exceptions.InsecureRequestWarning)
    return session


def _request_json(session: requests.Session, urls: Iterable[str], params: dict, timeout: float = 25.0) -> dict:
    errors = []
    url_list = list(urls)
    random.shuffle(url_list)
    for url in url_list:
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("rc") not in (None, 0):
                raise ValueError(f"remote rc={payload.get('rc')}")
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def latest_complete_date(now: datetime | None = None) -> date:
    """Exclude today's bar until a conservative 16:00 Asia/Shanghai cutoff."""
    shanghai = ZoneInfo("Asia/Shanghai")
    current = now.astimezone(shanghai) if now is not None else datetime.now(shanghai)
    if current.time() < datetime_time(16, 0):
        return current.date() - timedelta(days=1)
    return current.date()


def market_prefix(code: str, market_id: int | str | None = None) -> str:
    if market_id is not None and str(market_id) in {"0", "1"}:
        return "SH" if str(market_id) == "1" else "SZ"
    return "SH" if code.startswith(("5", "6")) else "SZ"


def eastmoney_secid(symbol: str) -> str:
    symbol = symbol.upper()
    return f"{1 if symbol.startswith('SH') else 0}.{symbol[2:]}"


def _diff_rows(payload: dict) -> list[dict]:
    diff = (payload.get("data") or {}).get("diff") or []
    return list(diff.values()) if isinstance(diff, dict) else list(diff)


def fetch_fund_master(session: requests.Session) -> dict[str, dict[str, str]]:
    response = session.get(FUND_MASTER_URL, timeout=40)
    response.raise_for_status()
    text = response.content.decode("utf-8-sig")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("cannot locate the fund master JSON array")
    records = json.loads(text[start : end + 1])
    return {
        str(record[0]).zfill(6): {
            "fund_name": str(record[2]),
            "fund_type": str(record[3]),
        }
        for record in records
        if len(record) >= 4
    }


def fetch_etf_snapshot(session: requests.Session, page_size: int = 100) -> list[dict]:
    rows: list[dict] = []
    page = 1
    total = math.inf
    while len(rows) < total:
        params = {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f12",
            "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
            "fields": "f2,f5,f6,f12,f13,f14,f20,f21,f124,f297",
        }
        payload = _request_json(session, (SPOT_URL,), params, timeout=30)
        data = payload.get("data") or {}
        batch = _diff_rows(payload)
        total = int(data.get("total") or len(rows) + len(batch))
        if not batch:
            break
        rows.extend(batch)
        page += 1
    if len(rows) < total:
        raise RuntimeError(f"ETF snapshot is incomplete: expected {total}, got {len(rows)}")
    return rows


def build_t1_universe(session: requests.Session, snapshot_date: date) -> pd.DataFrame:
    fund_master = fetch_fund_master(session)
    spot_rows = fetch_etf_snapshot(session)
    records = []
    for row in spot_rows:
        code = str(row.get("f12", "")).zfill(6)
        master = fund_master.get(code)
        if not master or master["fund_type"] != T1_FUND_TYPE:
            continue
        last_price = pd.to_numeric(row.get("f2"), errors="coerce")
        if pd.isna(last_price) or float(last_price) <= 0:
            continue
        prefix = market_prefix(code, row.get("f13"))
        update_value = pd.to_numeric(row.get("f124"), errors="coerce")
        update_time = ""
        if pd.notna(update_value) and float(update_value) > 0:
            update_time = datetime.fromtimestamp(float(update_value), ZoneInfo("Asia/Shanghai")).isoformat()
        records.append(
            {
                "symbol": f"{prefix}{code}",
                "code": code,
                "exchange": prefix,
                "name": str(row.get("f14") or master["fund_name"]),
                "fund_type": master["fund_type"],
                "settlement": "T+1",
                "snapshot_date": snapshot_date.isoformat(),
                "source_data_date": str(row.get("f297") or ""),
                "source_update_time": update_time,
                "last_price": last_price,
                "spot_volume_lots": pd.to_numeric(row.get("f5"), errors="coerce"),
                "spot_amount": pd.to_numeric(row.get("f6"), errors="coerce"),
                "total_market_value": pd.to_numeric(row.get("f20"), errors="coerce"),
                "float_market_value": pd.to_numeric(row.get("f21"), errors="coerce"),
            }
        )
    universe = pd.DataFrame(records).sort_values("symbol").reset_index(drop=True)
    if universe.empty:
        raise RuntimeError("T+1 ETF universe is empty")
    return universe


def _parse_kline(payload: dict, symbol: str, adjusted: bool) -> pd.DataFrame:
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame()
    rows = [item.split(",") for item in klines]
    columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume_lots",
        "amount",
        "amplitude",
        "pct_change",
        "price_change",
        "turnover_rate_pct",
    ]
    frame = pd.DataFrame(rows, columns=columns[: len(rows[0])])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in columns[1:]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = symbol
    if adjusted:
        return frame[["date", "symbol", "open", "close", "high", "low"]].rename(
            columns={column: f"qfq_{column}" for column in ("open", "close", "high", "low")}
        )
    frame["volume"] = frame["volume_lots"] * 100.0
    frame["turnover_rate"] = frame["turnover_rate_pct"] / 100.0
    frame["data_source"] = "eastmoney"
    frame["amount_quality"] = "reported"
    return frame[
        [
            "date",
            "symbol",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_change",
            "price_change",
            "turnover_rate",
            "data_source",
            "amount_quality",
        ]
    ].rename(columns={column: f"raw_{column}" for column in ("open", "close", "high", "low")})


def fetch_history_eastmoney(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    adjusted: bool,
) -> pd.DataFrame:
    params = {
        "secid": eastmoney_secid(symbol),
        "ut": EASTMONEY_TOKEN,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": 1 if adjusted else 0,
        "beg": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
    }
    payload = _request_json(session, HISTORY_URLS, params)
    return _parse_kline(payload, symbol, adjusted)


def fetch_history_tencent(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    adjusted: bool,
) -> pd.DataFrame:
    """Fetch Tencent history backwards because qfq responses are capped at 640 rows."""
    exchange_code = symbol.lower()
    page_size = 640 if adjusted else 2000
    cursor = end_date
    pages: list[pd.DataFrame] = []
    while cursor >= start_date:
        adjust_name = "qfq" if adjusted else ""
        params = {
            "param": (
                f"{exchange_code},day,{start_date.isoformat()},"
                f"{cursor.isoformat()},{page_size},{adjust_name}"
            )
        }
        page_errors = []
        payload = None
        for attempt in range(4):
            try:
                response = session.get(
                    TENCENT_HISTORY_URL,
                    params=params,
                    headers={"Referer": "https://gu.qq.com/"},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("code") not in (None, 0):
                    raise ValueError(f"remote code={payload.get('code')}: {payload.get('msg')}")
                break
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                page_errors.append(str(exc))
                if attempt < 3:
                    time.sleep(0.5 * (attempt + 1))
        if payload is None:
            raise RuntimeError(
                f"Tencent page failed for {symbol} through {cursor}: " + "; ".join(page_errors)
            )
        symbol_data = (payload.get("data") or {}).get(exchange_code) or {}
        if adjusted:
            # Tencent returns ``day`` for a qfq request when the instrument has
            # no recorded adjustment event; instruments with events use
            # ``qfqday``. In both cases the response is for the qfq request.
            rows = symbol_data.get("qfqday") or symbol_data.get("day") or []
        else:
            rows = symbol_data.get("day") or []
        if not rows:
            break
        frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume_lots"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "close", "high", "low", "volume_lots"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        if frame.empty:
            break
        pages.append(frame)
        earliest = frame["date"].min().date()
        if len(frame) < page_size or earliest <= start_date:
            break
        cursor = earliest - timedelta(days=1)
    if not pages:
        return pd.DataFrame()
    frame = pd.concat(pages, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    frame["symbol"] = symbol
    if adjusted:
        return frame[["date", "symbol", "open", "close", "high", "low"]].rename(
            columns={column: f"qfq_{column}" for column in ("open", "close", "high", "low")}
        )
    frame["volume"] = frame["volume_lots"] * 100.0
    typical_price = (frame["open"] + frame["close"] + frame["high"] + frame["low"]) / 4.0
    frame["amount"] = typical_price * frame["volume"]
    frame["amplitude"] = (frame["high"] - frame["low"]) / frame["close"].shift(1) * 100.0
    frame["pct_change"] = frame["close"].pct_change(fill_method=None) * 100.0
    frame["price_change"] = frame["close"].diff()
    frame["turnover_rate"] = np.nan
    frame["data_source"] = "tencent"
    frame["amount_quality"] = "estimated_typical_price"
    return frame[
        [
            "date",
            "symbol",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_change",
            "price_change",
            "turnover_rate",
            "data_source",
            "amount_quality",
        ]
    ].rename(columns={column: f"raw_{column}" for column in ("open", "close", "high", "low")})


def fetch_history(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    adjusted: bool,
    history_source: str,
) -> pd.DataFrame:
    if history_source == "tencent":
        return fetch_history_tencent(session, symbol, start_date, end_date, adjusted)
    if history_source == "eastmoney":
        return fetch_history_eastmoney(session, symbol, start_date, end_date, adjusted)
    try:
        return fetch_history_eastmoney(session, symbol, start_date, end_date, adjusted)
    except Exception as exc:
        LOGGER.warning("%s Eastmoney history failed (%s); using Tencent", symbol, exc)
        return fetch_history_tencent(session, symbol, start_date, end_date, adjusted)


def combine_histories(raw: pd.DataFrame, qfq: pd.DataFrame, end_date: date) -> pd.DataFrame:
    if raw.empty or qfq.empty:
        raise ValueError("raw or qfq history is empty")
    frame = raw.merge(qfq, on=["date", "symbol"], how="inner", validate="one_to_one")
    frame = frame[frame["date"].dt.date <= end_date]
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[RAW_COLUMNS].reset_index(drop=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    temp_path.replace(path)


def _overlap_changed(old: pd.DataFrame, new: pd.DataFrame, tolerance: float = 5e-4) -> bool:
    overlap = old[["date", "qfq_close"]].merge(new[["date", "qfq_close"]], on="date", suffixes=("_old", "_new"))
    if overlap.empty:
        return False
    relative = (overlap["qfq_close_new"] / overlap["qfq_close_old"] - 1).abs()
    return bool((relative > tolerance).any())


def download_one(
    symbol: str,
    raw_dir: Path,
    start_date: date,
    end_date: date,
    insecure: bool,
    full_refresh: bool,
    history_source: str,
) -> DownloadResult:
    path = raw_dir / f"{symbol.lower()}.csv"
    session = make_session(insecure)
    try:
        old = pd.DataFrame()
        fetch_start = start_date
        if path.exists() and not full_refresh:
            old = pd.read_csv(path, parse_dates=["date"])
            if not old.empty:
                fetch_start = max(start_date, old["date"].max().date() - timedelta(days=30))
        raw = fetch_history(session, symbol, fetch_start, end_date, adjusted=False, history_source=history_source)
        qfq = fetch_history(session, symbol, fetch_start, end_date, adjusted=True, history_source=history_source)
        fresh = combine_histories(raw, qfq, end_date)
        refreshed = full_refresh or old.empty
        if not old.empty and _overlap_changed(old, fresh):
            LOGGER.info("%s adjustment changed; refreshing full history", symbol)
            raw = fetch_history(session, symbol, start_date, end_date, adjusted=False, history_source=history_source)
            qfq = fetch_history(session, symbol, start_date, end_date, adjusted=True, history_source=history_source)
            fresh = combine_histories(raw, qfq, end_date)
            old = pd.DataFrame()
            refreshed = True
        combined = fresh if old.empty else pd.concat([old, fresh], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.drop_duplicates("date", keep="last").sort_values("date")
        combined = combined[combined["date"].dt.date <= end_date]
        if combined.empty:
            raise ValueError("no rows in requested date range")
        _atomic_csv(combined[RAW_COLUMNS], path)
        return DownloadResult(
            symbol=symbol,
            rows=len(combined),
            start_date=combined["date"].min().strftime("%Y-%m-%d"),
            end_date=combined["date"].max().strftime("%Y-%m-%d"),
            refreshed=refreshed,
        )
    except Exception as exc:  # individual failures are persisted and retried on the next run
        return DownloadResult(symbol=symbol, error=f"{type(exc).__name__}: {exc}")
    finally:
        session.close()


def download_dataset(
    data_dir: Path,
    start_date: date,
    end_date: date,
    workers: int = 8,
    limit: int | None = None,
    symbols: list[str] | None = None,
    insecure: bool = False,
    full_refresh: bool = False,
    history_source: str = "auto",
) -> tuple[pd.DataFrame, list[DownloadResult]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(insecure)
    try:
        snapshot_date = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        universe = build_t1_universe(session, snapshot_date)
    finally:
        session.close()
    universe["history_end_date"] = end_date.isoformat()
    if symbols:
        wanted = {symbol.upper() for symbol in symbols}
        universe = universe[universe["symbol"].isin(wanted)]
        missing = sorted(wanted - set(universe["symbol"]))
        if missing:
            raise ValueError(f"symbols are not in the current T+1 ETF universe: {missing}")
    if limit is not None:
        universe = universe.head(limit)
    universe = universe.reset_index(drop=True)
    _atomic_csv(universe, data_dir / "universe.csv")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    job_args = {
        "raw_dir": raw_dir,
        "start_date": start_date,
        "end_date": end_date,
        "insecure": insecure,
        "full_refresh": full_refresh,
        "history_source": history_source,
    }
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(download_one, symbol, **job_args): symbol for symbol in universe["symbol"]}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading ETF histories"):
            result = future.result()
            results.append(result)
            if result.error:
                LOGGER.error("%s: %s", result.symbol, result.error)
    result_frame = pd.DataFrame([result.__dict__ for result in results]).sort_values("symbol")
    _atomic_csv(result_frame, data_dir / "download_report.csv")
    failures = result_frame[result_frame["error"].notna()]
    _atomic_csv(failures, data_dir / "download_failures.csv")
    LOGGER.info(
        "downloaded %d/%d ETFs through %s",
        len(result_frame) - len(failures),
        len(result_frame),
        end_date,
    )
    return universe, results


def _valid_ohlc(frame: pd.DataFrame, prefix: str) -> pd.Series:
    open_ = frame[f"{prefix}_open"]
    close = frame[f"{prefix}_close"]
    high = frame[f"{prefix}_high"]
    low = frame[f"{prefix}_low"]
    present = frame[[f"{prefix}_{field}" for field in ("open", "close", "high", "low")]].notna().all(axis=1)
    positive = (open_ > 0) & (close > 0) & (high > 0) & (low > 0)
    ordered = (high + 1e-8 >= pd.concat([open_, close, low], axis=1).max(axis=1)) & (
        low - 1e-8 <= pd.concat([open_, close, high], axis=1).min(axis=1)
    )
    return present & positive & ordered


def normalize_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    numeric = [column for column in RAW_COLUMNS if column not in {"date", "symbol", "data_source", "amount_quality"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["date", "symbol"]).drop_duplicates("date", keep="last").sort_values("date")
    valid = _valid_ohlc(frame, "raw") & _valid_ohlc(frame, "qfq")
    valid &= frame["volume"].gt(0) & frame["amount"].ge(0)
    frame = frame[valid].copy()
    if frame.empty:
        raise ValueError("no valid OHLCV rows after cleaning")
    first_qfq_close = float(frame["qfq_close"].iloc[0])
    if not np.isfinite(first_qfq_close) or first_qfq_close <= 0:
        raise ValueError("invalid first qfq close")
    factor = (frame["qfq_close"] / frame["raw_close"]) / first_qfq_close
    if factor.isna().any() or (factor <= 0).any():
        raise ValueError("invalid adjustment factor")
    output = pd.DataFrame({"date": frame["date"], "symbol": frame["symbol"].str.upper()})
    for column in ("open", "close", "high", "low"):
        output[column] = frame[f"qfq_{column}"] / first_qfq_close
    output["factor"] = factor
    output["volume"] = frame["volume"] / factor
    output["change"] = frame["qfq_close"].pct_change(fill_method=None)
    output["amount"] = frame["amount"]
    output["vwap"] = (frame["amount"] / frame["volume"]) * factor
    output["turnover_rate"] = frame["turnover_rate"]
    output["amount_estimated"] = (frame["amount_quality"] != "reported").astype(float)
    output["paused"] = 0.0
    values = output[QLIB_FIELDS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("normalized data contains infinity")
    return output[["date", "symbol", *QLIB_FIELDS]].reset_index(drop=True)


def normalize_dataset(data_dir: Path, workers: int = 4) -> pd.DataFrame:
    raw_dir = data_dir / "raw"
    normalized_dir = data_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no raw CSV files under {raw_dir}")

    def normalize_one(path: Path) -> dict:
        try:
            raw = pd.read_csv(path, parse_dates=["date"])
            normalized = normalize_frame(raw)
            _atomic_csv(normalized, normalized_dir / path.name)
            return {
                "symbol": normalized["symbol"].iloc[0],
                "rows": len(normalized),
                "start_date": normalized["date"].min().strftime("%Y-%m-%d"),
                "end_date": normalized["date"].max().strftime("%Y-%m-%d"),
                "first_close": float(normalized["close"].iloc[0]),
                "data_sources": ",".join(sorted(raw["data_source"].dropna().astype(str).unique())),
                "amount_estimated": bool((normalized["amount_estimated"] > 0).any()),
                "error": None,
            }
        except Exception as exc:
            return {"symbol": path.stem.upper(), "error": f"{type(exc).__name__}: {exc}"}

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(normalize_one, path) for path in files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing for Qlib"):
            rows.append(future.result())
    report = pd.DataFrame(rows).sort_values("symbol")
    _atomic_csv(report, data_dir / "normalize_report.csv")
    failures = report[report["error"].notna()]
    if not failures.empty:
        raise RuntimeError(f"normalization failed for {len(failures)} ETF(s); see {data_dir / 'normalize_report.csv'}")
    return report


def validate_dataset(data_dir: Path, expected_end: date | None = None, max_stale_days: int = 7) -> dict:
    universe_path = data_dir / "universe.csv"
    normalized_dir = data_dir / "normalized"
    if not universe_path.exists():
        raise FileNotFoundError(universe_path)
    universe = pd.read_csv(universe_path, dtype={"code": str})
    files = sorted(normalized_dir.glob("*.csv"))
    issues: list[dict] = []
    total_rows = 0
    end_dates = []
    for path in tqdm(files, desc="Validating normalized data"):
        try:
            frame = pd.read_csv(path, parse_dates=["date"])
            total_rows += len(frame)
            missing = sorted(set(["date", "symbol", *QLIB_FIELDS]) - set(frame.columns))
            if missing:
                raise ValueError(f"missing columns: {missing}")
            if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
                raise ValueError("dates are duplicated or unsorted")
            if frame.empty:
                raise ValueError("empty file")
            first_close = float(frame["close"].iloc[0])
            if not np.isclose(first_close, 1.0, rtol=1e-6, atol=1e-6):
                raise ValueError(f"first close is not normalized to 1: {first_close}")
            numeric = frame[QLIB_FIELDS].to_numpy(dtype=float)
            if np.isinf(numeric).any():
                raise ValueError("contains infinity")
            if (frame[["open", "close", "high", "low", "volume", "factor"]].dropna() <= 0).any().any():
                raise ValueError("contains non-positive OHLCV/factor")
            end_date = frame["date"].max().date()
            end_dates.append(end_date)
            if expected_end and (expected_end - end_date).days > max_stale_days:
                raise ValueError(f"stale last date: {end_date}")
        except Exception as exc:
            issues.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "universe_count": int(len(universe)),
        "normalized_file_count": len(files),
        "total_rows": total_rows,
        "min_latest_date": min(end_dates).isoformat() if end_dates else None,
        "max_latest_date": max(end_dates).isoformat() if end_dates else None,
        "issue_count": len(issues),
        "issues": issues,
    }
    report_path = data_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if issues:
        raise RuntimeError(f"validation found {len(issues)} issue(s); see {report_path}")
    LOGGER.info("validation passed: %d ETFs, %d rows", len(files), total_rows)
    return report


def dump_to_qlib(data_dir: Path, qlib_dir: Path, workers: int = 4, overwrite: bool = False) -> None:
    normalized_dir = data_dir / "normalized"
    if not any(normalized_dir.glob("*.csv")):
        raise FileNotFoundError(f"no normalized CSV files under {normalized_dir}")
    qlib_dir = qlib_dir.resolve()
    if qlib_dir in {QLIB_ROOT.resolve(), data_dir.resolve(), normalized_dir.resolve()}:
        raise ValueError(f"refusing to overwrite unsafe qlib target: {qlib_dir}")
    if qlib_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{qlib_dir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(qlib_dir)
    scripts_dir = QLIB_ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from dump_bin import DumpDataAll

    dumper = DumpDataAll(
        data_path=str(normalized_dir),
        qlib_dir=str(qlib_dir),
        freq="day",
        max_workers=max(1, workers),
        date_field_name="date",
        file_suffix=".csv",
        symbol_field_name="symbol",
        include_fields=",".join(QLIB_FIELDS),
    )
    dumper.dump()
    all_file = qlib_dir / "instruments" / "all.txt"
    shutil.copy2(all_file, qlib_dir / "instruments" / "t1_etf.txt")
    metadata_dir = qlib_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / "universe.csv", metadata_dir / "t1_etf_universe.csv")
    validation_path = data_dir / "validation_report.json"
    if validation_path.exists():
        shutil.copy2(validation_path, metadata_dir / validation_path.name)
    LOGGER.info("Qlib dataset written to %s", qlib_dir)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    download = subparsers.add_parser("download", help="download the current T+1 universe and daily histories")
    add_common(download)
    download.add_argument("--start", type=parse_date, default=date(2005, 1, 1))
    download.add_argument("--end", type=parse_date, default=None)
    download.add_argument("--workers", type=int, default=8)
    download.add_argument("--limit", type=int)
    download.add_argument("--symbols", nargs="+")
    download.add_argument("--full-refresh", action="store_true")
    download.add_argument("--insecure", action="store_true")
    download.add_argument("--history-source", choices=("auto", "eastmoney", "tencent"), default="auto")

    normalize = subparsers.add_parser("normalize", help="clean raw histories and apply Qlib normalization")
    add_common(normalize)
    normalize.add_argument("--workers", type=int, default=4)

    validate = subparsers.add_parser("validate", help="validate normalized CSV data")
    add_common(validate)
    validate.add_argument("--expected-end", type=parse_date, default=None)
    validate.add_argument("--max-stale-days", type=int, default=7)

    dump = subparsers.add_parser("dump", help="convert normalized CSV files to Qlib binary format")
    add_common(dump)
    dump.add_argument("--qlib-dir", type=Path, default=DEFAULT_DATA_DIR / "qlib_data")
    dump.add_argument("--workers", type=int, default=4)
    dump.add_argument("--overwrite", action="store_true")

    all_parser = subparsers.add_parser("all", help="run download, normalize, validate and Qlib dump")
    add_common(all_parser)
    all_parser.add_argument("--qlib-dir", type=Path, default=DEFAULT_DATA_DIR / "qlib_data")
    all_parser.add_argument("--start", type=parse_date, default=date(2005, 1, 1))
    all_parser.add_argument("--end", type=parse_date, default=None)
    all_parser.add_argument("--workers", type=int, default=8)
    all_parser.add_argument("--limit", type=int)
    all_parser.add_argument("--symbols", nargs="+")
    all_parser.add_argument("--full-refresh", action="store_true")
    all_parser.add_argument("--insecure", action="store_true")
    all_parser.add_argument("--history-source", choices=("auto", "eastmoney", "tencent"), default="auto")
    all_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.command == "download":
        end_date = args.end or latest_complete_date()
        _, results = download_dataset(
            args.data_dir,
            args.start,
            end_date,
            args.workers,
            args.limit,
            args.symbols,
            args.insecure,
            args.full_refresh,
            args.history_source,
        )
        failures = [result for result in results if result.error]
        return 1 if failures else 0
    if args.command == "normalize":
        normalize_dataset(args.data_dir, args.workers)
        return 0
    if args.command == "validate":
        validate_dataset(args.data_dir, args.expected_end, args.max_stale_days)
        return 0
    if args.command == "dump":
        dump_to_qlib(args.data_dir, args.qlib_dir, args.workers, args.overwrite)
        return 0
    if args.command == "all":
        end_date = args.end or latest_complete_date()
        _, results = download_dataset(
            args.data_dir,
            args.start,
            end_date,
            args.workers,
            args.limit,
            args.symbols,
            args.insecure,
            args.full_refresh,
            args.history_source,
        )
        failures = [result for result in results if result.error]
        if failures:
            LOGGER.error("stopping because %d downloads failed; rerun to retry", len(failures))
            return 1
        normalize_dataset(args.data_dir, min(args.workers, 8))
        validate_dataset(args.data_dir, end_date)
        dump_to_qlib(args.data_dir, args.qlib_dir, min(args.workers, 8), args.overwrite)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
