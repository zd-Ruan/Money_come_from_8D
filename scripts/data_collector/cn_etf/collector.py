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
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
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
SINA_HISTORY_URL = "https://finance.sina.com.cn/realstock/company/{}/hisdata_klc2/klc_kl.js"
SINA_HFQ_FACTOR_URL = "https://finance.sina.com.cn/realstock/company/{}/hfq.js"
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
    "adj_open",
    "adj_close",
    "adj_high",
    "adj_low",
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
    "amount_estimated",
    "paused",
]
CORPORATE_ACTION_COLUMNS = [
    "symbol",
    "record_date",
    "ex_date",
    "cash_payment_date",
    "cash_dividend_per_old_share",
    "share_ratio",
    "fractional_share_treatment",
    "source_url",
    "source_sha256",
]
NO_SHARE_CHANGE_TREATMENT = "not_applicable_no_share_change"
EASTMONEY_UNKNOWN_FRACTIONAL_TREATMENT = "unknown_not_provided_by_eastmoney_archive"
DEFAULT_ACTION_REQUEST_DELAY_SECONDS = 0.25
DEFAULT_ACTION_ATTEMPTS = 5
ACTION_RETRY_STATUS_CODES = frozenset({429, 514, *range(500, 600)})


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


def make_session(insecure: bool = False, automatic_retries: bool = True) -> requests.Session:
    if automatic_retries:
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=0.7,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
    else:
        # Corporate-action requests implement visible, auditable retries below.
        retry = Retry(total=0, connect=0, read=0, redirect=0, status=0)
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
    """Exclude today's bar until a conservative 16:00 Asia/Shanghai cutoff.

    Weekend rollback ensures an in-session Monday run never selects Sunday as
    the completed data end date.
    """
    shanghai = ZoneInfo("Asia/Shanghai")
    current = now.astimezone(shanghai) if now is not None else datetime.now(shanghai)
    candidate = current.date()
    if current.time() < datetime_time(16, 0):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


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


def build_t1_universe(session: requests.Session, snapshot_date: date) -> tuple[pd.DataFrame, list[dict]]:
    """Build the T+1 ETF universe and return (universe, dropped_records).

    Dropped records are reported instead of silently shrinking the universe so
    an in-session snapshot missing quotes can never quietly change the pool.
    """
    fund_master = fetch_fund_master(session)
    spot_rows = fetch_etf_snapshot(session)
    records = []
    dropped: list[dict] = []
    for row in spot_rows:
        code = str(row.get("f12", "")).zfill(6)
        master = fund_master.get(code)
        if not master:
            dropped.append({"code": code, "reason": "missing_fund_master"})
            continue
        if master["fund_type"] != T1_FUND_TYPE:
            dropped.append(
                {
                    "code": code,
                    "name": str(master["fund_name"]),
                    "fund_type": str(master["fund_type"]),
                    "reason": "excluded_non_t1_fund_type",
                }
            )
            continue
        last_price = pd.to_numeric(row.get("f2"), errors="coerce")
        if pd.isna(last_price) or float(last_price) <= 0:
            dropped.append(
                {
                    "code": code,
                    "name": str(master["fund_name"]),
                    "reason": "missing_or_nonpositive_last_price",
                }
            )
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
    return universe, dropped


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
            columns={column: f"adj_{column}" for column in ("open", "close", "high", "low")}
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
        # Back-adjusted prices remain positive on long-lived dividend ETFs.
        "fqt": 2 if adjusted else 0,
        "beg": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
    }
    payload = _request_json(session, HISTORY_URLS, params)
    return _parse_kline(payload, symbol, adjusted)


def _decode_sina_klc(encoded: str) -> list[dict]:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Sina history decoding requires Node.js on PATH")
    decoder_path = CUR_DIR / "sina_decode.js"
    decoder = decoder_path.read_text(encoding="utf-8")
    script = decoder + "\nprocess.stdout.write(JSON.stringify(d(" + json.dumps(encoded) + ")));"
    result = subprocess.run(
        [node],
        input=script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=45,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Sina KLC decode failed: {result.stderr.strip()}")
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise ValueError("Sina KLC decoder returned a non-list payload")
    return rows


def _parse_sina_factor_payload(text: str) -> pd.DataFrame:
    start = text.find("{")
    end = text.find("/*", start)
    if end < 0:
        end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("invalid Sina adjustment factor payload")
    records = (json.loads(text[start:end].strip()).get("data") or [])
    if not records:
        raise ValueError("empty Sina adjustment factor payload")
    factors = pd.DataFrame(records).rename(columns={"d": "date"})
    required = ["date", "f", "s", "u"]
    if not set(required).issubset(factors.columns):
        raise ValueError(f"Sina adjustment factors lack columns: {required}")
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    factors[["f", "s", "u"]] = factors[["f", "s", "u"]].apply(pd.to_numeric, errors="coerce")
    factors = factors.dropna(subset=required).drop_duplicates("date", keep="last").sort_values("date")
    if factors.empty or (factors[["f", "s"]] <= 0).any().any():
        raise ValueError("invalid Sina adjustment factors")
    return factors[required].reset_index(drop=True)


def fetch_sina_factors(session: requests.Session, symbol: str) -> pd.DataFrame:
    sina_symbol = symbol.lower()
    response = session.get(
        SINA_HFQ_FACTOR_URL.format(sina_symbol),
        headers={"Referer": f"https://finance.sina.com.cn/realstock/company/{sina_symbol}/nc.shtml"},
        timeout=30,
    )
    response.raise_for_status()
    return _parse_sina_factor_payload(response.text)


def _fund_archive_code(symbol: str) -> str:
    code = str(symbol).upper()
    if len(code) != 8 or code[:2] not in {"SH", "SZ"} or not code[2:].isdigit():
        raise ValueError(f"invalid mainland ETF symbol: {symbol!r}")
    return code[2:]


def _eastmoney_corporate_action_url(symbol: str) -> str:
    return f"https://fundf10.eastmoney.com/fhsp_{_fund_archive_code(symbol)}.html"


def _parse_eastmoney_corporate_actions(html_text: str, symbol: str, source_url: str) -> pd.DataFrame:
    """Parse dated cash distributions and share conversions from a fund archive page."""
    from bs4 import BeautifulSoup

    import re

    records: dict[str, dict] = {}
    soup = BeautifulSoup(html_text, "html.parser")
    recognized_archive_tables = 0
    for table in soup.find_all("table"):
        headers = [cell.get_text(" ", strip=True) for cell in table.find_all("th")]
        if "除息日" in headers and "分红发放日" in headers:
            recognized_archive_tables += 1
            for row in table.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
                if len(cells) < 5:
                    continue
                amount = re.search(
                    r"每\s*10\s*份(?:基金份额)?\s*(?:派(?:发)?(?:现金|红利)|分配收益)\s*([0-9.]+)\s*元",
                    cells[3],
                )
                per_share_denominator = 10.0
                if amount is None:
                    amount = re.search(
                        r"每\s*份\s*(?:派(?:发)?(?:现金|红利)|分配收益)\s*([0-9.]+)\s*元",
                        cells[3],
                    )
                    per_share_denominator = 1.0
                if amount is None:
                    raise ValueError(f"unsupported cash distribution text for {symbol}: {cells[3]!r}")
                ex_date = pd.Timestamp(cells[2]).date().isoformat()
                records.setdefault(ex_date, {}).update(
                    {
                        "symbol": symbol.upper(),
                        "record_date": pd.Timestamp(cells[1]).date().isoformat(),
                        "ex_date": ex_date,
                        "cash_payment_date": pd.Timestamp(cells[4]).date().isoformat(),
                        "cash_dividend_per_old_share": float(amount.group(1)) / per_share_denominator,
                    }
                )
        if "拆分折算日" in headers and "拆分折算比例" in headers:
            recognized_archive_tables += 1
            for row in table.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
                if len(cells) < 4:
                    continue
                ratio = re.fullmatch(r"\s*([0-9.]+)\s*:\s*([0-9.]+)\s*", cells[3])
                if ratio is None:
                    ratio = re.fullmatch(
                        r"\s*每\s*([0-9.]+)\s*份\s*(?:基金份额)?折算为\s*([0-9.]+)\s*份\s*",
                        cells[3],
                    )
                if ratio is None or float(ratio.group(1)) <= 0 or float(ratio.group(2)) <= 0:
                    raise ValueError(f"unsupported share conversion text for {symbol}: {cells[3]!r}")
                ex_date = pd.Timestamp(cells[1]).date().isoformat()
                records.setdefault(ex_date, {}).update(
                    {
                        "symbol": symbol.upper(),
                        "ex_date": ex_date,
                        "share_ratio": float(ratio.group(2)) / float(ratio.group(1)),
                    }
                )

    if recognized_archive_tables == 0:
        raise ValueError(
            f"response for {symbol} is not a recognizable Eastmoney distribution archive"
        )

    source_hash = hashlib.sha256(html_text.encode("utf-8")).hexdigest()
    rows = []
    for ex_date, record in sorted(records.items()):
        rows.append(
            {
                "symbol": record["symbol"],
                "record_date": record.get("record_date"),
                "ex_date": ex_date,
                "cash_payment_date": record.get("cash_payment_date"),
                "cash_dividend_per_old_share": float(record.get("cash_dividend_per_old_share", 0.0)),
                "share_ratio": float(record.get("share_ratio", 1.0)),
                "fractional_share_treatment": (
                    EASTMONEY_UNKNOWN_FRACTIONAL_TREATMENT
                    if not math.isclose(float(record.get("share_ratio", 1.0)), 1.0, rel_tol=0.0, abs_tol=1e-12)
                    else NO_SHARE_CHANGE_TREATMENT
                ),
                "source_url": source_url,
                "source_sha256": source_hash,
            }
        )
    return pd.DataFrame(rows, columns=CORPORATE_ACTION_COLUMNS)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _RequestPacer:
    """Serialize request starts across workers without holding up parsing."""

    def __init__(self, interval_seconds: float):
        self._interval_seconds = interval_seconds
        self._next_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            remaining = self._next_request_at - now
            if remaining > 0:
                time.sleep(remaining)
            self._next_request_at = time.monotonic() + self._interval_seconds


class _CorporateActionRequestError(RuntimeError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


def _download_eastmoney_corporate_action_html(
    session: requests.Session,
    symbol: str,
    *,
    attempts: int = DEFAULT_ACTION_ATTEMPTS,
    request_delay_seconds: float = DEFAULT_ACTION_REQUEST_DELAY_SECONDS,
    request_pacer: _RequestPacer | None = None,
) -> tuple[str, int]:
    """Download one archive page with explicit throttling and retry behavior."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must be non-negative")

    url = _eastmoney_corporate_action_url(symbol)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if request_pacer is not None:
            request_pacer.wait()
        try:
            response = session.get(url, headers={"Referer": "https://fundf10.eastmoney.com/"}, timeout=30)
        except requests.RequestException as exc:
            last_error = exc
        else:
            status_code = int(getattr(response, "status_code", 200))
            if status_code in ACTION_RETRY_STATUS_CODES:
                last_error = requests.HTTPError(f"HTTP {status_code} for {url}", response=response)
            else:
                # Non-transient 4xx responses fail immediately instead of wasting attempts.
                try:
                    response.raise_for_status()
                except requests.RequestException as exc:
                    raise _CorporateActionRequestError(
                        f"non-retryable corporate-action response for {symbol}: {exc}", attempt
                    ) from exc
                html_text = response.text
                if html_text and html_text.strip():
                    return html_text, attempt
                last_error = ValueError(f"empty corporate-action page for {symbol}")

        if attempt == attempts:
            break
        base_delay = max(DEFAULT_ACTION_REQUEST_DELAY_SECONDS, request_delay_seconds)
        exponential_delay = min(30.0, base_delay * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, min(1.0, exponential_delay * 0.25))
        retry_delay = exponential_delay + jitter
        LOGGER.warning(
            "%s corporate-action request attempt %d/%d failed (%s); retrying in %.2fs",
            symbol,
            attempt,
            attempts,
            last_error,
            retry_delay,
        )
        time.sleep(retry_delay)

    raise _CorporateActionRequestError(
        f"corporate-action request failed for {symbol} after {attempts} attempt(s): {last_error}", attempts
    ) from last_error


def fetch_eastmoney_corporate_actions(
    session: requests.Session,
    symbol: str,
    attempts: int = DEFAULT_ACTION_ATTEMPTS,
    request_delay_seconds: float = DEFAULT_ACTION_REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """Fetch and parse one fund page; retained as the public single-symbol API."""
    html_text, _ = _download_eastmoney_corporate_action_html(
        session,
        symbol,
        attempts=attempts,
        request_delay_seconds=request_delay_seconds,
    )
    return _parse_eastmoney_corporate_actions(html_text, symbol, _eastmoney_corporate_action_url(symbol))


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def _resolve_corporate_action_symbols(
    universe_symbols: Iterable[str],
    symbols: Iterable[str] | str | None = None,
    symbols_file: Path | str | None = None,
) -> tuple[list[str], list[str]]:
    all_symbols = sorted({str(symbol).strip().upper() for symbol in universe_symbols if str(symbol).strip()})
    for symbol in all_symbols:
        _fund_archive_code(symbol)

    requested: list[str] = []
    if symbols is not None:
        requested.extend([symbols] if isinstance(symbols, str) else symbols)
    if symbols_file is not None:
        path = Path(symbols_file)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            value = line.split("#", 1)[0].strip()
            if value:
                requested.append(value)
    if symbols is None and symbols_file is None:
        return all_symbols, all_symbols

    selected = sorted({str(symbol).strip().upper() for symbol in requested if str(symbol).strip()})
    if not selected:
        raise ValueError("the requested corporate-action symbol scope is empty")
    for symbol in selected:
        _fund_archive_code(symbol)
    missing = sorted(set(selected) - set(all_symbols))
    if missing:
        raise ValueError(f"symbols are not in the current T+1 ETF universe: {missing}")
    return all_symbols, selected


def collect_corporate_actions(
    data_dir: Path,
    workers: int = 4,
    insecure: bool = False,
    symbols: Iterable[str] | str | None = None,
    symbols_file: Path | str | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    request_delay_seconds: float = DEFAULT_ACTION_REQUEST_DELAY_SECONDS,
    attempts: int = DEFAULT_ACTION_ATTEMPTS,
) -> pd.DataFrame:
    """Collect company actions, publishing only a complete-universe successful run."""
    data_dir = Path(data_dir)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must be non-negative")

    universe = pd.read_csv(data_dir / "universe.csv", dtype={"code": str})
    if "symbol" not in universe:
        raise ValueError("universe.csv lacks the symbol column")
    all_symbols, target_symbols = _resolve_corporate_action_symbols(
        universe["symbol"].dropna(), symbols, symbols_file
    )
    cache_root = Path(cache_dir) if cache_dir is not None else data_dir / "corporate_action_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    request_pacer = _RequestPacer(request_delay_seconds)

    def fetch_one(symbol: str) -> tuple[pd.DataFrame, dict]:
        cache_path = cache_root / f"{symbol}.html"
        source_url = _eastmoney_corporate_action_url(symbol)
        cache_exists = cache_path.is_file()
        cache_problem: str | None = None

        if cache_exists and not refresh:
            try:
                html_text = cache_path.read_text(encoding="utf-8")
                if not html_text.strip():
                    cache_problem = "invalid_empty"
                else:
                    frame = _parse_eastmoney_corporate_actions(html_text, symbol, source_url)
                    return frame, {
                        "symbol": symbol,
                        "events": len(frame),
                        "source": "cache",
                        "cache_status": "hit",
                        "cache_path": str(cache_path),
                        "cache_sha256": _text_sha256(html_text),
                        "request_attempts": 0,
                        "error": None,
                    }
            except Exception:
                cache_problem = "invalid_parse"

        session: requests.Session | None = None
        request_attempts = 0
        try:
            session = make_session(insecure, automatic_retries=False)
            html_text, request_attempts = _download_eastmoney_corporate_action_html(
                session,
                symbol,
                attempts=attempts,
                request_delay_seconds=request_delay_seconds,
                request_pacer=request_pacer,
            )
            frame = _parse_eastmoney_corporate_actions(html_text, symbol, source_url)
            _atomic_text(html_text, cache_path)
            if refresh:
                cache_status = "refreshed" if cache_exists else "refresh_miss_saved"
            elif cache_problem is not None:
                cache_status = f"{cache_problem}_replaced"
            else:
                cache_status = "miss_saved"
            return frame, {
                "symbol": symbol,
                "events": len(frame),
                "source": "network",
                "cache_status": cache_status,
                "cache_path": str(cache_path),
                "cache_sha256": _text_sha256(html_text),
                "request_attempts": request_attempts,
                "error": None,
            }
        except Exception as exc:
            request_attempts = int(getattr(exc, "attempts", request_attempts))
            if refresh:
                cache_status = "refresh_failed_preserved" if cache_exists else "refresh_miss_failed"
            elif cache_problem is not None:
                cache_status = f"{cache_problem}_failed_preserved"
            else:
                cache_status = "miss_failed"
            return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS), {
                "symbol": symbol,
                "events": 0,
                "source": "cache+network" if cache_problem is not None else "network",
                "cache_status": cache_status,
                "cache_path": str(cache_path),
                "cache_sha256": (
                    hashlib.sha256(cache_path.read_bytes()).hexdigest()
                    if cache_path.is_file() and cache_path.stat().st_size > 0
                    else None
                ),
                "request_attempts": request_attempts,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if session is not None:
                session.close()

    frames: list[pd.DataFrame] = []
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, symbol) for symbol in target_symbols]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading corporate actions"):
            frame, result = future.result()
            if result["error"] is None:
                frames.append(frame)
            rows.append(result)

    report = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    failures = report[report["error"].notna()]
    full_scope = target_symbols == all_symbols
    report["full_universe_scope"] = full_scope
    report["published"] = bool(full_scope and failures.empty)
    _atomic_csv(report, data_dir / "corporate_action_report.csv")
    if not failures.empty:
        raise RuntimeError(
            f"corporate-action download failed for {len(failures)} ETF(s); "
            f"see {data_dir / 'corporate_action_report.csv'}"
        )

    actions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
    actions = actions[CORPORATE_ACTION_COLUMNS]
    if not actions.empty:
        actions = actions.sort_values(["ex_date", "symbol"]).reset_index(drop=True)
    if full_scope:
        _atomic_csv(actions, data_dir / "corporate_actions.csv")
        LOGGER.info("published %d corporate actions for all %d ETFs", len(actions), len(target_symbols))
    else:
        LOGGER.info(
            "cached %d corporate actions for %d/%d ETFs; canonical table was not replaced",
            len(actions),
            len(target_symbols),
            len(all_symbols),
        )
    return actions


def _sina_total_return_multiplier(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Convert Sina's affine hfq events into a positive reinvested-return multiplier."""
    merged = pd.merge_asof(
        prices[["date", "close"]].sort_values("date"),
        factors.sort_values("date"),
        on="date",
        direction="backward",
    )
    if merged[["f", "s", "u"]].isna().any().any():
        raise ValueError("Sina factors do not cover the full price history")
    multipliers = np.ones(len(merged), dtype=float)
    current_multiplier = 1.0
    for index in range(1, len(merged)):
        previous = merged.iloc[index - 1]
        current = merged.iloc[index]
        previous_params = previous[["f", "s", "u"]].to_numpy(dtype=float)
        current_params = current[["f", "s", "u"]].to_numpy(dtype=float)
        if not np.allclose(previous_params, current_params, rtol=0.0, atol=1e-12):
            # Sina hfq maps a raw price P to P*f*s+u. Solve the old mapping in
            # the new unit system, then chain the event as a multiplicative
            # total-return factor. Hfq events become effective on the actual
            # ex-right date, including funds whose qfq metadata changes early.
            previous_scale = float(previous["f"] * previous["s"])
            current_scale = float(current["f"] * current["s"])
            equivalent_price = (
                float(previous["close"]) * previous_scale
                + float(previous["u"])
                - float(current["u"])
            ) / current_scale
            if not np.isfinite(equivalent_price) or equivalent_price <= 0:
                raise ValueError(f"invalid Sina corporate-action event at {current['date'].date()}")
            current_multiplier *= float(previous["close"]) / equivalent_price
        multipliers[index] = current_multiplier
    if not np.isfinite(multipliers).all() or (multipliers <= 0).any():
        raise ValueError("invalid Sina total-return multiplier")
    return pd.Series(multipliers, index=merged.index)


def fetch_history_sina_pair(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sina_symbol = symbol.lower()
    response = session.get(
        SINA_HISTORY_URL.format(sina_symbol),
        headers={"Referer": f"https://finance.sina.com.cn/realstock/company/{sina_symbol}/nc.shtml"},
        timeout=35,
    )
    response.raise_for_status()
    assignment = response.text.split("=", 1)[1].split(";", 1)[0].strip()
    encoded = json.loads(assignment)
    frame = pd.DataFrame(_decode_sina_klc(encoded))
    required = ["date", "open", "close", "high", "low", "volume", "amount"]
    if frame.empty or not set(required).issubset(frame.columns):
        raise ValueError(f"Sina history lacks columns: {required}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame[required[1:]] = frame[required[1:]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=required).drop_duplicates("date", keep="last").sort_values("date")

    factors = fetch_sina_factors(session, symbol)
    multiplier = _sina_total_return_multiplier(frame, factors).to_numpy()

    adjusted = frame[["date", "open", "close", "high", "low"]].copy()
    for column in ("open", "close", "high", "low"):
        adjusted[column] *= multiplier
    adjusted["symbol"] = symbol
    adjusted = adjusted[["date", "symbol", "open", "close", "high", "low"]].rename(
        columns={column: f"adj_{column}" for column in ("open", "close", "high", "low")}
    )

    raw = frame.copy()
    raw["symbol"] = symbol
    raw["amplitude"] = (raw["high"] - raw["low"]) / raw["close"].shift(1) * 100.0
    raw["pct_change"] = raw["close"].pct_change(fill_method=None) * 100.0
    raw["price_change"] = raw["close"].diff()
    raw["turnover_rate"] = np.nan
    raw["data_source"] = "sina"
    raw["amount_quality"] = "estimated_ohlc_average"
    raw = raw[
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
    requested = raw["date"].dt.date.between(start_date, end_date)
    return raw.loc[requested].reset_index(drop=True), adjusted.loc[requested].reset_index(drop=True)


def fetch_history_tencent(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    adjusted: bool,
) -> pd.DataFrame:
    """Fetch Tencent history backwards because adjusted responses are capped at 640 rows."""
    exchange_code = symbol.lower()
    page_size = 640 if adjusted else 2000
    cursor = end_date
    pages: list[pd.DataFrame] = []
    while cursor >= start_date:
        adjust_name = "hfq" if adjusted else ""
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
            # Tencent returns ``day`` for an adjusted request when the instrument has
            # no recorded adjustment event; instruments with events use
            # ``hfqday``. In both cases the response is for the hfq request.
            rows = symbol_data.get("hfqday") or symbol_data.get("day") or []
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
            columns={column: f"adj_{column}" for column in ("open", "close", "high", "low")}
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
    if history_source == "sina":
        raw, adjusted_frame = fetch_history_sina_pair(session, symbol, start_date, end_date)
        return adjusted_frame if adjusted else raw
    if history_source == "tencent":
        return fetch_history_tencent(session, symbol, start_date, end_date, adjusted)
    if history_source == "eastmoney":
        return fetch_history_eastmoney(session, symbol, start_date, end_date, adjusted)
    raw, adjusted_frame = fetch_history_sina_pair(session, symbol, start_date, end_date)
    return adjusted_frame if adjusted else raw


def fetch_history_pair(
    session: requests.Session,
    symbol: str,
    start_date: date,
    end_date: date,
    history_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if history_source == "sina":
        return fetch_history_sina_pair(session, symbol, start_date, end_date)
    if history_source in {"eastmoney", "tencent"}:
        raw = fetch_history(session, symbol, start_date, end_date, False, history_source)
        adjusted = fetch_history(session, symbol, start_date, end_date, True, history_source)
        return raw, adjusted
    try:
        return fetch_history_sina_pair(session, symbol, start_date, end_date)
    except Exception as exc:
        LOGGER.warning("%s Sina history failed (%s); using Eastmoney", symbol, exc)
        raw = fetch_history_eastmoney(session, symbol, start_date, end_date, False)
        adjusted = fetch_history_eastmoney(session, symbol, start_date, end_date, True)
        return raw, adjusted


def combine_histories(raw: pd.DataFrame, adjusted: pd.DataFrame, end_date: date) -> pd.DataFrame:
    if raw.empty or adjusted.empty:
        raise ValueError("raw or adjusted history is empty")
    frame = raw.merge(adjusted, on=["date", "symbol"], how="inner", validate="one_to_one")
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
    overlap = old[["date", "adj_close"]].merge(new[["date", "adj_close"]], on="date", suffixes=("_old", "_new"))
    if overlap.empty:
        return False
    relative = (overlap["adj_close_new"] / overlap["adj_close_old"] - 1).abs()
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
                incompatible = "adj_close" not in old.columns
                if history_source in {"auto", "sina"}:
                    old_quality = old.get("amount_quality", pd.Series(dtype=str))
                    incompatible |= not old_quality.isin(
                        {"reported", "estimated_ohlc_average"}
                    ).all()
                if incompatible:
                    old = pd.DataFrame()
                    fetch_start = start_date
                    refreshed = True
                else:
                    fetch_start = max(start_date, old["date"].max().date() - timedelta(days=30))
        raw, adjusted = fetch_history_pair(session, symbol, fetch_start, end_date, history_source)
        fresh = combine_histories(raw, adjusted, end_date)
        refreshed = full_refresh or old.empty
        if not old.empty and _overlap_changed(old, fresh):
            LOGGER.info("%s adjustment changed; refreshing full history", symbol)
            raw, adjusted = fetch_history_pair(session, symbol, start_date, end_date, history_source)
            fresh = combine_histories(raw, adjusted, end_date)
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
    frozen_universe: bool = False,
) -> tuple[pd.DataFrame, list[DownloadResult]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    universe_path = data_dir / "universe.csv"
    if frozen_universe:
        if limit is not None or symbols:
            raise ValueError("--frozen-universe cannot be combined with --limit or --symbols")
        if not universe_path.is_file():
            raise FileNotFoundError(f"frozen universe is missing: {universe_path}")
        universe = pd.read_csv(universe_path, dtype={"code": str})
        if "symbol" not in universe or universe["symbol"].isna().any():
            raise ValueError("frozen universe lacks a complete symbol column")
        universe["symbol"] = universe["symbol"].astype(str).str.upper()
        if universe["symbol"].duplicated().any():
            raise ValueError("frozen universe contains duplicate symbols")
    else:
        session = make_session(insecure)
        try:
            snapshot_date = datetime.now(ZoneInfo("Asia/Shanghai")).date()
            universe, dropped = build_t1_universe(session, snapshot_date)
        finally:
            session.close()
        if dropped:
            dropped_frame = pd.DataFrame(dropped)
            _atomic_csv(dropped_frame, data_dir / "universe_dropped.csv")
            quote_dropped = dropped_frame[
                dropped_frame["reason"] == "missing_or_nonpositive_last_price"
            ]
            LOGGER.warning(
                "universe snapshot dropped %d ETF(s), of which %d with missing/non-positive "
                "last price; details in universe_dropped.csv",
                len(dropped_frame),
                len(quote_dropped),
            )
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
    _atomic_csv(universe, universe_path)
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
    results.sort(key=lambda result: result.symbol)
    result_frame = pd.DataFrame([result.__dict__ for result in results])
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


def readjust_sina_dataset(data_dir: Path, workers: int = 4, insecure: bool = False) -> pd.DataFrame:
    """Rebuild adjusted OHLC from saved Sina raw prices without redownloading K-lines."""
    raw_dir = data_dir / "raw"
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no raw CSV files under {raw_dir}")

    def readjust_one(path: Path) -> dict:
        session = make_session(insecure)
        try:
            frame = pd.read_csv(path, parse_dates=["date"])
            required = {"date", "symbol", "raw_open", "raw_close", "raw_high", "raw_low", "data_source"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"missing columns: {missing}")
            if not frame["data_source"].eq("sina").all():
                raise ValueError("readjust only accepts Sina raw histories")
            symbol = str(frame["symbol"].iloc[0]).upper()
            factors = fetch_sina_factors(session, symbol)
            prices = frame[["date", "raw_close"]].rename(columns={"raw_close": "close"})
            multiplier = _sina_total_return_multiplier(prices, factors).to_numpy()
            for field in ("open", "close", "high", "low"):
                frame[f"adj_{field}"] = pd.to_numeric(frame[f"raw_{field}"], errors="raise") * multiplier
            _atomic_csv(frame[RAW_COLUMNS], path)
            return {"symbol": symbol, "rows": len(frame), "error": None}
        except Exception as exc:
            return {"symbol": path.stem.upper(), "rows": 0, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            session.close()

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(readjust_one, path) for path in files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rebuilding Sina adjustment factors"):
            rows.append(future.result())
    report = pd.DataFrame(rows).sort_values("symbol")
    _atomic_csv(report, data_dir / "readjust_report.csv")
    failures = report[report["error"].notna()]
    if not failures.empty:
        raise RuntimeError(f"readjustment failed for {len(failures)} ETF(s); see {data_dir / 'readjust_report.csv'}")
    LOGGER.info("readjusted %d Sina ETF histories", len(report))
    return report


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
    valid_ohlc = _valid_ohlc(frame, "raw") & _valid_ohlc(frame, "adj")
    if not valid_ohlc.all():
        bad_dates = frame.loc[~valid_ohlc, "date"].dt.strftime("%Y-%m-%d").head(5).tolist()
        raise ValueError(f"invalid raw/adjusted OHLC rows: {bad_dates}")
    active = frame["volume"].gt(0) & frame["amount"].ge(0)
    frame = frame[active].copy()
    if frame.empty:
        raise ValueError("no valid OHLCV rows after cleaning")
    first_adjusted_close = float(frame["adj_close"].iloc[0])
    if not np.isfinite(first_adjusted_close) or first_adjusted_close <= 0:
        raise ValueError("invalid first adjusted close")
    factor = (frame["adj_close"] / frame["raw_close"]) / first_adjusted_close
    if factor.isna().any() or (factor <= 0).any():
        raise ValueError("invalid adjustment factor")
    output = pd.DataFrame({"date": frame["date"], "symbol": frame["symbol"].str.upper()})
    for column in ("open", "close", "high", "low"):
        output[column] = frame[f"adj_{column}"] / first_adjusted_close
    output["factor"] = factor
    output["volume"] = frame["volume"] / factor
    output["change"] = frame["adj_close"].pct_change(fill_method=None).fillna(0.0)
    output["amount"] = frame["amount"]
    output["vwap"] = (frame["amount"] / frame["volume"]) * factor
    output["amount_estimated"] = (frame["amount_quality"] != "reported").astype(float)
    output["paused"] = 0.0
    values = output[QLIB_FIELDS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("normalized data contains infinity")
    return output[["date", "symbol", *QLIB_FIELDS]].reset_index(drop=True)


def _universe_symbols(data_dir: Path) -> list[str]:
    universe = pd.read_csv(data_dir / "universe.csv", dtype={"code": str})
    if "symbol" not in universe or universe["symbol"].isna().any():
        raise ValueError("universe.csv lacks a complete symbol column")
    symbols = universe["symbol"].astype(str).str.upper().tolist()
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("universe.csv must contain unique symbols")
    return symbols


def normalize_dataset(data_dir: Path, workers: int = 4) -> pd.DataFrame:
    raw_dir = data_dir / "raw"
    normalized_dir = data_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    symbols = _universe_symbols(data_dir)
    files = [raw_dir / f"{symbol.lower()}.csv" for symbol in symbols]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frozen-universe raw files are missing; examples={missing[:5]}")
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
    raw_dir = data_dir / "raw"
    normalized_dir = data_dir / "normalized"
    if not universe_path.exists():
        raise FileNotFoundError(universe_path)
    universe = pd.read_csv(universe_path, dtype={"code": str})
    symbols = _universe_symbols(data_dir)
    raw_files = [raw_dir / f"{symbol.lower()}.csv" for symbol in symbols]
    files = [normalized_dir / f"{symbol.lower()}.csv" for symbol in symbols]
    missing_raw_files = [path.name for path in raw_files if not path.is_file()]
    missing_normalized_files = [path.name for path in files if not path.is_file()]
    raw_files = [path for path in raw_files if path.is_file()]
    files = [path for path in files if path.is_file()]
    all_raw_files = sorted(raw_dir.glob("*.csv"))
    all_normalized_files = sorted(normalized_dir.glob("*.csv"))
    issues: list[dict] = []
    for filename in missing_raw_files:
        issues.append({"phase": "coverage", "file": filename, "error": "missing frozen-universe raw file"})
    for filename in missing_normalized_files:
        issues.append({"phase": "coverage", "file": filename, "error": "missing frozen-universe normalized file"})
    raw_total_rows = 0
    reported_amount_rows = 0
    source_counts: dict[str, int] = {}
    for path in tqdm(raw_files, desc="Validating raw data"):
        try:
            raw = pd.read_csv(path, parse_dates=["date"])
            raw_total_rows += len(raw)
            missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
            if missing:
                raise ValueError(f"missing columns: {missing}")
            if raw.empty:
                raise ValueError("empty file")
            if raw["date"].duplicated().any() or not raw["date"].is_monotonic_increasing:
                raise ValueError("dates are duplicated or unsorted")
            numeric_columns = [
                column for column in RAW_COLUMNS if column not in {"date", "symbol", "data_source", "amount_quality"}
            ]
            raw[numeric_columns] = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
            valid_ohlc = _valid_ohlc(raw, "raw") & _valid_ohlc(raw, "adj")
            if not valid_ohlc.all():
                bad_dates = raw.loc[~valid_ohlc, "date"].dt.strftime("%Y-%m-%d").head(5).tolist()
                raise ValueError(f"{int((~valid_ohlc).sum())} invalid OHLC rows, examples={bad_dates}")
            if (raw["volume"] < 0).any() or (raw["amount"] < 0).any():
                raise ValueError("contains negative volume/amount")
            returns = raw["adj_close"].pct_change(fill_method=None)
            jump_count = int((returns.abs() > 0.25).sum())
            if jump_count:
                jump_dates = raw.loc[returns.abs() > 0.25, "date"].dt.strftime("%Y-%m-%d").head(5).tolist()
                raise ValueError(f"{jump_count} adjusted return jumps above 25%, examples={jump_dates}")
            amount_reported = raw["amount_quality"].eq("reported")
            reported_amount_rows += int(amount_reported.sum())
            for source, count in raw["data_source"].fillna("missing").value_counts().items():
                source_counts[str(source)] = source_counts.get(str(source), 0) + int(count)
        except Exception as exc:
            issues.append({"phase": "raw", "file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    total_rows = 0
    estimated_rows = 0
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
            high = frame["high"]
            low = frame["low"]
            if (high + 1e-8 < frame[["open", "close", "low"]].max(axis=1)).any() or (
                low - 1e-8 > frame[["open", "close", "high"]].min(axis=1)
            ).any():
                raise ValueError("contains invalid normalized OHLC ordering")
            if (frame["change"].abs() > 0.25).any():
                raise ValueError("contains adjusted return jumps above 25%")
            estimated_rows += int(frame["amount_estimated"].fillna(1).ne(0).sum())
            end_date = frame["date"].max().date()
            end_dates.append(end_date)
            if expected_end and (expected_end - end_date).days > max_stale_days:
                raise ValueError(f"stale last date: {end_date}")
        except Exception as exc:
            issues.append({"phase": "normalized", "file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    training_ready = not issues
    report = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "training_ready": training_ready,
        "universe_count": int(len(universe)),
        "raw_file_count": len(raw_files),
        "raw_cache_file_count": len(all_raw_files),
        "pool_external_raw_file_count": len(all_raw_files) - len(raw_files),
        "raw_total_rows": raw_total_rows,
        "data_source_rows": source_counts,
        "reported_amount_ratio": reported_amount_rows / raw_total_rows if raw_total_rows else 0.0,
        "estimated_amount_ratio": estimated_rows / total_rows if total_rows else 0.0,
        "normalized_file_count": len(files),
        "normalized_cache_file_count": len(all_normalized_files),
        "pool_external_normalized_file_count": len(all_normalized_files) - len(files),
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
    LOGGER.info("validation passed and training-ready: %d ETFs, %d rows", len(files), total_rows)
    return report


def dump_to_qlib(data_dir: Path, qlib_dir: Path, workers: int = 4, overwrite: bool = False) -> None:
    normalized_dir = data_dir / "normalized"
    symbols = _universe_symbols(data_dir)
    selected_files = [normalized_dir / f"{symbol.lower()}.csv" for symbol in symbols]
    missing = [path.name for path in selected_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frozen-universe normalized files are missing; examples={missing[:5]}")
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
    dumper.df_files = selected_files
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
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--limit", type=int)
    download.add_argument("--symbols", nargs="+")
    download.add_argument(
        "--frozen-universe",
        action="store_true",
        help="reuse universe.csv exactly and update only its symbols",
    )
    download.add_argument("--full-refresh", action="store_true")
    download.add_argument("--insecure", action="store_true")
    download.add_argument("--history-source", choices=("auto", "sina", "eastmoney", "tencent"), default="auto")

    readjust = subparsers.add_parser("readjust", help="rebuild Sina adjustment factors from saved raw K-lines")
    add_common(readjust)
    readjust.add_argument("--workers", type=int, default=4)
    readjust.add_argument("--insecure", action="store_true")

    actions = subparsers.add_parser("actions", help="download auditable ETF cash distributions and share conversions")
    add_common(actions)
    actions.add_argument("--workers", type=int, default=4)
    actions.add_argument("--symbols", nargs="+")
    actions.add_argument("--symbols-file", type=Path)
    actions.add_argument("--cache-dir", type=Path)
    actions.add_argument("--refresh", action="store_true")
    actions.add_argument(
        "--request-delay-seconds",
        type=float,
        default=DEFAULT_ACTION_REQUEST_DELAY_SECONDS,
    )
    actions.add_argument("--attempts", type=int, default=DEFAULT_ACTION_ATTEMPTS)
    actions.add_argument("--insecure", action="store_true")

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
    all_parser.add_argument("--workers", type=int, default=4)
    all_parser.add_argument("--limit", type=int)
    all_parser.add_argument("--symbols", nargs="+")
    all_parser.add_argument(
        "--frozen-universe",
        action="store_true",
        help="reuse universe.csv exactly and update only its symbols",
    )
    all_parser.add_argument("--full-refresh", action="store_true")
    all_parser.add_argument("--insecure", action="store_true")
    all_parser.add_argument("--history-source", choices=("auto", "sina", "eastmoney", "tencent"), default="auto")
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
            args.frozen_universe,
        )
        failures = [result for result in results if result.error]
        return 1 if failures else 0
    if args.command == "readjust":
        readjust_sina_dataset(args.data_dir, args.workers, args.insecure)
        return 0
    if args.command == "actions":
        collect_corporate_actions(
            args.data_dir,
            args.workers,
            args.insecure,
            args.symbols,
            args.symbols_file,
            args.cache_dir,
            args.refresh,
            args.request_delay_seconds,
            args.attempts,
        )
        return 0
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
            args.frozen_universe,
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
