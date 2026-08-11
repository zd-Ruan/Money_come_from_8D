from datetime import date, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


COLLECTOR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data_collector" / "cn_etf" / "collector.py"
)
SPEC = spec_from_file_location("cn_etf_collector", COLLECTOR_PATH)
collector = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def test_latest_complete_date_excludes_intraday_bar():
    timezone = ZoneInfo("Asia/Shanghai")
    assert collector.latest_complete_date(datetime(2026, 8, 11, 14, 30, tzinfo=timezone)) == date(2026, 8, 10)
    assert collector.latest_complete_date(datetime(2026, 8, 11, 16, 1, tzinfo=timezone)) == date(2026, 8, 11)


def test_market_prefix():
    assert collector.market_prefix("510300", 1) == "SH"
    assert collector.market_prefix("159915", 0) == "SZ"
    assert collector.eastmoney_secid("SH510300") == "1.510300"
    assert collector.eastmoney_secid("SZ159915") == "0.159915"


def test_normalize_frame_matches_qlib_factor_contract():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-07", "2026-08-10"]),
            "symbol": ["SH510300", "SH510300"],
            "raw_open": [4.0, 4.2],
            "raw_close": [4.1, 4.3],
            "raw_high": [4.2, 4.4],
            "raw_low": [3.9, 4.1],
            "volume": [1_000_000.0, 1_200_000.0],
            "amount": [4_100_000.0, 5_160_000.0],
            "amplitude": [1.0, 1.0],
            "pct_change": [0.0, 4.878],
            "price_change": [0.0, 0.2],
            "turnover_rate": [0.01, 0.012],
            "qfq_open": [2.0, 4.2],
            "qfq_close": [2.05, 4.3],
            "qfq_high": [2.1, 4.4],
            "qfq_low": [1.95, 4.1],
        }
    )
    normalized = collector.normalize_frame(raw)
    assert np.isclose(normalized.loc[0, "close"], 1.0)
    reconstructed = normalized["close"] / normalized["factor"]
    assert np.allclose(reconstructed, raw["raw_close"])
    reconstructed_volume = normalized["volume"] * normalized["factor"]
    assert np.allclose(reconstructed_volume, raw["volume"])
    assert np.allclose(normalized["vwap"], (raw["amount"] / raw["volume"]) * normalized["factor"])


def test_invalid_ohlc_and_zero_volume_are_removed():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-07", "2026-08-08", "2026-08-10"]),
            "symbol": ["SZ159915"] * 3,
            "raw_open": [1.0, 1.0, 1.0],
            "raw_close": [1.0, 1.0, 1.0],
            "raw_high": [1.1, 0.9, 1.1],
            "raw_low": [0.9, 0.8, 0.9],
            "volume": [100.0, 100.0, 0.0],
            "amount": [100.0, 100.0, 0.0],
            "turnover_rate": [0.01, 0.01, 0.0],
            "qfq_open": [1.0, 1.0, 1.0],
            "qfq_close": [1.0, 1.0, 1.0],
            "qfq_high": [1.1, 1.1, 1.1],
            "qfq_low": [0.9, 0.9, 0.9],
        }
    )
    normalized = collector.normalize_frame(raw)
    assert normalized["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-07"]


def test_tencent_qfq_request_accepts_day_key_when_no_adjustment_exists():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "code": 0,
                "data": {
                    "sh510040": {
                        "day": [["2026-08-10", "1.196", "1.193", "1.201", "1.184", "73671"]]
                    }
                },
            }

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    frame = collector.fetch_history_tencent(
        FakeSession(), "SH510040", date(2005, 1, 1), date(2026, 8, 10), adjusted=True
    )
    assert frame.loc[0, "qfq_close"] == 1.193
    assert frame.loc[0, "symbol"] == "SH510040"
