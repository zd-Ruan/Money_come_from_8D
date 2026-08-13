from datetime import date, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest


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
            "adj_open": [2.0, 4.2],
            "adj_close": [2.05, 4.3],
            "adj_high": [2.1, 4.4],
            "adj_low": [1.95, 4.1],
            "data_source": ["sina", "sina"],
            "amount_quality": ["reported", "reported"],
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
            "adj_open": [1.0, 1.0, 1.0],
            "adj_close": [1.0, 1.0, 1.0],
            "adj_high": [1.1, 1.1, 1.1],
            "adj_low": [0.9, 0.9, 0.9],
            "data_source": ["sina"] * 3,
            "amount_quality": ["reported"] * 3,
        }
    )
    with pytest.raises(ValueError, match="invalid raw/adjusted OHLC"):
        collector.normalize_frame(raw)


def test_tencent_hfq_request_accepts_day_key_when_no_adjustment_exists():
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
    assert frame.loc[0, "adj_close"] == 1.193
    assert frame.loc[0, "symbol"] == "SH510040"


def test_sina_cash_distribution_becomes_positive_multiplicative_return():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "close": [1.0, 0.91],
        }
    )
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["1900-01-01", "2026-01-02"]),
            "f": [1.0, 1.0],
            "s": [1.0, 1.0],
            "u": [0.0, 0.1],
        }
    )
    multiplier = collector._sina_total_return_multiplier(prices, factors)
    adjusted = prices["close"] * multiplier
    assert np.allclose(multiplier, [1.0, 1.0 / 0.9])
    assert np.isclose(adjusted.pct_change().iloc[1], 0.91 / 0.9 - 1.0)


def test_eastmoney_corporate_actions_preserve_cash_dates_and_share_ratio():
    document = """
    <table><thead><tr><th>年份</th><th>权益登记日</th><th>除息日</th><th>每10份分红</th><th>分红发放日</th></tr></thead>
    <tbody><tr><td>2026年</td><td>2026-01-16</td><td>2026-01-19</td><td>每10份派现金1.2300元</td><td>2026-01-27</td></tr></tbody></table>
    <table><thead><tr><th>年份</th><th>拆分折算日</th><th>拆分类型</th><th>拆分折算比例</th></tr></thead>
    <tbody><tr><td>2026年</td><td>2026-01-19</td><td>份额折算</td><td>1:2.5000</td></tr></tbody></table>
    """
    frame = collector._parse_eastmoney_corporate_actions(document, "SH510300", "https://example.test")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["record_date"] == "2026-01-16"
    assert row["ex_date"] == "2026-01-19"
    assert row["cash_payment_date"] == "2026-01-27"
    assert row["cash_dividend_per_old_share"] == 0.123
    assert row["share_ratio"] == 2.5
    assert row["fractional_share_treatment"] == collector.EASTMONEY_UNKNOWN_FRACTIONAL_TREATMENT
    assert len(row["source_sha256"]) == 64


def test_corporate_action_parser_distinguishes_zero_events_from_a_200_error_page():
    valid_empty_archive = """
    <table><thead><tr><th>年份</th><th>权益登记日</th><th>除息日</th><th>每10份分红</th><th>分红发放日</th></tr></thead><tbody></tbody></table>
    <table><thead><tr><th>年份</th><th>拆分折算日</th><th>拆分类型</th><th>拆分折算比例</th></tr></thead><tbody></tbody></table>
    """
    frame = collector._parse_eastmoney_corporate_actions(
        valid_empty_archive, "SH510300", "https://example.test"
    )
    assert frame.empty
    assert list(frame.columns) == collector.CORPORATE_ACTION_COLUMNS

    with pytest.raises(ValueError, match="not a recognizable Eastmoney"):
        collector._parse_eastmoney_corporate_actions(
            "<html><title>request accepted</title></html>",
            "SH510300",
            "https://example.test",
        )


def _corporate_action_document(cash_amount: str = "1.2300") -> str:
    return f"""
    <table><thead><tr><th>年份</th><th>权益登记日</th><th>除息日</th><th>每10份分红</th><th>分红发放日</th></tr></thead>
    <tbody><tr><td>2026年</td><td>2026-01-16</td><td>2026-01-19</td><td>每10份派现金{cash_amount}元</td><td>2026-01-27</td></tr></tbody></table>
    """


def _write_universe(data_dir: Path, symbols: list[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols, "code": [symbol[2:] for symbol in symbols]}).to_csv(
        data_dir / "universe.csv", index=False
    )


def test_corporate_action_collection_reuses_nonempty_parsed_cache(monkeypatch, tmp_path):
    _write_universe(tmp_path, ["SH510300"])
    calls = []

    def download(_session, symbol, **_kwargs):
        calls.append(symbol)
        return _corporate_action_document(), 1

    monkeypatch.setattr(collector, "_download_eastmoney_corporate_action_html", download)
    first = collector.collect_corporate_actions(tmp_path, workers=1, request_delay_seconds=0, attempts=1)
    assert calls == ["SH510300"]
    assert len(first) == 1
    cache_path = tmp_path / "corporate_action_cache" / "SH510300.html"
    assert cache_path.read_text(encoding="utf-8").strip()
    assert not list(cache_path.parent.glob("*.tmp"))

    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("a valid cache hit must not use the network")

    monkeypatch.setattr(collector, "_download_eastmoney_corporate_action_html", unexpected_download)
    second = collector.collect_corporate_actions(tmp_path, workers=1, request_delay_seconds=0, attempts=1)
    assert len(second) == 1
    report = pd.read_csv(tmp_path / "corporate_action_report.csv")
    assert report.loc[0, "source"] == "cache"
    assert report.loc[0, "cache_status"] == "hit"
    assert len(report.loc[0, "cache_sha256"]) == 64
    assert report.loc[0, "request_attempts"] == 0


def test_corporate_action_symbol_and_file_scope_only_warms_cache(monkeypatch, tmp_path):
    _write_universe(tmp_path, ["SH510300", "SH510500", "SZ159915"])
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("SZ159915\n# ignored comment\n", encoding="utf-8")
    calls = []

    def download(_session, symbol, **_kwargs):
        calls.append(symbol)
        return _corporate_action_document(), 1

    monkeypatch.setattr(collector, "_download_eastmoney_corporate_action_html", download)
    actions = collector.collect_corporate_actions(
        tmp_path,
        workers=1,
        symbols=["sh510300"],
        symbols_file=symbols_file,
        request_delay_seconds=0,
        attempts=1,
    )
    assert calls == ["SH510300", "SZ159915"]
    assert set(actions["symbol"]) == {"SH510300", "SZ159915"}
    assert not (tmp_path / "corporate_actions.csv").exists()
    report = pd.read_csv(tmp_path / "corporate_action_report.csv")
    assert set(report["symbol"]) == {"SH510300", "SZ159915"}
    assert not report["full_universe_scope"].any()
    assert not report["published"].any()


def test_corporate_action_failure_preserves_table_and_success_cache_for_rerun(monkeypatch, tmp_path):
    symbols = ["SH510300", "SZ159915"]
    _write_universe(tmp_path, symbols)
    canonical_path = tmp_path / "corporate_actions.csv"
    original_contents = "existing,complete\n1,2\n"
    canonical_path.write_text(original_contents, encoding="utf-8")
    first_calls = []

    def partially_failing_download(_session, symbol, **_kwargs):
        first_calls.append(symbol)
        if symbol == "SZ159915":
            raise RuntimeError("temporary HTTP 514")
        return _corporate_action_document(), 1

    monkeypatch.setattr(
        collector,
        "_download_eastmoney_corporate_action_html",
        partially_failing_download,
    )
    with pytest.raises(RuntimeError, match="failed for 1 ETF"):
        collector.collect_corporate_actions(tmp_path, workers=1, request_delay_seconds=0, attempts=1)
    assert first_calls == symbols
    assert canonical_path.read_text(encoding="utf-8") == original_contents
    assert (tmp_path / "corporate_action_cache" / "SH510300.html").is_file()
    assert not (tmp_path / "corporate_action_cache" / "SZ159915.html").exists()
    failed_report = pd.read_csv(tmp_path / "corporate_action_report.csv")
    assert failed_report.set_index("symbol").loc["SH510300", "cache_status"] == "miss_saved"
    assert len(failed_report.set_index("symbol").loc["SH510300", "cache_sha256"]) == 64
    assert failed_report.set_index("symbol").loc["SZ159915", "cache_status"] == "miss_failed"

    rerun_calls = []

    def finish_download(_session, symbol, **_kwargs):
        rerun_calls.append(symbol)
        assert symbol == "SZ159915"
        return _corporate_action_document("0.8800"), 1

    monkeypatch.setattr(collector, "_download_eastmoney_corporate_action_html", finish_download)
    completed = collector.collect_corporate_actions(tmp_path, workers=1, request_delay_seconds=0, attempts=1)
    assert rerun_calls == ["SZ159915"]
    assert set(completed["symbol"]) == set(symbols)
    published = pd.read_csv(canonical_path)
    assert set(published["symbol"]) == set(symbols)
    rerun_report = pd.read_csv(tmp_path / "corporate_action_report.csv").set_index("symbol")
    assert rerun_report.loc["SH510300", "source"] == "cache"
    assert rerun_report.loc["SZ159915", "source"] == "network"
    assert rerun_report["published"].all()


def test_corporate_action_http_514_and_429_use_exponential_retry(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("retryable status should be handled explicitly")

    class FakeSession:
        def __init__(self):
            self.responses = [FakeResponse(514), FakeResponse(429), FakeResponse(200, "<html>ok</html>")]

        def get(self, *_args, **_kwargs):
            return self.responses.pop(0)

    sleeps = []
    monkeypatch.setattr(collector.time, "sleep", sleeps.append)
    monkeypatch.setattr(collector.random, "uniform", lambda _start, _end: 0.0)
    html_text, attempts = collector._download_eastmoney_corporate_action_html(
        FakeSession(), "SH510300", attempts=3, request_delay_seconds=0.25
    )
    assert html_text == "<html>ok</html>"
    assert attempts == 3
    assert sleeps == [0.25, 0.5]


def test_actions_cli_exposes_resumable_collection_options():
    args = collector.build_parser().parse_args(
        [
            "actions",
            "--symbols",
            "SH510300",
            "--symbols-file",
            "symbols.txt",
            "--cache-dir",
            "cache",
            "--refresh",
            "--attempts",
            "7",
        ]
    )
    assert args.symbols == ["SH510300"]
    assert args.symbols_file == Path("symbols.txt")
    assert args.cache_dir == Path("cache")
    assert args.refresh
    assert args.attempts == 7
    assert args.request_delay_seconds >= 0.25


def test_frozen_universe_download_does_not_refresh_pool_definition(monkeypatch, tmp_path):
    _write_universe(tmp_path, ["SH510300", "SZ159915"])
    captured = []

    def download_one(symbol, **kwargs):
        captured.append((symbol, kwargs["end_date"]))
        return collector.DownloadResult(symbol=symbol, rows=1, end_date="2026-08-12")

    monkeypatch.setattr(collector, "download_one", download_one)
    monkeypatch.setattr(
        collector,
        "build_t1_universe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not refresh universe")),
    )
    universe, results = collector.download_dataset(
        tmp_path,
        date(2026, 8, 1),
        date(2026, 8, 12),
        workers=1,
        frozen_universe=True,
    )
    assert universe["symbol"].tolist() == ["SH510300", "SZ159915"]
    assert [result.symbol for result in results] == ["SH510300", "SZ159915"]
    assert [item[0] for item in captured] == ["SH510300", "SZ159915"]


def test_frozen_universe_rejects_scope_changing_options(tmp_path):
    _write_universe(tmp_path, ["SH510300"])
    with pytest.raises(ValueError, match="cannot be combined"):
        collector.download_dataset(
            tmp_path,
            date(2026, 8, 1),
            date(2026, 8, 12),
            symbols=["SH510300"],
            frozen_universe=True,
        )


def test_normalize_and_validate_ignore_retained_pool_external_cache(monkeypatch, tmp_path):
    _write_universe(tmp_path, ["SH510300"])
    raw_dir = tmp_path / "raw"
    normalized_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    normalized_dir.mkdir()
    for directory in (raw_dir, normalized_dir):
        (directory / "sh510300.csv").write_text("placeholder", encoding="utf-8")
        (directory / "sh999999.csv").write_text("retained cache", encoding="utf-8")

    seen = []

    def fake_read_csv(path, *args, **kwargs):
        path = Path(path)
        if path.name == "universe.csv":
            return pd.DataFrame({"symbol": ["SH510300"], "code": ["510300"]})
        seen.append(path.name)
        if path.parent.name == "raw":
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-08-12"]),
                    "symbol": ["SH510300"],
                    **{column: [1.0] for column in collector.RAW_COLUMNS if column not in {"date", "symbol", "data_source", "amount_quality"}},
                    "data_source": ["sina"],
                    "amount_quality": ["reported"],
                }
            )
        values = {column: [1.0] for column in collector.QLIB_FIELDS}
        values.update({"change": [0.0], "amount_estimated": [0.0], "paused": [0.0]})
        return pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-12"]), "symbol": ["SH510300"], **values}
        )

    monkeypatch.setattr(collector.pd, "read_csv", fake_read_csv)
    report = collector.validate_dataset(tmp_path, expected_end=date(2026, 8, 12))
    assert seen == ["sh510300.csv", "sh510300.csv"]
    assert report["training_ready"] is True
    assert report["pool_external_raw_file_count"] == 1
    assert report["pool_external_normalized_file_count"] == 1


def test_latest_complete_date_rolls_back_over_weekends():
    timezone = ZoneInfo("Asia/Shanghai")
    monday_morning = datetime(2026, 8, 10, 9, 30, tzinfo=timezone)
    assert collector.latest_complete_date(monday_morning) == date(2026, 8, 7)
    monday_after_close = datetime(2026, 8, 10, 16, 5, tzinfo=timezone)
    assert collector.latest_complete_date(monday_after_close) == date(2026, 8, 10)
    sunday_evening = datetime(2026, 8, 9, 20, 0, tzinfo=timezone)
    assert collector.latest_complete_date(sunday_evening) == date(2026, 8, 7)


def test_eastmoney_corporate_actions_accept_red_dividend_and_conversion_phrasings():
    document = """
    <table><thead><tr><th>年份</th><th>权益登记日</th><th>除息日</th><th>每10份分红</th><th>分红发放日</th></tr></thead>
    <tbody><tr><td>2026年</td><td>2026-01-16</td><td>2026-01-19</td><td>每10份基金份额派发红利1.5000元</td><td>2026-01-27</td></tr></tbody></table>
    <table><thead><tr><th>年份</th><th>拆分折算日</th><th>拆分类型</th><th>拆分折算比例</th></tr></thead>
    <tbody><tr><td>2026年</td><td>2026-01-19</td><td>份额折算</td><td>每10份基金份额折算为25份</td></tr></tbody></table>
    """
    frame = collector._parse_eastmoney_corporate_actions(document, "SH510300", "https://example.test")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["cash_dividend_per_old_share"] == 0.15
    assert row["share_ratio"] == 2.5


def test_eastmoney_corporate_actions_accept_per_unit_cash_phrasing():
    document = """
    <table><thead><tr><th>年份</th><th>权益登记日</th><th>除息日</th><th>每10份分红</th><th>分红发放日</th></tr></thead>
    <tbody><tr><td>2026年</td><td>2026-01-16</td><td>2026-01-19</td><td>每份派现金0.0500元</td><td>2026-01-27</td></tr></tbody></table>
    """
    frame = collector._parse_eastmoney_corporate_actions(document, "SH510300", "https://example.test")
    assert len(frame) == 1
    assert frame.iloc[0]["cash_dividend_per_old_share"] == 0.05
