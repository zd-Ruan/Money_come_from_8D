import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "data_collector" / "cn_etf"))
import collector

DATA = Path(__file__).resolve().parent / "data" / "cn_etf"
CACHE = DATA / "corporate_action_cache"
CACHE.mkdir(parents=True, exist_ok=True)
SINA_HFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"

universe = pd.read_csv(DATA / "trainable_universe_2015.csv")
symbols = universe["symbol"].astype(str).str.upper().tolist()

jump_map = {}
for symbol in symbols:
    path = DATA / "normalized" / f"{symbol.lower()}.csv"
    frame = pd.read_csv(path, usecols=["date", "factor"], parse_dates=["date"])
    factor = frame["factor"].to_numpy(dtype=float)
    jump_positions = np.where(np.abs(np.diff(factor)) > 1e-10)[0]
    if len(jump_positions):
        jump_map[symbol] = [
            (frame["date"].iloc[int(i) + 1], float(factor[int(i)]), float(factor[int(i) + 1]))
            for i in jump_positions
        ]

print("symbols with jumps:", len(jump_map))


def fetch_one(symbol: str) -> dict:
    session = collector.make_session(insecure=True)
    try:
        url = SINA_HFQ_URL.format(symbol=symbol.lower())
        response = session.get(
            url,
            headers={"Referer": f"https://finance.sina.com.cn/realstock/company/{symbol.lower()}/nc.shtml"},
            timeout=30,
        )
        response.raise_for_status()
        text = response.text
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (CACHE / f"{symbol}.html").write_text(text, encoding="utf-8")
        feed = collector._parse_sina_factor_payload(text)
        changed = feed[(feed[["f", "s", "u"]].diff().abs() > 1e-9).any(axis=1)].reset_index(drop=True)
        rows = []
        jumps = jump_map.get(symbol, [])
        if not jumps:
            if len(changed) == 0:
                return {"symbol": symbol, "error": "", "rows": []}
            return {
                "symbol": symbol,
                "error": f"feed has {len(changed)} events but data has no factor jumps",
                "rows": [],
            }
        if len(changed) < len(jumps):
            return {
                "symbol": symbol,
                "error": f"count mismatch: feed_events={len(changed)} jumps={len(jumps)}",
                "rows": [],
            }
        if len(changed) > len(jumps):
            changed = changed.iloc[: len(jumps)]
        previous_u = 0.0
        previous_fs = 1.0
        for index, row in changed.iterrows():
            ex_date, factor_before, factor_after = jumps[index]
            current_u = float(row["u"])
            current_fs = float(row["f"]) * float(row["s"])
            dividend = (current_u - previous_u) / previous_fs
            ratio = current_fs / previous_fs if previous_fs > 0 else 1.0
            if dividend < 0:
                return {"symbol": symbol, "error": f"negative dividend delta at {row['date']}", "rows": []}
            rows.append(
                {
                    "symbol": symbol,
                    "record_date": ex_date,
                    "ex_date": ex_date,
                    "cash_payment_date": ex_date,
                    "cash_dividend_per_old_share": float(dividend),
                    "share_ratio": float(ratio),
                    "fractional_share_treatment": (
                        "unknown_not_provided_by_sina_hfq"
                        if abs(ratio - 1.0) > 1e-12
                        else "not_applicable_no_share_change"
                    ),
                    "event_id": f"{symbol}~{pd.Timestamp(ex_date).date().isoformat()}",
                    "source_url": url,
                    "source_sha256": digest,
                }
            )
            previous_u = current_u
            previous_fs = current_fs
        return {"symbol": symbol, "error": "", "rows": rows}
    except Exception as exc:
        return {"symbol": symbol, "error": f"{type(exc).__name__}: {exc}", "rows": []}
    finally:
        session.close()


results = {}
all_rows = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fetch_one, symbol): symbol for symbol in sorted(symbols)}
    done = 0
    for future in as_completed(futures):
        result = future.result()
        results[result["symbol"]] = result
        done += 1
        if result["error"]:
            print(f"[{done}/{len(symbols)}] {result['symbol']}: {result['error']}")
        else:
            all_rows.extend(result["rows"])
            if done % 200 == 0:
                print(f"[{done}/{len(symbols)}] ok, rows so far {len(all_rows)}")

errors = {symbol: result for symbol, result in results.items() if result["error"]}
print("derived rows:", len(all_rows), "errors:", len(errors))

actions = pd.DataFrame(
    all_rows,
    columns=[
        "symbol",
        "record_date",
        "ex_date",
        "cash_payment_date",
        "cash_dividend_per_old_share",
        "share_ratio",
        "fractional_share_treatment",
        "event_id",
        "source_url",
        "source_sha256",
    ],
).sort_values(["symbol", "ex_date"])

report_rows = []
for symbol in sorted(symbols):
    cache_path = CACHE / f"{symbol}.html"
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path.is_file() else ""
    error = errors.get(symbol, {}).get("error", "")
    report_rows.append(
        {
            "symbol": symbol,
            "error": error,
            "full_universe_scope": not bool(error),
            "published": not bool(error),
            "cache_sha256": cache_sha256,
        }
    )
report = pd.DataFrame(report_rows)

actions.to_csv(DATA / "corporate_actions.csv", index=False)
report.to_csv(DATA / "corporate_action_report.csv", index=False)
print("wrote corporate_actions.csv", len(actions), "and corporate_action_report.csv", len(report))
