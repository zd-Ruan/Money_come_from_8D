import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\zhendong ruan\Downloads\Code\My_Quant\qlib")
SNAPSHOT = ROOT / "pipeline" / "snapshots" / "20260811-2a82246e51e9"
DATA = ROOT / "data" / "cn_etf"

t1 = pd.read_csv(SNAPSHOT / "t1_etf.txt", encoding="utf-8-sig", header=None)
if t1.shape[1] == 1:
    parts = t1.iloc[:, 0].str.split("\t", expand=True)
    parts.columns = ["symbol", "list_date", "last_date"]
    t1 = parts
symbols = t1["symbol"].str.strip().str.upper()
symbols = symbols[symbols.str.fullmatch(r"SH\d{6}|SZ\d{6}")]
symbols = symbols.drop_duplicates().sort_values()
print("symbols:", len(symbols))

trainable = pd.DataFrame(
    {
        "symbol": symbols,
        "code": symbols.str[2:],
    }
)
trainable.to_csv(DATA / "trainable_universe_2015.csv", index=False, encoding="utf-8")
print("wrote trainable_universe_2015.csv", len(trainable))

instruments_path = DATA / "qlib_data" / "instruments" / "t1_etf.txt"
instruments_path.parent.mkdir(parents=True, exist_ok=True)
with open(instruments_path, "w", encoding="utf-8") as handle:
    for _, row in t1.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        list_date = str(row["list_date"]).strip()
        last_date = str(row["last_date"]).strip()
        handle.write(f"{symbol}\t{list_date}\t{last_date}\n")
print("wrote qlib instruments t1_etf.txt", len(symbols))

existing = pd.read_csv(DATA / "universe.csv", dtype={"code": str})
existing = existing[existing["symbol"].astype(str).str.upper().isin(set(symbols))]
if "history_end_date" in existing.columns:
    existing["history_end_date"] = "2026-08-11"
missing = set(symbols) - set(existing["symbol"].astype(str).str.upper())
if missing:
    fill = pd.DataFrame(
        {
            "symbol": sorted(missing),
            "code": [s[2:] for s in sorted(missing)],
        }
    )
    existing = pd.concat([existing, fill], ignore_index=True)
existing = existing.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
existing.to_csv(DATA / "universe.csv", index=False)
print("wrote universe.csv", len(existing))
