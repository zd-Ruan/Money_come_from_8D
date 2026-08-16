# -*- coding: utf-8 -*-
"""Build a point-in-time ETF universe (CORRECT filter: exchange column in SH/SZ)
and quantify survivorship bias. Writes NEW pit_* files only.
"""
import os
import pandas as pd

ROOT = r"E:\Code\Money_come_from_8D"
X = os.path.join(ROOT, "xianyu_data")
DATA = os.path.join(ROOT, "data", "cn_etf")

# ---- inputs
basic = pd.read_csv(os.path.join(X, "ETF数据更新至2026.8.7", "etf_basic_data.csv"),
                    encoding="utf-8", dtype={"ts_code": str})
# EXCHANGE-LISTED only (exchange column in SH/SZ), NOT the ts_code suffix
basic = basic[basic["exchange"].isin(["SH", "SZ"])].copy()
basic["code"] = basic["ts_code"].str.split(".").str[0]
basic["symbol"] = basic["exchange"] + basic["code"]
basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d", errors="coerce")
basic = basic.drop_duplicates("symbol", keep="first")

univ = pd.read_csv(os.path.join(DATA, "universe.csv"), dtype={"code": str})
univ["symbol"] = univ["symbol"].astype(str).str.upper()
univ_syms = set(univ["symbol"])

inst = pd.read_csv(os.path.join(DATA, "qlib_data", "instruments", "t1_etf.txt"),
                   sep="\t", header=None, names=["symbol", "list_date", "last_date"])
inst["symbol"] = inst["symbol"].astype(str).str.upper()
inst["list_date"] = pd.to_datetime(inst["list_date"], errors="coerce")

# last trade date per symbol from xianyu daily (proxy for delist / data end)
daily = pd.read_csv(os.path.join(X, "ETF数据更新至2026.8.7", "etf_daily.csv"),
                    encoding="utf-8", usecols=["ts_code", "trade_date"], dtype={"ts_code": str})
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
daily["symbol"] = daily["ts_code"].str.split(".").str[1] + daily["ts_code"].str.split(".").str[0]
last_trade = daily.groupby("symbol")["trade_date"].max().rename("last_trade_date")

# ---- build master = union(current universe, exchange-listed basic)
all_syms = set(univ_syms) | set(basic["symbol"])
rows = []
for sym in sorted(all_syms):
    row = {"symbol": sym, "code": sym[2:], "exchange": sym[:2],
           "in_current_universe": sym in univ_syms}
    b = basic[basic["symbol"] == sym]
    if not b.empty:
        b = b.iloc[0]
        row["list_date"] = b["list_date"] if pd.notna(b["list_date"]) else pd.NaT
        row["list_status"] = b["list_status"]
        row["fund_type"] = b["etf_type"]
        row["mgt_fee"] = b["mgt_fee"]
        row["name"] = b["cname"]
    else:
        row["list_date"] = pd.NaT
        row["list_status"] = None
        row["fund_type"] = None
        row["mgt_fee"] = None
        row["name"] = None
    i = inst[inst["symbol"] == sym]
    if pd.isna(row["list_date"]) and not i.empty:
        row["list_date"] = i.iloc[0]["list_date"]
    row["last_trade_date"] = last_trade.get(sym, pd.NaT)
    rows.append(row)

master = pd.DataFrame(rows)
master["list_date"] = pd.to_datetime(master["list_date"])
master["last_trade_date"] = pd.to_datetime(master["last_trade_date"])

# delist date = last trade date for D-status
master["delist_date"] = pd.NaT
d_mask = master["list_status"] == "D"
master.loc[d_mask, "delist_date"] = master.loc[d_mask, "last_trade_date"]

# ---- survivorship quantification (year-end active cross-section)
CAL_END = pd.Timestamp("2026-08-11")
report_rows = []
for y in range(2015, 2027):
    d = pd.Timestamp(f"{y}-12-31")
    listed_by = master["list_date"].notna() & (master["list_date"] <= d)
    delisted_by = master["delist_date"].notna() & (master["delist_date"] < d)
    active_pit = int((listed_by & ~delisted_by).sum())
    cur = int(master["in_current_universe"].sum())
    report_rows.append({"year": y, "active_pit": active_pit,
                        "current_snapshot": cur,
                        "overstated_by": cur - active_pit,
                        "overstatement_pct": round(100.0 * (cur - active_pit) / max(active_pit, 1), 1)})
report = pd.DataFrame(report_rows)

# ---- save
master.to_csv(os.path.join(DATA, "pit_universe.csv"), index=False, encoding="utf-8")
report.to_csv(os.path.join(DATA, "pit_survivorship_report.csv"), index=False, encoding="utf-8")
os.makedirs(os.path.join(DATA, "pit_instruments"), exist_ok=True)
pit_inst = master[master["list_date"].notna()].copy()
pit_inst["end"] = pit_inst["delist_date"].fillna(pd.Timestamp("2026-08-11"))
with open(os.path.join(DATA, "pit_instruments", "t1_etf.txt"), "w", encoding="utf-8") as fh:
    for _, r in pit_inst.sort_values("symbol").iterrows():
        fh.write(f"{r['symbol']}\t{r['list_date'].date()}\t{r['end'].date()}\n")

print("master rows:", len(master))
print("  in current universe:", int(master["in_current_universe"].sum()))
print("  additions (NOT in current universe):", int((~master["in_current_universe"]).sum()))
print("  of additions, delisted (D):", int((master["list_status"] == "D").sum()))
print("  of additions, live (L):", int((master["list_status"] == "L").sum()))
print("  with list_date:", int(master["list_date"].notna().sum()), "of", len(master))
print("  with delist_date:", int(master["delist_date"].notna().sum()))
print("\ndelist year distribution (D funds):")
print(pd.to_datetime(master.loc[d_mask, "delist_date"]).dt.year.value_counts().sort_index().to_string())
print("\nsurvivorship report:")
print(report.to_string(index=False))
print("\nwrote data/cn_etf/pit_universe.csv, pit_survivorship_report.csv, pit_instruments/t1_etf.txt")
