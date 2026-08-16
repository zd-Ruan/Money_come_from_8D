# -*- coding: utf-8 -*-
"""Correct survivorship-bias quantification: how many DELISTED ETFs were tradable
at each historical date but are absent from the current (survivor-only) pipeline.
"""
import os
import pandas as pd

ROOT = r"E:\Code\Money_come_from_8D"
DATA = os.path.join(ROOT, "data", "cn_etf")
master = pd.read_csv(os.path.join(DATA, "pit_universe.csv"))
master["list_date"] = pd.to_datetime(master["list_date"])
master["delist_date"] = pd.to_datetime(master["delist_date"])

survivors = master[master["in_current_universe"]].copy()
delisted = master[master["list_status"] == "D"].copy()

rows = []
for y in range(2015, 2027):
    d = pd.Timestamp(f"{y}-12-31")
    surv_n = int(((survivors["list_date"] <= d)).sum())  # survivors listed by year-end (none delisted)
    del_n = int(((delisted["list_date"] <= d) & (delisted["delist_date"] >= d)).sum())
    true_pit = surv_n + del_n
    rows.append({
        "year": y,
        "survivors_tradable": surv_n,
        "delisted_tradable_but_missing": del_n,
        "true_pit_universe": true_pit,
        "missing_share_pct": round(100.0 * del_n / max(true_pit, 1), 1),
    })
rep = pd.DataFrame(rows)
print(rep.to_string(index=False))
rep.to_csv(os.path.join(DATA, "pit_survivorship_report.csv"), index=False, encoding="utf-8")

print("\n--- delisted ETF list_date sanity (how far back they extend) ---")
print(delisted["list_date"].dt.year.value_counts().sort_index().to_string())
print("\ntotal delisted:", len(delisted))
