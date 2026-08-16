# -*- coding: utf-8 -*-
"""Per-fold survivorship gap: how many delisted ETFs were tradable inside each
train window but are absent from the current (survivor-only) pipeline.
"""
import os
import pandas as pd

ROOT = r"E:\Code\Money_come_from_8D"
DATA = os.path.join(ROOT, "data", "cn_etf")
master = pd.read_csv(os.path.join(DATA, "pit_universe.csv"))
master["list_date"] = pd.to_datetime(master["list_date"])
master["delist_date"] = pd.to_datetime(master["delist_date"])
delisted = master[master["list_status"] == "D"].copy()

folds = [
    ("fold1", "2015-01-05", "2024-06-21"),
    ("fold2", "2015-01-05", "2024-09-20"),
    ("fold3", "2015-01-05", "2024-12-25"),
    ("fold4", "2015-01-05", "2025-04-02"),
    ("fold5", "2015-01-05", "2025-07-07"),
    ("fold6", "2015-01-05", "2025-10-10"),
    ("fold7", "2015-01-05", "2026-01-09"),
]
rows = []
for name, t0, t1 in folds:
    t0 = pd.Timestamp(t0); t1 = pd.Timestamp(t1)
    # delisted ETF tradable for at least part of the train window
    mask = (delisted["list_date"] <= t1) & (delisted["delist_date"] >= t0)
    miss = delisted[mask]
    rows.append({
        "fold": name,
        "train_end": t1.date(),
        "delisted_tradable_in_window": int(mask.sum()),
        "their_total_list_date_range": f"{miss['list_date'].min().date()}..{miss['list_date'].max().date()}" if len(miss) else "-",
    })
rep = pd.DataFrame(rows)
print(rep.to_string(index=False))
rep.to_csv(os.path.join(DATA, "pit_fold_survivorship_gap.csv"), index=False, encoding="utf-8")

print("\nbacktest window (2025-01-01..2026-08-11) delisted ETFs still tradable at some point:")
bt = delisted[(delisted["delist_date"] >= pd.Timestamp("2025-01-01"))]
print("  count:", len(bt))
if len(bt):
    print(bt[["symbol", "list_date", "delist_date", "name"]].sort_values("delist_date").to_string(index=False))
