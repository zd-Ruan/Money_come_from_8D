# -*- coding: utf-8 -*-
"""Exact training-row impact of survivorship bias: how many (date, symbol) rows
the 127 delisted ETFs would add to each fold's training set."""
import os
import pandas as pd

ROOT = r"E:\Code\Money_come_from_8D"
DATA = os.path.join(ROOT, "data", "cn_etf")
master = pd.read_csv(os.path.join(DATA, "pit_universe.csv"))
master["list_date"] = pd.to_datetime(master["list_date"])
master["delist_date"] = pd.to_datetime(master["delist_date"])
delisted = master[master["list_status"] == "D"].copy()

cal = pd.read_csv(os.path.join(DATA, "qlib_data", "calendars", "day.txt"), header=None,
                  names=["date"], parse_dates=["date"])["date"]
cal = pd.DatetimeIndex(cal)

folds = [
    ("fold1", "2015-01-05", "2024-06-21"),
    ("fold2", "2015-01-05", "2024-09-20"),
    ("fold3", "2015-01-05", "2024-12-25"),
    ("fold4", "2015-01-05", "2025-04-02"),
    ("fold5", "2015-01-05", "2025-07-07"),
    ("fold6", "2015-01-05", "2025-10-10"),
    ("fold7", "2015-01-05", "2026-01-09"),
]
# total training rows per fold from the run's metrics (train rows incl. features)
train_rows = {  # from candidate_trend_crowding metrics.json "rows.train"
    "fold1": 319844, "fold2": 341274, "fold3": 371716,
    "fold4": 403147, "fold5": 435444, "fold6": 476091, "fold7": 519295,
}
rows = []
for name, t0s, t1s in folds:
    t0, t1 = pd.Timestamp(t0s), pd.Timestamp(t1s)
    added = 0
    for _, r in delisted.iterrows():
        lo = max(r["list_date"], t0)
        hi = min(r["delist_date"], t1)
        if lo <= hi:
            added += int(((cal >= lo) & (cal <= hi)).sum())
    rows.append({"fold": name, "train_end": t1.date(),
                 "delisted_added_rows": added,
                 "existing_train_rows": train_rows[name],
                 "added_pct": round(100.0 * added / train_rows[name], 2)})
rep = pd.DataFrame(rows)
print(rep.to_string(index=False))
rep.to_csv(os.path.join(DATA, "pit_training_row_impact.csv"), index=False, encoding="utf-8")
