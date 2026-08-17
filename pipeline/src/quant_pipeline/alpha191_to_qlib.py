"""Dump Alpha191 factors into a Qlib provider directory.

Reads the normalized ETF data over the full Alpha360 reference period
(2005..2026-08-11), computes the Alpha191 factors, and writes them into a new
Qlib provider ``features/<instrument>/alphaNNN.day.bin`` next to the copied
base kline bins (which keep the label/liquidity expressions working).  The
resulting provider is consumed by the pipeline runner with
``features.mode: alpha191`` (see ``alpha191_handler.py``).

Run (repository root, ``quant`` environment):

.. code-block:: powershell

    $env:PYTHONPATH = (Resolve-Path .\\qlib\\pipeline\\src).Path
    C:\\Exception\\quant\\python.exe -m quant_pipeline.alpha191_to_qlib ^
        --config .\\qlib\\pipeline\\configs\\alpha191_build.yaml ^
        --output-dir qlib\\data\\cn_etf\\qlib_data_alpha191_2005_20260811
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quant_pipeline.alpha191 import alpha191_registry
from quant_pipeline.alpha191_research import Alpha191BuildConfig, _resolve_repo_path

_BASE_FIELDS = [
    "amount",
    "amount_estimated",
    "change",
    "close",
    "factor",
    "high",
    "low",
    "open",
    "paused",
    "volume",
    "vwap",
]


def _read_instruments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=["symbol", "start_date", "end_date"])
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    return frame


def _write_bin(path: Path, values: np.ndarray, start_index: int) -> None:
    values = np.asarray(values, dtype=np.float32)
    header = np.asarray([start_index], dtype=np.float32)
    with path.open("wb") as handle:
        header.tofile(handle)
        values.tofile(handle)


def dump_alpha191_provider(config: Alpha191BuildConfig, output_dir: Path) -> dict:
    started = time.time()
    from quant_pipeline.alpha191_research import load_wide_data

    src_provider = _resolve_repo_path(Path(config.data_dir) / "qlib_data_alpha360_2005_20260811")
    if not (src_provider / "calendars" / "day.txt").exists():
        raise FileNotFoundError(f"reference provider not found: {src_provider}")

    data, calendar, symbols = load_wide_data(config)
    print(f"[load] {len(symbols)} symbols, {len(calendar)} sessions, "
          f"{calendar.min().date()}..{calendar.max().date()}")

    provider_calendar = pd.to_datetime(
        pd.read_csv(src_provider / "calendars" / "day.txt", header=None).iloc[:, 0]
    ).dt.normalize()
    instruments = _read_instruments(src_provider / "instruments" / "t1_etf.txt")
    calendar_positions = {ts: idx for idx, ts in enumerate(provider_calendar)}
    print(f"[provider] calendar {len(provider_calendar)} days; {len(instruments)} instruments")

    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("calendars", "instruments", "metadata"):
        src = src_provider / sub
        dst = output_dir / sub
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)

    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Per-instrument directories; copy the base kline bins so label and
    # liquidity expressions keep working.
    for symbol in instruments["symbol"].str.upper():
        symbol_dir = features_dir / symbol.lower()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        if not (symbol_dir / "close.day.bin").exists():
            for field in _BASE_FIELDS:
                src_bin = src_provider / "features" / symbol.lower() / f"{field}.day.bin"
                if src_bin.exists():
                    shutil.copy2(src_bin, symbol_dir / src_bin.name)

    errors: list[str] = []
    written = 0
    for factor in alpha191_registry():
        try:
            frame = factor.fn(data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{factor.name}: {type(exc).__name__}: {exc}")
            frame = None
        if frame is not None:
            frame = frame.reindex(provider_calendar)
        for row in instruments.itertuples(index=False):
            symbol = str(row.symbol).upper()
            start_index = calendar_positions.get(row.start_date.normalize())
            end_index = calendar_positions.get(row.end_date.normalize())
            if start_index is None or end_index is None or end_index < start_index:
                continue
            window = end_index - start_index + 1
            if frame is not None:
                column = frame.get(symbol)
                values = (
                    column.to_numpy(dtype=np.float32)
                    if column is not None
                    else np.full(window, np.nan, dtype=np.float32)
                )
            else:
                values = np.full(window, np.nan, dtype=np.float32)
            _write_bin(features_dir / symbol.lower() / f"{factor.name}.day.bin", values, start_index)
            written += 1
        del frame
    print(f"[dump] {written} alpha bins across {len(instruments)} instruments; {len(errors)} errors")
    for error in errors:
        print(f"  [error] {error}")

    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "source_provider": str(src_provider),
        "provider": str(output_dir),
        "calendar_days": len(provider_calendar),
        "instruments": len(instruments),
        "factors": len(alpha191_registry()),
        "bins_written": written,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    summary_path = output_dir / "dump_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] elapsed {summary['elapsed_seconds']}s -> {output_dir}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dump Alpha191 factors into a Qlib provider")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from quant_pipeline.alpha191_research import _parse_config

    config = _parse_config(args.config)
    dump_alpha191_provider(config, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
