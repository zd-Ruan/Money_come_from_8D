from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import now_shanghai, read_json, write_json_atomic


def update_registry(path: Path, run_record: dict[str, Any]) -> None:
    registry = read_json(path, {"updated_at": None, "runs": []})
    runs = [run for run in registry.get("runs", []) if run.get("run_id") != run_record.get("run_id")]
    runs.append(run_record)
    runs.sort(key=lambda run: run.get("created_at", ""), reverse=True)
    registry["updated_at"] = now_shanghai().isoformat()
    registry["runs"] = runs
    write_json_atomic(path, registry)


def list_runs(path: Path) -> list[dict[str, Any]]:
    return read_json(path, {"runs": []}).get("runs", [])

