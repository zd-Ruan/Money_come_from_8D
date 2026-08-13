from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import now_shanghai, read_json, write_json_atomic


def update_registry(path: Path, run_record: dict[str, Any]) -> None:
    registry = read_json(path, {"updated_at": None, "runs": []})
    run_dir = run_record.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ValueError("run_record.run_dir must be a non-empty string")
    resolved_dir = Path(run_dir).expanduser()
    if not resolved_dir.is_absolute():
        resolved_dir = (path.parent / resolved_dir).resolve()
    else:
        resolved_dir = resolved_dir.resolve()
    if not resolved_dir.is_dir():
        raise ValueError(f"run_record.run_dir does not exist: {resolved_dir}")
    runs = [run for run in registry.get("runs", []) if run.get("run_id") != run_record.get("run_id")]
    runs.append(run_record)
    runs.sort(key=lambda run: run.get("created_at", ""), reverse=True)
    registry["updated_at"] = now_shanghai().isoformat()
    registry["runs"] = runs
    write_json_atomic(path, registry)


def list_runs(path: Path) -> list[dict[str, Any]]:
    return read_json(path, {"runs": []}).get("runs", [])

