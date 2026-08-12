from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {
    "project",
    "paths",
    "data",
    "rolling",
    "model",
    "strategy",
    "execution",
    "gates",
    "report",
}


def workspace_root_from_config(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "configs" and resolved.parent.parent.name == "pipeline":
        return resolved.parent.parent.parent
    raise ValueError("config must live under <workspace>/pipeline/configs")


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    missing = sorted(REQUIRED_SECTIONS - set(config))
    if missing:
        raise ValueError(f"missing configuration sections: {missing}")

    workspace = workspace_root_from_config(config_path)
    config["_meta"] = {
        "config_path": str(config_path),
        "workspace_root": str(workspace),
    }
    for key, value in config["paths"].items():
        path = Path(value)
        config["paths"][key] = str(path if path.is_absolute() else (workspace / path).resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    rolling = config["rolling"]
    execution = config["execution"]
    if int(data["label_horizon_bars"]) < 1:
        raise ValueError("label_horizon_bars must be positive")
    if int(rolling["purge_bars"]) < int(data["label_horizon_bars"]):
        raise ValueError("purge_bars must be at least label_horizon_bars")
    if int(rolling["validation_days"]) < 20 or int(rolling["test_days"]) < 5:
        raise ValueError("rolling windows are too short")
    participation = float(execution["max_daily_volume_participation"])
    if not 0 < participation <= 0.25:
        raise ValueError("max_daily_volume_participation must be in (0, 0.25]")
    stress = sorted(set(int(value) for value in execution["stress_slippage_bps_per_side"]))
    if not stress or stress[0] < 0:
        raise ValueError("stress slippage values must be non-negative")
    execution["stress_slippage_bps_per_side"] = stress


def json_ready_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config, ensure_ascii=False, default=str))

