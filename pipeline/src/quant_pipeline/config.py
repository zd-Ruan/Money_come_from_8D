from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {
    "project",
    "features",
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
    features = config["features"]
    model = config["model"]
    strategy = config["strategy"]
    gates = config["gates"]
    if features.get("mode") not in {"alpha158", "alpha360", "alpha158_plus_original", "alpha191"}:
        raise ValueError(
            "features.mode must be alpha158, alpha360, alpha158_plus_original, or alpha191"
        )
    if features["mode"] in {"alpha158", "alpha360", "alpha191"} and features.get("families"):
        raise ValueError("standard Alpha feature modes must not select original factor families")
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
    if float(execution["account"]) <= 0:
        raise ValueError("execution.account must be positive")
    if int(execution.get("trade_unit", 0)) != 100:
        raise ValueError("execution.trade_unit must be 100 for this ETF pipeline")
    if int(execution.get("stamp_tax_bps", 0)) != 0:
        raise ValueError("stock ETF stamp tax must remain zero unless the instrument scope changes")
    if execution.get("price_limit_mode") != "ohlc_proven_tier_conservative":
        raise ValueError("execution.price_limit_mode must be ohlc_proven_tier_conservative")
    if float(execution.get("standard_limit_ratio", 0.0)) != 0.10:
        raise ValueError("execution.standard_limit_ratio must be 0.10")
    if float(execution.get("wide_limit_ratio", 0.0)) != 0.20:
        raise ValueError("execution.wide_limit_ratio must be 0.20")
    if float(execution.get("price_tick", 0.0)) != 0.001:
        raise ValueError("execution.price_tick must be 0.001 for the current ETF scope")
    if not 1 <= int(strategy["topk"]) <= 20:
        raise ValueError("strategy.topk must be between 1 and 20")
    if not 0 <= int(strategy["n_drop"]) <= int(strategy["topk"]):
        raise ValueError("strategy.n_drop must be between zero and topk")
    if int(strategy["hold_thresh"]) < 1:
        raise ValueError("strategy.hold_thresh must be positive")
    if model.get("device_type") not in {"cpu", "gpu"}:
        raise ValueError("model.device_type must be cpu or gpu")
    if int(gates["min_complete_folds"]) < 1:
        raise ValueError("gates.min_complete_folds must be positive")
    if int(gates.get("research_fold_days", 0)) < 5:
        raise ValueError("gates.research_fold_days must be at least 5")
    for key in (
        "min_research_fold_win_ratio",
        "max_single_etf_abs_contribution_share",
        "max_single_fold_abs_incremental_pnl_share",
    ):
        value = float(gates.get(key, -1.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"gates.{key} must be in (0, 1]")


def json_ready_config(config: dict[str, Any]) -> dict[str, Any]:
    ready = json.loads(json.dumps(config, ensure_ascii=False, default=str))
    metadata = ready.pop("_meta", {})
    workspace = Path(metadata.get("workspace_root", ".")).resolve()
    for key, value in ready.get("paths", {}).items():
        resolved = Path(value).resolve()
        try:
            ready["paths"][key] = resolved.relative_to(workspace).as_posix()
        except ValueError:
            ready["paths"][key] = resolved.as_posix()
    return ready
