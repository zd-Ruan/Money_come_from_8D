from __future__ import annotations

import ctypes.util
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import lightgbm

from .io import sha256_file


ENVIRONMENT_LOCK_SCHEMA_VERSION = 1
DEFAULT_ENVIRONMENT_LOCK = Path(__file__).resolve().parents[2] / "environment.lock.json"
_VERSION = re.compile(r"[^\s]+\Z")


class EnvironmentLockError(RuntimeError):
    pass


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvironmentLockError(f"environment lock is missing: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentLockError(f"environment lock is not valid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != ENVIRONMENT_LOCK_SCHEMA_VERSION:
        raise EnvironmentLockError("unsupported environment lock schema")
    if set(value) != {"schema_version", "python", "packages"}:
        raise EnvironmentLockError("environment lock contains unexpected top-level fields")
    python_lock = value.get("python")
    packages = value.get("packages")
    if not isinstance(python_lock, dict) or set(python_lock) != {"implementation", "version"}:
        raise EnvironmentLockError("environment lock Python contract is invalid")
    if not isinstance(packages, dict) or not packages:
        raise EnvironmentLockError("environment lock package contract is empty or invalid")
    for name, version in packages.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(version, str)
            or not _VERSION.fullmatch(version)
        ):
            raise EnvironmentLockError("environment lock package names and versions must be exact")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise EnvironmentLockError(f"runtime library is missing: {resolved.name}")
    return {
        "filename": resolved.name,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _lightgbm_build_identity() -> dict[str, Any]:
    library_name = getattr(getattr(lightgbm.basic, "_LIB", None), "_name", None)
    if not isinstance(library_name, str) or not library_name:
        raise EnvironmentLockError("LightGBM native library identity is unavailable")
    return _file_identity(Path(library_name))


def _opencl_loader_identity() -> dict[str, Any]:
    candidates: list[Path] = []
    loader = ctypes.util.find_library("OpenCL")
    if platform.system() == "Windows":
        windows_root = os.environ.get("WINDIR")
        if windows_root:
            candidates.append(Path(windows_root) / "System32" / "OpenCL.dll")
    if loader:
        raw = Path(loader)
        candidates.append(raw)
        if not raw.is_absolute():
            for root in (
                Path("/usr/lib/x86_64-linux-gnu"),
                Path("/lib/x86_64-linux-gnu"),
                Path("/usr/lib64"),
                Path("/usr/lib"),
                Path("/lib64"),
                Path("/lib"),
            ):
                candidates.append(root / raw)
    for candidate in candidates:
        if candidate.is_file():
            return {"available": True, **_file_identity(candidate)}
    return {
        "available": bool(loader),
        "filename": None if loader is None else Path(loader).name,
        "size": None,
        "sha256": None,
    }


def validate_locked_environment(lock_path: Path = DEFAULT_ENVIRONMENT_LOCK) -> dict[str, Any]:
    """Fail closed unless the active numerical runtime exactly matches the lock."""

    resolved_lock = Path(lock_path).resolve()
    lock = _read_lock(resolved_lock)
    expected_python = lock["python"]
    actual_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    mismatches: list[str] = []
    if actual_python != expected_python:
        mismatches.append(
            "Python "
            f"{actual_python['implementation']} {actual_python['version']} != "
            f"{expected_python['implementation']} {expected_python['version']}"
        )

    actual_packages: dict[str, str] = {}
    for name, expected_version in sorted(lock["packages"].items(), key=lambda item: item[0].lower()):
        try:
            actual_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name} is missing (expected {expected_version})")
            continue
        actual_packages[name] = actual_version
        if actual_version != expected_version:
            mismatches.append(f"{name} {actual_version} != {expected_version}")
    if mismatches:
        raise EnvironmentLockError("environment lock mismatch: " + "; ".join(mismatches))

    return {
        "python": actual_python,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": actual_packages,
        "lock": {
            "schema_version": ENVIRONMENT_LOCK_SCHEMA_VERSION,
            "filename": resolved_lock.name,
            "sha256": sha256_file(resolved_lock),
        },
        "lightgbm_build": _lightgbm_build_identity(),
        "opencl_loader": _opencl_loader_identity(),
    }
