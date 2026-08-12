from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .io import write_json_atomic


ARTIFACT_CHECKSUM_FILENAME = "artifact_checksums.json"
_SOURCE_HASH_DOMAIN = b"quant-pipeline-source-tree-v1\0"
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}\Z")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def validate_run_id(run_id: str) -> str:
    """Validate a run identifier as one portable, non-special directory name."""
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a single directory name")
    if not _SAFE_RUN_ID.fullmatch(run_id) or run_id.endswith("."):
        raise ValueError(
            "run_id must be 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    if run_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("run_id is a reserved Windows directory name")
    return run_id


def resolve_run_directory(runs_root: Path, run_id: str) -> Path:
    """Return a direct child of ``runs_root`` after lexical and resolved-path checks."""
    safe_run_id = validate_run_id(run_id)
    root = Path(runs_root).resolve()
    candidate = (root / safe_run_id).resolve()
    if candidate.parent != root:
        raise ValueError("run_id resolves outside the runs directory")
    return candidate


def _source_files(source_root: Path) -> list[Path]:
    files = []
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root)
        if "__pycache__" in relative.parts or path.suffix.lower() == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"source tree contains a symbolic link: {relative.as_posix()}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source_root).as_posix())


def source_tree_sha256(source_root: Path | None = None) -> str:
    """Hash relative paths and bytes in the quant_pipeline source tree deterministically."""
    root = (Path(source_root) if source_root is not None else Path(__file__).parent).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"source tree does not exist: {root}")

    digest = hashlib.sha256()
    digest.update(_SOURCE_HASH_DOMAIN)
    for path in _source_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _hash_and_size(path: Path, chunk_size: int = 1024 * 1024) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _checksum_target(run_root: Path, checksum_path: Path | None) -> tuple[Path, str]:
    raw_path = Path(checksum_path) if checksum_path is not None else Path(ARTIFACT_CHECKSUM_FILENAME)
    candidate = raw_path if raw_path.is_absolute() else run_root / raw_path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise ValueError("checksum path resolves outside the run directory") from exc
    if relative in {".", "manifest.json"}:
        raise ValueError("checksum path must be a file distinct from manifest.json")
    return resolved, relative


def _scan_artifacts(run_root: Path, excluded: set[str]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in run_root.rglob("*"):
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"run contains a symbolic link: {relative}")
        if not path.is_file():
            continue
        size, checksum = _hash_and_size(path)
        artifacts[relative] = {"path": relative, "size": size, "sha256": checksum}
    return dict(sorted(artifacts.items()))


def build_artifact_checksum_manifest(
    run_dir: Path, checksum_path: Path | None = None
) -> dict[str, Any]:
    """Build, but do not write, the deterministic artifact checksum manifest."""
    run_root = Path(run_dir).resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run_root}")
    _, checksum_relative = _checksum_target(run_root, checksum_path)
    artifacts = _scan_artifacts(run_root, {"manifest.json", checksum_relative})
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "artifacts": list(artifacts.values()),
    }


def generate_artifact_checksums(run_dir: Path, checksum_path: Path | None = None) -> Path:
    """Write a checksum manifest atomically and return its resolved path."""
    run_root = Path(run_dir).resolve()
    target, _ = _checksum_target(run_root, checksum_path)
    payload = build_artifact_checksum_manifest(run_root, target)
    write_json_atomic(target, payload)
    return target


def _validate_artifact_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact paths must be non-empty POSIX relative paths")
    parts = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe artifact path in checksum manifest: {value!r}")
    return value


def _load_expected_artifacts(checksum_file: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(checksum_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checksum manifest: {checksum_file}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported artifact checksum manifest schema")
    if payload.get("algorithm") != "sha256" or not isinstance(payload.get("artifacts"), list):
        raise ValueError("invalid artifact checksum manifest metadata")

    expected: dict[str, dict[str, Any]] = {}
    for record in payload["artifacts"]:
        if not isinstance(record, dict):
            raise ValueError("artifact checksum records must be objects")
        path = _validate_artifact_path(record.get("path"))
        size = record.get("size")
        checksum = record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid artifact size for {path!r}")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"invalid SHA-256 for {path!r}")
        if path in expected:
            raise ValueError(f"duplicate artifact path in checksum manifest: {path!r}")
        expected[path] = {"path": path, "size": size, "sha256": checksum}
    return expected


def verify_artifact_checksums(
    run_dir: Path,
    checksum_path: Path | None = None,
    *,
    check_unexpected: bool = True,
) -> dict[str, Any]:
    """Verify expected run artifacts and classify missing, modified, and extra files."""
    run_root = Path(run_dir).resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run_root}")
    checksum_file, checksum_relative = _checksum_target(run_root, checksum_path)
    expected = _load_expected_artifacts(checksum_file)
    actual = _scan_artifacts(run_root, {"manifest.json", checksum_relative})

    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    modified = sorted(
        path
        for path in expected.keys() & actual.keys()
        if expected[path]["size"] != actual[path]["size"]
        or expected[path]["sha256"] != actual[path]["sha256"]
    )
    modified_details = {
        path: {"expected": expected[path], "actual": actual[path]} for path in modified
    }
    valid = not missing and not modified and (not unexpected or not check_unexpected)
    return {
        "valid": valid,
        "check_unexpected": check_unexpected,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "modified": modified,
        "unexpected": unexpected,
        "modified_details": modified_details,
    }

