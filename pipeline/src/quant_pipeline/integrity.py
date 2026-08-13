from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .io import write_json_atomic


ARTIFACT_CHECKSUM_FILENAME = "artifact_checksums.json"
INTEGRITY_SEAL_FILENAME = "integrity_seal.json"
_SOURCE_HASH_DOMAIN = b"quant-pipeline-source-tree-v1\0"
_RUNTIME_CODE_HASH_DOMAIN = b"quant-pipeline-runtime-code-v1\0"
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
    """Hash relative paths and bytes in one imported package tree deterministically."""
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


def combine_runtime_code_sha256(
    pipeline_source_sha256: str, qlib_package_sha256: str
) -> str:
    """Bind the pipeline and the actually imported Qlib package into one identity."""

    digests = (pipeline_source_sha256, qlib_package_sha256)
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in digests
    ):
        raise ValueError("runtime code component digests must be lowercase SHA-256 values")
    digest = hashlib.sha256()
    digest.update(_RUNTIME_CODE_HASH_DOMAIN)
    for label, value in zip((b"pipeline", b"qlib"), digests):
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def runtime_code_identity(
    pipeline_source_root: Path, qlib_package_root: Path
) -> dict[str, str]:
    """Fingerprint all repository code imported by a model run."""

    pipeline_digest = source_tree_sha256(Path(pipeline_source_root))
    qlib_digest = source_tree_sha256(Path(qlib_package_root))
    return {
        "pipeline_source_sha256": pipeline_digest,
        "qlib_package_sha256": qlib_digest,
        "runtime_code_sha256": combine_runtime_code_sha256(
            pipeline_digest, qlib_digest
        ),
    }


def _hash_and_size(path: Path, chunk_size: int = 1024 * 1024) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _protected_output_target(
    run_root: Path,
    raw_path: Path,
    *,
    description: str,
    forbidden: set[str],
) -> tuple[Path, str]:
    candidate = raw_path if raw_path.is_absolute() else run_root / raw_path
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{description} path resolves outside the run directory") from exc
    resolved = lexical.resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"{description} path resolves outside the run directory") from exc
    if resolved != lexical:
        raise ValueError(f"{description} path must not traverse symbolic links or junctions")
    if relative in {".", *forbidden}:
        rendered = ", ".join(sorted(forbidden))
        raise ValueError(f"{description} path must be a file distinct from {rendered}")
    return lexical, relative


def _checksum_target(run_root: Path, checksum_path: Path | None) -> tuple[Path, str]:
    raw_path = Path(checksum_path) if checksum_path is not None else Path(ARTIFACT_CHECKSUM_FILENAME)
    return _protected_output_target(
        run_root,
        raw_path,
        description="checksum",
        forbidden={"manifest.json", INTEGRITY_SEAL_FILENAME},
    )


def _seal_target(run_root: Path, seal_path: Path | None) -> tuple[Path, str]:
    raw_path = Path(seal_path) if seal_path is not None else Path(INTEGRITY_SEAL_FILENAME)
    return _protected_output_target(
        run_root,
        raw_path,
        description="integrity seal",
        forbidden={"manifest.json", ARTIFACT_CHECKSUM_FILENAME},
    )


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
    artifacts = _scan_artifacts(
        run_root,
        {"manifest.json", checksum_relative, INTEGRITY_SEAL_FILENAME},
    )
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


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _validate_manifest_seal_contract(
    manifest_path: Path,
    checksum_path: Path,
    checksum_relative: str,
    seal_relative: str,
) -> list[str]:
    try:
        manifest = _load_json_object(manifest_path, "run manifest")
    except (FileNotFoundError, ValueError) as exc:
        return [str(exc)]

    errors: list[str] = []
    if manifest.get("status") != "completed":
        errors.append("run manifest status must be completed before sealing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("run manifest artifacts metadata is missing")
    else:
        if artifacts.get("artifact_checksums") != checksum_relative:
            errors.append("run manifest artifact checksum path does not match the seal")
        if artifacts.get("integrity_seal") != seal_relative:
            errors.append("run manifest integrity seal path does not match the seal")

    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        errors.append("run manifest integrity metadata is missing")
        return errors
    if integrity.get("checksum_manifest") != checksum_relative:
        errors.append("run manifest checksum declaration does not match the seal")
    if integrity.get("seal_manifest") != seal_relative:
        errors.append("run manifest seal declaration does not match the seal")
    if integrity.get("verified") is not True:
        errors.append("run manifest must declare successful integrity verification")

    declared_checksum = integrity.get("checksum_sha256")
    if not isinstance(declared_checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_checksum):
        errors.append("run manifest checksum SHA-256 declaration is invalid")
    elif checksum_path.is_file() and not checksum_path.is_symlink():
        _, actual_checksum = _hash_and_size(checksum_path)
        if declared_checksum != actual_checksum:
            errors.append("run manifest checksum SHA-256 does not match the checksum file")

    try:
        artifact_count = len(_load_expected_artifacts(checksum_path))
    except (FileNotFoundError, ValueError) as exc:
        errors.append(str(exc))
    else:
        declared_count = integrity.get("artifact_count")
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count != artifact_count
        ):
            errors.append("run manifest artifact count does not match the checksum file")
    return errors


def build_integrity_seal(
    run_dir: Path,
    checksum_path: Path | None = None,
    seal_path: Path | None = None,
) -> dict[str, Any]:
    """Build a detached seal over the final manifest and its artifact checksum list."""
    run_root = Path(run_dir).resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run_root}")
    checksum_file, checksum_relative = _checksum_target(run_root, checksum_path)
    seal_file, seal_relative = _seal_target(run_root, seal_path)
    if checksum_file == seal_file:
        raise ValueError("artifact checksum manifest and integrity seal must be distinct files")
    manifest_path = run_root / "manifest.json"
    for description, path in (
        ("run manifest", manifest_path),
        ("artifact checksum manifest", checksum_file),
    ):
        if path.is_symlink():
            raise ValueError(f"{description} must not be a symbolic link")
        if not path.is_file():
            raise FileNotFoundError(f"{description} is missing: {path}")

    contract_errors = _validate_manifest_seal_contract(
        manifest_path,
        checksum_file,
        checksum_relative,
        seal_relative,
    )
    if contract_errors:
        raise ValueError("cannot seal run: " + "; ".join(contract_errors))

    protected_files = []
    for relative, path in sorted(
        (("manifest.json", manifest_path), (checksum_relative, checksum_file))
    ):
        size, checksum = _hash_and_size(path)
        protected_files.append({"path": relative, "size": size, "sha256": checksum})
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "protected_files": protected_files,
    }


def generate_integrity_seal(
    run_dir: Path,
    checksum_path: Path | None = None,
    seal_path: Path | None = None,
) -> Path:
    """Atomically write the outer seal after the completed manifest is final."""
    run_root = Path(run_dir).resolve()
    target, _ = _seal_target(run_root, seal_path)
    payload = build_integrity_seal(run_root, checksum_path, target)
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
    payload = _load_json_object(checksum_file, "checksum manifest")
    if payload.get("schema_version") != 1:
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


def _load_seal_records(seal_file: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json_object(seal_file, "integrity seal")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported integrity seal schema")
    if payload.get("algorithm") != "sha256" or not isinstance(
        payload.get("protected_files"), list
    ):
        raise ValueError("invalid integrity seal metadata")

    expected: dict[str, dict[str, Any]] = {}
    for record in payload["protected_files"]:
        if not isinstance(record, dict):
            raise ValueError("integrity seal records must be objects")
        path = _validate_artifact_path(record.get("path"))
        size = record.get("size")
        checksum = record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid protected file size for {path!r}")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"invalid protected file SHA-256 for {path!r}")
        if path in expected:
            raise ValueError(f"duplicate protected path in integrity seal: {path!r}")
        expected[path] = {"path": path, "size": size, "sha256": checksum}
    return expected


def verify_integrity_seal(
    run_dir: Path,
    checksum_path: Path | None = None,
    seal_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the detached seal and the manifest's declarations without mutating the run."""
    run_root = Path(run_dir).resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run_root}")
    checksum_file, checksum_relative = _checksum_target(run_root, checksum_path)
    seal_file, seal_relative = _seal_target(run_root, seal_path)
    if checksum_file == seal_file:
        raise ValueError("artifact checksum manifest and integrity seal must be distinct files")
    if seal_file.is_symlink():
        raise ValueError("integrity seal must not be a symbolic link")
    expected = _load_seal_records(seal_file)
    required = {"manifest.json", checksum_relative}
    if set(expected) != required:
        raise ValueError(
            "integrity seal must protect exactly manifest.json and the artifact checksum manifest"
        )

    actual: dict[str, dict[str, Any]] = {}
    for relative in sorted(required):
        path = run_root / relative
        if path.is_symlink():
            raise ValueError(f"protected file must not be a symbolic link: {relative}")
        if path.is_file():
            size, checksum = _hash_and_size(path)
            actual[relative] = {"path": relative, "size": size, "sha256": checksum}
    missing = sorted(required - actual.keys())
    modified = sorted(
        path
        for path in required & actual.keys()
        if expected[path]["size"] != actual[path]["size"]
        or expected[path]["sha256"] != actual[path]["sha256"]
    )
    contract_errors = _validate_manifest_seal_contract(
        run_root / "manifest.json",
        checksum_file,
        checksum_relative,
        seal_relative,
    )
    return {
        "valid": not missing and not modified and not contract_errors,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "modified": modified,
        "contract_errors": contract_errors,
        "modified_details": {
            path: {"expected": expected[path], "actual": actual[path]} for path in modified
        },
    }


def verify_artifact_checksums(
    run_dir: Path,
    checksum_path: Path | None = None,
    *,
    check_unexpected: bool = True,
    require_seal: bool = True,
    seal_path: Path | None = None,
) -> dict[str, Any]:
    """Verify run artifacts and, by default, the detached final-manifest seal."""
    run_root = Path(run_dir).resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run_root}")
    checksum_file, checksum_relative = _checksum_target(run_root, checksum_path)
    seal_file, seal_relative = _seal_target(run_root, seal_path)
    if checksum_file == seal_file:
        raise ValueError("artifact checksum manifest and integrity seal must be distinct files")
    if checksum_file.is_symlink():
        raise ValueError("artifact checksum manifest must not be a symbolic link")
    expected = _load_expected_artifacts(checksum_file)
    actual = _scan_artifacts(run_root, {"manifest.json", checksum_relative, seal_relative})

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
    seal: dict[str, Any] | None = None
    if require_seal:
        if not seal_file.is_file():
            seal = {
                "valid": False,
                "expected_count": 2,
                "actual_count": 0,
                "missing": [seal_relative],
                "modified": [],
                "contract_errors": ["integrity seal is missing"],
                "modified_details": {},
            }
        else:
            seal = verify_integrity_seal(run_root, checksum_file, seal_file)

    protected_missing = [] if seal is None else seal["missing"]
    protected_modified = [] if seal is None else seal["modified"]
    all_missing = sorted(set(missing) | set(protected_missing))
    all_modified = sorted(set(modified) | set(protected_modified))
    valid = (
        not missing
        and not modified
        and (not unexpected or not check_unexpected)
        and (seal is None or seal["valid"])
    )
    return {
        "valid": valid,
        "check_unexpected": check_unexpected,
        "require_seal": require_seal,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": all_missing,
        "modified": all_modified,
        "unexpected": unexpected,
        "modified_details": {
            **modified_details,
            **({} if seal is None else seal["modified_details"]),
        },
        "seal": seal,
    }
