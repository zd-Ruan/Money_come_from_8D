from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .io import write_json_atomic


GITHUB_FILE_SIZE_LIMIT_BYTES = 100_000_000
DEFAULT_LARGEST_FILE_COUNT = 20


class GitHubReleaseSizeError(RuntimeError):
    """Raised when a GitHub publication candidate violates the file-size policy."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        blocked = report.get("blocked_files", [])
        rendered = ", ".join(
            f"{record['path']} ({record['size_bytes']} bytes)" for record in blocked[:10]
        )
        if len(blocked) > 10:
            rendered += f", ... ({len(blocked)} total)"
        super().__init__(
            f"GitHub release size gate failed; files must be smaller than "
            f"{report['file_size_limit_bytes']} bytes: {rendered}"
        )


def git_publish_candidates(repo_root: Path) -> list[str]:
    """Return tracked and non-ignored untracked files as repository-relative paths."""
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"repository root does not exist: {root}")
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        if Path(top_level).resolve() != root:
            raise ValueError(f"release root must be the Git repository root: {top_level}")
        output = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("git executable is unavailable; release size gate cannot run") from exc
    except subprocess.CalledProcessError as exc:
        stderr = os.fsdecode(exc.stderr or b"").strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"could not enumerate GitHub publication candidates{detail}") from exc

    candidates = [os.fsdecode(value) for value in output.split(b"\0") if value]
    if len(candidates) != len(set(candidates)):
        raise RuntimeError("git returned duplicate publication candidate paths")
    return sorted(candidates)


def _validated_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("file size limit must be a positive integer")
    return value


def _validated_largest_file_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("largest file count must be a non-negative integer")
    return value


def _resolve_candidate(root: Path, relative_value: str) -> tuple[Path, str]:
    if not isinstance(relative_value, str) or not relative_value or "\0" in relative_value:
        raise ValueError("release candidate paths must be non-empty strings")
    raw_path = Path(relative_value)
    if raw_path.is_absolute():
        raise ValueError(f"release candidate path must be relative: {relative_value!r}")
    path = Path(os.path.abspath(root / raw_path))
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"release candidate resolves outside the repository: {relative_value!r}") from exc
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"release candidate resolves outside the repository: {relative_value!r}") from exc
    if relative == ".":
        raise ValueError("release candidate must identify a file")
    return path, relative


def _traverses_link(root: Path, path: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def build_github_release_size_report(
    repo_root: Path,
    *,
    candidates: Iterable[str] | None = None,
    file_size_limit_bytes: int = GITHUB_FILE_SIZE_LIMIT_BYTES,
    largest_file_count: int = DEFAULT_LARGEST_FILE_COUNT,
) -> dict[str, Any]:
    """Inspect publishable files and return an exact, path-portable size report."""
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"repository root does not exist: {root}")
    limit = _validated_limit(file_size_limit_bytes)
    largest_count = _validated_largest_file_count(largest_file_count)
    relative_candidates = (
        git_publish_candidates(root) if candidates is None else list(candidates)
    )
    if len(relative_candidates) != len(set(relative_candidates)):
        raise ValueError("release candidate paths must not contain duplicates")

    files: list[dict[str, Any]] = []
    missing: list[str] = []
    unsupported: list[str] = []
    normalized: set[str] = set()
    for relative_value in sorted(relative_candidates):
        path, relative = _resolve_candidate(root, relative_value)
        if relative in normalized:
            raise ValueError(f"release candidate paths normalize to a duplicate: {relative!r}")
        normalized.add(relative)
        if _traverses_link(root, path):
            unsupported.append(relative)
            continue
        if not path.exists():
            missing.append(relative)
            continue
        if not path.is_file():
            unsupported.append(relative)
            continue
        size = path.stat().st_size
        files.append({"path": relative, "size_bytes": size})

    files.sort(key=lambda record: (-record["size_bytes"], record["path"]))
    blocked = [record for record in files if record["size_bytes"] >= limit]
    total_size = sum(record["size_bytes"] for record in files)
    valid = not blocked and not missing and not unsupported
    return {
        "schema_version": 1,
        "policy": "every publishable file must be smaller than file_size_limit_bytes",
        "scope": "git tracked plus untracked non-ignored working-tree files",
        "file_size_limit_bytes": limit,
        "valid": valid,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_mib": total_size / (1024 * 1024),
        "largest_files": files[:largest_count],
        "blocked_files": blocked,
        "missing_files": missing,
        "unsupported_paths": unsupported,
    }


def enforce_github_release_size(
    repo_root: Path,
    *,
    output_path: Path | None = None,
    candidates: Iterable[str] | None = None,
    file_size_limit_bytes: int = GITHUB_FILE_SIZE_LIMIT_BYTES,
    largest_file_count: int = DEFAULT_LARGEST_FILE_COUNT,
) -> dict[str, Any]:
    """Write an optional report and fail closed when publication is not valid."""
    report = build_github_release_size_report(
        repo_root,
        candidates=candidates,
        file_size_limit_bytes=file_size_limit_bytes,
        largest_file_count=largest_file_count,
    )
    if output_path is not None:
        write_json_atomic(Path(output_path), report)
    if not report["valid"]:
        raise GitHubReleaseSizeError(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed GitHub file-size gate with a total publication-size report"
    )
    parser.add_argument("repo_root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--largest-files", type=int, default=DEFAULT_LARGEST_FILE_COUNT)
    args = parser.parse_args()
    try:
        report = enforce_github_release_size(
            args.repo_root,
            output_path=args.output,
            largest_file_count=args.largest_files,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, GitHubReleaseSizeError):
            print(json.dumps(exc.report, ensure_ascii=False, indent=2))
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
