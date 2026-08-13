import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.release import (
    GITHUB_FILE_SIZE_LIMIT_BYTES,
    GitHubReleaseSizeError,
    build_github_release_size_report,
    enforce_github_release_size,
)


class GitHubReleaseSizeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _sparse_file(self, name: str, size: int) -> Path:
        path = self.root / name
        with path.open("wb") as handle:
            handle.truncate(size)
        return path

    def test_exact_github_limit_is_rejected_and_total_is_reported(self):
        self._sparse_file("allowed.bin", GITHUB_FILE_SIZE_LIMIT_BYTES - 1)
        self._sparse_file("blocked.bin", GITHUB_FILE_SIZE_LIMIT_BYTES)
        (self.root / "small.txt").write_bytes(b"abc")

        report = build_github_release_size_report(
            self.root,
            candidates=["small.txt", "blocked.bin", "allowed.bin"],
        )

        self.assertFalse(report["valid"])
        self.assertEqual(report["file_count"], 3)
        self.assertEqual(
            report["total_size_bytes"],
            2 * GITHUB_FILE_SIZE_LIMIT_BYTES + 2,
        )
        self.assertEqual(report["blocked_files"], [
            {"path": "blocked.bin", "size_bytes": GITHUB_FILE_SIZE_LIMIT_BYTES}
        ])
        self.assertEqual(report["largest_files"][0]["path"], "blocked.bin")

    def test_file_one_byte_under_limit_passes(self):
        self._sparse_file("weight.bin", GITHUB_FILE_SIZE_LIMIT_BYTES - 1)
        report = enforce_github_release_size(self.root, candidates=["weight.bin"])
        self.assertTrue(report["valid"])

    def test_gate_raises_with_report_and_persists_failure_report(self):
        self._sparse_file("weight.bin", GITHUB_FILE_SIZE_LIMIT_BYTES)
        output = self.root / "reports" / "github_release_size.json"
        with self.assertRaises(GitHubReleaseSizeError) as raised:
            enforce_github_release_size(
                self.root,
                candidates=["weight.bin"],
                output_path=output,
            )
        self.assertFalse(raised.exception.report["valid"])
        self.assertTrue(output.is_file())

    def test_missing_and_nonfile_candidates_fail_closed(self):
        (self.root / "directory").mkdir()
        report = build_github_release_size_report(
            self.root,
            candidates=["missing.bin", "directory"],
        )
        self.assertFalse(report["valid"])
        self.assertEqual(report["missing_files"], ["missing.bin"])
        self.assertEqual(report["unsupported_paths"], ["directory"])

    def test_candidate_cannot_escape_repository(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            build_github_release_size_report(self.root, candidates=["../escape.bin"])


if __name__ == "__main__":
    unittest.main()
