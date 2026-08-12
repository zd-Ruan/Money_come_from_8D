import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.integrity import (
    generate_artifact_checksums,
    resolve_run_directory,
    source_tree_sha256,
    validate_run_id,
    verify_artifact_checksums,
)


class RunPathTests(unittest.TestCase):
    def test_accepts_safe_run_id_and_resolves_direct_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            expected = root.resolve() / "20260812T165254-etf_alpha158_rolling"
            actual = resolve_run_directory(root, "20260812T165254-etf_alpha158_rolling")
            self.assertEqual(actual, expected)

    def test_rejects_unsafe_run_ids(self):
        unsafe = [
            "",
            ".",
            "..",
            "../escape",
            "nested/run",
            r"nested\run",
            "/absolute",
            r"C:\absolute",
            "C:drive-relative",
            ".hidden",
            "name with spaces",
            "CON",
            "aux.txt",
            "trailing.",
        ]
        for run_id in unsafe:
            with self.subTest(run_id=run_id), self.assertRaises((TypeError, ValueError)):
                validate_run_id(run_id)


class SourceTreeHashTests(unittest.TestCase):
    def test_hash_is_deterministic_and_excludes_python_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()

            (first / "a.py").write_bytes(b"A\n")
            (first / "nested").mkdir()
            (first / "nested" / "b.py").write_bytes(b"B\n")
            (second / "nested").mkdir()
            (second / "nested" / "b.py").write_bytes(b"B\n")
            (second / "a.py").write_bytes(b"A\n")

            expected = source_tree_sha256(first)
            self.assertEqual(source_tree_sha256(second), expected)

            cache = first / "__pycache__"
            cache.mkdir()
            (cache / "a.cpython-311.pyc").write_bytes(b"ignored")
            (first / "loose.pyc").write_bytes(b"also ignored")
            self.assertEqual(source_tree_sha256(first), expected)

            (first / "a.py").write_bytes(b"changed\n")
            self.assertNotEqual(source_tree_sha256(first), expected)


class ArtifactChecksumTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary_directory.name) / "run"
        (self.run_dir / "nested").mkdir(parents=True)
        (self.run_dir / "manifest.json").write_text('{"status":"running"}', encoding="utf-8")
        (self.run_dir / "alpha.txt").write_bytes(b"alpha")
        (self.run_dir / "nested" / "beta.bin").write_bytes(b"beta")
        self.checksum_file = generate_artifact_checksums(self.run_dir)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_manifest_is_deterministic_and_excludes_mutable_files(self):
        first_bytes = self.checksum_file.read_bytes()
        payload = json.loads(first_bytes)
        self.assertEqual(
            [record["path"] for record in payload["artifacts"]],
            ["alpha.txt", "nested/beta.bin"],
        )

        (self.run_dir / "manifest.json").write_text('{"status":"completed"}', encoding="utf-8")
        generate_artifact_checksums(self.run_dir)
        self.assertEqual(self.checksum_file.read_bytes(), first_bytes)
        self.assertTrue(verify_artifact_checksums(self.run_dir)["valid"])

    def test_reports_same_size_tampering_as_modified(self):
        (self.run_dir / "alpha.txt").write_bytes(b"ALPHA")
        result = verify_artifact_checksums(self.run_dir)
        self.assertFalse(result["valid"])
        self.assertEqual(result["modified"], ["alpha.txt"])
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["unexpected"], [])

    def test_reports_missing_artifact(self):
        (self.run_dir / "nested" / "beta.bin").unlink()
        result = verify_artifact_checksums(self.run_dir)
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing"], ["nested/beta.bin"])
        self.assertEqual(result["modified"], [])

    def test_reports_unexpected_artifact_and_can_ignore_it(self):
        (self.run_dir / "extra.txt").write_bytes(b"extra")
        strict = verify_artifact_checksums(self.run_dir)
        self.assertFalse(strict["valid"])
        self.assertEqual(strict["unexpected"], ["extra.txt"])

        permissive = verify_artifact_checksums(self.run_dir, check_unexpected=False)
        self.assertTrue(permissive["valid"])
        self.assertEqual(permissive["unexpected"], ["extra.txt"])


if __name__ == "__main__":
    unittest.main()
