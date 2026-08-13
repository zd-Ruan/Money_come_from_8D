import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.integrity import (
    combine_runtime_code_sha256,
    generate_artifact_checksums,
    generate_integrity_seal,
    resolve_run_directory,
    runtime_code_identity,
    source_tree_sha256,
    validate_run_id,
    verify_artifact_checksums,
    verify_integrity_seal,
)
from quant_pipeline.io import sha256_file


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

    def test_runtime_identity_binds_pipeline_and_imported_qlib_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "pipeline"
            qlib = root / "qlib"
            pipeline.mkdir()
            qlib.mkdir()
            (pipeline / "runner.py").write_bytes(b"pipeline\n")
            (qlib / "__init__.py").write_bytes(b"qlib\n")

            identity = runtime_code_identity(pipeline, qlib)
            self.assertEqual(
                identity["runtime_code_sha256"],
                combine_runtime_code_sha256(
                    identity["pipeline_source_sha256"],
                    identity["qlib_package_sha256"],
                ),
            )
            (qlib / "data.py").write_bytes(b"changed imported behavior\n")
            changed = runtime_code_identity(pipeline, qlib)
            self.assertEqual(
                changed["pipeline_source_sha256"], identity["pipeline_source_sha256"]
            )
            self.assertNotEqual(changed["qlib_package_sha256"], identity["qlib_package_sha256"])
            self.assertNotEqual(changed["runtime_code_sha256"], identity["runtime_code_sha256"])

    def test_runtime_digest_rejects_noncanonical_components(self):
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            combine_runtime_code_sha256("A" * 64, "b" * 64)


class ArtifactChecksumTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary_directory.name) / "run"
        (self.run_dir / "nested").mkdir(parents=True)
        (self.run_dir / "manifest.json").write_text('{"status":"running"}', encoding="utf-8")
        (self.run_dir / "alpha.txt").write_bytes(b"alpha")
        (self.run_dir / "nested" / "beta.bin").write_bytes(b"beta")
        self.checksum_file = generate_artifact_checksums(self.run_dir)
        checksum_payload = json.loads(self.checksum_file.read_text(encoding="utf-8"))
        self.manifest = {
            "status": "completed",
            "artifacts": {
                "artifact_checksums": "artifact_checksums.json",
                "integrity_seal": "integrity_seal.json",
            },
            "integrity": {
                "checksum_manifest": "artifact_checksums.json",
                "checksum_sha256": sha256_file(self.checksum_file),
                "artifact_count": len(checksum_payload["artifacts"]),
                "seal_manifest": "integrity_seal.json",
                "verified": True,
            },
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        self.seal_file = generate_integrity_seal(self.run_dir)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_checksum_is_deterministic_but_outer_seal_protects_manifest(self):
        first_bytes = self.checksum_file.read_bytes()
        payload = json.loads(first_bytes)
        self.assertEqual(
            [record["path"] for record in payload["artifacts"]],
            ["alpha.txt", "nested/beta.bin"],
        )

        self.manifest["classification"] = "candidate"
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        generate_artifact_checksums(self.run_dir)
        self.assertEqual(self.checksum_file.read_bytes(), first_bytes)
        result = verify_artifact_checksums(self.run_dir)
        self.assertFalse(result["valid"])
        self.assertEqual(result["modified"], ["manifest.json"])

    def test_completed_run_verifies_both_checksum_layers(self):
        result = verify_artifact_checksums(self.run_dir)
        self.assertTrue(result["valid"])
        self.assertTrue(result["seal"]["valid"])
        self.assertEqual(result["seal"]["expected_count"], 2)
        self.assertEqual(
            [record["path"] for record in json.loads(self.seal_file.read_text())["protected_files"]],
            ["artifact_checksums.json", "manifest.json"],
        )

    def test_missing_outer_seal_fails_closed_by_default(self):
        self.seal_file.unlink()
        result = verify_artifact_checksums(self.run_dir)
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing"], ["integrity_seal.json"])
        self.assertTrue(verify_artifact_checksums(self.run_dir, require_seal=False)["valid"])

    def test_seal_cannot_be_generated_for_noncompleted_manifest(self):
        self.manifest["status"] = "reporting"
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "status must be completed"):
            generate_integrity_seal(self.run_dir)

    def test_manifest_declarations_are_part_of_the_seal_contract(self):
        self.manifest["integrity"]["artifact_count"] += 1
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        result = verify_integrity_seal(self.run_dir)
        self.assertFalse(result["valid"])
        self.assertIn("manifest.json", result["modified"])
        self.assertTrue(any("artifact count" in error for error in result["contract_errors"]))

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
