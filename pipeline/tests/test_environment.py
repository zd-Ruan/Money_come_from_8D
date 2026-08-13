import importlib.metadata
import json
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.environment import (
    DEFAULT_ENVIRONMENT_LOCK,
    EnvironmentLockError,
    validate_locked_environment,
)


class EnvironmentLockTests(unittest.TestCase):
    def _runtime_lock(self, directory: str) -> Path:
        repository_lock = json.loads(DEFAULT_ENVIRONMENT_LOCK.read_text(encoding="utf-8"))
        repository_lock["python"] = {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        }
        repository_lock["packages"] = {
            name: importlib.metadata.version(name)
            for name in repository_lock["packages"]
        }
        path = Path(directory) / "environment.lock.json"
        path.write_text(json.dumps(repository_lock), encoding="utf-8")
        return path

    def test_exact_runtime_lock_records_native_build(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_locked_environment(self._runtime_lock(directory))
        self.assertEqual(report["lock"]["filename"], "environment.lock.json")
        self.assertEqual(len(report["lock"]["sha256"]), 64)
        self.assertEqual(len(report["lightgbm_build"]["sha256"]), 64)
        self.assertIn("numpy", report["packages"])
        self.assertIn("pandas", report["packages"])
        self.assertIn("pyarrow", report["packages"])

    def test_repository_lock_contains_a10_verified_build_versions(self):
        lock = json.loads(DEFAULT_ENVIRONMENT_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(lock["packages"]["lightgbm"], "4.6.0")
        self.assertEqual(lock["packages"]["pyqlib"], "0.1.dev5")

    def test_version_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._runtime_lock(directory)
            lock = json.loads(path.read_text(encoding="utf-8"))
            lock["packages"]["numpy"] = "0.0.invalid"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(EnvironmentLockError, "numpy .* != 0.0.invalid"):
                validate_locked_environment(path)

    def test_python_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._runtime_lock(directory)
            with patch("quant_pipeline.environment.platform.python_version", return_value="0.0.0"):
                with self.assertRaisesRegex(EnvironmentLockError, "Python CPython 0.0.0"):
                    validate_locked_environment(path)

    def test_lock_rejects_extra_unvalidated_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = json.loads(DEFAULT_ENVIRONMENT_LOCK.read_text(encoding="utf-8"))
            lock["ignored"] = True
            path = Path(directory) / "environment.lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(EnvironmentLockError, "unexpected top-level"):
                validate_locked_environment(path)


if __name__ == "__main__":
    unittest.main()
