import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.audit import (
    REQUIRED_QLIB_FIELDS,
    _seal_snapshot_manifest,
    _scoped_inventory,
    audit_and_snapshot,
    ensure_future_calendar_boundary,
    evaluate_corporate_action_collection,
    evaluate_provider_features,
    evaluate_upstream_validation,
)
from quant_pipeline.io import write_json_atomic


class AuditTests(unittest.TestCase):
    def _write_provider_field(
        self, path: Path, *, start_index: int = 0, values: int = 3
    ) -> None:
        path.write_bytes(
            struct.pack("<f", float(start_index))
            + struct.pack(f"<{values}f", *([1.0] * values))
        )

    def _audit_fixture(self, root: Path) -> dict:
        data = root / "data"
        raw = data / "raw"
        normalized = data / "normalized"
        cache = data / "corporate_action_cache"
        provider = root / "provider"
        features = provider / "features" / "sh510300"
        instruments = provider / "instruments"
        calendars = provider / "calendars"
        for path in (raw, normalized, cache, features, instruments, calendars):
            path.mkdir(parents=True)

        (data / "universe.csv").write_text(
            "code,symbol\n510300,SH510300\n", encoding="utf-8"
        )
        (instruments / "t1_etf.txt").write_text(
            "SH510300\t2026-08-07\t2026-08-11\n", encoding="utf-8"
        )
        (calendars / "day.txt").write_text(
            "2026-08-07\n2026-08-10\n2026-08-11\n", encoding="utf-8"
        )
        for field in REQUIRED_QLIB_FIELDS:
            self._write_provider_field(features / f"{field}.day.bin")
        raw_frame = "date,open,close,high,low,volume\n2026-08-11,1,1,1,1,100\n"
        (raw / "sh510300.csv").write_text(raw_frame, encoding="utf-8")
        (normalized / "sh510300.csv").write_text(raw_frame, encoding="utf-8")
        source = cache / "SH510300.html"
        source.write_text("source", encoding="utf-8")
        import hashlib

        source_sha = hashlib.sha256(b"source").hexdigest()
        pd.DataFrame(
            {
                "symbol": ["SH510300"],
                "error": [""],
                "full_universe_scope": [True],
                "published": [True],
                "cache_sha256": [source_sha],
            }
        ).to_csv(data / "corporate_action_report.csv", index=False)
        pd.DataFrame(
            columns=[
                "symbol",
                "record_date",
                "ex_date",
                "cash_payment_date",
                "cash_dividend_per_old_share",
                "share_ratio",
                "fractional_share_treatment",
                "source_url",
                "source_sha256",
            ]
        ).to_csv(data / "corporate_actions.csv", index=False)
        write_json_atomic(
            data / "validation_report.json",
            {
                "training_ready": True,
                "universe_count": 1,
                "min_latest_date": "2026-08-11",
                "max_latest_date": "2026-08-11",
                "issues": [],
            },
        )
        return {
            "paths": {
                "universe": str(data / "universe.csv"),
                "instruments": str(instruments / "t1_etf.txt"),
                "qlib_provider": str(provider),
                "validation_report": str(data / "validation_report.json"),
                "snapshots": str(root / "snapshots"),
            },
            "data": {"end_date": "2026-08-10", "universe_mode": "current_snapshot"},
        }

    def test_scoped_inventory_is_portable_deterministic_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            second = nested / "b.bin"
            first = root / "a.csv"
            first.write_bytes(b"alpha")
            second.write_bytes(b"beta")

            records = _scoped_inventory("data", [second, first], root)
            self.assertEqual([record["path"] for record in records], ["a.csv", "nested/b.bin"])
            self.assertTrue(all(record["scope"] == "data" for record in records))
            self.assertEqual([record["size"] for record in records], [5, 4])
            self.assertTrue(all(len(record["sha256"]) == 64 for record in records))

    def test_future_calendar_contains_data_calendar_and_one_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            calendar = Path(directory) / "day.txt"
            calendar.write_text("2026-08-07\n2026-08-10\n", encoding="utf-8")
            future = ensure_future_calendar_boundary(calendar)
            values = pd.read_csv(future, header=None).iloc[:, 0].tolist()
            self.assertEqual(values, ["2026-08-07", "2026-08-10", "2026-08-11"])
            self.assertEqual(ensure_future_calendar_boundary(calendar), future)

    def test_provider_features_require_all_float32_fields_and_equal_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features" / "sh510300"
            features.mkdir(parents=True)
            instruments = pd.DataFrame(
                {
                    "symbol": ["SH510300"],
                    "start_date": ["2026-08-07"],
                    "end_date": ["2026-08-11"],
                }
            )
            calendar = pd.Series(pd.to_datetime(["2026-08-07", "2026-08-10", "2026-08-11"]))
            for field in REQUIRED_QLIB_FIELDS:
                self._write_provider_field(features / f"{field}.day.bin")

            issues, summary = evaluate_provider_features(features.parent, instruments, calendar)
            self.assertEqual(issues, [])
            self.assertEqual(summary["validated_symbols"], 1)

            (features / "paused.day.bin").unlink()
            self._write_provider_field(features / "volume.day.bin", values=2)
            (features / "vwap.day.bin").write_bytes(b"bad")
            (features / "amount.day.bin").write_bytes(
                struct.pack("<f", 0.0) + struct.pack("<3f", *([math.nan] * 3))
            )
            issues, summary = evaluate_provider_features(features.parent, instruments, calendar)
            self.assertTrue(any("missing required fields" in issue for issue in issues))
            self.assertTrue(any("malformed feature files" in issue for issue in issues))
            self.assertTrue(any("inconsistent field lengths" in issue for issue in issues))
            self.assertEqual(summary["missing_fields"], {"SH510300": ["paused"]})

    def test_historical_stage_end_may_precede_fresh_provider_end(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._audit_fixture(Path(directory))
            result = audit_and_snapshot(config)
            self.assertTrue(result.report["data_valid"])
            self.assertEqual(result.report["calendar"]["end"], "2026-08-11")
            self.assertEqual(result.report["calendar"]["requested_end"], "2026-08-10")
            self.assertEqual(
                result.report["audit_identity"]["configured_end"], "2026-08-10"
            )

    def test_dry_audit_writes_neither_future_calendar_nor_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._audit_fixture(Path(directory))
            provider = Path(config["paths"]["qlib_provider"])
            future = provider / "calendars" / "day_future.txt"
            snapshots = Path(config["paths"]["snapshots"])

            dry = audit_and_snapshot(config, persist=False)

            self.assertTrue(dry.report["data_valid"])
            self.assertFalse(future.exists())
            self.assertFalse(snapshots.exists())
            persisted = audit_and_snapshot(config)
            self.assertEqual(dry.snapshot_id, persisted.snapshot_id)
            self.assertEqual(
                dry.report["source_fingerprint"], persisted.report["source_fingerprint"]
            )

    def test_existing_snapshot_cannot_override_a_new_audit_result(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._audit_fixture(Path(directory))
            result = audit_and_snapshot(config)
            manifest_path = result.snapshot_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["data_valid"] = False
            manifest["blocking_issues"] = ["forged stale result"]
            write_json_atomic(manifest_path, _seal_snapshot_manifest(manifest))

            with self.assertRaisesRegex(RuntimeError, "differs from the current audit"):
                audit_and_snapshot(config)

    def test_existing_snapshot_revalidates_copied_control_files(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._audit_fixture(Path(directory))
            result = audit_and_snapshot(config)
            (result.snapshot_dir / "universe.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "snapshot control file mismatch"):
                audit_and_snapshot(config)

    def test_pool_external_cache_explains_upstream_count_warnings(self):
        report = {
            "training_ready": False,
            "universe_count": 3,
            "min_latest_date": "2026-08-10",
            "max_latest_date": "2026-08-10",
            "issues": [
                {"error": "raw file count 5 != universe 3"},
                {"error": "normalized file count 5 != universe 3"},
            ],
        }
        blocking, warnings = evaluate_upstream_validation(
            report,
            universe_count=3,
            configured_end=pd.Timestamp("2026-08-10"),
            raw_file_count=5,
            normalized_file_count=5,
            external_raw_count=2,
            external_normalized_count=2,
        )
        self.assertEqual(blocking, [])
        self.assertEqual(len(warnings), 1)

    def test_unexpected_upstream_issue_blocks_training(self):
        report = {
            "training_ready": False,
            "universe_count": 3,
            "min_latest_date": "2026-08-10",
            "max_latest_date": "2026-08-10",
            "issues": [{"error": "non-positive close in sh510300.csv"}],
        }
        blocking, _ = evaluate_upstream_validation(
            report,
            universe_count=3,
            configured_end=pd.Timestamp("2026-08-10"),
            raw_file_count=3,
            normalized_file_count=3,
            external_raw_count=0,
            external_normalized_count=0,
        )
        self.assertIn("upstream validation: non-positive close in sh510300.csv", blocking)

    def test_corporate_action_collection_requires_exact_scope_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            document = "<html>audited</html>"
            source = cache / "SH510300.html"
            source.write_text(document, encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
            report = pd.DataFrame(
                {
                    "symbol": ["SH510300"],
                    "error": [None],
                    "full_universe_scope": ["True"],
                    "published": ["True"],
                    "cache_sha256": [digest],
                }
            )
            actions = pd.DataFrame(
                {
                    "symbol": ["SH510300"],
                    "record_date": ["2026-01-05"],
                    "ex_date": ["2026-01-06"],
                    "cash_payment_date": ["2026-01-07"],
                    "cash_dividend_per_old_share": [0.1],
                    "share_ratio": [1.0],
                    "fractional_share_treatment": ["not_applicable_no_share_change"],
                    "source_url": ["https://example.test"],
                    "source_sha256": [digest],
                }
            )
            issues, summary, files = evaluate_corporate_action_collection(
                report, actions, {"SH510300"}, cache
            )
            self.assertEqual(issues, [])
            self.assertTrue(summary["complete_universe"])
            self.assertEqual(files, [source])

            report.loc[0, "published"] = "False"
            issues, summary, _ = evaluate_corporate_action_collection(
                report, actions, {"SH510300"}, cache
            )
            self.assertIn(
                "corporate-action collection was not published for the full universe", issues
            )
            self.assertFalse(summary["complete_universe"])

    def test_corporate_action_collection_rejects_string_false_and_tampered_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "SH510300.html").write_text("tampered", encoding="utf-8")
            report = pd.DataFrame(
                {
                    "symbol": ["SH510300"],
                    "error": [None],
                    "full_universe_scope": ["False"],
                    "published": ["False"],
                    "cache_sha256": ["a" * 64],
                }
            )
            actions = pd.DataFrame(
                columns=[
                    "symbol",
                    "record_date",
                    "ex_date",
                    "cash_payment_date",
                    "cash_dividend_per_old_share",
                    "share_ratio",
                    "fractional_share_treatment",
                    "source_url",
                    "source_sha256",
                ]
            )
            issues, _, _ = evaluate_corporate_action_collection(
                report, actions, {"SH510300"}, cache
            )
            self.assertIn(
                "corporate-action collection was not published for the full universe", issues
            )
            self.assertTrue(any("hash-mismatched" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
