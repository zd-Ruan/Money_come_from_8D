import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline import cli


class ComparisonOutputTests(unittest.TestCase):
    def test_default_output_is_a_direct_comparisons_child(self):
        output = cli.resolve_comparison_output(None, "baseline", "candidate")
        self.assertEqual(output, (cli.PIPELINE_ROOT / "comparisons" / "baseline__vs__candidate.json").resolve())

    def test_rejects_output_outside_comparisons_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "direct child"):
                cli.resolve_comparison_output(Path(directory) / "result.json", "baseline", "candidate")

    def test_accepts_explicit_direct_child_json(self):
        target = cli.PIPELINE_ROOT / "comparisons" / "paired.json"
        with patch.object(Path, "cwd", return_value=cli.PIPELINE_ROOT.parent):
            output = cli.resolve_comparison_output(target, "baseline", "candidate")
        self.assertEqual(output, target.resolve())


if __name__ == "__main__":
    unittest.main()
