import math
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.factors import (
    FACTOR_CATALOG_VERSION,
    FACTOR_FAMILIES,
    ORIGINAL_RESEARCH_CANDIDATES,
    RESEARCH_PROTOCOL,
    FactorDefinition,
    build_alpha158_factor_handler,
    combined_alpha158_feature_config,
    factor_catalog_manifest,
    factor_config,
    factors_by_family,
    select_factor_definitions,
    validate_factor_definitions,
)


@contextmanager
def fake_qlib_alpha158():
    class FakeAlpha158DL:
        @staticmethod
        def get_feature_config(config):
            if config["price"]["feature"] != ["OPEN", "HIGH", "LOW", "VWAP"]:
                raise AssertionError("unexpected Alpha158 feature configuration")
            return ["$close", "Mean($close,5)"], ["ALPHA_RAW", "ALPHA_MEAN"]

    class FakeAlpha158:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.feature_config = self.get_feature_config()

    modules = {
        "qlib": types.ModuleType("qlib"),
        "qlib.contrib": types.ModuleType("qlib.contrib"),
        "qlib.contrib.data": types.ModuleType("qlib.contrib.data"),
        "qlib.contrib.data.loader": types.ModuleType("qlib.contrib.data.loader"),
        "qlib.contrib.data.handler": types.ModuleType("qlib.contrib.data.handler"),
    }
    modules["qlib.contrib.data.loader"].Alpha158DL = FakeAlpha158DL
    modules["qlib.contrib.data.handler"].Alpha158 = FakeAlpha158
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield FakeAlpha158
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class FactorDefinitionTests(unittest.TestCase):
    def test_builtin_candidates_are_valid_and_not_direct_alpha158_names(self):
        validated = validate_factor_definitions()
        self.assertGreaterEqual(len(validated), 12)
        self.assertLessEqual(len(validated), 20)
        self.assertEqual(len({factor.name for factor in validated}), len(validated))
        self.assertTrue(all(factor.name.startswith("ORC_") for factor in validated))
        self.assertTrue(all("Ref(" not in factor.name for factor in validated))

    def test_all_families_are_nonempty_and_selection_is_stable(self):
        grouped = factors_by_family()
        self.assertEqual(tuple(grouped), FACTOR_FAMILIES)
        self.assertTrue(all(grouped[family] for family in FACTOR_FAMILIES))
        requested = ("session_structure", "trend_crowding")
        selected = select_factor_definitions(requested)
        self.assertTrue(all(factor.family in requested for factor in selected))
        self.assertEqual(
            [factor.name for factor in selected],
            [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES if factor.family in requested],
        )

    def test_factor_config_returns_aligned_qlib_fields_and_names(self):
        fields, names = factor_config(["compression_release"])
        expected = select_factor_definitions(["compression_release"])
        self.assertEqual(fields, [factor.expression for factor in expected])
        self.assertEqual(names, [factor.name for factor in expected])
        self.assertEqual(len(fields), len(names))

    def test_catalog_manifest_is_deterministic_and_covers_protocol(self):
        first = factor_catalog_manifest()
        second = factor_catalog_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["catalog_version"], FACTOR_CATALOG_VERSION)
        self.assertEqual(first["protocol"]["catalog_version"], FACTOR_CATALOG_VERSION)
        self.assertEqual(first["protocol"]["stage_order"], list(RESEARCH_PROTOCOL.stage_order))
        self.assertEqual(len(first["sha256"]), 64)
        subset = factor_catalog_manifest(["session_structure"])
        self.assertNotEqual(first["sha256"], subset["sha256"])
        self.assertTrue(all(item["family"] == "session_structure" for item in subset["factors"]))

    def test_named_factor_selection_is_exact_ordered_and_mutually_exclusive(self):
        names = [
            ORIGINAL_RESEARCH_CANDIDATES[2].name,
            ORIGINAL_RESEARCH_CANDIDATES[0].name,
        ]
        selected = select_factor_definitions(factor_names=names)
        self.assertEqual([factor.name for factor in selected], names)
        manifest = factor_catalog_manifest(factor_names=names)
        self.assertEqual([factor["name"] for factor in manifest["factors"]], names)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            select_factor_definitions(["volume_impact"], names)
        with self.assertRaisesRegex(ValueError, "unknown factor names"):
            select_factor_definitions(factor_names=["ORC_DOES_NOT_EXIST"])

    def test_combined_config_preserves_alpha_order_and_only_appends_candidates(self):
        with fake_qlib_alpha158():
            fields, names = combined_alpha158_feature_config(["trend_crowding"])
        candidates = select_factor_definitions(["trend_crowding"])
        self.assertEqual(fields[:2], ["$close", "Mean($close,5)"])
        self.assertEqual(names[:2], ["ALPHA_RAW", "ALPHA_MEAN"])
        self.assertEqual(fields[2:], [factor.expression for factor in candidates])
        self.assertEqual(names[2:], [factor.name for factor in candidates])

    def test_handler_factory_reuses_alpha158_arguments_and_combined_features(self):
        label = (["Ref($close,-2)/Ref($close,-1)-1"], ["LABEL0"])
        with fake_qlib_alpha158() as fake_handler_type:
            handler = build_alpha158_factor_handler(
                instruments="t1_etf",
                start_time="2020-01-01",
                end_time="2026-08-10",
                fit_start_time="2020-01-01",
                fit_end_time="2024-12-31",
                label=label,
                families=["volume_impact"],
                filter_pipe=[{"filter_type": "ExpressionDFilter"}],
            )
        self.assertIsInstance(handler, fake_handler_type)
        self.assertEqual(handler.kwargs["label"], label)
        self.assertEqual(handler.kwargs["instruments"], "t1_etf")
        self.assertEqual(handler.kwargs["filter_pipe"], [{"filter_type": "ExpressionDFilter"}])
        self.assertEqual(handler.feature_config[1][:2], ["ALPHA_RAW", "ALPHA_MEAN"])
        self.assertEqual(
            handler.feature_config[1][2:],
            [factor.name for factor in select_factor_definitions(["volume_impact"])],
        )

    def test_future_reference_is_rejected_even_with_nested_first_argument(self):
        unsafe = replace(
            ORIGINAL_RESEARCH_CANDIDATES[0],
            expression="Ref(Mean($close,5)/Ref($volume,1), -2)",
            lookback=20,
        )
        with self.assertRaisesRegex(ValueError, "future Ref offset"):
            validate_factor_definitions([unsafe])

    def test_duplicate_names_are_rejected(self):
        original = ORIGINAL_RESEARCH_CANDIDATES[0]
        duplicate = replace(ORIGINAL_RESEARCH_CANDIDATES[1], name=original.name)
        with self.assertRaisesRegex(ValueError, "duplicate factor name"):
            validate_factor_definitions([original, duplicate])

    def test_non_finite_metadata_and_expression_constants_are_rejected(self):
        original = ORIGINAL_RESEARCH_CANDIDATES[0]
        for change in (
            {"direction": math.nan},
            {"lookback": math.inf},
            {"expression": "$close+1e999"},
        ):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "finite|non-finite"):
                    validate_factor_definitions([replace(original, **change)])

    def test_unsupported_non_ohlcv_field_is_rejected(self):
        invalid = replace(ORIGINAL_RESEARCH_CANDIDATES[0], expression="$close/$vwap")
        with self.assertRaisesRegex(ValueError, "unsupported source field"):
            validate_factor_definitions([invalid])

    def test_understated_lookback_is_rejected(self):
        invalid = replace(
            ORIGINAL_RESEARCH_CANDIDATES[0],
            expression="Mean($close/Ref($close,3)-1,10)",
            lookback=11,
        )
        with self.assertRaisesRegex(ValueError, "understates required historical lag 12"):
            validate_factor_definitions([invalid])

    def test_unknown_operator_and_family_are_rejected(self):
        original = ORIGINAL_RESEARCH_CANDIDATES[0]
        with self.assertRaisesRegex(ValueError, "unsupported operator"):
            validate_factor_definitions([replace(original, expression="Future($close,1)")])
        with self.assertRaisesRegex(ValueError, "operator Ref must be called"):
            validate_factor_definitions([replace(original, expression="Ref+1")])
        with self.assertRaisesRegex(ValueError, "unknown factor famil"):
            validate_factor_definitions([replace(original, family="uncontrolled")])
        with self.assertRaisesRegex(ValueError, "unknown factor families"):
            select_factor_definitions(["uncontrolled"])

    def test_zero_ref_and_nonpositive_rolling_windows_are_rejected(self):
        original = ORIGINAL_RESEARCH_CANDIDATES[0]
        with self.assertRaisesRegex(ValueError, "offset zero"):
            validate_factor_definitions([replace(original, expression="Ref($close,0)")])
        with self.assertRaisesRegex(ValueError, "rolling windows must be positive"):
            validate_factor_definitions([replace(original, expression="Mean($close,0)")])


if __name__ == "__main__":
    unittest.main()
