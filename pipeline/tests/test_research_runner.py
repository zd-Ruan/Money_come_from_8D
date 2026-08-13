import hashlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.factor_research import analyze_confirmation, build_research_plan
from quant_pipeline.factors import FACTOR_FAMILIES, ORIGINAL_RESEARCH_CANDIDATES
from quant_pipeline.io import sha256_file
from quant_pipeline.metrics import evaluation_frame, max_drawdown
from quant_pipeline.research_runner import (
    BASELINE_EXPERIMENT_ID,
    build_alpha158_named_factor_handler,
    build_baseline_experiment_spec,
    build_discovery_experiment_specs,
    build_frozen_candidate_experiment_spec,
    build_research_experiment_manifest,
    build_stage_run_request,
    combined_alpha158_named_feature_config,
    enforce_valid_metric_coverage,
    factor_catalog_manifest_by_name,
    factor_config_by_name,
    _load_stage_portfolio_evidence,
    _load_stage_raw_factor_evidence,
    load_stage_metric_pair,
    load_stage_signal_metric,
    prepare_stage_pipeline_config,
    select_factor_definitions_by_name,
    validate_discovery_experiment_specs,
    validate_experiment_spec,
    validate_stage_run_request,
)


def make_plan():
    dates = pd.bdate_range("2025-01-02", periods=258)
    return build_research_plan(
        dates,
        discovery_end=dates[125],
        confirmation_end=dates[190],
        plan_id="orchestrated-factor-study",
        base_config_sha256="f" * 64,
    )


def frozen_record(names):
    unsigned = {
        "selected_factor_names": list(names),
        "specification": {
            "config_sha256": "a" * 64,
            "model": "lightgbm",
            "model_parameters_frozen": True,
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {**unsigned, "sha256": digest}


def _write_portfolio_bundle(
    root,
    records,
    *,
    prefix,
    raw_sessions,
    evaluation_sessions,
    gross,
    contract,
    summary_identity,
    filled_intents=100,
    zero_fill_intents=0,
):
    raw_index = pd.DatetimeIndex(raw_sessions, name="date")
    gross = np.asarray(gross, dtype=float)
    if len(gross) != len(raw_index) or gross[0] != 0.0:
        raise AssertionError("test gross returns must match the raw sessions and start at zero")
    cost = np.zeros(len(raw_index), dtype=float)
    benchmark = np.zeros(len(raw_index), dtype=float)
    account = 20_000.0 * np.cumprod(1.0 + gross - cost)
    report = pd.DataFrame(
        {
            "return": gross,
            "cost": cost,
            "bench": benchmark,
            "account": account,
        },
        index=raw_index,
    )
    aligned = evaluation_frame(report)
    if not pd.DatetimeIndex(aligned.index).equals(pd.DatetimeIndex(evaluation_sessions)):
        raise AssertionError("test report does not match its evaluation contract")

    account_change = pd.Series(account, index=raw_index).diff()
    account_change.iloc[0] = account[0] - 20_000.0
    attribution_rows = []
    symbols = ("SH510300", "SH510500", "SH512100")
    weights = (0.34, 0.33, 0.33)
    for date, pnl in account_change.items():
        for symbol, weight in zip(symbols, weights):
            value = float(pnl) * weight
            attribution_rows.append(
                {"date": date, "symbol": symbol, "net_pnl": value, "net_pnl_cny": value}
            )
    attribution = pd.DataFrame(attribution_rows)
    gross_by_symbol = attribution.groupby("symbol")["net_pnl"].apply(
        lambda values: float(values.abs().sum())
    )
    denominator = float(gross_by_symbol.sum())
    concentration = {
        "zero_denominator_policy": "concentration_null_fail_closed",
        "gross_abs": {
            "symbol": "SH510300",
            "numerator_cny": float(gross_by_symbol.loc["SH510300"]),
            "denominator_cny": denominator,
            "share": float(gross_by_symbol.loc["SH510300"] / denominator),
        },
        "net_abs": {
            "symbol": "SH510300",
            "numerator_cny": float(
                attribution.groupby("symbol")["net_pnl"].sum().abs().loc["SH510300"]
            ),
            "denominator_cny": float(
                attribution.groupby("symbol")["net_pnl"].sum().abs().sum()
            ),
            "share": float(
                attribution.groupby("symbol")["net_pnl"].sum().abs().loc["SH510300"]
                / attribution.groupby("symbol")["net_pnl"].sum().abs().sum()
            ),
        },
    }
    intent_count = filled_intents + zero_fill_intents
    target_notional = 100_000.0 if intent_count else 0.0
    fill_notional = target_notional * (filled_intents / intent_count if intent_count else 0.0)
    notional_fill_rate = fill_notional / target_notional if target_notional else 0.0
    benchmark_terminal = 20_000.0 * float((1.0 + aligned["benchmark"]).prod())
    strategy_drawdown = float(max_drawdown(aligned["strategy_net"]))
    execution = {
        "engine": contract["engine"],
        "nav_reconciled": True,
        "initial_account": 20_000.0,
        "final_account": float(account[-1]),
        "intent_count": intent_count,
        "filled_intent_count": filled_intents,
        "zero_fill_intent_count": zero_fill_intents,
        "intent_fill_rate": filled_intents / intent_count if intent_count else 0.0,
        "zero_fill_intent_rate": zero_fill_intents / intent_count if intent_count else 0.0,
        "target_notional": target_notional,
        "fill_notional": fill_notional,
        "fill_rate": notional_fill_rate,
        "notional_fill_rate": notional_fill_rate,
        "max_single_etf_gross_abs_contribution_share": concentration["gross_abs"]["share"],
        "symbol_attribution_concentration": concentration,
        "config": {
            "initial_cash": 20_000.0,
            "slippage_bps_per_side": 10.0,
        },
    }
    summary = {
        "slippage_bps_per_side": 10,
        "alignment_method": contract["alignment_method"],
        "initial_execution_date": raw_index[0].date().isoformat(),
        "evaluation_start_date": pd.Timestamp(evaluation_sessions[0]).date().isoformat(),
        "evaluation_end_date": pd.Timestamp(evaluation_sessions[-1]).date().isoformat(),
        "days": len(aligned),
        "raw_execution_days": len(report),
        "terminal_account": float(account[-1]),
        "benchmark_terminal_account": benchmark_terminal,
        "net_cumulative_return": float(account[-1] / 20_000.0 - 1.0),
        "benchmark_cumulative_return": float(benchmark_terminal / 20_000.0 - 1.0),
        "strategy_max_drawdown": strategy_drawdown,
        "max_drawdown": strategy_drawdown,
        "fill_rate": notional_fill_rate,
        "notional_fill_rate": notional_fill_rate,
        "single_etf_abs_contribution_share": concentration["gross_abs"]["share"],
        "symbol_attribution_concentration": concentration,
        "execution": execution,
        **summary_identity,
    }
    if "fold" in summary_identity:
        summary.update(
            {
                "terminal_account_value": float(account[-1]),
                "benchmark_terminal_account_value": benchmark_terminal,
                "initial_account_value": 20_000.0,
            }
        )

    output = root / prefix
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output / "report.parquet",
        "summary": output / "summary.json",
        "attribution": output / "symbol_attribution.parquet",
    }
    report.to_parquet(paths["report"])
    attribution.to_parquet(paths["attribution"], index=False)
    paths["summary"].write_text(json.dumps(summary), encoding="utf-8")
    for path in paths.values():
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "sha256": sha256_file(path)})
    return report, summary, attribution, paths


def write_stage_portfolio_artifacts(
    root,
    request,
    gross,
    *,
    filled_intents=100,
    zero_fill_intents=0,
    fold_daily_return=0.0001,
):
    contract = request["metric_contract"]["portfolio"]
    records = []
    main = _write_portfolio_bundle(
        root,
        records,
        prefix=Path(contract["artifact"]).parent.as_posix(),
        raw_sessions=contract["raw_report_sessions"],
        evaluation_sessions=contract["evaluation_sessions"],
        gross=gross,
        contract=contract,
        summary_identity={},
        filled_intents=filled_intents,
        zero_fill_intents=zero_fill_intents,
    )
    folds = []
    for fold in contract["research_folds"]:
        fold_gross = np.r_[
            0.0,
            np.full(len(fold["raw_report_sessions"]) - 1, fold_daily_return),
        ]
        folds.append(
            _write_portfolio_bundle(
                root,
                records,
                prefix=f"folds/research_fold_{fold['fold']:02d}/backtest",
                raw_sessions=fold["raw_report_sessions"],
                evaluation_sessions=fold["evaluation_sessions"],
                gross=fold_gross,
                contract=contract,
                summary_identity={
                    "fold": fold["fold"],
                    "signal_start": fold["signal_start"],
                    "signal_end": fold["signal_end"],
                    "signal_observations": fold["signal_observations"],
                    "signal_sessions": fold["signal_sessions"],
                    "raw_report_start": fold["raw_report_start"],
                    "raw_report_end": fold["raw_report_end"],
                    "raw_report_sessions": fold["raw_report_sessions"],
                    "evaluation_start": fold["evaluation_start"],
                    "evaluation_end": fold["evaluation_end"],
                    "evaluation_sessions": fold["evaluation_sessions"],
                    "complete_for_gate": fold["complete_for_gate"],
                },
                filled_intents=filled_intents,
                zero_fill_intents=zero_fill_intents,
            )
        )
    return records, main, folds


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

        def get_feature_config(self):
            return ["$close"], ["ALPHA_RAW"]

    modules = {
        "qlib": types.ModuleType("qlib"),
        "qlib.contrib": types.ModuleType("qlib.contrib"),
        "qlib.contrib.data": types.ModuleType("qlib.contrib.data"),
        "qlib.contrib.data.handler": types.ModuleType("qlib.contrib.data.handler"),
        "qlib.contrib.data.loader": types.ModuleType("qlib.contrib.data.loader"),
    }
    modules["qlib.contrib.data.loader"].Alpha158DL = FakeAlpha158DL
    modules["qlib.contrib.data.handler"].Alpha158 = FakeAlpha158
    with patch.dict(sys.modules, modules):
        yield FakeAlpha158


class NamedFactorSelectionTests(unittest.TestCase):
    def test_exact_name_selection_is_catalog_ordered_and_does_not_expand_family(self):
        first = ORIGINAL_RESEARCH_CANDIDATES[0]
        last = ORIGINAL_RESEARCH_CANDIDATES[-1]
        selected = select_factor_definitions_by_name([last.name, first.name])
        self.assertEqual([factor.name for factor in selected], [first.name, last.name])
        fields, names = factor_config_by_name([first.name])
        self.assertEqual(names, [first.name])
        self.assertEqual(fields, [first.expression])
        manifest = factor_catalog_manifest_by_name([first.name])
        self.assertEqual([factor["name"] for factor in manifest["factors"]], [first.name])
        self.assertEqual(manifest["families"], [first.family])

    def test_name_selection_rejects_ambiguous_or_unknown_input(self):
        name = ORIGINAL_RESEARCH_CANDIDATES[0].name
        with self.assertRaisesRegex(TypeError, "individual names"):
            select_factor_definitions_by_name(name)
        with self.assertRaisesRegex(ValueError, "unique"):
            select_factor_definitions_by_name([name, name])
        with self.assertRaisesRegex(ValueError, "unknown factor names"):
            select_factor_definitions_by_name(["ORC_NOT_IN_FROZEN_CATALOG"])

    def test_named_handler_appends_only_requested_factor(self):
        factor = ORIGINAL_RESEARCH_CANDIDATES[4]
        with fake_qlib_alpha158() as fake_handler_type:
            fields, names = combined_alpha158_named_feature_config([factor.name])
            handler = build_alpha158_named_factor_handler(
                factor_names=[factor.name],
                instruments="t1_etf",
                fit_end_time="2025-01-31",
                label=(["LABEL_EXPR"], ["LABEL0"]),
            )
        self.assertEqual(names, ["ALPHA_RAW", "ALPHA_MEAN", factor.name])
        self.assertEqual(fields[-1], factor.expression)
        self.assertIsInstance(handler, fake_handler_type)
        self.assertEqual(handler.feature_config[1], names)
        self.assertEqual(handler.kwargs["instruments"], "t1_etf")


class ExperimentSpecificationTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()

    def test_discovery_manifest_is_deterministic_complete_and_json_friendly(self):
        first = build_discovery_experiment_specs(self.plan)
        second = build_discovery_experiment_specs(self.plan)
        self.assertEqual(first, second)
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(first["experiment_count"], 24)
        self.assertEqual(first["baseline"]["experiment_id"], BASELINE_EXPERIMENT_ID)
        self.assertEqual(len(first["family_ablations"]), 5)
        self.assertEqual(len(first["single_factor_tests"]), 18)
        self.assertTrue(all(spec["status"] == "not_run" for spec in first["family_ablations"]))
        self.assertTrue(all(spec["status"] == "not_run" for spec in first["single_factor_tests"]))
        self.assertEqual(
            [spec["family"] for spec in first["family_ablations"]], list(FACTOR_FAMILIES)
        )
        self.assertTrue(
            all(len(spec["features"]["factor_names"]) == 1 for spec in first["single_factor_tests"])
        )
        self.assertNotIn("run_id", json.dumps(first))
        self.assertEqual(validate_discovery_experiment_specs(self.plan, first), first)

    def test_family_specs_contain_whole_family_and_singles_contain_one_factor(self):
        manifest = build_discovery_experiment_specs(self.plan)
        for spec in manifest["family_ablations"]:
            expected = [
                factor.name
                for factor in ORIGINAL_RESEARCH_CANDIDATES
                if factor.family == spec["family"]
            ]
            self.assertEqual(spec["features"]["factor_names"], expected)
            self.assertEqual(validate_experiment_spec(self.plan, spec), spec)
        for factor, spec in zip(ORIGINAL_RESEARCH_CANDIDATES, manifest["single_factor_tests"]):
            self.assertEqual(spec["factor_name"], factor.name)
            self.assertEqual(spec["features"]["factor_names"], [factor.name])
            self.assertEqual(spec["expected_direction"], factor.direction)

    def test_frozen_candidate_is_unchanged_for_confirmation_and_holdout(self):
        names = [ORIGINAL_RESEARCH_CANDIDATES[0].name, ORIGINAL_RESEARCH_CANDIDATES[-1].name]
        record = frozen_record(names)
        candidate = build_frozen_candidate_experiment_spec(self.plan, record)
        self.assertEqual(candidate["allowed_stages"], ["confirmation", "locked_holdout"])
        self.assertEqual(candidate["features"]["factor_names"], names)
        self.assertEqual(candidate["frozen_confirmation_spec"], record)
        self.assertEqual(validate_experiment_spec(self.plan, candidate), candidate)

        full = build_research_experiment_manifest(self.plan, record)
        self.assertEqual(full["frozen_candidate_status"], "frozen_not_run")
        self.assertEqual(full["frozen_candidate"], candidate)
        waiting = build_research_experiment_manifest(self.plan)
        self.assertEqual(waiting["frozen_candidate_status"], "awaiting_discovery_freeze")
        self.assertIsNone(waiting["frozen_candidate"])

    def test_tampered_or_ad_hoc_frozen_candidate_fails_closed(self):
        name = ORIGINAL_RESEARCH_CANDIDATES[0].name
        record = frozen_record([name])
        record["specification"]["model"] = "changed_after_freeze"
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_frozen_candidate_experiment_spec(self.plan, record)

        spec = build_baseline_experiment_spec(self.plan)
        spec["allowed_stages"] = ["locked_holdout"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_experiment_spec(self.plan, spec)


class StageBoundInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()
        self.discovery = build_discovery_experiment_specs(self.plan)
        self.baseline = self.discovery["baseline"]
        self.family = self.discovery["family_ablations"][0]
        self.candidate = build_frozen_candidate_experiment_spec(
            self.plan, frozen_record([ORIGINAL_RESEARCH_CANDIDATES[0].name])
        )

    def request(self, spec, stage):
        return build_stage_run_request(self.plan, spec, self.plan["partitions"][stage])

    def test_requests_are_exactly_stage_bound_and_do_not_claim_training(self):
        discovery = self.request(self.family, "discovery")
        self.assertEqual(discovery["status"], "not_run")
        self.assertEqual(discovery["partition"], self.plan["partitions"]["discovery"])
        self.assertEqual(
            discovery["training_contract"]["later_partition_training_forbidden"],
            ["confirmation", "locked_holdout"],
        )
        self.assertEqual(
            discovery["metric_contract"]["signal"]["sessions_sha256"],
            self.plan["partitions"]["discovery"]["sessions_sha256"],
        )
        portfolio = discovery["metric_contract"]["portfolio"]
        self.assertEqual(portfolio["minimum_candidate_terminal_account"], 20_000.0)
        self.assertEqual(portfolio["minimum_intent_fill_rate"], 0.95)
        self.assertEqual(portfolio["maximum_zero_fill_intent_rate"], 0.05)
        self.assertEqual(validate_stage_run_request(self.plan, discovery), discovery)

        confirmation = self.request(self.candidate, "confirmation")
        holdout = self.request(self.candidate, "locked_holdout")
        self.assertEqual(
            confirmation["experiment"]["spec_sha256"], holdout["experiment"]["spec_sha256"]
        )
        discovery_end = self.plan["partitions"]["discovery"]["end"]
        self.assertEqual(
            confirmation["training_contract"][
                "feature_selection_and_hyperparameter_tuning_date_not_after"
            ],
            discovery_end,
        )
        self.assertEqual(
            holdout["training_contract"][
                "feature_selection_and_hyperparameter_tuning_date_not_after"
            ],
            discovery_end,
        )
        self.assertEqual(
            confirmation["training_contract"]["source_data_end_not_after"],
            self.plan["partitions"]["confirmation"]["source_data_end"],
        )
        self.assertEqual(
            confirmation["training_contract"]["label_maturity_sessions"],
            self.plan["partitions"]["confirmation"]["label_maturity_sessions"],
        )
        self.assertTrue(confirmation["training_contract"]["frozen_hyperparameters_required"])
        self.assertTrue(holdout["training_contract"]["frozen_hyperparameters_required"])

    def test_stage_config_keeps_provider_end_but_freezes_internal_source_bound(self):
        request = self.request(self.baseline, "discovery")
        base = {
            "_meta": {"workspace_root": ".", "config_path": "baseline.yaml"},
            "data": {
                "start_date": "2020-01-01",
                "test_start_date": "2025-01-02",
                "end_date": "2026-12-31",
                "label_horizon_bars": 2,
                "benchmark": "SH510300",
            },
            "features": {"mode": "alpha158", "families": []},
            "execution": {
                "account": 20_000,
                "stress_slippage_bps_per_side": [0, 5, 10],
            },
            "gates": {
                "required_stress_slippage_bps": 10,
                "research_fold_days": 21,
                "min_research_fold_win_ratio": 0.60,
                "max_strategy_drawdown": 0.25,
                "max_single_etf_abs_contribution_share": 0.35,
                "max_single_fold_abs_incremental_pnl_share": 0.50,
            },
        }
        from quant_pipeline.config import json_ready_config

        plan = make_plan()
        plan["base_config_sha256"] = hashlib.sha256(
            json.dumps(
                json_ready_config(base), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("ascii")
        ).hexdigest()
        unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
        plan["plan_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        ).hexdigest()
        baseline = build_baseline_experiment_spec(plan)
        request = build_stage_run_request(plan, baseline, plan["partitions"]["discovery"])
        prepared, _ = prepare_stage_pipeline_config(base, plan, request)
        self.assertEqual(prepared["data"]["end_date"], "2026-12-31")
        self.assertEqual(
            prepared["_research_stage"]["source_data_end"],
            plan["partitions"]["discovery"]["source_data_end"],
        )
        self.assertEqual(
            prepared["_research_stage"]["prediction_end"],
            plan["partitions"]["discovery"]["end"],
        )

    def test_valid_metric_coverage_requires_at_least_ninety_percent(self):
        metric = pd.Series(np.arange(100, dtype=float))
        metric.iloc[:10] = np.nan
        result = enforce_valid_metric_coverage(metric, experiment_id="exactly-90")
        self.assertEqual(result["valid_metric_coverage"], 0.90)
        metric.iloc[10] = np.nan
        with self.assertRaisesRegex(RuntimeError, "below 90.00%"):
            enforce_valid_metric_coverage(metric, experiment_id="below-90")
        with self.assertRaisesRegex(ValueError, "cannot be below"):
            enforce_valid_metric_coverage(metric, experiment_id="bad-threshold", minimum=0.89)

    def test_wrong_stage_or_broadened_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not allowed in confirmation"):
            self.request(self.family, "confirmation")
        with self.assertRaisesRegex(ValueError, "not allowed in discovery"):
            self.request(self.candidate, "discovery")
        broadened = dict(self.plan["partitions"]["discovery"])
        broadened["end"] = self.plan["partitions"]["confirmation"]["end"]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_stage_run_request(self.plan, self.baseline, broadened)

    def test_metric_reader_uses_only_request_partition(self):
        request = self.request(self.baseline, "discovery")
        partition = self.plan["partitions"]["discovery"]
        index = pd.DatetimeIndex(partition["sessions"], name="datetime")
        frame = pd.DataFrame({"rank_ic": np.arange(len(index), dtype=float)}, index=index)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.parquet"
            path.write_bytes(b"stage-bound-placeholder")
            with patch("quant_pipeline.factor_research.pd.read_parquet", return_value=frame) as read:
                values = load_stage_signal_metric(
                    path,
                    self.plan,
                    request,
                    expected_sha256=sha256_file(path),
                )
        filters = read.call_args.kwargs["filters"]
        self.assertEqual(filters[0], ("datetime", ">=", pd.Timestamp(partition["start"])))
        self.assertEqual(filters[1], ("datetime", "<=", pd.Timestamp(partition["end"])))
        self.assertEqual(list(values.index), list(index))

    def test_raw_share_evidence_uses_required_stress_and_reconciles_terminal_account(self):
        request = self.request(self.baseline, "discovery")
        contract = request["metric_contract"]["portfolio"]
        raw_index = pd.DatetimeIndex(contract["raw_report_sessions"], name="date")
        gross = np.r_[0.0, np.full(len(raw_index) - 1, 0.001)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, main, folds = write_stage_portfolio_artifacts(root, request, gross)
            report, summary, attribution, paths = main
            aligned = evaluation_frame(report)
            strategy_net, portfolio = _load_stage_portfolio_evidence(
                root.resolve(), records, request
            )
            self.assertEqual(list(strategy_net.index), list(aligned.index.rename("datetime")))
            self.assertAlmostEqual(portfolio["terminal_account"], float(report["account"].iloc[-1]))
            self.assertEqual(portfolio["benchmark_terminal_account"], 20_000.0)
            self.assertAlmostEqual(
                portfolio["strategy_max_drawdown"], max_drawdown(strategy_net)
            )
            self.assertEqual(portfolio["stress_slippage_bps_per_side"], 10)
            self.assertEqual(portfolio["intent_fill_rate"], 1.0)
            self.assertEqual(portfolio["notional_fill_rate"], 1.0)
            self.assertEqual(portfolio["zero_fill_intent_rate"], 0.0)
            self.assertEqual(len(portfolio["research_folds"]), len(folds))
            self.assertEqual(portfolio["single_etf_abs_contribution_symbol"], "SH510300")

            bad_summary = json.loads(json.dumps(summary))
            bad_summary["slippage_bps_per_side"] = 5
            paths["summary"].write_text(json.dumps(bad_summary), encoding="utf-8")
            next(item for item in records if item["path"] == contract["summary_artifact"])[
                "sha256"
            ] = sha256_file(paths["summary"])
            with self.assertRaisesRegex(ValueError, "slippage_bps_per_side"):
                _load_stage_portfolio_evidence(root.resolve(), records, request)

            for field, value in (
                ("intent_fill_rate", None),
                ("intent_fill_rate", float("nan")),
                ("zero_fill_intent_rate", 1.01),
            ):
                with self.subTest(field=field, value=value):
                    bad_summary = json.loads(json.dumps(summary))
                    if value is None:
                        del bad_summary["execution"][field]
                    else:
                        bad_summary["execution"][field] = value
                    paths["summary"].write_text(json.dumps(bad_summary), encoding="utf-8")
                    next(
                        item for item in records if item["path"] == contract["summary_artifact"]
                    )["sha256"] = sha256_file(paths["summary"])
                    with self.assertRaisesRegex(ValueError, field):
                        _load_stage_portfolio_evidence(root.resolve(), records, request)

            paths["summary"].write_text(json.dumps(summary), encoding="utf-8")
            next(item for item in records if item["path"] == contract["summary_artifact"])[
                "sha256"
            ] = sha256_file(paths["summary"])
            tampered = attribution.copy()
            tampered.loc[0, "net_pnl"] += 1.0
            tampered.loc[0, "net_pnl_cny"] += 1.0
            tampered.to_parquet(paths["attribution"], index=False)
            next(
                item
                for item in records
                if item["path"] == contract["attribution_artifact"]
            )["sha256"] = sha256_file(paths["attribution"])
            with self.assertRaisesRegex(ValueError, "does not reconcile"):
                _load_stage_portfolio_evidence(root.resolve(), records, request)

            attribution.to_parquet(paths["attribution"], index=False)
            next(
                item
                for item in records
                if item["path"] == contract["attribution_artifact"]
            )["sha256"] = sha256_file(paths["attribution"])
            fold_summary_path = folds[0][3]["summary"]
            fold_summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
            fold_summary["terminal_account_value"] += 100.0
            fold_summary_path.write_text(json.dumps(fold_summary), encoding="utf-8")
            fold_relative = fold_summary_path.relative_to(root).as_posix()
            next(item for item in records if item["path"] == fold_relative)["sha256"] = sha256_file(
                fold_summary_path
            )
            with self.assertRaisesRegex(ValueError, "terminal_account_value"):
                _load_stage_portfolio_evidence(root.resolve(), records, request)

    def test_raw_factor_evidence_requires_exact_columns_dates_and_coverage(self):
        request = self.request(self.family, "discovery")
        contract = request["metric_contract"]["signal"]
        names = request["experiment"]["features"]["factor_names"]
        index = pd.DatetimeIndex(contract["sessions"], name="datetime")
        columns = [f"{name}{contract['raw_factor_column_suffix']}" for name in names]
        frame = pd.DataFrame(
            {column: np.linspace(-0.02, 0.03, len(index)) for column in columns},
            index=index,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / contract["raw_factor_artifact"]
            frame.to_parquet(path)
            records = [{"path": path.name, "sha256": sha256_file(path)}]
            loaded = _load_stage_raw_factor_evidence(root.resolve(), records, request)
            self.assertEqual(list(loaded), names)
            self.assertTrue(all(series.index.equals(index) for series in loaded.values()))

            extra = frame.assign(UNDECLARED__rank_ic=0.0)
            extra.to_parquet(path)
            records[0]["sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "columns differ"):
                _load_stage_raw_factor_evidence(root.resolve(), records, request)

            low_coverage = frame.copy()
            low_coverage.iloc[: int(len(index) * 0.11), 0] = np.nan
            low_coverage.to_parquet(path)
            records[0]["sha256"] = sha256_file(path)
            with self.assertRaisesRegex(RuntimeError, "below 90.00%"):
                _load_stage_raw_factor_evidence(root.resolve(), records, request)

            shifted = frame.copy()
            shifted.index = shifted.index + pd.Timedelta(days=1)
            shifted.to_parquet(path)
            records[0]["sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "dates differ"):
                _load_stage_raw_factor_evidence(root.resolve(), records, request)

    def test_baseline_raw_factor_evidence_must_be_empty(self):
        request = self.request(self.baseline, "discovery")
        relative = request["metric_contract"]["signal"]["raw_factor_artifact"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / relative
            pd.DataFrame(index=pd.DatetimeIndex([], name="datetime")).to_parquet(path)
            records = [{"path": relative, "sha256": sha256_file(path)}]
            self.assertEqual(
                _load_stage_raw_factor_evidence(root.resolve(), records, request), {}
            )
            pd.DataFrame(
                index=pd.DatetimeIndex(
                    request["metric_contract"]["signal"]["sessions"], name="datetime"
                )
            ).to_parquet(path)
            records[0]["sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "must be empty"):
                _load_stage_raw_factor_evidence(root.resolve(), records, request)

    def test_loaded_execution_quality_flows_into_confirmation_decision(self):
        baseline_request = self.request(self.baseline, "confirmation")
        candidate_request = self.request(self.candidate, "confirmation")
        contract = baseline_request["metric_contract"]["portfolio"]
        raw_index = pd.DatetimeIndex(contract["raw_report_sessions"], name="date")

        def load_portfolio(
            root,
            request,
            gross,
            *,
            fill_rate,
            zero_fill_rate,
            fold_daily_return,
        ):
            intent_count = 100
            filled = round(fill_rate * intent_count)
            zero = round(zero_fill_rate * intent_count)
            records, _, _ = write_stage_portfolio_artifacts(
                root,
                request,
                gross,
                filled_intents=filled,
                zero_fill_intents=zero,
                fold_daily_return=fold_daily_return,
            )
            return _load_stage_portfolio_evidence(root.resolve(), records, request)

        baseline_gross = np.r_[
            0.0, np.sin(np.arange(len(raw_index) - 1) / 9.0) * 0.0002
        ]
        candidate_gross = baseline_gross.copy()
        candidate_gross[1:] += 0.0002 + np.cos(
            np.arange(len(raw_index) - 1) / 8.0
        ) * 0.00002
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_net, baseline_portfolio = load_portfolio(
                root / "baseline",
                baseline_request,
                baseline_gross,
                fill_rate=1.0,
                zero_fill_rate=0.0,
                fold_daily_return=0.00005,
            )
            candidate_net, candidate_portfolio = load_portfolio(
                root / "candidate",
                candidate_request,
                candidate_gross,
                fill_rate=0.96,
                zero_fill_rate=0.04,
                fold_daily_return=0.00015,
            )

        signal_index = pd.DatetimeIndex(
            self.plan["partitions"]["confirmation"]["sessions"], name="datetime"
        )
        baseline_rank_ic = pd.Series(
            np.sin(np.arange(len(signal_index))) * 0.01,
            index=signal_index,
            name="rank_ic",
        )
        candidate_rank_ic = baseline_rank_ic + 0.002 + np.cos(
            np.arange(len(signal_index)) / 7.0
        ) * 0.0002
        raw_factor_rank_ic = pd.Series(
            -0.012 + np.sin(np.arange(len(signal_index)) / 4.0) * 0.001,
            index=signal_index,
        )
        result = analyze_confirmation(
            {
                "rank_ic": baseline_rank_ic,
                "raw_factor_rank_ic": {},
                "strategy_net": baseline_net,
                "benchmark": baseline_portfolio.pop("_benchmark"),
                "portfolio": baseline_portfolio,
            },
            {
                "rank_ic": candidate_rank_ic,
                "raw_factor_rank_ic": {
                    ORIGINAL_RESEARCH_CANDIDATES[0].name: raw_factor_rank_ic
                },
                "strategy_net": candidate_net,
                "benchmark": candidate_portfolio.pop("_benchmark"),
                "portfolio": candidate_portfolio,
            },
            self.plan,
            frozen_spec_sha256="a" * 64,
            candidate_factor_names=[ORIGINAL_RESEARCH_CANDIDATES[0].name],
        )
        self.assertTrue(result["criteria"]["candidate_execution_quality_passed"])
        self.assertTrue(result["confirmation_passed"], result["criteria"])

    def test_paired_reader_rejects_cross_stage_requests_before_reading(self):
        baseline_request = self.request(self.baseline, "confirmation")
        candidate_request = self.request(self.candidate, "locked_holdout")
        with patch("quant_pipeline.research_runner.load_stage_signal_metric") as loader:
            with self.assertRaisesRegex(ValueError, "same exact partition"):
                load_stage_metric_pair(
                    "baseline.parquet",
                    "candidate.parquet",
                    self.plan,
                    baseline_request,
                    candidate_request,
                    baseline_sha256="a" * 64,
                    candidate_sha256="b" * 64,
                )
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
