import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_pipeline.factor_research import (
    CATALOG_HYPOTHESIS_COUNT,
    DISCOVERY_FDR_Q,
    DISCOVERY_MULTIPLICITY_APPLIED_TO,
    FAMILY_ABLATION_COUNT,
    MIN_PAIRED_OBSERVATIONS,
    RESEARCH_STAGES,
    _one_sided_hac_p_value,
    _student_t_cdf,
    analyze_confirmation,
    analyze_discovery,
    analyze_factor_discovery,
    analyze_family_ablations,
    analyze_locked_holdout,
    benjamini_hochberg,
    build_research_plan,
    catalog_benjamini_hochberg,
    evaluate_stage_once,
    freeze_confirmation_spec,
    initialize_research_state,
    load_partition_signal_metric,
    paired_hac_test,
    read_research_state,
    validate_research_plan,
)
from quant_pipeline.exposure import stage_exposure_fields
from quant_pipeline.factors import FACTOR_FAMILIES, ORIGINAL_RESEARCH_CANDIDATES
from quant_pipeline.io import sha256_file
from quant_pipeline.metrics import max_drawdown


def make_plan():
    dates = pd.bdate_range("2025-01-02", periods=258)
    return build_research_plan(
        dates,
        discovery_end=dates[125],
        confirmation_end=dates[190],
        plan_id="frozen-factor-study",
        base_config_sha256="f" * 64,
    )


def stage_series(plan, stage, values):
    index = pd.DatetimeIndex(plan["partitions"][stage]["sessions"], name="datetime")
    if np.isscalar(values):
        values = np.full(len(index), float(values))
    return pd.Series(values, index=index, dtype=float)


def stage_evidence(
    plan,
    stage,
    signal_values,
    *,
    net_values=None,
    factor_names=(),
    raw_factor_values=None,
    benchmark_values=None,
):
    signal = stage_series(plan, stage, signal_values)
    index = pd.DatetimeIndex(
        plan["partitions"][stage]["portfolio_evaluation_sessions"], name="datetime"
    )
    if net_values is None:
        net_values = np.sin(np.arange(len(index)) / 9.0) * 0.001
    if np.isscalar(net_values):
        net_values = np.full(len(index), float(net_values))
    strategy_net = pd.Series(net_values, index=index, dtype=float, name="strategy_net")
    if benchmark_values is None:
        benchmark_values = np.full(len(index), -0.0001)
    if np.isscalar(benchmark_values):
        benchmark_values = np.full(len(index), float(benchmark_values))
    benchmark = pd.Series(benchmark_values, index=index, dtype=float, name="benchmark")
    terminal = 20_000.0 * float((1.0 + strategy_net).prod())
    benchmark_terminal = 20_000.0 * float((1.0 + benchmark).prod())
    factor_by_name = {factor.name: factor for factor in ORIGINAL_RESEARCH_CANDIDATES}
    raw_metrics = {}
    for position, name in enumerate(factor_names):
        if raw_factor_values is None:
            raw = factor_by_name[name].direction * (
                0.02 + np.sin(np.arange(len(signal)) / (7.0 + position)) * 0.001
            )
        elif isinstance(raw_factor_values, dict):
            raw = raw_factor_values[name]
        else:
            raw = raw_factor_values
        raw_metrics[name] = stage_series(plan, stage, raw)

    folds = []
    for fold in plan["partitions"][stage]["research_folds"]:
        fold_number = int(fold["fold"])
        fold_start = (fold_number - 1) * 21
        fold_end = min(fold_start + 21, len(strategy_net))
        fold_net = strategy_net.iloc[fold_start:fold_end]
        fold_benchmark = benchmark.iloc[fold_start:fold_end]
        fold_terminal = 20_000.0 * float((1.0 + fold_net).prod())
        fold_benchmark_terminal = 20_000.0 * float((1.0 + fold_benchmark).prod())
        folds.append(
            {
                "fold": fold["fold"],
                "signal_start": fold["signal_start"],
                "signal_end": fold["signal_end"],
                "evaluation_start": fold["evaluation_start"],
                "evaluation_end": fold["evaluation_end"],
                "complete_for_gate": fold["complete_for_gate"],
                "initial_account": 20_000.0,
                "stress_slippage_bps_per_side": 10,
                "engine": "raw_share_daily_v1",
                "terminal_account": fold_terminal,
                "benchmark_terminal_account": fold_benchmark_terminal,
                "single_etf_abs_contribution_share": 0.20,
                "single_etf_abs_contribution_symbol": "SH510300",
                "single_etf_abs_contribution_numerator_cny": 20.0,
                "single_etf_abs_contribution_denominator_cny": 100.0,
            }
        )
    return {
        "rank_ic": signal,
        "raw_factor_rank_ic": raw_metrics,
        "strategy_net": strategy_net,
        "benchmark": benchmark,
        "portfolio": {
            "benchmark": "SH510300",
            "account_currency": "CNY",
            "initial_account": 20_000.0,
            "stress_slippage_bps_per_side": 10,
            "engine": "raw_share_daily_v1",
            "alignment_method": "initial_cost_compounded_into_first_realized_return",
            "evaluation_sessions_sha256": plan["partitions"][stage][
                "portfolio_evaluation_sessions_sha256"
            ],
            "terminal_account": terminal,
            "benchmark_terminal_account": benchmark_terminal,
            "intent_fill_rate": 1.0,
            "notional_fill_rate": 1.0,
            "zero_fill_intent_rate": 0.0,
            "strategy_max_drawdown": max_drawdown(strategy_net),
            "single_etf_abs_contribution_share": 0.20,
            "single_etf_abs_contribution_symbol": "SH510300",
            "single_etf_abs_contribution_numerator_cny": 20.0,
            "single_etf_abs_contribution_denominator_cny": 100.0,
            "research_folds": folds,
        },
    }


def confirmation_criteria(value):
    return {
        "rank_ic_mean_difference_above_minimum": value,
        "rank_ic_one_sided_p_value_below_alpha": value,
        "strategy_net_mean_difference_above_minimum": value,
        "strategy_net_one_sided_p_value_below_alpha": value,
        "terminal_account_improvement_positive": value,
        "terminal_relative_wealth_improvement_positive": value,
        "candidate_terminal_account_not_below_initial": value,
        "candidate_execution_quality_passed": value,
        "baseline_execution_quality_passed": value,
        "candidate_beats_benchmark_at_10bps": value,
        "candidate_max_drawdown_within_limit": value,
        "paired_complete_fold_majority": value,
        "single_etf_abs_contribution_share_within_limit": value,
        "single_fold_abs_incremental_pnl_share_within_limit": value,
        "all_signed_raw_factor_rank_ic_positive": value,
        "all_signed_raw_factor_rank_ic_p_values_below_alpha": value,
        "all_signed_raw_factor_fold_majorities": value,
    }


class ResearchPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_explicit_and_does_not_claim_runs(self):
        first = make_plan()
        second = make_plan()
        self.assertEqual(first, second)
        self.assertEqual(tuple(first["partitions"]), RESEARCH_STAGES)
        self.assertEqual(len(first["family_ablations"]), FAMILY_ABLATION_COUNT)
        self.assertEqual(len(first["factor_hypotheses"]), CATALOG_HYPOTHESIS_COUNT)
        self.assertTrue(all(item["status"] == "not_run" for item in first["family_ablations"]))
        self.assertTrue(all(item["status"] == "not_run" for item in first["factor_hypotheses"]))
        self.assertEqual(first["experiment_status"], "not_run")
        self.assertEqual(first["label_horizon_bars"], 2)
        self.assertEqual(first["execution_evidence"]["initial_account"], 20_000.0)
        self.assertEqual(
            first["execution_evidence"]["required_stress_slippage_bps_per_side"], 10
        )
        discovery = first["partitions"]["discovery"]
        confirmation = first["partitions"]["confirmation"]
        self.assertEqual(discovery["label_maturity_sessions"], [
            pd.bdate_range("2025-01-02", periods=258)[126].date().isoformat(),
            pd.bdate_range("2025-01-02", periods=258)[127].date().isoformat(),
        ])
        self.assertEqual(
            confirmation["start"],
            pd.bdate_range("2025-01-02", periods=258)[128].date().isoformat(),
        )
        self.assertTrue(
            set(discovery["label_maturity_sessions"]).isdisjoint(confirmation["sessions"])
        )
        self.assertNotIn("run_id", json.dumps(first))
        self.assertEqual(validate_research_plan(first), first)

    def test_plan_rejects_too_short_or_nonmember_partitions(self):
        dates = pd.bdate_range("2025-01-02", periods=257)
        with self.assertRaisesRegex(ValueError, "locked_holdout requires at least"):
            build_research_plan(
                dates,
                discovery_end=dates[125],
                confirmation_end=dates[190],
                plan_id="short",
                base_config_sha256="f" * 64,
            )
        with self.assertRaisesRegex(ValueError, "members"):
            build_research_plan(
                pd.bdate_range("2025-01-02", periods=258),
                discovery_end="2025-01-04",
                confirmation_end="2025-10-01",
                plan_id="bad-cut",
                base_config_sha256="f" * 64,
            )

    def test_plan_tampering_is_detected(self):
        plan = make_plan()
        plan["partitions"]["locked_holdout"]["end"] = "2099-01-01"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_research_plan(plan)


class BenjaminiHochbergTests(unittest.TestCase):
    def test_standard_step_up_and_adjusted_values(self):
        result = benjamini_hochberg(
            [("a", 0.01), ("b", 0.03), ("c", 0.02), ("d", 0.20)], q=0.05
        )
        records = {item["hypothesis_id"]: item for item in result["results"]}
        self.assertEqual(result["rejected_count"], 3)
        self.assertEqual(result["cutoff_p_value"], 0.03)
        self.assertTrue(all(records[name]["rejected"] for name in ("a", "b", "c")))
        self.assertFalse(records["d"]["rejected"])
        self.assertAlmostEqual(records["a"]["q_value"], 0.04)
        self.assertAlmostEqual(records["b"]["q_value"], 0.04)

    def test_catalog_wrapper_requires_exactly_eighteen(self):
        names = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
        values = {name: 0.5 for name in names}
        result = catalog_benjamini_hochberg(values)
        self.assertEqual(result["hypothesis_count"], 18)
        self.assertEqual(result["fdr_q"], DISCOVERY_FDR_Q)
        del values[names[0]]
        with self.assertRaisesRegex(ValueError, "exactly the eighteen"):
            catalog_benjamini_hochberg(values)

    def test_invalid_probabilities_fail_closed(self):
        for value in (-0.1, 1.1, float("nan"), True):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    benjamini_hochberg([("x", value)])


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan()

    def test_paired_test_requires_exact_partition_and_missing_mask(self):
        baseline = stage_series(self.plan, "discovery", np.sin(np.arange(126)) * 0.01)
        candidate = baseline + 0.002 + np.cos(np.arange(126)) * 0.0002
        result = paired_hac_test(
            baseline, candidate, partition=self.plan["partitions"]["discovery"]
        )
        self.assertGreater(result["mean_difference"], 0)
        self.assertLess(result["one_sided_p_value"], 0.05)
        with self.assertRaisesRegex(ValueError, "dates must equal"):
            paired_hac_test(
                baseline.iloc[:-1],
                candidate.iloc[:-1],
                partition=self.plan["partitions"]["discovery"],
            )
        candidate.iloc[3] = np.nan
        with self.assertRaisesRegex(ValueError, "missing-value masks"):
            paired_hac_test(
                baseline, candidate, partition=self.plan["partitions"]["discovery"]
            )

    def test_five_family_screen_is_discovery_only(self):
        baseline_signal = stage_series(self.plan, "discovery", np.sin(np.arange(126)) * 0.01)
        baseline = stage_evidence(self.plan, "discovery", baseline_signal)
        metrics = {
            family: stage_evidence(
                self.plan,
                "discovery",
                baseline_signal + (position + 1) * 0.0001 + np.cos(np.arange(126)) * 0.00005,
                net_values=baseline["strategy_net"] + (position + 1) * 0.00001,
                factor_names=tuple(
                    factor.name
                    for factor in ORIGINAL_RESEARCH_CANDIDATES
                    if factor.family == family
                ),
            )
            for position, family in enumerate(FACTOR_FAMILIES)
        }
        result = analyze_family_ablations(baseline, metrics, self.plan)
        self.assertEqual(result["family_count"], 5)
        self.assertEqual(result["scope"], "discovery_only_family_screen")
        self.assertNotIn("confirmed", json.dumps(result))
        del metrics[FACTOR_FAMILIES[0]]
        with self.assertRaisesRegex(ValueError, "exactly the five"):
            analyze_family_ablations(baseline, metrics, self.plan)

    def test_eighteen_factor_discovery_applies_catalog_bh(self):
        baseline_signal = stage_series(self.plan, "discovery", np.sin(np.arange(126)) * 0.01)
        baseline = stage_evidence(self.plan, "discovery", baseline_signal)
        names = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
        metrics = {}
        for position, name in enumerate(names):
            effect = 0.002 if position < 3 else 0.00001
            noise = np.cos(np.arange(126) * (position + 1) / 17) * 0.0002
            net_effect = 0.0002 if position < 3 else -0.00001
            metrics[name] = stage_evidence(
                self.plan,
                "discovery",
                baseline_signal + effect + noise,
                net_values=baseline["strategy_net"]
                + net_effect
                + np.cos(np.arange(len(baseline["strategy_net"])) * (position + 1) / 19)
                * 0.00002,
                factor_names=(name,),
            )
        result = analyze_factor_discovery(baseline, metrics, self.plan)
        self.assertEqual(result["hypothesis_count"], 18)
        self.assertEqual(result["bh"]["hypothesis_count"], 18)
        self.assertEqual(result["stage"], "discovery")
        self.assertIn("confirmation has not occurred", result["claim"])

    def test_joint_discovery_rejects_rank_ic_only_and_terminal_loss(self):
        signal = stage_series(self.plan, "discovery", np.sin(np.arange(126)) * 0.01)
        baseline = stage_evidence(self.plan, "discovery", signal)
        names = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
        candidates = {}
        for position, name in enumerate(names):
            rank_effect = 0.002 if position < 2 else -0.0002
            net_effect = -0.0002 if position == 0 else (0.0002 if position == 1 else -0.0002)
            candidates[name] = stage_evidence(
                self.plan,
                "discovery",
                signal + rank_effect + np.cos(np.arange(126) / 11.0) * 0.00002,
                net_values=baseline["strategy_net"]
                + net_effect
                + np.cos(np.arange(126) / 13.0) * 0.00002,
                factor_names=(name,),
            )
        result = analyze_factor_discovery(baseline, candidates, self.plan)
        records = {record["hypothesis_id"]: record for record in result["results"]}
        self.assertLess(records[names[0]]["rank_ic"]["one_sided_p_value"], 0.05)
        self.assertGreater(records[names[0]]["joint_p_value"], 0.95)
        self.assertNotIn(names[0], result["selected_factor_names"])
        self.assertIn(names[1], result["selected_factor_names"])
        self.assertEqual(
            result["multiplicity_applied_to"],
            DISCOVERY_MULTIPLICITY_APPLIED_TO,
        )

    def test_discovery_rejects_relative_improvement_that_still_loses_capital(self):
        signal = stage_series(self.plan, "discovery", np.sin(np.arange(126)) * 0.01)
        baseline_net = np.full(126, (18_000.0 / 20_000.0) ** (1.0 / 126) - 1.0)
        candidate_net = np.full(126, (18_780.0 / 20_000.0) ** (1.0 / 126) - 1.0)
        baseline = stage_evidence(
            self.plan, "discovery", signal, net_values=baseline_net
        )
        candidates = {}
        names = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
        for position, name in enumerate(names):
            candidates[name] = stage_evidence(
                self.plan,
                "discovery",
                signal
                + (0.002 if position == 0 else -0.0002)
                + np.cos(np.arange(126) / 11.0) * 0.00002,
                net_values=(
                    candidate_net
                    + np.cos(np.arange(126) / 13.0) * 0.0000001
                    if position == 0
                    else baseline_net
                    - 0.0001
                    + np.cos(np.arange(126) / 13.0) * 0.0000001
                ),
                factor_names=(name,),
            )

        result = analyze_factor_discovery(baseline, candidates, self.plan)
        record = next(
            item for item in result["results"] if item["hypothesis_id"] == names[0]
        )
        self.assertTrue(record["selection_criteria"]["joint_bh_rejected"])
        self.assertTrue(
            record["selection_criteria"]["terminal_account_improvement_positive"]
        )
        self.assertAlmostEqual(record["terminal"]["candidate_terminal_account"], 18_780.0, delta=1.0)
        self.assertFalse(
            record["selection_criteria"]["candidate_terminal_account_not_below_initial"]
        )
        self.assertNotIn(names[0], result["selected_factor_names"])

    def test_discovery_rejects_low_execution_quality_without_aborting_battery(self):
        signal = stage_series(self.plan, "discovery", np.sin(np.arange(126)) * 0.01)
        baseline = stage_evidence(self.plan, "discovery", signal)
        names = [factor.name for factor in ORIGINAL_RESEARCH_CANDIDATES]
        candidates = {}
        for position, name in enumerate(names):
            candidate = stage_evidence(
                self.plan,
                "discovery",
                signal
                + (0.002 if position == 0 else -0.0002)
                + np.cos(np.arange(126) / 11.0) * 0.00002,
                net_values=baseline["strategy_net"]
                + (0.0002 if position == 0 else -0.0002)
                + np.cos(np.arange(126) / 13.0) * 0.00002,
                factor_names=(name,),
            )
            if position == 0:
                candidate["portfolio"]["intent_fill_rate"] = 0.94
            candidates[name] = candidate

        result = analyze_factor_discovery(baseline, candidates, self.plan)
        record = next(
            item for item in result["results"] if item["hypothesis_id"] == names[0]
        )
        self.assertEqual(result["hypothesis_count"], CATALOG_HYPOTHESIS_COUNT)
        self.assertTrue(record["selection_criteria"]["joint_bh_rejected"])
        self.assertTrue(
            record["selection_criteria"]["candidate_terminal_account_not_below_initial"]
        )
        self.assertFalse(
            record["selection_criteria"]["candidate_execution_quality_passed"]
        )
        self.assertNotIn(names[0], result["selected_factor_names"])

    def test_confirmation_requires_both_metrics_and_terminal_improvement(self):
        name = ORIGINAL_RESEARCH_CANDIDATES[0].name
        signal = stage_series(self.plan, "confirmation", np.sin(np.arange(63)) * 0.01)
        baseline = stage_evidence(self.plan, "confirmation", signal)
        rank_only = stage_evidence(
            self.plan,
            "confirmation",
            signal + 0.002 + np.cos(np.arange(63) / 7.0) * 0.0002,
            net_values=baseline["strategy_net"]
            - 0.0002
            + np.cos(np.arange(63) / 8.0) * 0.00002,
            factor_names=(name,),
        )
        result = analyze_confirmation(
            baseline,
            rank_only,
            self.plan,
            frozen_spec_sha256="a" * 64,
            candidate_factor_names=(name,),
        )
        self.assertFalse(result["confirmation_passed"])
        self.assertFalse(result["criteria"]["strategy_net_mean_difference_above_minimum"])
        self.assertFalse(result["criteria"]["terminal_account_improvement_positive"])
        self.assertEqual(result["initial_account"], 20_000.0)
        self.assertEqual(result["stress_slippage_bps_per_side"], 10)

    def test_confirmation_rejects_relative_improvement_that_still_loses_capital(self):
        name = ORIGINAL_RESEARCH_CANDIDATES[0].name
        signal = stage_series(self.plan, "confirmation", np.sin(np.arange(63)) * 0.01)
        baseline_net = np.full(63, (18_000.0 / 20_000.0) ** (1.0 / 63) - 1.0)
        candidate_net = np.full(63, (18_780.0 / 20_000.0) ** (1.0 / 63) - 1.0)
        baseline = stage_evidence(
            self.plan, "confirmation", signal, net_values=baseline_net
        )
        candidate = stage_evidence(
            self.plan,
            "confirmation",
            signal + 0.002 + np.cos(np.arange(63) / 7.0) * 0.0002,
            net_values=candidate_net + np.cos(np.arange(63) / 8.0) * 0.0000001,
            factor_names=(name,),
        )

        result = analyze_confirmation(
            baseline,
            candidate,
            self.plan,
            frozen_spec_sha256="a" * 64,
            candidate_factor_names=(name,),
        )
        self.assertTrue(result["criteria"]["rank_ic_one_sided_p_value_below_alpha"])
        self.assertTrue(result["criteria"]["strategy_net_one_sided_p_value_below_alpha"])
        self.assertTrue(result["criteria"]["terminal_account_improvement_positive"])
        self.assertAlmostEqual(
            result["tests"]["terminal"]["candidate_terminal_account"], 18_780.0, delta=1.0
        )
        self.assertFalse(
            result["criteria"]["candidate_terminal_account_not_below_initial"]
        )
        self.assertFalse(result["confirmation_passed"])

    def test_execution_quality_gates_fail_closed(self):
        name = ORIGINAL_RESEARCH_CANDIDATES[0].name
        signal = stage_series(self.plan, "confirmation", np.sin(np.arange(63)) * 0.01)
        baseline = stage_evidence(self.plan, "confirmation", signal)
        candidate = stage_evidence(
            self.plan,
            "confirmation",
            signal + 0.002 + np.cos(np.arange(63) / 7.0) * 0.0002,
            net_values=baseline["strategy_net"] + 0.0002,
            factor_names=(name,),
        )
        for field, value in (
            ("intent_fill_rate", 0.949999),
            ("notional_fill_rate", 0.949999),
            ("zero_fill_intent_rate", 0.050001),
        ):
            poor_execution = {
                **candidate,
                "portfolio": {**candidate["portfolio"], field: value},
            }
            with self.subTest(field=field):
                result = analyze_confirmation(
                    baseline,
                    poor_execution,
                    self.plan,
                    frozen_spec_sha256="a" * 64,
                    candidate_factor_names=(name,),
                )
                self.assertFalse(
                    result["criteria"]["candidate_execution_quality_passed"]
                )
                self.assertFalse(result["confirmation_passed"])

    def test_confirmation_and_holdout_use_distinct_exact_partitions(self):
        name = ORIGINAL_RESEARCH_CANDIDATES[0].name
        confirmation_signal = stage_series(self.plan, "confirmation", np.sin(np.arange(63)) * 0.01)
        confirmation_baseline = stage_evidence(self.plan, "confirmation", confirmation_signal)
        confirmation_candidate = stage_evidence(
            self.plan,
            "confirmation",
            confirmation_signal + 0.002 + np.cos(np.arange(63)) * 0.0002,
            net_values=confirmation_baseline["strategy_net"]
            + 0.0002
            + np.cos(np.arange(63) / 8.0) * 0.00002,
            factor_names=(name,),
        )
        result = analyze_confirmation(
            confirmation_baseline,
            confirmation_candidate,
            self.plan,
            frozen_spec_sha256="a" * 64,
            candidate_factor_names=(name,),
        )
        self.assertTrue(result["confirmation_passed"])
        holdout_signal = stage_series(self.plan, "locked_holdout", np.sin(np.arange(63)) * 0.01)
        holdout_baseline = stage_evidence(self.plan, "locked_holdout", holdout_signal)
        holdout_candidate = stage_evidence(
            self.plan,
            "locked_holdout",
            holdout_signal + 0.002 + np.cos(np.arange(63)) * 0.0002,
            net_values=holdout_baseline["strategy_net"]
            + 0.0002
            + np.cos(np.arange(63) / 8.0) * 0.00002,
            factor_names=(name,),
        )
        result = analyze_locked_holdout(
            holdout_baseline,
            holdout_candidate,
            self.plan,
            frozen_spec_sha256="a" * 64,
            candidate_factor_names=(name,),
        )
        self.assertTrue(result["locked_holdout_passed"])
        self.assertEqual(result["stage"], "locked_holdout")


class ArtifactPartitionTests(unittest.TestCase):
    def test_predicate_loader_requires_exact_dates_and_frozen_checksum(self):
        plan = make_plan()
        all_dates = pd.DatetimeIndex(
            [
                date
                for stage in RESEARCH_STAGES
                for date in plan["partitions"][stage]["sessions"]
            ],
            name="datetime",
        )
        frame = pd.DataFrame({"rank_ic": np.arange(len(all_dates), dtype=float)}, index=all_dates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.parquet"
            path.write_bytes(b"frozen parquet placeholder")
            discovery = frame.loc[
                plan["partitions"]["discovery"]["start"] : plan["partitions"]["discovery"]["end"]
            ]
            with patch("quant_pipeline.factor_research.pd.read_parquet", return_value=discovery) as read:
                values = load_partition_signal_metric(
                    path,
                    plan["partitions"]["discovery"],
                    expected_sha256=sha256_file(path),
                )
            kwargs = read.call_args.kwargs
            self.assertEqual(kwargs["columns"], ["rank_ic"])
            self.assertEqual(kwargs["engine"], "pyarrow")
            self.assertEqual(kwargs["filters"][0][0:2], ("datetime", ">="))
            self.assertEqual(len(values), 126)
            self.assertEqual(values.index.max().date().isoformat(), plan["partitions"]["discovery"]["end"])
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_partition_signal_metric(
                    path,
                    plan["partitions"]["discovery"],
                    expected_sha256="0" * 64,
                )


class OneShotStateTests(unittest.TestCase):
    _discovery_cache = {}

    def setUp(self):
        self.plan = make_plan()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "factor_research_state.json"
        initialize_research_state(self.path, self.plan)

    def _discovery_result(self, selected=None):
        selected = tuple(selected or ())
        cached = self._discovery_cache.get(selected)
        if cached is not None:
            return deepcopy(cached)
        baseline_signal = stage_series(
            self.plan, "discovery", np.sin(np.arange(126)) * 0.01
        )
        baseline = stage_evidence(self.plan, "discovery", baseline_signal)
        family_evidence = {}
        for position, family in enumerate(FACTOR_FAMILIES):
            factor_names = tuple(
                factor.name
                for factor in ORIGINAL_RESEARCH_CANDIDATES
                if factor.family == family
            )
            family_evidence[family] = stage_evidence(
                self.plan,
                "discovery",
                baseline_signal
                + 0.0001 * (position + 1)
                + np.cos(np.arange(126) / (9.0 + position)) * 0.00002,
                net_values=baseline["strategy_net"] + 0.00001 * (position + 1),
                factor_names=factor_names,
            )
        factor_evidence = {}
        for position, factor in enumerate(ORIGINAL_RESEARCH_CANDIDATES):
            passed = factor.name in selected
            rank_effect = 0.002 if passed else -0.0002
            strategy_effect = 0.0002 if passed else -0.0002
            factor_evidence[factor.name] = stage_evidence(
                self.plan,
                "discovery",
                baseline_signal
                + rank_effect
                + np.cos(np.arange(126) / (8.0 + position)) * 0.00002,
                net_values=baseline["strategy_net"]
                + strategy_effect
                + np.cos(np.arange(126) / (10.0 + position)) * 0.000002,
                factor_names=(factor.name,),
            )
        result = analyze_discovery(
            baseline, family_evidence, factor_evidence, self.plan
        )
        self.assertEqual(result["selected_factor_names"], list(selected))
        self._discovery_cache[selected] = deepcopy(result)
        return result

    def _stage_result(
        self, stage, factor_names, *, passed, frozen_spec_sha256
    ):
        observations = len(self.plan["partitions"][stage]["sessions"])
        baseline_signal = stage_series(
            self.plan, stage, np.sin(np.arange(observations)) * 0.01
        )
        baseline = stage_evidence(self.plan, stage, baseline_signal)
        direction = 1.0 if passed else -1.0
        candidate = stage_evidence(
            self.plan,
            stage,
            baseline_signal
            + direction * 0.002
            + np.cos(np.arange(observations) / 7.0) * 0.00002,
            net_values=baseline["strategy_net"]
            + direction * 0.0002
            + np.cos(np.arange(observations) / 8.0) * 0.000002,
            factor_names=tuple(factor_names),
        )
        analyzer = (
            analyze_confirmation if stage == "confirmation" else analyze_locked_holdout
        )
        result = analyzer(
            baseline,
            candidate,
            self.plan,
            frozen_spec_sha256=frozen_spec_sha256,
            candidate_factor_names=tuple(factor_names),
        )
        decision = (
            result["confirmation_passed"]
            if stage == "confirmation"
            else result["locked_holdout_passed"]
        )
        self.assertIs(decision, passed)
        return result

    def _new_state_path(self, name):
        path = Path(self.temporary_directory.name) / name
        initialize_research_state(path, self.plan)
        return path

    def test_failure_consumes_stage_and_cannot_be_retried(self):
        def fail(_partition):
            raise RuntimeError("analysis failed")

        with self.assertRaisesRegex(RuntimeError, "analysis failed"):
            evaluate_stage_once(self.path, self.plan, "discovery", fail)
        state = read_research_state(self.path, self.plan)
        self.assertEqual(state["stages"]["discovery"]["status"], "failed")
        self.assertEqual(state["stages"]["discovery"]["attempts"], 1)
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            evaluate_stage_once(
                self.path, self.plan, "discovery", lambda _: self._discovery_result()
            )

    def test_order_freeze_and_single_holdout_access(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        discovery = evaluate_stage_once(
            self.path,
            self.plan,
            "discovery",
            lambda _: self._discovery_result([selected]),
        )
        self.assertEqual(discovery["selected_factor_names"], [selected])
        with self.assertRaisesRegex(RuntimeError, "specification has not been frozen"):
            evaluate_stage_once(
                self.path,
                self.plan,
                "confirmation",
                lambda _: {
                    "stage": "confirmation",
                    "plan_sha256": self.plan["plan_sha256"],
                    "partition_sha256": self.plan["partitions"]["confirmation"]["sessions_sha256"],
                    **stage_exposure_fields(self.plan, "confirmation"),
                    "confirmation_passed": True,
                },
            )
        spec_sha256 = freeze_confirmation_spec(
            self.path,
            self.plan,
            selected_factor_names=[selected],
            frozen_spec={"config_sha256": "a" * 64, "model": "lightgbm"},
        )
        confirmation = self._stage_result(
            "confirmation",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        evaluate_stage_once(
            self.path,
            self.plan,
            "confirmation",
            lambda _: confirmation,
        )
        holdout = self._stage_result(
            "locked_holdout",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        evaluate_stage_once(
            self.path,
            self.plan,
            "locked_holdout",
            lambda _: holdout,
        )
        state = read_research_state(self.path, self.plan)
        self.assertEqual(state["stages"]["locked_holdout"]["attempts"], 1)
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            evaluate_stage_once(
                self.path,
                self.plan,
                "locked_holdout",
                lambda _: {},
            )

    def test_freeze_requires_all_and_only_joint_selected_factors(self):
        selected = [
            ORIGINAL_RESEARCH_CANDIDATES[0].name,
            ORIGINAL_RESEARCH_CANDIDATES[1].name,
        ]
        evaluate_stage_once(
            self.path,
            self.plan,
            "discovery",
            lambda _: self._discovery_result(selected),
        )
        with self.assertRaisesRegex(ValueError, "complete recorded joint-selected set"):
            freeze_confirmation_spec(
                self.path,
                self.plan,
                selected_factor_names=selected[:1],
                frozen_spec={"config_sha256": "c" * 64},
            )

    def test_failed_confirmation_keeps_holdout_locked(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        evaluate_stage_once(
            self.path,
            self.plan,
            "discovery",
            lambda _: self._discovery_result([selected]),
        )
        spec_sha256 = freeze_confirmation_spec(
            self.path,
            self.plan,
            selected_factor_names=[selected],
            frozen_spec={"config_sha256": "b" * 64},
        )
        failed = self._stage_result(
            "confirmation",
            [selected],
            passed=False,
            frozen_spec_sha256=spec_sha256,
        )
        evaluate_stage_once(
            self.path,
            self.plan,
            "confirmation",
            lambda _: failed,
        )
        with self.assertRaisesRegex(RuntimeError, "must remain unopened"):
            evaluate_stage_once(
                self.path,
                self.plan,
                "locked_holdout",
                lambda _: {},
            )

    def test_stage_results_reject_incomplete_frozen_criteria(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        incomplete = self._discovery_result([selected])
        del incomplete["factor_discovery"]["results"][0]["selection_criteria"][
            "candidate_execution_quality_passed"
        ]
        with self.assertRaisesRegex(ValueError, "invalid joint selection criteria"):
            evaluate_stage_once(
                self.path, self.plan, "discovery", lambda _: incomplete
            )

    def test_discovery_state_rejects_single_field_joint_bh_and_criteria_tampering(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        original = self._discovery_result([selected])

        def tamper_joint(result):
            result["factor_discovery"]["results"][0]["joint_iut"][
                "one_sided_p_value"
            ] = 0.5

        def tamper_bh(result):
            result["factor_discovery"]["results"][0]["bh_q_value"] = 1.0

        def tamper_criteria(result):
            result["factor_discovery"]["results"][0]["selection_criteria"][
                "candidate_execution_quality_passed"
            ] = False

        for position, (name, mutator) in enumerate(
            (
                ("joint", tamper_joint),
                ("bh", tamper_bh),
                ("criteria", tamper_criteria),
            )
        ):
            with self.subTest(field=name):
                path = self._new_state_path(f"tampered-discovery-{position}.json")
                tampered = deepcopy(original)
                mutator(tampered)
                with self.assertRaises(ValueError):
                    evaluate_stage_once(
                        path, self.plan, "discovery", lambda _: tampered
                    )
                state = read_research_state(path, self.plan)
                self.assertEqual(state["stages"]["discovery"]["status"], "failed")

    def test_confirmation_state_rejects_single_field_metric_tampering(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        evaluate_stage_once(
            self.path,
            self.plan,
            "discovery",
            lambda _: self._discovery_result([selected]),
        )
        spec_sha256 = freeze_confirmation_spec(
            self.path,
            self.plan,
            selected_factor_names=[selected],
            frozen_spec={"config_sha256": "1" * 64},
        )
        tampered = self._stage_result(
            "confirmation",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        tampered["tests"]["rank_ic"]["mean_difference"] += 0.001
        with self.assertRaisesRegex(ValueError, "HAC result does not reconcile"):
            evaluate_stage_once(
                self.path, self.plan, "confirmation", lambda _: tampered
            )
        state = read_research_state(self.path, self.plan)
        self.assertEqual(state["stages"]["confirmation"]["status"], "failed")

    def test_locked_holdout_state_rejects_single_field_terminal_tampering(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        evaluate_stage_once(
            self.path,
            self.plan,
            "discovery",
            lambda _: self._discovery_result([selected]),
        )
        spec_sha256 = freeze_confirmation_spec(
            self.path,
            self.plan,
            selected_factor_names=[selected],
            frozen_spec={"config_sha256": "2" * 64},
        )
        confirmation = self._stage_result(
            "confirmation",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        evaluate_stage_once(
            self.path, self.plan, "confirmation", lambda _: confirmation
        )
        tampered = self._stage_result(
            "locked_holdout",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        tampered["tests"]["terminal"]["account_improvement"] += 1.0
        with self.assertRaisesRegex(ValueError, "terminal result does not reconcile"):
            evaluate_stage_once(
                self.path, self.plan, "locked_holdout", lambda _: tampered
            )
        state = read_research_state(self.path, self.plan)
        self.assertEqual(state["stages"]["locked_holdout"]["status"], "failed")

    def test_confirmation_result_rejects_incomplete_frozen_criteria(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        evaluate_stage_once(
            self.path,
            self.plan,
            "discovery",
            lambda _: self._discovery_result([selected]),
        )
        spec_sha256 = freeze_confirmation_spec(
            self.path,
            self.plan,
            selected_factor_names=[selected],
            frozen_spec={"config_sha256": "d" * 64},
        )
        incomplete = self._stage_result(
            "confirmation",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        del incomplete["criteria"]["candidate_execution_quality_passed"]
        with self.assertRaisesRegex(ValueError, "frozen joint rule"):
            evaluate_stage_once(
                self.path,
                self.plan,
                "confirmation",
                lambda _: incomplete,
            )

    def test_locked_holdout_result_rejects_incomplete_frozen_criteria(self):
        selected = ORIGINAL_RESEARCH_CANDIDATES[0].name
        evaluate_stage_once(
            self.path,
            self.plan,
            "discovery",
            lambda _: self._discovery_result([selected]),
        )
        spec_sha256 = freeze_confirmation_spec(
            self.path,
            self.plan,
            selected_factor_names=[selected],
            frozen_spec={"config_sha256": "e" * 64},
        )
        confirmation = self._stage_result(
            "confirmation",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        evaluate_stage_once(
            self.path,
            self.plan,
            "confirmation",
            lambda _: confirmation,
        )
        incomplete = self._stage_result(
            "locked_holdout",
            [selected],
            passed=True,
            frozen_spec_sha256=spec_sha256,
        )
        del incomplete["criteria"]["candidate_terminal_account_not_below_initial"]
        with self.assertRaisesRegex(ValueError, "frozen joint rule"):
            evaluate_stage_once(
                self.path,
                self.plan,
                "locked_holdout",
                lambda _: incomplete,
            )

    def test_state_tampering_and_reinitialization_fail_closed(self):
        with self.assertRaises(FileExistsError):
            initialize_research_state(self.path, self.plan)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["stages"]["locked_holdout"]["status"] = "completed"
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_research_state(self.path, self.plan)


class StudentTApproximationTests(unittest.TestCase):
    def test_known_student_t_quantiles(self):
        self.assertAlmostEqual(_student_t_cdf(0.0, 5.0), 0.5, places=12)
        self.assertAlmostEqual(_student_t_cdf(1.0, 1.0), 0.75, places=12)
        self.assertAlmostEqual(_student_t_cdf(-1.0, 1.0), 0.25, places=12)
        self.assertAlmostEqual(_student_t_cdf(1.4759, 5.0), 0.9, places=4)
        self.assertAlmostEqual(_student_t_cdf(2.2281, 10.0), 0.975, places=4)
        self.assertAlmostEqual(_student_t_cdf(2.0, 1e7), 0.9772498680518208, places=8)

    def test_small_sample_hac_p_value_is_more_conservative_than_normal(self):
        from statistics import NormalDist

        normal_p = float(NormalDist().cdf(-1.96))
        small_sample_p = _one_sided_hac_p_value(1.96, 63, 5)
        self.assertGreater(small_sample_p, normal_p)
        self.assertLess(small_sample_p, 0.05)

    def test_paired_hac_test_rejects_below_minimum_observations(self):
        plan = make_plan()
        partition = plan["partitions"]["discovery"]
        short = stage_series(plan, "discovery", np.zeros(len(partition["sessions"])))[
            : MIN_PAIRED_OBSERVATIONS - 1
        ]
        short_partition = {**partition, "sessions": list(short.index.strftime("%Y-%m-%d"))}
        with self.assertRaisesRegex(ValueError, "at least"):
            paired_hac_test(short, short, partition=short_partition)


if __name__ == "__main__":
    unittest.main()
