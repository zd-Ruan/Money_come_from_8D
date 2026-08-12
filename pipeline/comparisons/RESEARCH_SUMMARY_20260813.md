# 2026-08-13 隔夜因子研究摘要

## 结论（先说重点）

本次新增 7 个 ORC 候选因子，并按 4 个 family 分别与 Alpha158 CPU 基线做配对实验。

严格配对判定下，**4 个候选 family 均未达到“可晋升”标准**（`comparison_status: not_improved`）。
因此当前不建议把这些新因子直接用于实盘/晋升；Alpha158 基线仍是这批实验里更稳的对照。

## 实验设置

- 基线：`baseline_cpu_20260813`（Alpha158，CPU）
- 候选：
  - `candidate_trend_crowding_cpu_20260813`
  - `candidate_volume_impact_cpu_20260813`
  - `candidate_price_volume_divergence_cpu_20260813`
  - `candidate_session_structure_cpu_20260813`
- 因子：本次新增 7 个
  - `ORC_TREND_RSQR_STRESS_20`
  - `ORC_TREND_ACCEL_GAP_10_30`
  - `ORC_VOLUME_CLIMAX_10`
  - `ORC_VOLUME_STABILITY_TREND_20`
  - `ORC_NET_VOLUME_PRESSURE_20`
  - `ORC_VOLUME_TREND_DIVERGENCE_10`
  - `ORC_INTRADAY_RANGE_RETURN_10`
- 窗口：测试期 2025-01-06 至 2026-08-11，387 个收益观察日，6 个完整 fold + 1 个残余 fold。
- 成本：单边佣金 3bps（最低 5 元）、单边基础滑点 5bps。

## 核心结果

| 候选 family | 期末相对财富 vs 基线 | IC vs 基线 | Rank IC vs 基线 | fold 胜率 | 判定 |
| --- | --- | --- | --- | --- | --- |
| trend_crowding | 1.1835 vs 1.3700（差） | 0.00831 vs 0.01037（差） | 0.01391 vs 0.01573（差） | 4/6 | not_improved |
| volume_impact | 1.1547 vs 1.3700（差） | 0.00885 vs 0.01037（差） | 0.01460 vs 0.01573（差） | 4/6 | not_improved |
| price_volume_divergence | 1.3206 vs 1.3700（略差） | 0.01255 vs 0.01037（升） | 0.02039 vs 0.01573（升） | 3/6 | not_improved |
| session_structure | 1.3833 vs 1.3700（升） | 0.00905 vs 0.01037（差） | 0.01440 vs 0.01573（差） | 4/6 | not_improved |

基线 Alpha158 的 5bps 滑点期末净值为 35,415.73 元（本金 20,000 元，净收益约 77.1%）。

## 观察与解读

1. **price_volume_divergence** 是唯一在 IC 和 Rank IC 上都超过基线的 family，但组合收益略差、fold 胜率只有 3/6，HAC 不显著，因此严格标准下不能声称改进。
2. **session_structure** 在期末相对财富和日收益上微幅超过基线，fold 胜率 4/6，但 IC/Rank IC 反而下降，HAC 不显著。
3. **trend_crowding / volume_impact** 多项指标低于基线，当前看更像引入噪声。
4. 所有候选的 `daily_return_hac_significant` 均未达到 1.96，说明这些增量在统计上还不足够可信。

## 下一步建议

- 对 `price_volume_divergence` 和 `session_structure` 做**单因子消融**，而不是整 family 一起加；先剔除拖累因子。
- 检查新增因子的缺失率/覆盖，确认 `Ref` 偏移和分母 `1e-12` 没有带来异常值。
- 继续解决之前的 point-in-time universe 和压力滑点非单调问题。
- 若单因子通过消融，再进入 `candidate_confirmation`，而不是直接晋升。

## 复现命令

```bash
cd /e/Code
PYTHONPATH="E:\Code\Money_come_from_8D\pipeline\src" \
  /d/annconda/envs/qlib/python.exe -m quant_pipeline.cli run \
  --config "E:\Code\Money_come_from_8D\pipeline\configs\<config>.yaml" \
  --run-id <run_id>
```

配对比较：

```bash
PYTHONPATH="E:\Code\Money_come_from_8D\pipeline\src" \
  /d/annconda/envs/qlib/python.exe -m quant_pipeline.cli compare \
  --baseline-run baseline_cpu_20260813 \
  --candidate-run <candidate_run_id>
```
