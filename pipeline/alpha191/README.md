# Alpha191 因子库（ETF 研究流水线）

在现有 ETF 研究流水线内实现了完整的 Alpha191 因子集：

- **Alpha001–101**：Kakushadze, *101 Formulaic Alphas*（arXiv:1601.00991，即 WorldQuant 101 公式）。
- **Alpha102–191**：国内量化社区流通的扩展 90 因子（聚宽/BigQuant 语法）。
- 全部 191 个因子均带规范公式文本（见 `alpha191.py` 各函数 docstring 与
  `factor_metadata_table()`），算子语义对齐聚宽（JoinQuant）风格。

## 代码位置

| 文件 | 作用 |
|---|---|
| `pipeline/src/quant_pipeline/alpha191.py` | 算子层 + 191 个因子函数 + 注册表 + 批计算 |
| `pipeline/src/quant_pipeline/alpha191_research.py` | 数据加载、批计算、parquet 落盘、IC 报告、CLI |
| `pipeline/configs/alpha191_build.yaml` | 构建配置 |
| `pipeline/tests/test_alpha191_ops.py` | 算子语义单元测试（13 项） |

## 运行

```powershell
# 仓库根目录（My_Quant），使用已安装 qlib 的环境
$env:PYTHONPATH = (Resolve-Path .\qlib\pipeline\src).Path

# 全量 191 因子 + IC 报告
C:\Exception\quant\python.exe -m quant_pipeline.alpha191_research `
  --config .\qlib\pipeline\configs\alpha191_build.yaml

# 只算部分因子
C:\Exception\quant\python.exe -m quant_pipeline.alpha191_research `
  --config .\qlib\pipeline\configs\alpha191_build.yaml --numbers 1,2,101,143

# 单元测试
C:\Exception\quant\python.exe -m pytest .\qlib\pipeline\tests\test_alpha191_ops.py -q
```

## 输出

运行结果写入 `qlib/pipeline/runs/alpha191/`：

- `factors/alpha001.parquet` … `alpha191.parquet`：每个因子一个宽表文件
  （index=date，columns=标的），全历史 2015–2026，便于直接复用/喂模型。
- `alpha191_metadata.csv`：因子号、族、最大回看窗口、公式文本。
- `alpha191_ic_report.csv` / `.md`：每个因子对 1 日 / 5 日前瞻收益的截面 IC、RankIC、
  t 统计量、IC 为正比例与覆盖率（全样本 + 最近 2 年）。
- `summary.json`：运行元信息与产物路径。

> 说明：默认不再合并 191 因子长表（约 6GB 内存），按因子宽表落盘。
> 如需小规模长表，可用 `quant_pipeline.alpha191.stack_factors({...})`。

## 算子语义约定

| 算子 | 语义 |
|---|---|
| `RANK(x)` | 截面百分位排名（按日期跨标的，平均并列） |
| `DELAY(x,n)` / `DELTA(x,n)` | `shift(n)` / `diff(n)` |
| `SUM/MEAN/STD/TS_MAX/TS_MIN` | 滚动窗（满窗，预热期 NaN） |
| `MAX(a,b)` / `MIN(a,b)` | 逐元素两序列取大 / 取小 |
| `CORR/COVARIANCE(x,y,n)` | 滚动相关 / 协方差；常数输入 → NaN（±inf 归一为 NaN） |
| `TS_RANK(x,n)` | 窗口内最新值的滚动百分位排名 |
| `SMA(x,n,m)` | 递归平滑均线 `Y_t=(m*X_t+(n-m)*Y_{t-1})/n` |
| `WMA/DECAYLINEAR(x,n)` | 线性衰减加权均线（权重 `i/(n(n+1)/2)`，满窗） |
| `REGBETA(x,y,n)` | `cov(x,y)/var(y)`（x 对 y 回归斜率） |
| `HIGHDAY/LOWDAY(x,n)` | 距最近一次窗口极值的交易日数（今天=0） |
| `IFELSE(cond,a,b)` | 条件取值，保留 DataFrame 一侧的 NaN |

## 与公开实现的差异（刻意设计，均已在 docstring 标注）

- **Alpha030**：原公式需要 MKT/SMB/HML 因子；本实现以基准 ETF（SH510300）
  收益率作为 MKT、SMB/HML 置 0，残差平方后做 WMA(20)。
- **Alpha075/149/181/182**：基准相关因子使用真实基准 ETF 的行情，
  而非参考实现中的截面均值近似。
- **Alpha029/144/150**：按公式用 `volume`/`amount`（参考实现误用 `log(volume)`）。
- **Alpha033/062**：按公式用 `volume`（参考实现误用 turnover）。
- **Alpha073**：按公式 `-1*(A-B)`；**Alpha164**：按公式补全括号。
- **Alpha166**：使用精确常数 `-20·19^1.5/(19·18)`。
- **Alpha143**：SELF 递归用向量化累积乘积实现（初值 1）；参考实现未实现该因子。
- **Alpha165/183**：`SUMAC` 按参考实现读作同窗口滚动和（该算子本身路径依赖）。

## 后续接入

- 因子 parquet 可直接喂给 `factor_research` 单因子消融框架做 LightGBM 增量验证，
  或作为 Alpha360 之外的备选特征族。
- 因子为复权价计算、未中性化；接入模型前建议先做行业 / 市值 / 流动性中性化
  （ETF 场景可对规模与成交额分组中性化）。
- IC 报告为研究口径（`research_only`）：universe 是当前快照，存在幸存者偏差，
  结论不能直接当作实盘承诺。
