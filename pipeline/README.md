# My Quant Pipeline

这是围绕现有 Qlib 数据和 Alpha158 构建的独立 ETF 研究 Pipeline。Qlib 仓库、行情数据和已有 MLflow 记录均保留不动。

## 可信度设计

- 原始数据、ETF 白名单、交易日历和特征文件生成版本化指纹；
- 上游 `validation_report.json` 纳入指纹和门禁；目录中保留的池外缓存会被单独统计，但不会混入 `t1_etf` 训练池；
- 训练、验证、测试之间按交易日执行 purge；
- 每个滚动窗口使用三个固定随机种子的 LightGBM 集成；
- 所有预测均为严格样本外预测，拼接后只执行一次连续回测；
- 回测只按最后一个已完整实现收益的信号日期截断；日期内不会用未来标签是否存在筛选 ETF，最新未实现日期的信号只保存、不计绩效；
- 回测包含 T+1、停牌、涨跌停、佣金、滑点和每日 5% 成交量上限；
- 自动执行多档滑点压力测试、统计显著性、折叠级组合超额与回撤门禁；
- 只有模型、回测、门禁和网页报告全部成功落盘，运行才会标记为 `completed`；
- 当前时点 ETF 池在历史回测中存在幸存者偏差，因此系统会自动标记为 `research_only`；
- 网页只读取冻结产物，不在浏览器重新计算核心指标。

## 命令

在 `My_Quant` 目录运行，使用已安装 Qlib 的 Python 环境：

```powershell
$env:PYTHONPATH = (Resolve-Path pipeline/src).Path

# 数据审计和快照
C:\Exception\quant\python.exe -m quant_pipeline.cli audit

# 完整滚动训练、压力回测、质量门禁和报告
C:\Exception\quant\python.exe -m quant_pipeline.cli run

# 网页看板
C:\Exception\quant\python.exe -m quant_pipeline.cli serve --port 8765
```

网页地址：`http://127.0.0.1:8765`

配置入口为 `configs/baseline.yaml`。以后研究新因子时，应复制配置建立新的实验基线，不直接修改已经完成的运行目录。

## 目录

```text
pipeline/
  configs/       版本化实验配置
  snapshots/     数据与 ETF 池快照
  runs/          不可变运行产物
  src/           Pipeline、回测、门禁和网页代码
  tests/         关键口径测试
  registry.json  实验索引
```
