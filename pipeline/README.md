# My Quant ETF Pipeline

这是在现有 Qlib 仓库内独立维护的 ETF 研究 Pipeline。Qlib 继续负责复权特征、标签和模型信号；最终组合收益由原始价格、真实份额和人民币现金账本计算。旧 Qlib 数据、旧实验和 Qlib 源码均保留。

## 适用边界

- 初始资金固定为人民币 20,000 元。
- 当前 ETF 池是 2026-08-12 的现时快照，历史回测存在幸存者偏差，结果只能标记为 `research_only`，不能直接作为实盘收益承诺。
- GitHub 已有研究在 2026-08-13 前查看过截至 2026-08-11 的完整历史结果。因此，本轮所有覆盖该日期的发现、确认和留出阶段都强制标记为 `retrospective_exposed`；一次性状态账本只能防止重复运行，不能把已经看过的历史重新变成盲测。
- 真正的前向验证只能使用候选规格冻结后、2026-08-13 起新增且此前未查看的数据。至少积累完整的标签成熟期和 63 个新交易日后，才有资格执行一次前向确认；在此之前禁止实盘晋级。
- 18 个新增特征是“原创研究候选”。这表示它们为本项目独立构造并预注册，不表示市场上从未出现过经济含义相似的信号。
- 回测不是撮合所仿真。停牌、涨跌停和成交量限制只能根据日线公开数据保守重建；实盘前仍需仿真盘、券商接口核对和小额前向验证。

## 可靠性口径

- 信号在 T 日收盘后生成，T+1 收盘执行，标签为 T+1 到 T+2 收益。
- 训练、验证、发现、确认和锁定留出之间按交易日隔离，并为标签成熟保留 2 个交易日。
- 最终账本使用未复权 OHLC、真实份额和人民币现金，不调用 Qlib 的一千万元默认账户回测。
- 双边佣金 3 bp、每笔最低 5 元；滑点单独计算；100 份整手；单日成交不超过公开成交量的 5%。
- 买入后至少持有 5 个交易日；停牌、涨跌停、冻结卖出和零成交均留下逐单记录，失败订单不会由低排名 ETF 替补。
- 分红在登记日冻结权利，除息日成为应收并计入净值，发放日才转为可用现金；份额折算直接更新真实份额。
- 公司行动原文、事件表、原始行情、Qlib 特征和逐文件 SHA-256 清单共同进入不可变数据快照。
- 每次运行保存配置、代码版本、环境、模型、预测、持仓、逐单成交、公司行动账本、压力测试、门禁结果及离线 HTML 报告。

## 正式运行顺序

在仓库外层 `My_Quant` 目录运行，并使用已安装 Qlib 的环境：

```powershell
$env:PYTHONPATH = (Resolve-Path .\qlib\pipeline\src).Path

# 1. 更新冻结 ETF 池行情，不改变 universe.csv
C:\Exception\quant\python.exe .\qlib\scripts\data_collector\cn_etf\collector.py download `
  --data-dir .\qlib\data\cn_etf --frozen-universe --history-source sina `
  --end 2026-08-12 --workers 8

# 2. 低频、可续跑地采集全池公司行动
C:\Exception\quant\python.exe .\qlib\scripts\data_collector\cn_etf\collector.py actions `
  --data-dir .\qlib\data\cn_etf --workers 1 --request-delay-seconds 1.5 --attempts 7

# 3. 清洗、严格验证，并写入新的版本化 Qlib 目录
C:\Exception\quant\python.exe .\qlib\scripts\data_collector\cn_etf\collector.py normalize `
  --data-dir .\qlib\data\cn_etf --workers 8
C:\Exception\quant\python.exe .\qlib\scripts\data_collector\cn_etf\collector.py validate `
  --data-dir .\qlib\data\cn_etf --expected-end 2026-08-12 --max-stale-days 0
C:\Exception\quant\python.exe .\qlib\scripts\data_collector\cn_etf\collector.py dump `
  --data-dir .\qlib\data\cn_etf --qlib-dir .\qlib\data\cn_etf\qlib_data_20260812 --workers 8

# 4. 数据审计通过后才允许训练
C:\Exception\quant\python.exe -m quant_pipeline.cli audit `
  --config .\qlib\pipeline\configs\baseline.yaml
C:\Exception\quant\python.exe -m quant_pipeline.cli run `
  --config .\qlib\pipeline\configs\baseline.yaml
```

预注册因子研究先用 `research init` 冻结交易日分区、基础配置、24 个发现期实验规格和一次性状态账本。发现期包含 1 个 Alpha158 基线、5 个因子族和 18 个单因子实验；18 个正式假设使用 Benjamini-Hochberg `q=0.10`。通过者作为一个不可变候选进入一次确认；确认失败则锁定留出不会打开。固定的 `research/exposure_registry.json` 会把已公开研究的截止日和证据提交绑定到计划、请求、状态、运行清单、结果与网页；删除、改写或伪造为未见数据都会失败关闭。

## 网页可视化

```powershell
$env:PYTHONPATH = (Resolve-Path .\qlib\pipeline\src).Path
C:\Exception\quant\python.exe -m quant_pipeline.cli serve --host 127.0.0.1 --port 8765
```

访问 `http://127.0.0.1:8765`。网页只读取已经冻结并通过校验的产物，不在浏览器内重算核心指标。

## 目录

```text
pipeline/
  configs/       版本化实验配置
  research/      预注册计划、一次性状态和阶段产物
  snapshots/     数据指纹、逐文件哈希与冻结池元数据
  runs/          模型、预测、账本、门禁和 HTML 报告
  comparisons/   严格配对比较结果
  src/           Pipeline、回测、研究协议和网页代码
  tests/         关键行为与失败关闭测试
  registry.json  已完成实验索引
```
