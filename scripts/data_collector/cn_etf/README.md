# A 股 T+1 ETF 数据管线

这套管线获取沪深交易所当前可交易的 ETF，再与基金主数据交叉验证，只保留基金类型为
`指数型-股票` 的境内股票 ETF。跨境、债券、黄金/商品、货币 ETF 不在这个数据集中，
因为它们的交易和结算规则与目标 T+1 股票 ETF 不一致。

数据默认下载到仓库内的 `data/cn_etf`：

- `universe.csv`：当次可交易的 T+1 ETF 清单和快照信息；
- `raw/*.csv`：未复权价格、前复权价格、成交量和成交额，便于审计；
- `normalized/*.csv`：符合 Qlib 约定的输入 CSV；
- `qlib_data/`：可直接给 `qlib.init` 使用的二进制数据；
- `*_report.*`：下载、清洗和新鲜度检查结果。

## 环境

建议在独立环境安装当前 Qlib 和采集依赖：

```powershell
python -m pip install -e .
python -m pip install -r scripts/data_collector/cn_etf/requirements.txt
```

## 下载并转换

在仓库根目录运行完整流程：

```powershell
python scripts/data_collector/cn_etf/collector.py all --workers 8
```

交易日 16:00（Asia/Shanghai）前运行时，脚本自动排除当天尚未完成的日 K 线。例如
2026-08-11 盘中运行，数据截止日是 2026-08-10。重复运行会从最近 30 个自然日增量检查；
如果检测到分红等导致前复权历史变化，会自动重拉该 ETF 的完整历史。

历史行情默认优先使用东方财富，节点限流或断开时自动切换腾讯行情。批量下载时也可以直接使用
稳定的腾讯源：

```powershell
python scripts/data_collector/cn_etf/collector.py all `
  --history-source tencent --insecure --workers 8
```

`--insecure` 只用于当前网络有可信 HTTPS 代理、但 Python 不认识其根证书的情况。腾讯历史接口
不提供成交额和换手率：回退到该源时，`amount` 用 OHLC 典型价乘成交量估算，
`amount_estimated=1`；`turnover_rate` 保持 NaN，不伪造换手率。核心 OHLCV、复权因子和标签不受影响。

调试时可以只取指定标的：

```powershell
python scripts/data_collector/cn_etf/collector.py all `
  --symbols SH510300 SZ159915 --workers 2 --overwrite
```

也可以分阶段运行：

```powershell
python scripts/data_collector/cn_etf/collector.py download --workers 8
python scripts/data_collector/cn_etf/collector.py normalize --workers 4
python scripts/data_collector/cn_etf/collector.py validate --expected-end 2026-08-10
python scripts/data_collector/cn_etf/collector.py dump --overwrite
```

若公司网络替换 HTTPS 证书且本机证书链无法验证，仅在确认网络可信时给下载命令增加
`--insecure`。

## Qlib 字段

价格使用前复权序列，并按每只 ETF 第一个有效交易日的收盘价归一化为 1。`factor` 满足：

```text
原始成交价 = Qlib 复权价 / factor
```

`volume` 同步反向复权，原始成交量可由 `volume * factor` 还原。成交量单位统一为“份”，
`amount` 单位为人民币元，`turnover_rate` 为小数比例，`vwap` 与复权价格处在同一尺度，
`amount_estimated` 标记成交额是否为备用源估算值。
停牌或零成交日不伪造价格，Qlib 对应位置保持 NaN。

## T+1 训练

示例配置是 `workflow_config_lightgbm_Alpha158_T1.yaml`。标签为：

```text
Ref($close, -2) / Ref($close, -1) - 1
```

也就是在 D 日收盘后产生信号，使用 D+1 到 D+2 的收益，避免把当天收盘价同时当作已知信号和
成交价。回测设置 `hold_thresh: 1`，禁止买入当日卖出；ETF 卖出不收股票印花税，因此买卖成本
都按佣金 0.03% 配置。训练前还用过去 20 日的收盘价成交额近似值大于 1,000 万元做动态流动性过滤，避免
静态使用未来流动性造成前视偏差。

运行训练：

```powershell
qrun scripts/data_collector/cn_etf/workflow_config_lightgbm_Alpha158_T1.yaml
```

当前可交易清单天然不包含已经退市的 ETF，所以用于很长历史回测时仍有“退市幸存者偏差”。
本数据适合训练当前可投 ETF 池；若研究 ETF 策略的历史可实现收益，需要另行维护历年上市、
清盘和退市清单。

## 可视化报告

生成一个可离线打开的中文交互报告：

```powershell
python scripts/data_collector/cn_etf/visualize.py
```

报告输出到 `data/cn_etf/etf_data_report.html`，包含 ETF 数量、数据覆盖、上市年份、代表性走势、
流动性排行，以及可搜索的全部 ETF 明细。
