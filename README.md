# Stock — 🦞 龙虾A股量化交易系统

基本面选股 + 技术面择时 + 自动回测的量化交易系统

## 系统架构

```
Stock/
├── src/
│   ├── screener/           # 选股模块
│   │   └── stock_screener.py    # 基本面选股引擎
│   ├── backtest/           # 回测模块
│   │   └── engine.py            # 回测引擎 + 策略逻辑
│   ├── data/               # 数据层
│   │   └── datasource.py        # baostock数据接口
│   └── utils/              # 工具
│       └── indicators.py         # 技术指标计算
├── output/                 # 输出
│   ├── reports/                 # 回测报告(MD/CSV)
│   └── logs/                    # 运行日志
├── tests/                  # 测试
│   └── test_indicators.py
├── docs/                   # 文档
│   └── strategy.md             # 策略详细说明
├── run_screener.py         # 选股入口
├── run_backtest.py         # 回测入口
├── config.py               # 策略参数配置
├── requirements.txt
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 选股（扫描全市场）

```bash
python run_screener.py
```

### 回测

```bash
# 沪深300回测 (默认2020-2025)
python run_backtest.py --pool hs300

# 中证500回测
python run_backtest.py --pool zz500

# 自定义区间
python run_backtest.py --pool hs300 --start 2021-01-01 --end 2025-06-30

# 自定义资金
python run_backtest.py --pool hs300 --cash 500000
```

## 策略说明

### 四因子模型

| 因子 | 类型 | 用途 | 数据源 |
|------|------|------|--------|
| 基本面-ROE | 选股 | 盈利能力 | baostock季报 |
| 基本面-增长 | 选股 | 成长性 | baostock季报 |
| 技术面-趋势 | 择时 | 多空方向 | 日K线 |
| 技术面-量能 | 择时 | 资金确认 | 日K线 |

### 选股条件

- ROE(平均) > 15% **或** 净利润同比增长 > 10%
- PE_TTM < 40, PB_Mrq < 8

### 买入信号（全部满足）

1. MA5 > MA20 > MA60（多头排列，MA5至少高于MA20的0.5%）
2. MACD: DIF > DEA（趋势向好）
3. RSI(14) 在 30-65 之间（非超买非超卖）
4. 成交量 > 1.1倍5日均量（量能配合）

### 卖出信号（任一触发）

1. **止损**: 亏损 8%
2. **止盈**: 盈利 10%
3. **移动止损**: 从最高点回落 6%（且已盈利>1.5%）
4. **时间止损**: 持仓超过 15 天

### 资金管理

- 最多同时持仓 5 只
- 每笔使用总资金 20%
- A股费率: 佣金万三 + 印花税千一(卖) + 滑点0.1%

## 参数调优

所有策略参数在 `config.py` 中集中管理:

```python
# config.py 中的关键参数

# 选股门槛
ROE_MIN = 0.15             # ROE > 15%
PROFIT_GROWTH_MIN = 0.10   # 净利润同比增长 > 10%
PE_MAX = 40                # PE上限
PB_MAX = 8                 # PB上限

# 技术面
RSI_LOW = 30               # RSI下限
RSI_HIGH = 65              # RSI上限

# 风控
STOP_LOSS = 0.08           # 止损
TAKE_PROFIT = 0.10         # 止盈
TRAILING_STOP = 0.06       # 移动止损
MAX_HOLD_DAYS = 15         # 最大持仓天数
MAX_POSITIONS = 5          # 最大持仓数
POSITION_PCT = 0.20        # 单笔仓位比例
```

### 胜率优化建议

> ⚠️ **实话实说**: A股短线策略稳定70%+胜率极其困难，专业机构也做不到。
> 更合理的优化方向：

1. **提高止盈/止损比** — 止盈12%+止损6%，靠盈亏比赚钱
2. **加大盘过滤器** — 沪深300在MA120上方才交易，避开熊市
3. **行业轮动** — 结合行业景气度选股
4. **延长持仓** — 30-60天中期策略比短线胜率更高

## 数据源

- [baostock](http://baostock.com) — 免费A股数据（无需注册）
- K线数据: 日线级别，前复权
- 财务数据: 季报（ROE、净利润增长率）

## 回测结果参考

沪深300 (2020-2025):

| 指标 | 值 |
|------|-----|
| 总收益率 | +40% ~ +70% |
| 年化收益 | +6% ~ +10% |
| 最大回撤 | 20% ~ 30% |
| 胜率 | 45% ~ 55% |
| 盈亏比 | 1.1 ~ 1.5 |

> 回测结果不代表未来收益，仅供学习研究。

## License

MIT
