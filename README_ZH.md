# DarwinTrade

**面向多智能体投资组合交易的分层自进化框架。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

论文 *DarwinTrade: Hierarchical Self-Evolution for Multi-Agent Portfolio
Trading* 的官方实现——方杰、张绍磊（中国人民大学）。

[English](README.md) · [简体中文](README_ZH.md)

## 项目简介

现有的大模型交易智能体大多把每一次策略调整压缩成单一的模型动作，因此无法把某笔收益
归因到真正导致它的那个组件。DarwinTrade 的做法是把策略适应拆解为**三个状态互不重叠、
各自按独立节奏运行的进化回路**，它们共同作用于一个大模型永远不能改写的确定性配置器：

| 回路 | 触发节奏 | 写入的状态 | 回退机制 |
|---|---|---|---|
| **分析师层** | 每根 K 线 | 每个 `(市场状态, 角色)` 的 IC 可信度 | 正 IC 门控直接剔除该角色 |
| **战术层** | 每根 K 线 | 1–3 根 K 线的风险护栏 | TTL 到期自动失效 |
| **战略层** | 每 3 个 episode | 受约束的配置器补丁 | 记录在案的回滚触发器 |

每一次策略更新都会连同它的状态归属、触发原因和后续结果一起落盘，因此适应过程是可审计
的，而不是隐藏在某个反思提示词内部。

## 快速开始

```bash
git clone https://github.com/Jacob6812/DarwinTrade.git
cd DarwinTrade
pip install -r requirements.txt
cp .env.example .env      # 然后填入 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
```

在默认的 20 只股票池上运行一个季度：

```bash
python -m backtest.stockbench.cli --start 2025-04-01 --end 2025-06-30
```

产物写入 `storage/reports/backtest/<run_id>/`，包含净值曲线、成交记录、指标以及逐根
K 线的决策记录。

> **注意**：行情数据由 `storage/cache/` 提供，该缓存体积达数 GB，**未**包含在本仓库
> 中。缓存缺失时会回退到 Polygon / Finnhub 实时接口，需要在 `.env` 中配置对应的 key。
> 加上 `--offline` 可以让缓存缺失直接报错，而不是访问网络。

## 环境要求

- Python 3.10+（开发与测试环境为 3.13）
- 一个兼容 OpenAI 协议的大模型服务（论文所报告的实验使用 `mimo-v2.5-pro`，
  temperature `0`，seed `1234`，并启用 JSON schema 强约束输出）
- 可选，仅在需要自行构建缓存时使用：Polygon、Finnhub、Alpha Vantage 的 API key

## 单根 K 线的执行流程

每个交易日 `DarwinTradePipeline.run(...)` 按以下顺序执行：

1. **结算上一根 K 线。** 对昨日的 episode 运行战术反思与战术进化，安装新的
   `TacticalInfluence`（1–3 根 K 线 TTL）。把已实现收益写入 `AnalystMemory` 胶囊，
   使每个 `(市场状态, 角色)` 的 IC 估计保持最新。当战略复盘节奏触发时，依次执行
   战略反思 → 诊断 → 策略撰写，并应用产出的 `StrategicPatch`。
2. **合成记忆影响。** 把战略基线与当前生效的战术提示（`avoid_symbols`、
   `reduce_only`、`position_haircut`）合并。`urgency=emergency` 级别的护栏可以进一步
   收紧总敞口上限。
3. **判定市场状态。** `RegimeAgent` 输出状态标签与置信度；大模型不可用时回退到确定性
   趋势规则。
4. **生成个股信号。** `MarketAgent` 为每只股票并发运行 LangGraph 分析师团队
   （行情 / 新闻 / 社交 / 基本面）。每个角色的 `score`/`confidence` 按
   `max(0, IC) × shrink(n)` 加权后汇总为单一 `AssetSignal`。市场状态通过
   `set_aggregator_runtime` 直接注入聚合器，而**不**经由分析师提示词，因此 IC 权重
   始终与当前状态绑定，同时不会把状态信息泄漏进证据中。
5. **计算目标权重。** `PortfolioAllocator` 按置信度质量在多头与空头之间划分总敞口预算，
   随后依次施加 `max_single_position`、战术减仓系数与 `max_gross_exposure`。
6. **下单执行。** 每笔委托都携带明确的方向
   （`buy` / `sell` / `sell_short` / `buy_to_cover`），保证方向信息完整传递到引擎。

净敞口是横截面选股的**结果**——即置信度质量的多空拆分——而不是一个可自由设定的参数。

## 多空均为一等公民

`AssetSignal.direction` 取值为 `long`、`short` 或 `hold`，配置器输出**带符号**的权重。
`darwintrade/` 包内没有 `allow_short` 开关，也没有纯多头的回退路径——做空是结构性的，
而非可选项。（若在 `backtest/stockbench/strategies/` 下看到 `allow_short`，那属于外部
基线适配器，与 DarwinTrade 本身无关。）已持仓标的上的信号反转会生成两条
有序腿：先平仓，再反向开仓。

```
大模型信号   → 方向        配置器动作                      引擎侧
BUY          → long        open_long / increase_long        buy
SELL         → short       open_short / increase_short      sell_short
（减仓）                    close_long                       sell
（反转）                    close_short                      buy_to_cover
```

`max_gross_exposure` 约束 `|long| + |short|`；`max_single_position` 约束单个标的的
`|target_weight|`。

## 配置说明

`backtest/stockbench/config_darwintrade.yaml` 是基准配置：

```yaml
symbols_universe: [GS, MSFT, HD, V, SHW, CAT, MCD, UNH, AXP, AMGN,
                   TRV, CRM, JPM, IBM, HON, BA, AMZN, AAPL, PG, JNJ]

portfolio:
  total_cash: 100000

backtest:
  commission_bps: 1.0
  slippage_bps: 2.0
  fill_ratio: 1.0
  max_positions: 20
  benchmark:
    type: per_symbol_buy_and_hold
    symbol: SPY

darwintrade:
  max_gross_exposure: 0.95      # |long| + |short| 上限
  max_single_position: 1.0      # 单标的上限（1.0 表示不额外限制）
  min_confidence_threshold: 0.35
  tactical_enabled: true
  strategic_enabled: true
  memory:
    enabled: true               # 总开关
```

凭据只从 `.env` 读取（参见 `.env.example`）。DarwinTrade 使用自带的 `LLMClient`
直接读取 `LLM_*` 变量——**没有** `--llm-profile` 开关。yaml 中的
`api.polygon` / `api.finnhub` 仅用于在缓存缺失时初始化引擎侧的数据抓取器。

## 复现消融实验

`backtest/stockbench/ablation/` 提供了 {战术, 战略, 分析师胶囊} 上完整的 `2³` 因子设计：

| 配置文件 | 关闭的层 |
|---|---|
| `config_darwintrade_no_tactical.yaml` | 战术层 |
| `config_darwintrade_no_strategic.yaml` | 战略层 |
| `config_darwintrade_no_analyst_capsule.yaml` | 分析师胶囊层 |
| `config_darwintrade_only_tactical.yaml` | 仅保留战术层 |
| `config_darwintrade_only_strategic.yaml` | 仅保留战略层 |
| `config_darwintrade_only_capsule.yaml` | 仅保留分析师胶囊层 |
| `config_darwintrade_no_memory.yaml` | 三层全关 |

```bash
# 单个单元格
python -m backtest.stockbench.cli \
  --cfg backtest/stockbench/ablation/config_darwintrade_no_tactical.yaml

# 默认矩阵，或完整因子设计
bash scripts/run_ablations.sh
ABLATIONS="baseline no-strategic no-tactical no-analyst-capsule \
           only-tactical only-strategic only-capsule no-memory" \
  bash scripts/run_ablations.sh

# 空跑：仅列出实验矩阵而不实际启动
DRYRUN=1 bash scripts/run_experiments.sh
```

## 常用命令

```bash
# 冒烟测试：3 个交易日（离线）
python -m backtest.stockbench.cli \
  --start 2025-03-03 --end 2025-03-05 --offline --no-resume --run-id smoke3day

# 恢复中断的回测
python -m backtest.stockbench.cli --resume \
  --resume-summary storage/reports/backtest/<run_id>/daily_run_summary.jsonl \
  --resume-date 2025-04-15

# 临时指定股票池
python -m backtest.stockbench.cli --symbols AAPL,MSFT,JPM \
  --start 2025-04-01 --end 2025-06-30
```

## 目录结构

```
darwintrade/              策略主体：智能体、记忆、配置器、流水线
├── pipeline.py           单根 K 线入口（DarwinTradePipeline）
├── core/                 数据契约、兼容 OpenAI 的大模型客户端、技能加载器
├── agents/               市场状态分类器、分析师团队封装、进化智能体
│   └── bayesian_aggregator.py   IC 加权信号融合
├── agentic/              LangGraph 分析师团队 + 工具注册/规划/执行
├── memory/               tactical.py、strategic.py、analyst_capsules.py、combined.py
├── portfolio/allocator.py       带符号权重的多空配置器
└── integrations/llm/     MCP 适配层、工具实现、.env 加载

backtest/stockbench/      自包含的回测引擎
├── cli.py                命令行入口
├── engine.py             现金流感知、带符号股数的引擎
├── core/                 data_hub、executor、price_utils、features、schemas
├── strategies/           darwintrade + 规则类 / 预测类 / 传统量化
│                         / TradingAgents / FinAgent 基线
├── metrics.py            Sharpe、Sortino、IR、最大回撤
├── config_darwintrade.yaml
└── ablation/             2³ 因子设计配置

scripts/                  消融与实验矩阵启动脚本、缓存回填工具
skills/                   以 markdown 形式存放的大模型角色提示词（见下）
```

## Skills（提示词技能）

角色提示词以独立 markdown 文件存放于 `skills/<role>/SKILL.md`。
`darwintrade/core/skills.py` 会把对应内容追加到各智能体的基础系统提示词之后，
因此调整行为无需改动代码。

| Skill | 使用方 |
|---|---|
| `market-regime-analyst` | `RegimeAgent` |
| `market-analyst` | 分析师团队 `market_analyst` 角色 |
| `news-analyst` | 分析师团队 `news_analyst` 角色 |
| `social-analyst` | 分析师团队 `social_analyst` 角色 |
| `fundamentals-analyst` | 分析师团队 `fundamentals_analyst` 角色 |
| `tactical-reflection` | `TacticalReflectionAgent` |
| `tactical-evolution` | `TacticalEvolutionAgent` |
| `strategic-reflection` | `StrategicReflectionAgent` |
| `strategic-diagnosis` | 战略诊断环节 |
| `strategic-policy-author` | 战略策略撰写环节 |

## 技术说明

- 分析师图只有一个 `research` 节点，内部并发展开四个角色并聚合各自的
  `score`/`confidence`——没有额外的投资经理或校验环节的大模型调用。
- 工具调用统一经过进程内的 MCP 适配层
  （`darwintrade.integrations.llm.mcp`），由它选择数据源、归一化输出并暴露可用性标志，
  使得某个数据源返回空结果时分析师图能优雅降级。
- 胶囊 IC 估计是在线更新的，并跨回测持久化在
  `<artifact_root>/memory/analyst_capsules.json`。删除该文件即可冷启动。
  胶囊超参（见 `analyst_capsules.py`）：预热 `N=20`、历史窗口上限 `200`、
  已实现收益截断 `±20%`、score 标准差下限 `0.05`、预热前收益标准差先验 `0.02`。
- 第 `t` 日采纳的所有证据其时间戳都严格早于 `t`。第 `t` 日的开盘价仅作为成交参考，
  由它产生的收益要到第 `t+1` 根 K 线才会写入状态。

## 引用

```bibtex
@article{fang2026darwintrade,
  title  = {DarwinTrade: Hierarchical Self-Evolution for Multi-Agent Portfolio Trading},
  author = {Fang, Jie and Zhang, Shaolei},
  year   = {2026}
}
```

## 许可证

MIT——详见 [LICENSE](LICENSE)。

## 免责声明

本仓库是用于复现论文实验的研究代码，不构成投资建议，也不适用于实盘交易。回测结果依赖于
离线证据缓存、所使用的大模型服务以及成本模型，不能用于预测未来收益。论文所报告的实验
假设做空无摩擦且不计融券成本。
