# DarwinTrade

**一个自我进化的多空股票交易系统。**

DarwinTrade 是一套自我进化的智能体交易系统：多角色 LLM 分析师团队在每根 K 线上产生方向性信号，确定性的资金分配器将其转化为多空组合，三层记忆系统（战术层、战略层、分析师胶囊层）随时间推移不断重写策略自身的约束参数。仓库内置了独立的回测引擎，一条命令即可完成端到端运行。

```bash
# 运行一个季度的基线回测
python -m backtest.stockbench.cli --start 2025-04-01 --end 2025-06-30
```

LLM 凭证从 `.env` 文件中读取（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`）。
行情数据优先读取本地缓存 `storage/cache/`，对已缓存的标的和日期无需消耗外部 API 配额。

---

## 目录结构

```
darwintrade/
├── darwintrade/                 ← 策略核心：智能体、记忆、分配器、管道
│   ├── pipeline.py              ← 每根 K 线入口（DarwinTradePipeline）
│   ├── core/
│   │   ├── contracts.py         ← 数据契约：AssetSignal、ExecutionPlan、MemoryInfluence …
│   │   ├── llm.py               ← OpenAI 兼容客户端，读取 .env
│   │   └── skills.py            ← Skill 提示词加载器
│   ├── agents/
│   │   ├── regime.py            ← LLM 市场状态分类器
│   │   ├── market.py            ← 单标的多角色分析师团队封装
│   │   └── evolution.py         ← TacticalEvolutionAgent + StrategicEvolutionAgent
│   ├── agentic/                 ← 基于 LangGraph 的多角色分析师团队
│   │   ├── analysts/            ← market_analyst / news_analyst /
│   │   │                          social_analyst / fundamentals_analyst 图
│   │   └── tooling/             ← 工具注册表、规划器、执行器、MCP 垫片
│   ├── memory/
│   │   ├── tactical.py          ← 短期护栏（1-3 天 TTL）
│   │   ├── strategic.py         ← 长期补丁、胶囊、episode 日志
│   │   ├── reflection.py        ← TacticalReflectionAgent +
│   │   │                          StrategicReflectionAgent
│   │   ├── analyst_memory.py    ← AnalystMemory：（状态, 角色）胶囊注册表
│   │   ├── analyst_capsules.py  ← AnalystCapsule：在线 IC 估计器
│   │   └── combined.py          ← 合并战略基线 + 战术提示
│   ├── portfolio/
│   │   └── allocator.py         ← 带符号权重分配器（多头 & 空头）
│   └── integrations/llm/        ← MCP 服务器、工具实现、.env 加载器
│
├── backtest/                    ← 回测引擎命名空间
│   ├── stockbench/              ← 自包含引擎（默认）
│   │   ├── cli.py               ← argparse 入口，单策略
│   │   ├── pipeline.py          ← run_backtest(...)
│   │   ├── engine.py            ← BacktestEngine（现金流感知，带符号份额）
│   │   ├── reports.py           ← 输出 NAV / 指标 / 交易记录
│   │   ├── visualization.py     ← matplotlib 图表（NAV / 回撤 / 热力图）
│   │   ├── summarize.py         ← 自然语言运行摘要
│   │   ├── slippage.py          ← 滑点模型
│   │   ├── datasets.py          ← data_hub 的数据集门面
│   │   ├── metrics.py           ← Sharpe / Sortino / IR / 回撤等
│   │   ├── strategies/darwintrade.py  ← DarwinTrade 的引擎适配器
│   │   ├── strategies/external_llm_adapters.py ← FinAgent 基线适配器
│   │   ├── strategies/tradingagents_baseline.py ← TradingAgents 基线
│   │   ├── core/
│   │   │   ├── data_hub.py      ← K线 / 新闻 / 基本面（parquet + finnhub）
│   │   │   ├── executor.py      ← 订单执行：buy/sell/sell_short/buy_to_cover
│   │   │   ├── price_utils.py   ← 统一的开/收/vwap 访问器
│   │   │   ├── schemas.py       ← Order、FeatureInput 等
│   │   │   └── features.py      ← 技术指标构建器
│   │   ├── adapters/            ← polygon_client、finnhub_client
│   │   ├── agents/backtest_report_llm.py  ← 可选的自然语言摘要智能体
│   │   ├── llm/llm_client.py    ← 报告智能体使用的 LLM 客户端
│   │   ├── utils/               ← 日志、IO、格式化工具
│   │   ├── config_darwintrade.yaml ← 引擎 + 股票宇宙 + 基准配置
│   │   └── ablation/            ← 消融实验 YAML（2^3 因子设计单元格）
│   │       ├── config_darwintrade_no_strategic.yaml
│   │       ├── config_darwintrade_no_tactical.yaml
│   │       ├── config_darwintrade_no_analyst_capsule.yaml
│   │       ├── config_darwintrade_only_tactical.yaml
│   │       ├── config_darwintrade_only_strategic.yaml
│   │       ├── config_darwintrade_only_capsule.yaml
│   │       └── config_darwintrade_no_memory.yaml
│
├── scripts/
│   ├── _common.sh               ← 共享辅助函数（python/env/launch/wait）
│   ├── run_ablations.sh         ← 参数化消融矩阵（单窗口）
│   ├── run_experiments.sh       ← 完整实验矩阵（重复 × 窗口）
│   ├── run_external_agents.sh   ← FinAgent 基线运行脚本
│   ├── precache_external_agents_data.py ← 离线数据预缓存
│   └── backfill_news_by_day.py  ← 新闻缓存回填工具
│
├── skills/                      ← LLM Skill 提示词（Markdown）
│   ├── market-regime-analyst/
│   ├── market-analyst/
│   ├── news-analyst/
│   ├── social-analyst/
│   ├── fundamentals-analyst/
│   ├── tactical-reflection/
│   ├── tactical-evolution/
│   ├── strategic-reflection/
│   ├── strategic-diagnosis/
│   └── strategic-policy-author/
│
├── storage/
│   ├── cache/
│   │   ├── bars/                ← 每标的 OHLCV parquet
│   │   ├── news_by_day/         ← 每标的每日新闻缓存
│   │   ├── news/                ← 旧版哈希键新闻缓存
│   │   ├── fundamentals/        ← Finnhub 财务数据缓存
│   │   ├── corporate_actions/   ← 拆股、分红
│   │   └── stock_indicators/    ← 衍生指标缓存
│   ├── reports/backtest/<run_id>/        ← stockbench NAV、交易、图表
│   └── logs/
│       └── backtest/            ← stockbench 引擎日志
│
├── tests/                       ← 85 个测试，分布在 7 个文件
├── docs/                        ← 设计文档、基准选取说明
├── pytest.ini
├── requirements.txt
└── .env                         ← LLM 凭证 + API 密钥
```

---

## 单根 K 线的执行流程

每个交易日，`DarwinTradePipeline.run(...)` 按以下顺序执行：

1. **收盘上一根 K 线。** 若有 NAV 结果，对前日 episode 运行战术反思智能体，由战术进化智能体写入新的 `TacticalInfluence`（1–3 天 TTL）。同时将已实现收益率提交到 `AnalystMemory` 胶囊，更新各 `(状态, 角色)` 的 IC 估计。当战略审查周期触发（每 3 个交易日）时，运行战略反思 → 诊断 → 政策撰写，并将生成的 `StrategicPatch` 应用到长期策略配置。

2. **构建记忆影响。** 合并战略基线（长期约束、首选优化器）与活跃的战术提示（避免标的、仅减仓、仓位折扣）。战术 `urgency=emergency` 会进一步收紧战略层的总风险敞口上限。

3. **分类市场状态。** `RegimeAgent` 利用 LLM 产生 `bull / bear / sideways / volatile` 及置信度（LLM 不可用时回退到启发式规则）。

4. **产生各标的信号。** `MarketAgent` 并发地对每个标的运行完整的 LangGraph 多角色分析师团队：`market_analyst → news_analyst → social_analyst → fundamentals_analyst`。当前市场状态通过 `set_aggregator_runtime` 直接注入聚合器（而非写入 LLM 上下文），使胶囊 IC 权重始终锁定当前状态，同时不污染分析师提示词。每位分析师的 `score / confidence` 以 `max(0, IC) × shrink(n)` 为权重加权求和，生成单一 `AssetSignal`（`direction = long / short / hold`）。

5. **资金分配。** `PortfolioAllocator` 按置信度比例将总风险敞口预算分配给多头和空头，应用 `max_single_position` 上限、战术 `position_haircut` 及 `max_gross_exposure` 约束。方向翻转（多头↔空头）自动拆分为两腿：先平仓再反向开仓。

6. **生成引擎载荷。** 每个 `ExecutionOrder` 包装为 `sim_order`，携带明确的方向（`buy / sell / sell_short / buy_to_cover`），确保引擎正确执行。

7. **写入 episode。** 每根 K 线的结果写入战术与战略记忆；单根 K 线的结构化产物落地于 `storage/reports/backtest/<run_id>/bars/<YYYYMMDD>/result.json`。

---

## 多空均为一等公民

`AssetSignal.direction` 取值为 `long / short / hold`，分配器产出**带符号**的目标权重。系统不存在 `allow_short` 开关，也没有纯多头回退路径。已持仓标的信号翻转时，系统自动生成两腿订单：先平旧方向，再开新方向。

```
LLM 信号         → 方向             分配器动作                  引擎侧
─────────────    ─────────────      ──────────────────────      ─────────────
BUY              → long             open_long / increase_long   side=buy
SELL             → short            open_short / increase_short side=sell_short
（减仓或翻转）                       close_long                  side=sell
                                    close_short                 side=buy_to_cover
```

`max_gross_exposure` 约束 `|多头| + |空头|`；`max_single_position` 约束单标的 `|目标权重|`。

---

## 三层记忆系统

记忆是 DarwinTrade 自我进化的核心。三层各自从不同信号、不同时间尺度学习，并通过 `MemoryInfluence` 直接影响每根 K 线的分配决策。

### 战术记忆（每日）

`TacticalMemory` 存储由 `TacticalReflectionAgent` 和 `TacticalEvolutionAgent` 在每根 K 线收盘后产生的短期护栏。

| 字段                    | 效果                                        |
|-------------------------|---------------------------------------------|
| `avoid_symbols`         | 禁止在这些标的上开新仓                      |
| `reduce_only_symbols`   | 禁止增加敞口                                |
| `position_haircut`      | 所有目标权重乘以此系数（< 1.0 表示收紧）    |
| `expires_in_days`       | 1–3 天；过期条目每根 K 线自动清理           |
| `urgency`               | `emergency` 会进一步收紧战略层的总敞口上限  |

战术记忆是临时性的，永远不会写入长期策略配置。

### 战略记忆（每 3 个交易日）

`StrategicMemory` 维护滚动的 `EpisodeRecord` 日志，并运行三智能体管道：反思 → 诊断 → 政策撰写。

| 智能体                      | 输出                  | 效果                                                    |
|-----------------------------|-----------------------|---------------------------------------------------------|
| `StrategicReflectionAgent`  | `StrategicReflection` | 识别多日规律                                            |
| 诊断智能体                  | 问题列表              | 定位需要调整的具体配置项                                |
| `strategic-policy-author`   | `StrategicPatch`      | 修改 `max_gross_exposure`、`max_single_position`、`preferred_optimizer`、回撤阈值 |

补丁须通过置信度门控并经过允许字段白名单校验后才会生效。战略记忆同时提供 `MemoryInfluence.strategic_baseline`，每根 K 线调节分配器约束。

### 分析师记忆（按状态 × 角色持续更新）

`AnalystMemory` 是第三层记忆，跟踪每位 `(状态, 角色)` 分析师的在线**信息系数**（IC），并以 IC 收缩权重缩放信号聚合器中各分析师的投票。

**胶囊工作原理：**

每个 `AnalystCapsule` 维护最多 200 条滚动样本 `(predicted_score, predicted_confidence, realized_return)`。K 线收盘时，`commit_realized(prev_date, realized_returns)` 消耗待定预测队列并更新胶囊统计量。下一根 K 线时，聚合器调用 `skill_weight_for(regime, role)` 获取最新权重。

**IC 收缩公式：**

```
skill_weight = max(0, IC) × shrink(n)
shrink(n)    = n / (n + WARMUP_N)    # WARMUP_N = 20
```

- **负 IC 分析师权重归零**，而非方向取反。这消除了小样本下回归估计符号翻转的风险。
- **收缩项**使观测数不足 20 的分析师权重向零收缩，防止冷启动噪声主导聚合结果。
- **每日截面去均值**：计算 IC 前先减去当日截面均值，确保指标捕捉的是真实的**截面**选股能力，而非整体市场方向的偏差。
- **状态通过 `set_aggregator_runtime` 注入**（直接运行时调用，而非 LLM 上下文），使胶囊权重始终锁定当前活跃状态，同时不向分析师提示词泄露状态信息。

**超参数：**

| 常量                    | 值    | 作用                                              |
|-------------------------|-------|---------------------------------------------------|
| `CAPSULE_WARMUP_N`      | 20    | 达到完全 IC 信任所需样本数                        |
| `CAPSULE_HISTORY_CAP`   | 200   | 滚动窗口上限；丢弃旧历史以适应状态漂移            |
| `REALIZED_RETURN_CLIP`  | 0.20  | 截断 ±20%，防止单日涨跌停主导方差估计             |
| `SCORE_SD_FLOOR`        | 0.05  | 防止所有预测相同时的除零错误                      |
| `PRIOR_REALIZED_SD`     | 0.02  | 预热前的默认波动率先验（约 2% 日波动）            |

---

## 自我进化总览

三层记忆共同构成 DarwinTrade 的自我进化回路。

| 层级       | 频率              | 输出                 | 作用范围                                                                 |
|------------|-------------------|----------------------|--------------------------------------------------------------------------|
| 战术记忆   | 每根 K 线         | `TacticalInfluence`  | 1–3 天 TTL：避免标的、仅减仓、仓位折扣                                   |
| 战略记忆   | 每 3 个交易日     | `StrategicPatch`     | 长期：总敞口上限、单仓上限、首选优化器、回撤阈值                         |
| 分析师记忆 | 每根 K 线（在线） | 胶囊 IC 权重         | 持续：每 (状态, 角色) 的信号可信度                                       |

战术层永远不修改长期策略配置。战略补丁须经过置信度门控和字段白名单校验。三层均汇聚到 `MemoryInfluence`，分配器每根 K 线读取一次。

---

## 数据层

DarwinTrade 优先读取本地缓存，缓存未命中时才调用实时 API。

| 数据类型              | 供应商                            | 缓存路径                                                  |
|-----------------------|-----------------------------------|-----------------------------------------------------------|
| OHLCV K 线            | Polygon.io / yfinance / AKShare   | `storage/cache/bars/<symbol>.parquet`                     |
| 新闻                  | Finnhub                           | `storage/cache/news_by_day/<YYYY-MM-DD>/<symbol>.json`    |
| 基本面                | Finnhub                           | `storage/cache/fundamentals/<symbol>.json`                |
| 公司行动              | 本地                              | `storage/cache/corporate_actions/<symbol>.json`           |
| 衍生指标              | 本地计算                          | `storage/cache/stock_indicators/<symbol>.parquet`         |

---

## 配置说明

`backtest/stockbench/config_darwintrade.yaml` 控制引擎行为：

```yaml
strategy:
  name: darwintrade
  universe:
    symbols: [AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, GS, BAC,
              JNJ, PFE, UNH, XOM, CVX, CAT, BA, GE, HD, WMT]
  benchmark: SPY
  start_date: '2025-04-01'
  end_date:   '2025-06-30'

engine:
  initial_capital: 1000000
  commission_rate: 0.001
  slippage_bps: 5

darwintrade:
  memory_enabled: true        # 总开关
  memory_root: ''             # 默认：<artifact_root>/memory
```

`.env`（项目根目录）：

```dotenv
LLM_BASE_URL=http://127.0.0.1:3000/v1
LLM_API_KEY=sk-...
LLM_MODEL=openai/gpt-oss-120b

POLYGON_API_KEY=...
FINNHUB_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
```

`.env` 是 LLM 凭证的唯一来源，没有 `--llm-profile` 开关。DarwinTrade 使用自己的 `LLMClient` 直接读取 `LLM_*` 变量。YAML 中的 `api.polygon` / `api.finnhub` 仅在缓存未命中时用于引擎侧数据获取。

---

## 消融实验

`backtest/stockbench/ablation/` 下的配置文件可隔离每个自进化层，
共同构成 {战术、战略、分析师胶囊} 的完整 2^3 因子设计：

| 配置文件                                          | 关闭的层                 |
|---------------------------------------------------|--------------------------|
| `config_darwintrade_no_tactical.yaml`             | 关闭战术层               |
| `config_darwintrade_no_strategic.yaml`            | 关闭战略层               |
| `config_darwintrade_no_analyst_capsule.yaml`      | 关闭分析师胶囊层         |
| `config_darwintrade_only_tactical.yaml`           | 仅开启战术层             |
| `config_darwintrade_only_strategic.yaml`          | 仅开启战略层             |
| `config_darwintrade_only_capsule.yaml`            | 仅开启分析师胶囊层       |
| `config_darwintrade_no_memory.yaml`               | 三层全部关闭             |

可通过 `--cfg` 传入单个配置，或用 `scripts/run_ablations.sh`（配合 `ABLATIONS`
环境变量）驱动整个矩阵：

```bash
# 默认 4 单元格矩阵
bash scripts/run_ablations.sh
# 完整 2^3 因子设计
ABLATIONS="baseline no-strategic no-tactical no-analyst-capsule \
           only-tactical only-strategic only-capsule no-memory" \
  bash scripts/run_ablations.sh
```

---

## 测试

```bash
pytest -q      # 85 个测试，分布在 7 个文件
```

覆盖范围：

- `TacticalMemory`：安装/过期、持久化、近期窗口聚合
- `StrategicMemory`：胶囊激活、回撤收紧、审查周期、补丁持久化
- `combine_influence`：优化器透传、折扣透传、紧急收紧
- `_dd_metrics` / `_outcome_score`：回撤检测 + 分数惩罚
- `PortfolioAllocator`：
  - 纯多头 / 纯空头信号各产生正确的动作组合
  - 混合多空信号共享总敞口预算
  - 多空翻转发出两腿订单
  - no-trade 对多空开仓均匀阻断
  - `position_haircut` 对两腿等比缩减
  - `max_gross_exposure` 限制 `|多头| + |空头|`
- `PipelineConfig`：补丁应用 + 未知字段拒绝

---

## Skills（提示词技能）

LLM 提示词以独立 Markdown 文件形式存放于 `skills/<角色>/SKILL.md`。Skill 加载器（`darwintrade/core/skills.py`）将对应 Skill 内容追加到每个智能体的基础系统提示词之后。调整行为只需编辑 Markdown，无需改动代码。

| Skill                       | 加载方                                      |
|-----------------------------|---------------------------------------------|
| `market-regime-analyst`     | `RegimeAgent`                               |
| `market-analyst`            | 分析师团队，`market_analyst` 角色           |
| `news-analyst`              | 分析师团队，`news_analyst` 角色             |
| `social-analyst`            | 分析师团队，`social_analyst` 角色           |
| `fundamentals-analyst`      | 分析师团队，`fundamentals_analyst` 角色     |
| `tactical-reflection`       | `TacticalReflectionAgent`                   |
| `tactical-evolution`        | `TacticalEvolutionAgent`                    |
| `strategic-reflection`      | `StrategicReflectionAgent`                  |
| `strategic-diagnosis`       | 诊断智能体                                  |
| `strategic-policy-author`   | 政策撰写智能体                              |

---

## 常用命令

```bash
# 冒烟测试：3 个交易日（离线）
python -m backtest.stockbench.cli \
  --start 2025-03-03 --end 2025-03-05 --offline --no-resume \
  --run-id smoke3day

# 完整季度回测，使用默认 DJIA-20 股票池
python -m backtest.stockbench.cli --start 2025-04-01 --end 2025-06-30

# 恢复中断的回测
python -m backtest.stockbench.cli --resume \
  --resume-summary storage/reports/backtest/<run_id>/daily_run_summary.jsonl \
  --resume-date 2025-04-15

# 消融实验：关闭战术层
python -m backtest.stockbench.cli \
  --cfg backtest/stockbench/ablation/config_darwintrade_no_tactical.yaml

# 实验矩阵——仅列出将要运行的任务而不启动（空跑）
DRYRUN=1 bash scripts/run_experiments.sh
```

---

## 技术说明

- 分析师图使用 LangGraph 实现，核心是单个 `research` 节点并发扇出四个角色，将各角色的 `score / confidence` 聚合为一个信号，不存在独立的投资组合经理或校验 LLM 节点。
- 工具调用通过进程内 MCP 垫片（`darwintrade.integrations.llm.mcp`）路由，负责选择数据提供商、归一化输出，并暴露可用性标志，使分析师图在某个提供商无数据时能够优雅降级。
- `backtest/` 包完全自包含，`external/stockbench` 已移除，仓库内没有任何代码导入 `stockbench.*`。
- 分析师胶囊 IC 估计为在线计算，跨回测运行持久化于 `<artifact_root>/memory/analyst_capsules.json`。删除此文件可从零开始，不携带任何分析师历史。
