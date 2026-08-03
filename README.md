# DarwinTrade

**A self-evolving long/short equity trading system.**

Official implementation of *DarwinTrade: Hierarchical Self-Evolution for
Multi-Agent Portfolio Trading* (Jie Fang, Shaolei Zhang; Renmin University of
China). See [Citation](#citation).

DarwinTrade is a self-evolving agentic trading system: a multi-role LLM analyst team produces directional signals on each bar, a deterministic allocator turns them into a long/short portfolio, and a three-layer memory system (tactical, strategic, and analyst-capsule) rewrites the strategy's own constraints over time. The repo bundles its own backtest engine, so a full end-to-end run is a single command.

```bash
# One quarterly baseline run
python -m backtest.stockbench.cli --start 2025-04-01 --end 2025-06-30
```

Copy `.env.example` to `.env` and fill in your credentials
(`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`, plus market-data keys).
Market data is read from local cache under `storage/cache/` — no external API quotas needed for in-cache symbols and dates.

---

## Repository layout

```
darwintrade/
├── darwintrade/                 ← strategy: agents, memory, allocator, pipeline
│   ├── pipeline.py              ← per-bar entrypoint (DarwinTradePipeline)
│   ├── core/
│   │   ├── contracts.py         ← AssetSignal, ExecutionPlan, MemoryInfluence, ...
│   │   ├── llm.py               ← OpenAI-compatible client, reads .env
│   │   └── skills.py            ← skill prompt loader
│   ├── agents/
│   │   ├── regime.py            ← LLM market-regime classifier
│   │   ├── market.py            ← per-symbol multi-role analyst team wrapper
│   │   └── evolution.py         ← TacticalEvolutionAgent + StrategicEvolutionAgent
│   ├── agentic/                 ← LangGraph multi-role analyst team
│   │   ├── analysts/            ← market_analyst / news_analyst /
│   │   │                          social_analyst / fundamentals_analyst graph
│   │   └── tooling/             ← tool registry, planner, executor, MCP shims
│   ├── memory/
│   │   ├── tactical.py          ← short-lived guardrails (1-3 day TTL)
│   │   ├── strategic.py         ← long-lived patches, capsules, episode log
│   │   ├── reflection.py        ← TacticalReflectionAgent +
│   │   │                          StrategicReflectionAgent
│   │   ├── analyst_memory.py    ← AnalystMemory: (regime, role) capsule registry
│   │   ├── analyst_capsules.py  ← AnalystCapsule: online IC estimator
│   │   └── combined.py          ← merge strategic baseline + tactical hints
│   ├── portfolio/
│   │   └── allocator.py         ← signed-weight allocator (long & short)
│   └── integrations/llm/        ← MCP server, tool implementations, .env loader
│
├── backtest/                    ← namespace for backtest engines
│   ├── stockbench/              ← self-contained engine (default)
│   │   ├── cli.py               ← argparse entrypoint, single strategy
│   │   ├── pipeline.py          ← run_backtest(...)
│   │   ├── engine.py            ← BacktestEngine (cash-flow aware, signed shares)
│   │   ├── reports.py           ← write NAV / metrics / trades artifacts
│   │   ├── visualization.py     ← matplotlib charts (NAV / drawdown / heatmaps)
│   │   ├── summarize.py         ← natural-language run summary
│   │   ├── slippage.py          ← slippage model
│   │   ├── datasets.py          ← datasets façade over data_hub
│   │   ├── metrics.py           ← Sharpe / Sortino / IR / drawdown / etc.
│   │   ├── strategies/darwintrade.py  ← engine adapter for DarwinTrade
│   │   ├── strategies/external_llm_adapters.py ← FinAgent baseline adapter
│   │   ├── strategies/tradingagents_baseline.py ← TradingAgents baseline
│   │   ├── core/
│   │   │   ├── data_hub.py      ← bars / news / fundamentals (parquet + finnhub)
│   │   │   ├── executor.py      ← plan_orders → buy/sell/sell_short/buy_to_cover
│   │   │   ├── price_utils.py   ← unified open/close/vwap accessors
│   │   │   ├── schemas.py       ← Order, FeatureInput, etc.
│   │   │   └── features.py      ← technical-indicator builders
│   │   ├── adapters/            ← polygon_client, finnhub_client
│   │   ├── agents/backtest_report_llm.py  ← optional natural-language summary
│   │   ├── llm/llm_client.py    ← LLM client used by the report agent
│   │   ├── utils/               ← logging_setup, io, formatting helpers
│   │   ├── config_darwintrade.yaml ← engine + symbol-universe + benchmark
│   │   ├── ablation/            ← memory-ablation YAMLs (2^3 factorial cells)
│   │   │   ├── config_darwintrade_no_strategic.yaml
│   │   │   ├── config_darwintrade_no_tactical.yaml
│   │   │   ├── config_darwintrade_no_analyst_capsule.yaml
│   │   │   ├── config_darwintrade_only_tactical.yaml
│   │   │   ├── config_darwintrade_only_strategic.yaml
│   │   │   ├── config_darwintrade_only_capsule.yaml
│   │   │   └── config_darwintrade_no_memory.yaml
│   │   └── scripts/run_benchmark.sh   ← internal wrapper
│
├── scripts/
│   ├── _common.sh               ← shared helpers (python/env/launch/wait)
│   ├── run_ablations.sh         ← parameterized ablation matrix (one window)
│   ├── run_experiments.sh       ← full experiment matrix (repeats × windows)
│   ├── run_external_agents.sh   ← FinAgent baseline runner
│   ├── precache_external_agents_data.py ← offline data pre-cache
│   └── backfill_news_by_day.py  ← news cache backfill utility
│
├── skills/                      ← LLM skill prompts (markdown)
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
│   │   ├── bars/                ← parquet OHLCV per symbol
│   │   ├── news_by_day/         ← per-symbol per-day news cache
│   │   ├── news/                ← legacy hash-keyed news cache
│   │   ├── fundamentals/        ← finnhub-reported financials cache
│   │   ├── corporate_actions/   ← splits, dividends
│   │   └── stock_indicators/    ← derived indicators cache
│   ├── reports/backtest/<run_id>/        ← stockbench NAV, trades, charts
│   └── logs/
│       └── backtest/            ← stockbench engine logs
│
├── tests/                       ← 85 tests across 7 files: pipeline contracts,
│                                  memory, combined influence, allocator,
│                                  long/short flips, gross-exposure caps,
│                                  Bayesian aggregator, look-ahead safety,
│                                  regime beta control, strategic self-evolution
├── docs/                        ← design notes, benchmark selection notes
├── pytest.ini
├── requirements.txt
└── .env                         ← LLM_BASE_URL / LLM_API_KEY / LLM_MODEL +
                                   POLYGON_API_KEY / FINNHUB_API_KEY
```

---

## How a single bar runs

Each trading day, `DarwinTradePipeline.run(...)` executes in this order:

1. **Close the previous bar.** If a NAV outcome is available, run the tactical
   reflection agent on yesterday's episode and let the tactical evolution agent
   install fresh `TacticalInfluence` (1–3 day TTL). Commit realized returns into
   `AnalystMemory` capsules so each `(regime, role)` IC estimate stays current.
   When the strategic review cadence triggers (every 3 trade days), run strategic
   reflection → diagnosis → policy author and apply the resulting `StrategicPatch`
   to long-lived strategy config.
2. **Build memory influence.** Combine the strategic baseline (long-term
   constraints, preferred optimizer) with active tactical hints (avoid_symbols,
   reduce_only, position_haircut). Tactical `urgency=emergency` can tighten
   the strategic gross-exposure cap further.
3. **Classify regime.** `RegimeAgent` produces `bull / bear / sideways /
   volatile` plus confidence using the LLM (heuristic fallback if the LLM is
   unavailable).
4. **Produce per-symbol signals.** `MarketAgent` runs the full LangGraph
   multi-role analyst team for each symbol concurrently:
   `market_analyst → news_analyst → social_analyst → fundamentals_analyst`.
   The current regime is injected into the aggregator directly via
   `set_aggregator_runtime` (not as LLM context) so capsule IC weights are
   keyed to the active regime without polluting the analyst prompts.
   Each analyst's `score` / `confidence` is weighted by `max(0, IC) × shrink(n)`
   and summed into a single `AssetSignal` (`direction = long / short / hold`).
5. **Allocate.** `PortfolioAllocator` splits the gross-exposure budget between
   long and short legs proportionally to confidence, applies
   `max_single_position` caps, the tactical `position_haircut`, and the
   `max_gross_exposure` constraint. Crossing zero (long↔short) is split into
   `close_long + open_short` (or symmetric) two-leg orders.
6. **Emit engine payload.** Each `ExecutionOrder` is wrapped with a `sim_order`
   that carries the explicit side (`buy / sell / sell_short / buy_to_cover`) so
   `plan_orders_from_sim_order` preserves direction through the engine.
7. **Write episodes.** Per-bar results are written to tactical and strategic
   memory; each bar artifact lands under
   `storage/reports/backtest/<run_id>/bars/<YYYYMMDD>/result.json`.

---

## Long & short are first-class

`AssetSignal.direction` is one of `long / short / hold`. The allocator produces
**signed** target weights. There is no `allow_short` flag and no long-only
fallback path. A signal flip on a held name automatically generates two legs in
order: a closing leg, then an opening leg in the new direction.

```
LLM signal       → direction       allocator action           engine sees
─────────────    ─────────────     ──────────────────────     ─────────────
BUY              → long            open_long / increase_long  side=buy
SELL             → short           open_short / increase_short side=sell_short
(reduce or flip)                   close_long                 side=sell
                                   close_short                side=buy_to_cover
```

`max_gross_exposure` constrains `|long| + |short|`; `max_single_position`
constrains `|target_weight|` per symbol.

---

## Three-layer memory

Memory is the core of DarwinTrade's self-evolution. Three distinct layers learn
from different signals at different timescales and feed directly into allocation
decisions via `MemoryInfluence`.

### Tactical memory (daily)

`TacticalMemory` stores short-lived guardrails produced by `TacticalReflectionAgent`
and `TacticalEvolutionAgent` after each bar closes.

| Field              | Effect                                          |
|--------------------|-------------------------------------------------|
| `avoid_symbols`    | Bars opening new positions in those names       |
| `reduce_only_symbols` | Prevents increasing exposure                 |
| `position_haircut` | Multiplies all target weights (< 1.0 = tighter) |
| `expires_in_days`  | 1–3; stale entries are pruned each bar          |
| `urgency`          | `emergency` tightens the strategic gross-exposure cap |

Tactical influence is ephemeral — it never writes to long-lived strategy config.

### Strategic memory (every 3 trade days)

`StrategicMemory` maintains a rolling `EpisodeRecord` log and runs a
three-agent pipeline: reflection → diagnosis → policy author.

| Agent                       | Output              | Effect                                        |
|-----------------------------|---------------------|-----------------------------------------------|
| `StrategicReflectionAgent`  | `StrategicReflection` | Identifies multi-day patterns                |
| diagnosis agent             | issues list         | Pinpoints which config knob to adjust         |
| `strategic-policy-author`   | `StrategicPatch`    | Mutates `max_gross_exposure`, `max_single_position`, `preferred_optimizer`, drawdown thresholds |

Patches are gated by a confidence threshold and validated against an
allowed-fields whitelist before being applied. Strategic memory also feeds the
`MemoryInfluence.strategic_baseline` that modulates allocator constraints every
bar.

### Analyst memory (per regime × role)

`AnalystMemory` is the third memory layer. It tracks each `(regime, role)`
analyst's online **Information Coefficient** (IC) and uses IC-shrinkage weights
to scale individual analyst votes in the signal aggregator.

**How capsules work:**

Each `AnalystCapsule` maintains a rolling window of `(predicted_score,
predicted_confidence, realized_return)` tuples (capped at 200 samples).
At bar close, `commit_realized(prev_date, realized_returns)` drains the pending
prediction queue and updates capsule statistics. On the next bar, the aggregator
calls `skill_weight_for(regime, role)` to retrieve the current weight for each
analyst.

**IC-shrinkage formula:**

```
skill_weight = max(0, IC) × shrink(n)
shrink(n)    = n / (n + WARMUP_N)         # WARMUP_N = 20
```

- **Negative-IC analysts are dropped** (weight → 0), not inverted. This
  eliminates the small-sample sign-flip hazard where a noisy regression estimate
  could literally reverse an analyst's direction.
- **Shrinkage blends toward zero** for analysts with fewer than 20 observations,
  preventing warm-start noise from dominating.
- **Cross-sectional daily demean** is applied before computing IC, so a day when
  every analyst agreed the market was going up does not inflate IC estimates — the
  metric captures true *cross-sectional* skill.
- **Regime is injected via `set_aggregator_runtime`** (a direct runtime call, not
  LLM context), so capsule weights are always keyed to the currently active regime
  without leaking regime information into analyst prompts.

**Hyper-priors:**

| Constant              | Value | Purpose                                         |
|-----------------------|-------|-------------------------------------------------|
| `CAPSULE_WARMUP_N`    | 20    | Samples needed for full IC trust                |
| `CAPSULE_HISTORY_CAP` | 200   | Rolling window; older history dropped to adapt to regime drift |
| `REALIZED_RETURN_CLIP`| 0.20  | Clips ±20% to prevent limit-move days dominating variance |
| `SCORE_SD_FLOOR`      | 0.05  | Guards divide-by-zero when all predictions are identical |
| `PRIOR_REALIZED_SD`   | 0.02  | Pre-warmup scale fallback (~2% daily vol)       |

---

## Three-layer self-evolution

The tactical, strategic, and analyst-capsule layers together form DarwinTrade's
self-evolution loop — three learning channels operating at different timescales.

| Layer        | Cadence            | Output              | Scope                                                              |
|--------------|--------------------|---------------------|--------------------------------------------------------------------|
| Tactical     | every bar          | `TacticalInfluence` | 1–3 day TTL: avoid_symbols, reduce_only, position_haircut          |
| Strategic    | every 3 trade days | `StrategicPatch`    | long-lived: max_gross_exposure, max_single_position, preferred_optimizer, drawdown thresholds |
| Analyst      | every bar (online) | capsule IC weights  | continuous: per-(regime, role) signal trust                        |

Tactical never modifies long-lived strategy config. Strategic patches are
gated by a confidence threshold and validated against an allowed-fields
whitelist before being applied. All three layers converge into `MemoryInfluence`
which the allocator reads every bar.

---

## Data layer

DarwinTrade reads everything it can from local cache before touching live APIs.

| Tool                       | Vendor                       | Source                                                 |
|----------------------------|------------------------------|--------------------------------------------------------|
| OHLCV bars                 | Polygon.io / yfinance / AKShare | `storage/cache/bars/<symbol>.parquet`               |
| News                       | Finnhub                      | `storage/cache/news_by_day/<YYYY-MM-DD>/<symbol>.json` |
| Fundamentals               | Finnhub                      | `storage/cache/fundamentals/<symbol>.json`             |
| Corporate actions          | local                        | `storage/cache/corporate_actions/<symbol>.json`        |
| Derived indicators         | local compute                | `storage/cache/stock_indicators/<symbol>.parquet`      |

Cache misses fall back to live API calls (requires `POLYGON_API_KEY` /
`FINNHUB_API_KEY` in `.env`).

---

## Configuration

`backtest/stockbench/config_darwintrade.yaml` controls the engine:

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
  memory_enabled: true        # master switch
  memory_root: ''             # default: <artifact_root>/memory
```

`.env` (project root):

```dotenv
LLM_BASE_URL=http://127.0.0.1:3000/v1
LLM_API_KEY=sk-...
LLM_MODEL=openai/gpt-oss-120b

POLYGON_API_KEY=...
FINNHUB_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
```

The `.env` is the single source of truth for LLM credentials. There is **no**
`--llm-profile` switch — DarwinTrade uses its own `LLMClient` and reads
`LLM_*` directly. The yaml `api.polygon` / `api.finnhub` blocks are only used
to seed engine-side data fetchers when cache misses occur.

---

## Memory ablation

Configs in `backtest/stockbench/ablation/` isolate each self-evolving layer.
Together they form the full 2^3 factorial over {tactical, strategic, analyst-capsule}:

| Config file                                     | What's disabled                          |
|-------------------------------------------------|------------------------------------------|
| `config_darwintrade_no_tactical.yaml`           | Tactical layer off                       |
| `config_darwintrade_no_strategic.yaml`          | Strategic layer off                      |
| `config_darwintrade_no_analyst_capsule.yaml`    | Analyst-capsule layer off                |
| `config_darwintrade_only_tactical.yaml`         | Only tactical on                         |
| `config_darwintrade_only_strategic.yaml`        | Only strategic on                        |
| `config_darwintrade_only_capsule.yaml`          | Only analyst-capsule on                  |
| `config_darwintrade_no_memory.yaml`             | All three layers off                     |

Pass any of them with `--cfg`, or drive the whole matrix with
`scripts/run_ablations.sh` (via the `ABLATIONS` env var):

```bash
# Default 4-cell matrix
bash scripts/run_ablations.sh
# Full 2^3 factorial
ABLATIONS="baseline no-strategic no-tactical no-analyst-capsule \
           only-tactical only-strategic only-capsule no-memory" \
  bash scripts/run_ablations.sh
```

---

## Tests

```bash
# 79 offline unit tests — no credentials or data cache needed
pytest -q --ignore=tests/test_offline_cache_completeness.py

# Full suite (203 tests). test_offline_cache_completeness.py additionally
# requires a populated storage/cache/ for the evaluated windows.
pytest -q
```

Coverage:

- `TacticalMemory`: install/expire, persistence, recent-window aggregation
- `StrategicMemory`: capsule activation, drawdown tightening, review cadence,
  patch persistence
- `combine_influence`: optimizer carry-through, haircut pass-through, emergency
  tightening
- `_dd_metrics` / `_outcome_score`: drawdown detection + score penalties
- `PortfolioAllocator`:
  - long-only and short-only signals each produce the right action mix
  - mixed long+short share the gross-exposure budget
  - long↔short flips emit two legs (`close + open` of opposite sides)
  - no-trade blocks long and short opens symmetrically
  - `position_haircut` halves both legs symmetrically
  - `max_gross_exposure` caps `|long| + |short|`
- `PipelineConfig`: patch application + unknown-field rejection

---

## Skills

LLM prompts live as standalone markdown under `skills/<role>/SKILL.md`. The
skill loader (`darwintrade/core/skills.py`) appends the relevant skill body to
each agent's base system prompt. To tune behavior, edit the markdown — no code
change needed.

| Skill                       | Loaded by                                 |
|-----------------------------|-------------------------------------------|
| `market-regime-analyst`     | `RegimeAgent`                             |
| `market-analyst`            | analyst team, `market_analyst` role       |
| `news-analyst`              | analyst team, `news_analyst` role         |
| `social-analyst`            | analyst team, `social_analyst` role       |
| `fundamentals-analyst`      | analyst team, `fundamentals_analyst` role |
| `tactical-reflection`       | `TacticalReflectionAgent`                 |
| `tactical-evolution`        | `TacticalEvolutionAgent`                  |
| `strategic-reflection`      | `StrategicReflectionAgent`                |
| `strategic-diagnosis`       | diagnosis agent                           |
| `strategic-policy-author`   | policy authoring agent                    |

---

## Common operations

```bash
# Smoke test: 3 trading days offline
python -m backtest.stockbench.cli \
  --start 2025-03-03 --end 2025-03-05 --offline --no-resume \
  --run-id smoke3day

# Full quarterly run with default DJIA-20 universe
python -m backtest.stockbench.cli --start 2025-04-01 --end 2025-06-30

# Resume an interrupted run
python -m backtest.stockbench.cli --resume \
  --resume-summary storage/reports/backtest/<run_id>/daily_run_summary.jsonl \
  --resume-date 2025-04-15

# Memory ablation: disable the tactical layer
python -m backtest.stockbench.cli \
  --cfg backtest/stockbench/ablation/config_darwintrade_no_tactical.yaml

# Experiment matrix — list runs without launching (dry run)
DRYRUN=1 bash scripts/run_experiments.sh
```

---

## Notes

- The pipeline uses LangGraph for the agentic analyst graph. The graph is a
  single `research` node that fans out four roles in parallel and aggregates
  their per-role `score / confidence` into one signal — no separate
  portfolio-manager / verification LLM passes.
- Tool calls go through an in-process MCP shim
  (`darwintrade.integrations.llm.mcp`) which selects providers, normalises
  outputs, and surfaces availability flags so the analyst graph can degrade
  gracefully when a provider returns no data.
- The `backtest/` package is fully self-contained. `external/stockbench` was
  removed; nothing in this repo imports `stockbench.*` anymore.
- Analyst capsule IC estimates are online and persist across backtest runs under
  `<artifact_root>/memory/analyst_capsules.json`. Delete this file to start
  cold with no analyst history.

---

## Citation

```bibtex
@article{fang2026darwintrade,
  title  = {DarwinTrade: Hierarchical Self-Evolution for Multi-Agent Portfolio Trading},
  author = {Fang, Jie and Zhang, Shaolei},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This is a research artifact for reproducing the experiments in the paper. It is
not investment advice and is not intended for live trading. Backtested results
depend on the offline evidence cache, the LLM provider, and the cost model, and
do not predict future returns. Reported runs assume frictionless shorting with
no borrow cost.
