# DarwinTrade

**Hierarchical self-evolution for multi-agent portfolio trading.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Official implementation of *DarwinTrade: Hierarchical Self-Evolution for
Multi-Agent Portfolio Trading* — Jie Fang, Shaolei Zhang (Renmin University of
China).

[English](README.md) · [简体中文](README_ZH.md)

## Overview

Most LLM trading agents compress every policy adaptation into a single model
action, so you cannot trace a return back to the component that caused it.
DarwinTrade instead decomposes adaptation into **three loops that own disjoint
state and run on their own clocks**, over a deterministic allocator the LLM
never edits:

| Loop | Cadence | Writes | Reverts by |
|---|---|---|---|
| **Analyst** | every bar | per-`(regime, role)` IC reliability | positive-IC gate drops a role |
| **Tactical** | every bar | 1–3 bar risk guards | automatic TTL expiry |
| **Strategic** | every 3 episodes | bounded allocator patch | logged rollback trigger |

Each update is logged with its state owner, trigger, and outcome, which makes
adaptation inspectable rather than hidden inside a reflection prompt.

## Quickstart

```bash
git clone https://github.com/Jacob6812/DarwinTrade.git
cd DarwinTrade
pip install -r requirements.txt
cp .env.example .env      # then fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
```

Run one quarter on the default 20-stock universe:

```bash
python -m backtest.stockbench.cli --start 2025-04-01 --end 2025-06-30
```

Artifacts land in `storage/reports/backtest/<run_id>/` (NAV, trades, metrics,
per-bar decision records).

> **Note.** Market data is served from `storage/cache/`, which is **not**
> included in this repository (it is multi-GB). Cache misses fall back to live
> Polygon/Finnhub calls and need the corresponding keys in `.env`. Use
> `--offline` to fail loudly instead of hitting the network.

## Requirements

- Python 3.10+ (developed and tested on 3.13)
- An OpenAI-compatible LLM endpoint (reported runs used `mimo-v2.5-pro`,
  temperature `0`, seed `1234`, JSON-schema enforced output)
- Optional, only to build the cache: Polygon, Finnhub, Alpha Vantage keys

## How one bar runs

`DarwinTradePipeline.run(...)` executes in this order each trading day:

1. **Close the previous bar.** Run tactical reflection/evolution on yesterday's
   episode and install fresh `TacticalInfluence` (1–3 bar TTL). Commit realized
   returns into `AnalystMemory` capsules so each `(regime, role)` IC stays
   current. On the strategic cadence, run reflection → diagnosis → policy author
   and apply the resulting `StrategicPatch`.
2. **Build memory influence.** Merge the strategic baseline with active tactical
   hints (`avoid_symbols`, `reduce_only`, `position_haircut`). An
   `urgency=emergency` guard can tighten the gross-exposure cap further.
3. **Classify regime.** `RegimeAgent` emits a label plus confidence, with a
   deterministic trend-rule fallback when the LLM is unavailable.
4. **Score symbols.** `MarketAgent` runs the LangGraph analyst team
   (market / news / social / fundamentals) per symbol concurrently. Each role's
   `score`/`confidence` is weighted by `max(0, IC) × shrink(n)` and summed into
   one `AssetSignal`. The regime reaches the aggregator through
   `set_aggregator_runtime`, not through analyst prompts, so IC weights stay
   keyed to the active regime without leaking it into the evidence.
5. **Allocate.** `PortfolioAllocator` splits the gross budget between long and
   short legs by confidence mass, then applies `max_single_position`, the
   tactical haircut, and `max_gross_exposure`.
6. **Execute.** Each order carries an explicit side
   (`buy` / `sell` / `sell_short` / `buy_to_cover`) so direction survives into
   the engine.

Net exposure is a *consequence* of cross-sectional selection — the
confidence-mass long/short split — not a free parameter.

## Long and short are first-class

`AssetSignal.direction` is `long`, `short`, or `hold`, and the allocator emits
**signed** weights. The `darwintrade/` package has no `allow_short` flag and no
long-only fallback path — shorting is structural, not opt-in. (The
`allow_short` you may find under `backtest/stockbench/strategies/` belongs to the
external baseline adapters, not to DarwinTrade.) A signal flip on a held name
produces two ordered legs: close, then open.

```
LLM signal   → direction   allocator action              engine side
BUY          → long        open_long / increase_long      buy
SELL         → short       open_short / increase_short    sell_short
(reduce)                   close_long                     sell
(flip)                     close_short                    buy_to_cover
```

`max_gross_exposure` bounds `|long| + |short|`; `max_single_position` bounds
`|target_weight|` per name.

## Configuration

`backtest/stockbench/config_darwintrade.yaml` is the reference config:

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
  max_gross_exposure: 0.95      # |long| + |short| ceiling
  max_single_position: 1.0      # per-name cap (1.0 == permissive)
  min_confidence_threshold: 0.35
  tactical_enabled: true
  strategic_enabled: true
  memory:
    enabled: true               # master switch
```

Credentials come from `.env` only (see `.env.example`). DarwinTrade uses its own
`LLMClient` and reads `LLM_*` directly — there is no `--llm-profile` switch. The
`api.polygon` / `api.finnhub` blocks only seed engine-side fetchers on cache
misses.

## Reproducing the ablations

`backtest/stockbench/ablation/` holds the full `2³` factorial over
{tactical, strategic, analyst-capsule}:

| Config | Disabled |
|---|---|
| `config_darwintrade_no_tactical.yaml` | tactical |
| `config_darwintrade_no_strategic.yaml` | strategic |
| `config_darwintrade_no_analyst_capsule.yaml` | analyst capsules |
| `config_darwintrade_only_tactical.yaml` | all but tactical |
| `config_darwintrade_only_strategic.yaml` | all but strategic |
| `config_darwintrade_only_capsule.yaml` | all but analyst capsules |
| `config_darwintrade_no_memory.yaml` | all three |

```bash
# Single cell
python -m backtest.stockbench.cli \
  --cfg backtest/stockbench/ablation/config_darwintrade_no_tactical.yaml

# Default matrix, or the full factorial
bash scripts/run_ablations.sh
ABLATIONS="baseline no-strategic no-tactical no-analyst-capsule \
           only-tactical only-strategic only-capsule no-memory" \
  bash scripts/run_ablations.sh

# Preview the experiment matrix without launching
DRYRUN=1 bash scripts/run_experiments.sh
```

## Common operations

```bash
# 3-day offline smoke test
python -m backtest.stockbench.cli \
  --start 2025-03-03 --end 2025-03-05 --offline --no-resume --run-id smoke3day

# Resume an interrupted run
python -m backtest.stockbench.cli --resume \
  --resume-summary storage/reports/backtest/<run_id>/daily_run_summary.jsonl \
  --resume-date 2025-04-15

# Override the universe ad hoc
python -m backtest.stockbench.cli --symbols AAPL,MSFT,JPM \
  --start 2025-04-01 --end 2025-06-30
```

## Repository layout

```
darwintrade/              Strategy: agents, memory, allocator, pipeline
├── pipeline.py           Per-bar entrypoint (DarwinTradePipeline)
├── core/                 Contracts, OpenAI-compatible LLM client, skill loader
├── agents/               Regime classifier, analyst-team wrapper, evolution agents
│   └── bayesian_aggregator.py   IC-weighted signal fusion
├── agentic/              LangGraph analyst team + tool registry/planner/executor
├── memory/               tactical.py, strategic.py, analyst_capsules.py, combined.py
├── portfolio/allocator.py       Signed-weight long/short allocator
└── integrations/llm/     MCP shim, tool implementations, .env loader

backtest/stockbench/      Self-contained backtest engine
├── cli.py                Entrypoint
├── engine.py             Cash-flow aware, signed-share engine
├── core/                 data_hub, executor, price_utils, features, schemas
├── strategies/           darwintrade + rule-based / predictive / classical-quant
│                         / TradingAgents / FinAgent baselines
├── metrics.py            Sharpe, Sortino, IR, drawdown
├── config_darwintrade.yaml
└── ablation/             2³ factorial configs

scripts/                  Ablation + experiment matrix runners, cache backfill
skills/                   LLM role prompts as markdown (see below)
```

## Skills

Role prompts live as standalone markdown in `skills/<role>/SKILL.md`.
`darwintrade/core/skills.py` appends the relevant body to each agent's base
system prompt, so behavior can be tuned without touching code.

| Skill | Loaded by |
|---|---|
| `market-regime-analyst` | `RegimeAgent` |
| `market-analyst` | analyst team, `market_analyst` role |
| `news-analyst` | analyst team, `news_analyst` role |
| `social-analyst` | analyst team, `social_analyst` role |
| `fundamentals-analyst` | analyst team, `fundamentals_analyst` role |
| `tactical-reflection` | `TacticalReflectionAgent` |
| `tactical-evolution` | `TacticalEvolutionAgent` |
| `strategic-reflection` | `StrategicReflectionAgent` |
| `strategic-diagnosis` | strategic diagnosis step |
| `strategic-policy-author` | strategic policy author step |

## Implementation notes

- The analyst graph is a single `research` node that fans out four roles in
  parallel and aggregates their `score`/`confidence` — there are no separate
  portfolio-manager or verification LLM passes.
- Tool calls route through an in-process MCP shim
  (`darwintrade.integrations.llm.mcp`) that selects providers, normalizes
  outputs, and surfaces availability flags so the graph degrades gracefully when
  a provider returns nothing.
- Capsule IC estimates are online and persist across runs at
  `<artifact_root>/memory/analyst_capsules.json`. Delete it to start cold.
  Capsule hyper-priors (`analyst_capsules.py`): warm-up `N=20`, history cap
  `200`, realized-return clip `±20%`, score-SD floor `0.05`, prior realized SD
  `0.02`.
- All evidence admitted on day `t` is timestamped strictly before `t`. The
  day-`t` open is an execution reference only, and the return it realizes enters
  state on bar `t+1`.

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

Research artifact for reproducing the paper's experiments. Not investment
advice, and not intended for live trading. Results depend on the offline
evidence cache, the LLM provider, and the cost model, and do not predict future
returns. Reported runs assume frictionless shorting with no borrow cost.
