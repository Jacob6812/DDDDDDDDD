# DarwinTrade

**An explained stock-research workspace powered by a self-evolving analyst team.**

[中文说明](README_ZH.md) · [Research and reproduction](docs/RESEARCH.md) · [MIT license](LICENSE)

Enter US stock tickers, choose starting capital and a trading date, and review a proposed long/short portfolio in your browser. DarwinTrade combines market, news, social and fundamental analysis with portfolio constraints and memory from previous decisions.

It produces directional signals and allocation weights, not guaranteed future prices. It does not connect to a broker or place orders. Confidence is a model score, not a calibrated probability of profit.

## Quick start

Python 3.11 or 3.12 is recommended. No Node.js or frontend build is required.

```bash
git clone https://github.com/Jacob6812/DarwinTrade.git
cd DarwinTrade
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

```bash
# macOS / Linux
source .venv/bin/activate
cp .env.example .env
```

Install and configure:

```bash
python -m pip install -e .
```

Edit `.env` with your own OpenAI-compatible provider settings:

```dotenv
LLM_BASE_URL=https://your-provider.example.com/v1
LLM_API_KEY=your-key
LLM_MODEL=your-provider-model-name
POLYGON_API_KEY=your-market-data-key
FINNHUB_API_KEY=your-finnhub-key
DARWINTRADE_DATA_MODE=auto
```

The endpoint must support chat completions and structured responses. Data availability depends on provider entitlements, rate limits and historical coverage. The repository does not include market-data caches or credentials. A missing or unavailable provider can prevent a run or leave part of the evidence unavailable.

```bash
darwintrade serve
```

Open **http://127.0.0.1:8000**. If the command is not on your PATH, use `python -m darwintrade.live.cli serve`. Start the command from the directory containing your `.env`.

Use **Explore sample report** to inspect the interface without credentials. Its prices and allocations are explicitly illustrative; it makes no provider calls and changes no session.

## Your first analysis

1. Enter tickers such as `AAPL, MSFT, NVDA`, or select a sample watchlist. Requests accept up to 30 distinct US tickers.
2. Enter starting equity in USD. Leave the date blank to use the latest US trading day, or choose a historical trading day covered by your data source.
3. Select **Run analysis**. The elapsed timer shows that the request is pending; runtime depends on the number of stocks, provider latency and retries. Keep the page open.
4. Review the regime, long/short weights, signed notionals, reference prices, analyst thesis and risk flags. Missing symbols are reported separately. A no-position result is possible.
5. Use **Download JSON** to save the full report.

Reference prices use the selected day's open when available, otherwise a prior cached close. They are not streaming execution quotes. The benchmark trend and return history use earlier bars. Historical analysis is research replay; it is not a guarantee that every external information source provides a point-in-time snapshot.

### Continue learning across days

Keep **Keep memory across runs** enabled to reuse a session. The browser remembers its session ID across refreshes; session files remain on the server. Continue with a strictly later trading date. To compare the same day, change starting capital, or analyse earlier history, choose **New session**.

The recommended portfolio is simulated forward using observed price changes. This is hypothetical equity, not a broker balance: it excludes actual fills, fees, borrowing costs and slippage. The first run has no outcome feedback; later runs accumulate it with a one-bar release delay. Missing prices do not contribute returns, so keep a consistent watchlist for meaningful comparisons.

## Terminal and API

```bash
darwintrade predict AAPL MSFT NVDA --capital 100000 --date 2025-06-25
darwintrade predict AAPL MSFT NVDA --session SESSION_ID --date 2025-06-26
```

CLI reports are JSON on stdout; session information is printed to stderr. Omit `--capital` when continuing a session so its simulated equity compounds.

Interactive API documentation is at **http://127.0.0.1:8000/docs**.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Server configuration and model-key readiness |
| `POST /api/decide` | Analyse `symbols`, optional `trade_date`, `capital`, `session_id` |
| `POST /api/sessions` | Create a session with optional capital and label |
| `GET /api/sessions` | List persisted sessions |
| `GET /api/sessions/{id}` | Read a session's state and recent history |

Health confirms configuration, not successful provider connectivity. Invalid symbols, dates, missing credentials and missing prices produce actionable errors. Run a small analysis to check your own provider access.

## Configuration and storage

| Variable | Default | Purpose |
| --- | --- | --- |
| `DARWINTRADE_DATA_MODE` | `auto` | Use providers on cache miss; `offline_only` restricts market data to cache |
| `DARWINTRADE_CACHE_DIR` | `storage/cache` | Market-data cache |
| `DARWINTRADE_SESSION_DIR` | `storage/live/sessions` | Session state, memory and decision JSON |
| `DARWINTRADE_DEFAULT_CAPITAL` | `100000` | Starting equity in USD |
| `DARWINTRADE_MAX_GROSS_EXPOSURE` | `0.95` | Initial gross exposure cap |
| `DARWINTRADE_MAX_SINGLE_POSITION` | `1.0` | Initial per-stock cap |
| `DARWINTRADE_MIN_CONFIDENCE` | `0.35` | Initial signal confidence threshold |
| `LLM_REQUEST_TIMEOUT` | `120` in example | Per-provider-request timeout in seconds |

`--offline` still requires an LLM endpoint. It is not a free demo and only works for symbols/dates already cached. `--cache-dir` and `--session-dir` override the paths for both CLI commands.

Use one server process for a session directory. The server serializes decisions within each session; independent worker processes do not share those locks. Back up session storage to preserve memory.

## Docker

After creating `.env`:

```bash
docker compose up --build
```

Open **http://127.0.0.1:8000**. Compose keeps sessions in a named volume and mounts `storage/cache` for market data. The service runs as a non-root user. Bind-mounted cache directories must be writable by UID 10001 on Linux.

The app is designed for local use. The API has no authentication and analysis consumes provider quota. Keep the default localhost binding; network deployment requires an authenticated proxy and request limits.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| No LLM key configured | Edit `.env` in your launch directory, then restart the server. |
| No usable price | Check ticker/date, provider credentials and coverage; offline mode requires a populated cache. |
| Same or earlier date rejected | Start a new session, or continue on a later trading day. |
| Analysis failed | Inspect the server terminal for provider errors; check model name, endpoint, quota and network access. |
| Long-running request | A run may include multiple calls and retries per analyst. The browser timer does not imply completion. |
| Port already in use | Run `darwintrade serve --port 8001`. |

## Development and verification

```bash
python -m pip install -e ".[dev]"
python -m pytest -q --ignore=tests/test_offline_cache_completeness.py
python -m build --wheel
```

Install `build` separately with `python -m pip install build`. Tests stub LLM calls. Cache-dependent tests skip when private cached data is absent; the API continuation test uses synthetic inputs and exercises the actual pipeline and allocator. The cache-completeness test is intended for a fully populated research dataset, not a fresh clone. Passing offline tests does not measure forecasting performance or provider availability.

The frontend lives in `darwintrade/live/static/`, API and session handling in `darwintrade/live/`, strategy code in `darwintrade/`, and the research engine in `backtest/`. See [research documentation](docs/RESEARCH.md) for architecture, ablations and experiment commands.

Optional frontend regression tests use Node.js 22: `npm ci --ignore-scripts` then `npm test`. They cover restored sessions, continued equity, sample isolation, duplicate submissions, error recovery and safe rendering of provider text. Node.js is not needed to run the application.

## Research and citation

This project implements *DarwinTrade: Hierarchical Self-Evolution for Multi-Agent Portfolio Trading* by Jie Fang and Shaolei Zhang. See [CITATION.cff](CITATION.cff) for citation metadata and [research documentation](docs/RESEARCH.md) for reproduction.

Licensed under [MIT](LICENSE). Research support only; model outputs can be wrong and require independent review.
