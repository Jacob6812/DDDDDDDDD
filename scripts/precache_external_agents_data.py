"""Pre-cache all market data the external agent backtests need.

Runs before `scripts/run_external_agents.sh` so the backtests can execute
completely offline — every bar, news item, fundamentals record, and ticker
overview is fetched once here and written into the DarwinTrade caches. During
the actual backtest the LLM API is the only network call.

Data pulled:
  1. Daily OHLCV bars (data_hub parquet cache) — 20 symbols × wA/wD +
     the training look-back window the agent needs, so we fetch from
     window_start − 400 calendar days.
  2. Per-symbol news (data_hub news_by_day cache) — same date range.
  3. Polygon financials (tradingagents_polygon_vendor JSON cache) —
     four quarter keys covering wA + wD.
  4. Polygon ticker details (same cache) — two year keys.
  5. SPY macro news for the global-news vendor.

Concurrent by symbol; retries are handled by the underlying clients
(now bounded by POLYGON_CLIENT_MAX_ATTEMPTS / _RETRY_INTERVAL_SEC and
the vendor's own _cached_or_fetch).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_envmm() -> None:
    """Load .envmm into os.environ so the Polygon/Finnhub clients see keys."""
    env_path = REPO_ROOT / ".envmm"
    if not env_path.exists():
        print(f"[precache] .envmm not found at {env_path}", flush=True)
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_envmm()
# Polygon free tier caps around 5 req/minute. Use the client's default
# exponential backoff with a large attempt budget so we survive the
# retry-heavy cadence without exhausting the retry limit.
os.environ.setdefault("POLYGON_CLIENT_MAX_ATTEMPTS", "200")
# Deliberately leave POLYGON_CLIENT_RETRY_INTERVAL_SEC unset so the
# client uses exponential backoff — hammering with a fixed 3 s interval
# just re-hits the rate limiter every retry.
os.environ.pop("POLYGON_CLIENT_RETRY_INTERVAL_SEC", None)


UNIVERSE = [
    "GS", "MSFT", "HD", "V", "SHW", "CAT", "MCD", "UNH", "AXP", "AMGN",
    "TRV", "CRM", "JPM", "IBM", "HON", "BA", "AMZN", "AAPL", "PG", "JNJ",
]
BENCHMARK_SYMBOL = "SPY"

# Windows to pre-cache. Extend the lookback so the FinAgent training
# loop never hits the network.
WINDOWS = [
    ("w1", "2025-03-03", "2025-06-30"),
    ("w2", "2025-12-01", "2026-03-31"),
    ("w3", "2026-01-01", "2026-04-30"),
]
TRAIN_LOOKBACK_YEARS = 0.5


def _extended_window(start: str, end: str) -> tuple[str, str]:
    start_d = datetime.fromisoformat(start).date()
    end_d = datetime.fromisoformat(end).date()
    extended_start = start_d - timedelta(days=int(365 * TRAIN_LOOKBACK_YEARS) + 30)
    return extended_start.isoformat(), end_d.isoformat()


def _prefetch_bars(symbol: str, start: str, end: str) -> tuple[str, int]:
    from backtest.stockbench.core.data_hub import get_bars

    ext_start, ext_end = _extended_window(start, end)
    df = get_bars(symbol.upper(), ext_start, ext_end, 1, "day", True)
    return symbol, 0 if df is None or df.empty else len(df)


def _prefetch_news(symbol: str, start: str, end: str, limit: int = 50) -> tuple[str, int]:
    from backtest.stockbench.core.data_hub import get_news

    ext_start, ext_end = _extended_window(start, end)
    try:
        raw = get_news(symbol.upper(), ext_start, ext_end, limit)
    except Exception as exc:
        print(f"[precache-news] {symbol}: {type(exc).__name__}: {exc}", flush=True)
        return symbol, 0
    if isinstance(raw, tuple) and raw:
        items = raw[0]
    else:
        items = raw
    return symbol, len(list(items or []))




def _prefetch_financials(symbol: str, window_start: str) -> tuple[str, int]:
    from backtest.stockbench.strategies.tradingagents_polygon_vendor import _local_financials

    rows = _local_financials(symbol.upper(), curr_date=window_start)
    return symbol, len(rows)


def _prefetch_ticker_details(symbol: str, window_start: str) -> tuple[str, bool]:
    from backtest.stockbench.strategies.tradingagents_polygon_vendor import get_polygon_fundamentals

    text = get_polygon_fundamentals(symbol.upper(), curr_date=window_start)
    return symbol, bool(text and "unavailable" not in text.lower())


def _run_parallel(label: str, fn, args_list, workers: int) -> None:
    print(f"[precache] {label}: {len(args_list)} tasks × {workers} workers", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *args) for args in args_list]
        for fut in as_completed(futures):
            try:
                symbol, count = fut.result()
                print(f"  [{label}] {symbol}: {count}", flush=True)
            except Exception as exc:
                print(f"  [{label}] FAIL: {type(exc).__name__}: {exc}", flush=True)
    print(f"[precache] {label} done in {time.monotonic() - started:.1f}s", flush=True)


def main() -> int:
    from backtest.stockbench.core.data_hub import refresh_api_clients

    refresh_api_clients(
        polygon_api_key=os.getenv("POLYGON_API_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
    )

    # Polygon free tier caps at ~5 req/min. Splitting into two pools:
    # cache-mostly-hits (bars, news) go 20-wide, HTTP-bound (financials,
    # ticker_details) drop to 1 worker so we don't self-DoS.
    fast_workers = int(os.getenv("PRECACHE_FAST_WORKERS", "20"))
    slow_workers = int(os.getenv("PRECACHE_SLOW_WORKERS", "1"))

    for win_label, win_start, win_end in WINDOWS:
        print(f"\n{'=' * 70}\n  Window {win_label}: {win_start} → {win_end}\n{'=' * 70}", flush=True)

        _run_parallel(
            f"bars/{win_label}",
            _prefetch_bars,
            [(sym, win_start, win_end) for sym in UNIVERSE + [BENCHMARK_SYMBOL]],
            fast_workers,
        )
        _run_parallel(
            f"news/{win_label}",
            _prefetch_news,
            [(sym, win_start, win_end) for sym in UNIVERSE + [BENCHMARK_SYMBOL]],
            fast_workers,
        )
        _run_parallel(
            f"financials/{win_label}",
            _prefetch_financials,
            [(sym, win_start) for sym in UNIVERSE],
            slow_workers,
        )
        _run_parallel(
            f"ticker_details/{win_label}",
            _prefetch_ticker_details,
            [(sym, win_start) for sym in UNIVERSE],
            slow_workers,
        )

    print("\n[precache] all done — external agent runs can now execute offline.", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
