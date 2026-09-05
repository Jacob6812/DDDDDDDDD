"""Guardrail: run before launching batch backtests to verify every data path
the analyst LLM will touch has a cache hit for every (symbol, trade_day) in
the experiment window. Fails loudly if anything is missing — the goal is to
never watch a 4-hour backtest die at hour 3 because Polygon rate-limited.

Scope: stockbench 20-symbol universe + SPY benchmark, 2025-03-03..2025-06-30.
Data types checked:
  - parquet bars: full window + 200 trading days of history for SMA200
  - news_by_day: file exists per (symbol, trade_day); an empty items list is OK
  - fundamentals: at least one cached statement per symbol per type
  - stock_indicators: per-day precomputed indicators for each (symbol, trade_day)

We deliberately do NOT check filing_text: statements act as fallback.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "storage" / "cache"

UNIVERSE = [
    "GS", "MSFT", "HD", "V", "SHW", "CAT", "MCD", "UNH", "AXP", "AMGN",
    "TRV", "CRM", "JPM", "IBM", "HON", "BA", "AMZN", "AAPL", "PG", "JNJ",
]
BENCH = ["SPY"]
START, END = "2025-03-03", "2025-06-30"
HISTORY_START = "2024-05-01"  # ~200 trading days before START, comfortable margin for SMA200

# Contrast window D (2025-12-01 → 2026-03-31): moderate-vol drawdown regime
# — net negative return, sustained drawdown, distinct from the primary
# window's V-shape tariff-shock recovery pattern.
WD_START, WD_END = "2025-12-01", "2026-03-31"
WD_HISTORY_START = "2025-01-01"


def _trade_days_in_window() -> list[str]:
    spy_dir = CACHE_ROOT / "parquet" / "SPY" / "day"
    dates = sorted(p.stem for p in spy_dir.glob("*.parquet"))
    return [d for d in dates if START <= d <= END]


@pytest.fixture(scope="module")
def trade_days() -> list[str]:
    tds = _trade_days_in_window()
    assert tds, "no trade days found under storage/cache/parquet/SPY/day"
    return tds


@pytest.mark.parametrize("symbol", UNIVERSE + BENCH)
def test_bars_cover_history_and_window(symbol: str, trade_days: list[str]) -> None:
    d = CACHE_ROOT / "parquet" / symbol / "day"
    assert d.exists(), f"parquet dir missing: {d}"
    files = sorted(p.stem for p in d.glob("*.parquet"))
    assert files, f"no parquet files for {symbol}"
    first, last = files[0], files[-1]
    # SPY is only used for the benchmark change (12-day window) plus buy-and-hold
    # metrics; the universe symbols feed SMA200 which needs ~200 trading days.
    required_history = "2024-11-01" if symbol in BENCH else HISTORY_START
    assert first <= required_history, (
        f"{symbol}: history starts {first}, need <= {required_history}"
    )
    assert last >= END, f"{symbol}: bars stop at {last}, need >= {END}"
    present = set(files)
    missing = [t for t in trade_days if t not in present]
    assert not missing, f"{symbol}: {len(missing)} missing trade-day bars: {missing[:5]}..."


@pytest.mark.parametrize("symbol", UNIVERSE)
def test_news_by_day_covers_window(symbol: str, trade_days: list[str]) -> None:
    d = CACHE_ROOT / "news_by_day" / symbol
    assert d.exists(), f"news dir missing: {d}"
    present = {p.stem for p in d.glob("*.json")}
    missing = [t for t in trade_days if t not in present]
    assert not missing, (
        f"{symbol}: {len(missing)}/{len(trade_days)} news-by-day gaps "
        f"first={missing[0] if missing else None}"
    )
    # Every file must be a well-formed JSON envelope
    for t in trade_days:
        p = d / f"{t}.json"
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and "items" in payload, f"bad news file: {p}"


@pytest.mark.parametrize("symbol", UNIVERSE)
def test_fundamentals_present(symbol: str) -> None:
    for kind in ("get_balance_sheet", "get_cashflow", "get_income_statement"):
        d = CACHE_ROOT / "fundamentals" / kind / symbol
        assert d.exists(), f"{kind} dir missing for {symbol}"
        files = list(d.glob("*.json"))
        assert files, f"{kind}/{symbol}: no cached statements"


@pytest.mark.parametrize("symbol", UNIVERSE)
def test_indicators_cover_window(symbol: str, trade_days: list[str]) -> None:
    d = CACHE_ROOT / "stock_indicators"
    present = {p.stem.split("_", 1)[1] for p in d.glob(f"{symbol}_*.json")}
    missing = [t for t in trade_days if t not in present]
    assert not missing, (
        f"{symbol}: {len(missing)}/{len(trade_days)} indicator gaps "
        f"first={missing[0] if missing else None}"
    )


def test_offline_only_mode_reads_bars() -> None:
    """End-to-end: with data_hub in offline_only mode, get_bars must succeed
    for a sample symbol across the target window without touching the network.
    """
    from backtest.stockbench.core import data_hub

    prev_mode = data_hub._DATA_MODE  # type: ignore[attr-defined]
    data_hub.set_data_mode("offline_only")
    try:
        df = data_hub.get_bars(
            "AAPL", start="2025-06-01", end="2025-06-30",
            multiplier=1, timespan="day", adjusted=True,
            cfg={"data": {"mode": "offline_only"}},
        )
        assert df is not None and len(df) > 0, "offline_only get_bars returned empty"
        assert "close" in df.columns
    finally:
        data_hub.set_data_mode(prev_mode)


def test_offline_only_mode_reads_news() -> None:
    from backtest.stockbench.core import data_hub

    prev_mode = data_hub._DATA_MODE  # type: ignore[attr-defined]
    data_hub.set_data_mode("offline_only")
    try:
        items, _ = data_hub.get_news(
            "AAPL", "2025-06-01", "2025-06-30", limit=10,
            cfg={"data": {"mode": "offline_only"}},
        )
        assert items is not None
    finally:
        data_hub.set_data_mode(prev_mode)


# ---------------------------------------------------------------------------
# Contrast window D (2025-12 → 2026-03) — robustness rerun on different regime
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wd_trade_days() -> list[str]:
    spy_dir = CACHE_ROOT / "parquet" / "SPY" / "day"
    dates = sorted(p.stem for p in spy_dir.glob("*.parquet"))
    tds = [d for d in dates if WD_START <= d <= WD_END]
    assert tds, "no trade days for window D in SPY parquet"
    return tds


@pytest.mark.parametrize("symbol", UNIVERSE + BENCH)
def test_wd_bars_cover_window(symbol: str, wd_trade_days: list[str]) -> None:
    d = CACHE_ROOT / "parquet" / symbol / "day"
    assert d.exists(), f"parquet dir missing: {d}"
    files = sorted(p.stem for p in d.glob("*.parquet"))
    first, last = files[0], files[-1]
    required_history = "2025-06-01" if symbol in BENCH else WD_HISTORY_START
    assert first <= required_history, (
        f"{symbol}: window-D history starts {first}, need <= {required_history}"
    )
    assert last >= WD_END, f"{symbol}: bars stop at {last}, need >= {WD_END}"
    missing = [t for t in wd_trade_days if t not in set(files)]
    assert not missing, f"{symbol}: {len(missing)} missing window-D bars: {missing[:5]}..."


@pytest.mark.parametrize("symbol", UNIVERSE)
def test_wd_news_by_day_covers_window(symbol: str, wd_trade_days: list[str]) -> None:
    d = CACHE_ROOT / "news_by_day" / symbol
    assert d.exists(), f"news dir missing: {d}"
    present = {p.stem for p in d.glob("*.json")}
    missing = [t for t in wd_trade_days if t not in present]
    assert not missing, (
        f"{symbol}: {len(missing)}/{len(wd_trade_days)} window-D news gaps"
    )
