"""
Live inference layer — run DarwinTrade as a stock-picking tool, not a backtest.

The backtest adapter (`backtest/stockbench/strategies/darwintrade.py`) drives the
same `DarwinTradePipeline` from an engine `ctx`. This package drives it from live
market data instead, so a user can post a symbol list and get today's long/short
book back. The two paths share the pipeline and `data_hub`; they build their
market context independently, and nothing here is imported by the backtest.

Entry points:
    LiveSession    persistent memory + per-decision state, keyed by session id
    MarketContext  builds regime / asset / returns-history inputs from live bars
    decide(...)    one-shot convenience wrapper around a throwaway session
"""
from __future__ import annotations

from .context import MarketContext, resolve_trade_date
from .session import LiveSession, SessionStore, decide

__all__ = [
    "LiveSession",
    "MarketContext",
    "SessionStore",
    "decide",
    "resolve_trade_date",
]
