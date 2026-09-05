"""
Live market context — the inputs `DarwinTradePipeline.run` needs, built from
market data instead of a backtest engine's per-bar `ctx`.

The backtest adapter gets `open_map`, `portfolio` and `datasets` handed to it by
the engine. Here there is no engine, so this module reconstructs the same four
inputs directly from `data_hub`:

    open_map          today's execution price per symbol
    market_data       benchmark trend snapshot for the regime classifier
    asset_data        per-symbol decision-time price
    returns_history   trailing log returns for the allocator's optimizers

Look-ahead rules are the same as the backtest path, and for the same reason: the
tool is meant to be pointed at a historical date as easily as at today, and a
leak would silently flatter the result. Benchmark trend fields read closes
strictly before `trade_date`; returns history ends at `trade_date - 1`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd

from .config import LiveSettings

logger = logging.getLogger(__name__)

# The allocator's quant optimizers need ~20 samples for a usable covariance
# estimate; 60 trading days gives margin without making the fetch expensive.
RETURNS_LOOKBACK_DAYS = 60


class MarketDataUnavailable(RuntimeError):
    """Raised when no symbol in the request has a usable price.

    Distinct from a partial failure: if some symbols resolve, the run proceeds on
    those and the rest are reported as skipped.
    """


def _data_hub():
    from backtest.stockbench.core import data_hub

    return data_hub


def resolve_trade_date(
    requested: str | None, settings: LiveSettings
) -> tuple[str, bool]:
    """Resolve the bar to trade.

    Returns `(trade_date, was_inferred)`. An explicit date is honoured as-is so a
    user can replay history. When omitted, we take the most recent NYSE trading
    day on or before today — the bar whose open is the next realistic fill.
    """
    hub = _data_hub()
    today = datetime.now(ZoneInfo("America/New_York")).date()
    if requested:
        try:
            parsed_date = date.fromisoformat(requested)
        except (ValueError, TypeError) as exc:
            raise MarketDataUnavailable("Use a valid trade date in YYYY-MM-DD format.") from exc
        if parsed_date > today:
            raise MarketDataUnavailable("Future dates have no observed market data. Choose today or an earlier trading day.")
        parsed = pd.Timestamp(parsed_date)
        if not hub.is_trading_day(parsed):
            raise MarketDataUnavailable("The selected date is not a US trading day. Choose a trading day or leave it blank.")
        return parsed.strftime("%Y-%m-%d"), False

    cursor = pd.Timestamp(today)
    for _ in range(10):
        if hub.is_trading_day(cursor):
            return cursor.strftime("%Y-%m-%d"), True
        cursor -= pd.Timedelta(days=1)
    return cursor.strftime("%Y-%m-%d"), True


@dataclass
class MarketContext:
    """Resolved decision-time market inputs for one bar."""

    trade_date: str
    open_map: dict[str, float]
    market_data: dict[str, Any]
    asset_data: dict[str, Any]
    returns_history: dict[str, list[float]]
    usable_symbols: list[str] = field(default_factory=list)
    skipped_symbols: dict[str, str] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        trade_date: str,
        symbols: list[str],
        settings: LiveSettings,
    ) -> "MarketContext":
        cfg = settings.data_hub_cfg()
        open_map: dict[str, float] = {}
        skipped: dict[str, str] = {}

        for symbol in symbols:
            try:
                price = _execution_price(symbol, trade_date, cfg)
            except Exception as exc:  # one bad symbol must not sink the request
                logger.warning("price lookup failed for %s: %s", symbol, exc)
                skipped[symbol] = f"price lookup failed: {exc}"
                continue
            if price is None or price <= 0.0:
                skipped[symbol] = "no price available for this date"
                continue
            open_map[symbol] = price

        if not open_map:
            raise MarketDataUnavailable(
                "No usable price for any requested symbol on "
                f"{trade_date}. Check the symbols and the date, or widen the "
                "cache / API access."
            )

        usable = sorted(open_map)
        snapshot = _benchmark_snapshot(trade_date, settings)
        return cls(
            trade_date=trade_date,
            open_map=open_map,
            market_data=_market_data(trade_date, open_map, snapshot, settings),
            asset_data={
                sym: {"price": open_map[sym], "trade_date": trade_date}
                for sym in usable
            },
            returns_history=_returns_history(usable, trade_date, cfg),
            usable_symbols=usable,
            skipped_symbols=skipped,
            benchmark=snapshot,
        )


def _execution_price(symbol: str, trade_date: str, cfg: dict) -> float | None:
    """Decision-time execution price for `symbol` on `trade_date`.

    Prefers `trade_date`'s open — that is the fill price and the market fixed it
    before we act, so it is legitimate decision-time information (same rule the
    analyst tools follow). When the bar has not printed yet (a request made
    before the open, or a stale feed), falls back to the last close strictly
    before `trade_date`, which is also leak-free.
    """
    hub = _data_hub()
    end = pd.Timestamp(trade_date).normalize()
    start = end - pd.Timedelta(days=14)
    bars = hub.get_bars(
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        1,
        "day",
        True,
        cfg=cfg,
    )
    if bars is None or len(bars) == 0:
        return None

    frame = bars.copy()
    if "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce")
    target = end.date()

    same_day = frame.loc[dates.dt.date == target]
    if len(same_day) > 0:
        open_px = _first_positive(same_day, "open")
        if open_px is not None:
            return open_px

    prior = frame.loc[dates.dt.date < target]
    if len(prior) > 0:
        return _first_positive(prior.tail(1), "close") or _first_positive(
            prior.tail(1), "adjusted_close"
        )
    return None


def _first_positive(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    for value in frame[column].tolist():
        try:
            px = float(value)
        except (TypeError, ValueError):
            continue
        if px == px and px > 0:  # NaN-safe
            return px
    return None


def _benchmark_snapshot(trade_date: str, settings: LiveSettings) -> dict[str, Any]:
    """Benchmark trend as of the last CLOSED bar before `trade_date`.

    Mirrors the backtest adapter: a single prior-day return flips the regime label
    on roughly 70% of days, which fragments the regime-keyed memory. The 5- and
    20-bar trends give the classifier a stable backbone.
    """
    snapshot: dict[str, Any] = {
        "symbol": settings.benchmark,
        "change_pct": None,
        "trend_5d": None,
        "trend_20d": None,
    }
    try:
        hub = _data_hub()
        # end < trade_date so today's benchmark close cannot leak in.
        end = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=40)
        bars = hub.get_bars(
            settings.benchmark,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            1,
            "day",
            True,
            cfg=settings.data_hub_cfg(),
        )
        if bars is None or len(bars) < 2:
            return snapshot
        column = (
            "close"
            if "close" in bars.columns
            else ("adjusted_close" if "adjusted_close" in bars.columns else None)
        )
        if column is None:
            return snapshot
        closes = [float(c) for c in bars[column].dropna().tolist() if float(c) > 0]
        if len(closes) >= 2:
            snapshot["change_pct"] = round((closes[-1] / closes[-2] - 1.0) * 100.0, 4)
        if len(closes) >= 6:
            snapshot["trend_5d"] = round((closes[-1] / closes[-6] - 1.0) * 100.0, 4)
        if len(closes) >= 21:
            snapshot["trend_20d"] = round((closes[-1] / closes[-21] - 1.0) * 100.0, 4)
    except Exception as exc:
        logger.warning("benchmark snapshot unavailable for %s: %s", trade_date, exc)
    return snapshot


def _market_data(
    trade_date: str,
    open_map: dict[str, float],
    snapshot: dict[str, Any],
    settings: LiveSettings,
) -> dict[str, Any]:
    change = snapshot.get("change_pct")
    return {
        "trade_date": trade_date,
        # canonical key the RegimeAgent heuristic reads
        "spy_change_pct": round(float(change), 4) if change is not None else 0.0,
        "benchmark_change_pct": round(float(change), 4) if change is not None else 0.0,
        "spy_trend_5d_pct": snapshot.get("trend_5d"),
        "spy_trend_20d_pct": snapshot.get("trend_20d"),
        "prices_snapshot": dict(list(open_map.items())[:20]),
        # the HMM regime baseline fetches its own bars and needs the same cfg
        "cfg": settings.data_hub_cfg(),
    }


def _returns_history(
    symbols: list[str], trade_date: str, cfg: dict
) -> dict[str, list[float]]:
    """Trailing close-to-close log returns per symbol, ending before trade_date."""
    try:
        end_date = date.fromisoformat(trade_date)
    except ValueError:
        logger.warning("bad trade_date for returns history: %s", trade_date)
        return {}

    # ~3 calendar months clears the 20-trading-day floor with margin.
    start = (end_date - timedelta(days=RETURNS_LOOKBACK_DAYS * 2 + 14)).isoformat()
    end = (end_date - timedelta(days=1)).isoformat()

    hub = _data_hub()
    out: dict[str, list[float]] = {}
    for symbol in symbols:
        try:
            bars = hub.get_bars(symbol, start, end, 1, "day", True, cfg=cfg)
            if bars is None or len(bars) < 2 or "close" not in bars.columns:
                continue
            closes = [
                float(c) for c in bars["close"].tolist() if c == c and float(c) > 0
            ]
            rets = [
                math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            ]
            if rets:
                out[symbol] = rets[-RETURNS_LOOKBACK_DAYS:]
        except Exception as exc:
            logger.debug("returns history failed for %s: %s", symbol, exc)
    return out


__all__ = [
    "RETURNS_LOOKBACK_DAYS",
    "MarketContext",
    "MarketDataUnavailable",
    "resolve_trade_date",
]
