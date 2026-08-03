"""Polygon-backed TradingAgents data vendor.

TradingAgents' upstream `yfinance` vendor rate-limits under parallel load
and its `alpha_vantage` free tier caps at 25 requests/day. Neither is
usable for a 20-symbol, 82-day backtest. This module registers a third
vendor — `polygon` — that reads from the DarwinTrade repo's local
parquet cache and Polygon client (already provisioned via
POLYGON_API_KEY in .envmm). Register once at TradingAgents import time
via `register_polygon_vendor()`.

The functions return the same string / dict shapes that TradingAgents'
downstream LLM prompts expect from the existing vendors.

All Polygon HTTP calls (fundamentals, ticker details, news) are wrapped
in a persistent JSON cache under `storage/cache/tradingagents_polygon_vendor/`
so a given (method, symbol, quarter) is fetched at most once across the
entire backtest — the second call in the same process, and every call
in resumed runs, is a disk read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time as _time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_VENDOR_CACHE_DIR = _REPO_ROOT / "storage" / "cache" / "tradingagents_polygon_vendor"
_VENDOR_CACHE_LOCK = threading.Lock()
# 100 attempts × 3 s fixed interval matches the LLM-side contract set
# in tradingagents_baseline._TA_DATA_RETRY_*.
_HTTP_RETRY_ATTEMPTS = 100
_HTTP_RETRY_INTERVAL_SEC = 3.0


def _vendor_cache_key(method: str, symbol: str, extra: str = "") -> Path:
    """Deterministic on-disk location for a (method, symbol, extra) tuple."""
    _VENDOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{method}::{symbol}::{extra}".encode("utf-8")).hexdigest()[:16]
    safe_symbol = "".join(ch for ch in symbol if ch.isalnum() or ch in "._-")[:12] or "x"
    return _VENDOR_CACHE_DIR / f"{method}_{safe_symbol}_{digest}.json"


def _cached_or_fetch(method: str, symbol: str, extra: str, fetch: Callable[[], Any]) -> Any:
    """Return cached JSON if present; otherwise call fetch(), cache, return.

    fetch() may raise — we retry it with the module-level 100/3s policy.
    """
    path = _vendor_cache_key(method, symbol, extra)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            path.unlink(missing_ok=True)

    last_exc: Exception | None = None
    for attempt in range(1, _HTTP_RETRY_ATTEMPTS + 1):
        try:
            value = fetch()
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= _HTTP_RETRY_ATTEMPTS:
                raise
            _time.sleep(_HTTP_RETRY_INTERVAL_SEC)
    else:
        raise last_exc if last_exc else RuntimeError("fetch failed")

    # Best-effort cache write (json-serialisable payloads only).
    try:
        with _VENDOR_CACHE_LOCK:
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=True)
            tmp.replace(path)
    except Exception:
        pass
    return value


def _local_bars(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    from backtest.stockbench.core.data_hub import get_bars

    df = get_bars(symbol.upper(), start_date, end_date, 1, "day", True)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _local_news(symbol: str, start_date: str, end_date: str, limit: int = 25) -> List[Dict[str, Any]]:
    from backtest.stockbench.core.data_hub import get_news

    try:
        raw = get_news(symbol.upper(), start_date, end_date, limit)
    except Exception:
        return []
    # data_hub.get_news returns (items, next_page_token). Legacy code paths
    # may return a bare list. Handle both.
    if isinstance(raw, tuple) and raw:
        items = raw[0]
    else:
        items = raw
    return list(items or [])


def _fmt_bars_csv(symbol: str, start_date: str, end_date: str, df: pd.DataFrame) -> str:
    if df.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
    out = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    numeric = ["Open", "High", "Low", "Close"]
    for col in numeric:
        if col in out.columns:
            out[col] = out[col].round(2)
    out = out[["date"] + [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]]
    header = (
        f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        f"# Total records: {len(out)}\n"
    )
    return header + out.to_csv(index=False)


def _clamp_end_date(end_date: str) -> str:
    """Never let the LLM query beyond TA_RUN_TRADE_DATE (set per bar).

    The graph analysts feed `state["trade_date"]` (the current bar) into
    prompts, but the LLM sometimes synthesises an `end_date` that runs
    past it. We honour whatever cap the strategy adapter injects into
    the environment as `TA_RUN_TRADE_DATE` and clamp `end_date` down so
    the backtest can't accidentally see the future.
    """
    cap_raw = os.getenv("TA_RUN_TRADE_DATE") or ""
    if not cap_raw:
        return end_date
    try:
        cap = pd.Timestamp(cap_raw)
        end_ts = pd.Timestamp(end_date)
        if end_ts > cap:
            return cap.strftime("%Y-%m-%d")
    except Exception:
        pass
    return end_date


def get_polygon_stock(symbol: str, start_date: str, end_date: str) -> str:
    """core_stock_apis.get_stock_data — daily OHLCV as CSV."""
    end_date = _clamp_end_date(end_date)
    df = _local_bars(symbol, start_date, end_date)
    return _fmt_bars_csv(symbol, start_date, end_date, df)


_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50-day simple moving average of close.",
    "close_200_sma": "200-day simple moving average of close.",
    "close_10_ema": "10-day exponential moving average of close.",
    "macd": "MACD line (EMA12 minus EMA26).",
    "macds": "MACD signal line (EMA9 of MACD).",
    "macdh": "MACD histogram (MACD minus signal).",
    "rsi": "14-day relative strength index.",
    "boll": "Bollinger middle band (20 SMA).",
    "boll_ub": "Bollinger upper band (middle + 2 stdev).",
    "boll_lb": "Bollinger lower band (middle - 2 stdev).",
    "atr": "14-day average true range.",
    "vwma": "Volume-weighted moving average.",
    "mfi": "Money flow index.",
}


def _compute_indicator(df: pd.DataFrame, indicator: str) -> pd.Series:
    close = df["close"].astype(float)
    high = df.get("high", close).astype(float)
    low = df.get("low", close).astype(float)
    volume = df.get("volume", pd.Series(0, index=df.index)).astype(float)

    if indicator == "close_50_sma":
        return close.rolling(50, min_periods=1).mean()
    if indicator == "close_200_sma":
        return close.rolling(200, min_periods=1).mean()
    if indicator == "close_10_ema":
        return close.ewm(span=10, adjust=False).mean()
    if indicator == "macd":
        return close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    if indicator == "macds":
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        return macd.ewm(span=9, adjust=False).mean()
    if indicator == "macdh":
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        return macd - macd.ewm(span=9, adjust=False).mean()
    if indicator == "rsi":
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, float("nan"))
        return 100 - (100 / (1 + rs))
    if indicator == "boll":
        return close.rolling(20, min_periods=1).mean()
    if indicator == "boll_ub":
        m = close.rolling(20, min_periods=1).mean()
        s = close.rolling(20, min_periods=1).std().fillna(0.0)
        return m + 2 * s
    if indicator == "boll_lb":
        m = close.rolling(20, min_periods=1).mean()
        s = close.rolling(20, min_periods=1).std().fillna(0.0)
        return m - 2 * s
    if indicator == "atr":
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / 14, adjust=False).mean()
    if indicator == "vwma":
        pv = (close * volume).rolling(14, min_periods=1).sum()
        vv = volume.rolling(14, min_periods=1).sum().replace(0.0, float("nan"))
        return pv / vv
    if indicator == "mfi":
        tp = (high + low + close) / 3.0
        raw_flow = tp * volume
        direction = (tp.diff() > 0).astype(int) - (tp.diff() < 0).astype(int)
        pos = raw_flow.where(direction > 0, 0.0).rolling(14, min_periods=1).sum()
        neg = raw_flow.where(direction < 0, 0.0).rolling(14, min_periods=1).sum().replace(0.0, float("nan"))
        return 100 - (100 / (1 + pos / neg))
    return pd.Series(dtype=float)


def get_polygon_indicator(symbol: str, indicator: str, curr_date: str, look_back_days: int) -> str:
    """technical_indicators.get_indicators — window of one named indicator."""
    curr_date = _clamp_end_date(curr_date)
    curr = pd.Timestamp(curr_date).date()
    start_lookback = (curr - timedelta(days=int(look_back_days) + 250)).isoformat()
    end_str = curr.isoformat()
    df = _local_bars(symbol, start_lookback, end_str)
    if df.empty:
        return f"No stock data for {symbol} around {curr_date}."
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    series = _compute_indicator(df, indicator)
    if series.empty:
        return f"Indicator '{indicator}' not supported by polygon vendor."
    series.index = df["date"]
    window_start = pd.Timestamp(curr) - pd.Timedelta(days=int(look_back_days))
    window = series[(series.index >= window_start) & (series.index <= pd.Timestamp(curr))]
    if window.empty:
        return f"No indicator values in the requested window for {symbol}."
    lines = [
        f"# {indicator} for {symbol.upper()} — last {len(window)} rows through {curr_date}",
        _INDICATOR_DESCRIPTIONS.get(indicator, ""),
        "date,value",
    ]
    for ts, val in window.items():
        if pd.isna(val):
            continue
        lines.append(f"{ts.date().isoformat()},{float(val):.4f}")
    return "\n".join(line for line in lines if line)


def _polygon_client():
    from backtest.stockbench.adapters.polygon_client import PolygonClient

    return PolygonClient(api_key=os.getenv("POLYGON_API_KEY", "") or None)


def _local_financials(symbol: str, curr_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Best-effort financials via Polygon vX/reference/financials.

    Fundamentals shift slowly, so we key the cache by *quarter* — a given
    quarter's financials get pulled once regardless of how many bars are
    inside that quarter.
    """
    curr = pd.Timestamp(curr_date) if curr_date else pd.Timestamp.utcnow()
    quarter_key = f"{curr.year}Q{((curr.month - 1) // 3) + 1}"

    def _fetch() -> Any:
        client = _polygon_client()
        return client.list_financials(symbol.upper(), timeframe=None, limit=8) or []

    try:
        return _cached_or_fetch("financials", symbol.upper(), quarter_key, _fetch) or []
    except Exception:
        return []


def _financial_statement_text(symbol: str, statement_key: str, curr_date: Optional[str] = None) -> str:
    rows = _local_financials(symbol, curr_date)
    if not rows:
        return f"No {statement_key} data available for {symbol} via polygon vendor."
    # Strict as-of filter: drop any statement whose filing_date/end_date is
    # not strictly < curr_date so the LLM never sees financials that were
    # published on or after the trade date.
    if curr_date:
        curr_ts = pd.Timestamp(curr_date)
        filtered = []
        for entry in rows:
            report_date = entry.get("filing_date") or entry.get("end_date") or ""
            try:
                report_ts = pd.Timestamp(report_date)
            except Exception:
                continue
            if report_ts < curr_ts:
                filtered.append(entry)
        rows = filtered
    if not rows:
        return f"# {statement_key} for {symbol.upper()} (polygon vendor)\n  No filings dated before {curr_date}."
    lines = [f"# {statement_key} for {symbol.upper()} (polygon vendor, as-of {curr_date})"]
    for entry in rows[:4]:
        period_end = entry.get("end_date") or entry.get("filing_date") or ""
        fiscal = entry.get("fiscal_period") or ""
        stmt = ((entry.get("financials") or {}).get(statement_key) or {})
        lines.append(f"\n## {period_end} ({fiscal})")
        if not stmt:
            lines.append("  (no data)")
            continue
        for name, payload in stmt.items():
            if isinstance(payload, dict) and "value" in payload:
                lines.append(f"  {name}: {payload.get('value')} {payload.get('unit') or ''}")
    return "\n".join(lines)


def get_polygon_fundamentals(ticker: str, curr_date: Optional[str] = None) -> str:
    """fundamental_data.get_fundamentals — company overview.

    Ticker overview is nearly static; cache by year so a single call
    covers all bars for that symbol × year.
    """
    curr = pd.Timestamp(curr_date) if curr_date else pd.Timestamp.utcnow()
    year_key = str(curr.year)

    def _fetch() -> Any:
        client = _polygon_client()
        return client.get_ticker_details(ticker.upper(), date=curr_date) or {}

    try:
        details = _cached_or_fetch("ticker_details", ticker.upper(), year_key, _fetch) or {}
    except Exception as exc:
        return f"Fundamentals for {ticker.upper()} unavailable via polygon vendor: {exc}"
    if not details:
        return f"No fundamentals available for {ticker.upper()}."
    fields = [
        "name",
        "market_cap",
        "phone_number",
        "address",
        "description",
        "sic_description",
        "primary_exchange",
        "total_employees",
        "list_date",
        "share_class_shares_outstanding",
    ]
    lines = [f"# Fundamentals for {ticker.upper()} (polygon vendor, {curr_date or 'latest'})"]
    for key in fields:
        if key in details and details[key] is not None:
            lines.append(f"{key}: {details[key]}")
    return "\n".join(lines)


def get_polygon_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None) -> str:
    return _financial_statement_text(ticker, "balance_sheet", curr_date)


def get_polygon_cashflow(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None) -> str:
    return _financial_statement_text(ticker, "cash_flow_statement", curr_date)


def get_polygon_income_statement(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None) -> str:
    return _financial_statement_text(ticker, "income_statement", curr_date)


def get_polygon_news(ticker: str, start_date: str, end_date: str) -> str:
    end_date = _clamp_end_date(end_date)
    items = _local_news(ticker, start_date, end_date, limit=50)
    if not items:
        return f"No news found for {ticker.upper()} between {start_date} and {end_date}."
    lines = [f"# News for {ticker.upper()} from {start_date} to {end_date} (polygon vendor)"]
    for item in items[:50]:
        title = item.get("title") or ""
        published = (
            item.get("published_utc")
            or item.get("datetime")
            or item.get("date")
            or ""
        )
        source = item.get("publisher", {}).get("name") if isinstance(item.get("publisher"), dict) else item.get("source", "")
        summary = item.get("description") or item.get("summary") or ""
        lines.append(f"\n- [{published}] {title}")
        if source:
            lines.append(f"  source: {source}")
        if summary:
            lines.append(f"  summary: {str(summary)[:400]}")
    return "\n".join(lines)


def get_polygon_global_news(curr_date: str, look_back_days: int = 7, limit: int = 50) -> str:
    # Polygon news is per-ticker; approximate global by pulling SPY headlines.
    # Under offline_only mode the news cache may not cover every calendar
    # day, so we swallow miss/errors and hand the LLM an explicit "no
    # macro news available" line instead of blowing up the whole run.
    curr_date = _clamp_end_date(curr_date)
    end = pd.Timestamp(curr_date).date()
    start = end - timedelta(days=int(look_back_days))
    try:
        items = _local_news("SPY", start.isoformat(), end.isoformat(), limit=limit)
    except Exception:
        items = []
    if not items:
        return f"# Global market news around {curr_date}\nNo macro headlines available in the offline cache for this window."
    lines = [f"# Global market news around {curr_date} (polygon vendor, SPY-linked headlines)"]
    for item in items[:limit]:
        title = item.get("title") or ""
        published = item.get("published_utc") or item.get("date") or ""
        lines.append(f"- [{published}] {title}")
    return "\n".join(lines)


def get_polygon_insider_transactions(symbol: str, curr_date: Optional[str] = None) -> str:
    # Not available for free tier; return a neutral placeholder so downstream
    # LLM prompts can still consume it without crashing.
    return (
        f"# Insider transactions for {symbol.upper()} (polygon vendor)\n"
        "  polygon vendor does not surface insider transactions in this "
        "backtest environment; treat as no material insider activity."
    )


def register_polygon_vendor() -> None:
    """Register 'polygon' in TradingAgents' VENDOR_METHODS.

    Idempotent — safe to call multiple times. Registers a callable for every
    method TradingAgents' vendor router might dispatch, so a config of
    `data_vendors.<category> = "polygon"` fully resolves.
    """
    from tradingagents.dataflows import interface as _iface

    if _iface.VENDOR_METHODS.get("get_stock_data", {}).get("polygon") is get_polygon_stock:
        return
    if "polygon" not in _iface.VENDOR_LIST:
        _iface.VENDOR_LIST.append("polygon")
    mapping = {
        "get_stock_data": get_polygon_stock,
        "get_indicators": get_polygon_indicator,
        "get_fundamentals": get_polygon_fundamentals,
        "get_balance_sheet": get_polygon_balance_sheet,
        "get_cashflow": get_polygon_cashflow,
        "get_income_statement": get_polygon_income_statement,
        "get_news": get_polygon_news,
        "get_global_news": get_polygon_global_news,
        "get_insider_transactions": get_polygon_insider_transactions,
    }
    for method, impl in mapping.items():
        _iface.VENDOR_METHODS.setdefault(method, {})["polygon"] = impl
