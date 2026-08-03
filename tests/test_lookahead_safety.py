"""
Regression tests pinning the anti-lookahead contract.

Any change that reintroduces same-day close leakage into the analyst LLM,
regime signal, or intraday adapters must fail these tests.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. real_data_provider._get_history must cut off at trade_date - 1
# ---------------------------------------------------------------------------

def test_get_history_default_stops_before_trade_date() -> None:
    # _get_history() with default include_trade_date_open=False must never
    # request a bar dated ON or AFTER trade_date. This is the safety net for
    # _get_indicators (SMA/MACD/RSI/ATR feed) which cannot tolerate a partial
    # T-day bar.
    from darwintrade.integrations.llm import real_data_provider as rdp

    provider = rdp.DarwinTradeMCPProvider()

    with patch.object(rdp, "_load_data_hub") as data_hub_loader:
        data_hub = MagicMock()
        data_hub.get_bars.return_value = _minimal_price_frame(
            dates=["2025-06-26", "2025-06-27", "2025-06-29"]
        )
        data_hub_loader.return_value = data_hub

        provider._get_history("AAPL", end_date="2025-06-30", look_back_days=60)
        kwargs = data_hub.get_bars.call_args.kwargs
        assert kwargs["end"] == "2025-06-29", (
            "history default end must be strictly before trade_date, "
            f"got end={kwargs['end']}"
        )


def test_get_history_with_trade_date_open_masks_future_fields() -> None:
    # _get_history(include_trade_date_open=True) may fetch bar T, but T's
    # close/high/low/volume MUST be masked to NA so only the open (which is
    # already-realised decision-time info) survives.
    from darwintrade.integrations.llm import real_data_provider as rdp
    import pandas as pd

    provider = rdp.DarwinTradeMCPProvider()

    with patch.object(rdp, "_load_data_hub") as data_hub_loader:
        data_hub = MagicMock()
        # Frame includes T-1, T-2, and T (2025-06-30)
        data_hub.get_bars.return_value = _minimal_price_frame(
            dates=["2025-06-26", "2025-06-27", "2025-06-30"]
        )
        data_hub_loader.return_value = data_hub

        result = provider._get_history(
            "AAPL", end_date="2025-06-30", look_back_days=60,
            include_trade_date_open=True,
        )

        kwargs = data_hub.get_bars.call_args.kwargs
        assert kwargs["end"] == "2025-06-30", "must fetch through T to grab T's open"

        t_row = result[result["date"] == pd.Timestamp("2025-06-30")].iloc[0]
        assert not pd.isna(t_row["open"]), "T's open must survive"
        for field in ("close", "high", "low", "volume"):
            assert pd.isna(t_row[field]), f"T's {field} must be masked (future info)"


def _minimal_price_frame(dates=None):
    import pandas as pd

    dates = dates or ["2025-06-27", "2025-06-28", "2025-06-29"]
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000_000] * n,
        }
    )


# ---------------------------------------------------------------------------
# 2. Benchmark daily change must use only closed bars (t-1 and t-2)
# ---------------------------------------------------------------------------

def test_benchmark_daily_change_uses_past_closes_only() -> None:
    import pandas as pd

    from backtest.stockbench.strategies.darwintrade import DarwinTradeStrategy

    strategy = DarwinTradeStrategy.__new__(DarwinTradeStrategy)
    strategy._spy_change_cache_date = None
    strategy._spy_change_cache_val = None

    captured: dict = {}

    def fake_get_bars(sym, start, end, *args, **kwargs):
        captured["end"] = end
        captured["start"] = start
        return pd.DataFrame(
            {
                "close": [500.0, 505.0, 510.0],
                "date": pd.to_datetime(
                    ["2025-06-25", "2025-06-26", "2025-06-27"]
                ),
            }
        )

    with patch(
        "backtest.stockbench.core.data_hub.get_bars",
        side_effect=fake_get_bars,
    ):
        strategy._benchmark_snapshot(
            "2025-06-30", {"cfg": {"backtest": {"benchmark": {"symbol": "SPY"}}}}
        )

    end = pd.Timestamp(captured["end"])
    assert end < pd.Timestamp("2025-06-30"), (
        f"benchmark change window must not include trade_date, "
        f"got end={captured['end']}"
    )


# ---------------------------------------------------------------------------
# 3. Akshare intraday adapter must not default to trade_date's session
# ---------------------------------------------------------------------------

def test_akshare_intraday_defaults_to_previous_session() -> None:
    from darwintrade.agentic.tooling.toolbox.akshare_adapter import (
        stock_us_hist_min_em,
    )

    captured: dict = {}

    def fake_call_akshare(function_name, payload, **kwargs):
        captured.update(kwargs)
        return {"available": True, "provider": "akshare", "records": [], "row_count": 0}

    with patch(
        "darwintrade.agentic.tooling.toolbox.akshare_adapter._call_akshare",
        side_effect=fake_call_akshare,
    ):
        stock_us_hist_min_em({"symbol": "AAPL", "trade_date": "2025-06-30"})

    assert captured["start_date"].startswith("2025-06-29"), captured["start_date"]
    assert captured["end_date"].startswith("2025-06-29"), captured["end_date"]


# ---------------------------------------------------------------------------
# 4. Akshare realtime snapshots must refuse during backtest
# ---------------------------------------------------------------------------

def test_realtime_snapshot_blocked_when_backtesting() -> None:
    from darwintrade.agentic.tooling.toolbox.akshare_adapter import (
        stock_us_famous_spot_em,
        stock_value_em,
    )

    r1 = stock_us_famous_spot_em({"trade_date": "2025-06-30"})
    assert r1["available"] is False
    assert "realtime" in r1["reason"] or "backtest" in r1["reason"]

    r2 = stock_value_em({"symbol": "000001", "trade_date": "2025-06-30"})
    assert r2["available"] is False
    assert "backtest" in r2["reason"] or "point_in_time" in r2["reason"]


# ---------------------------------------------------------------------------
# 5. Akshare daily-history adapter must filter out bars on or after trade_date
# ---------------------------------------------------------------------------

def test_stock_us_daily_filters_trade_date_and_future() -> None:
    from darwintrade.agentic.tooling.toolbox import akshare_adapter

    fake_records = [
        {"date": "2025-06-27", "close": 100.0},
        {"date": "2025-06-28", "close": 101.0},
        {"date": "2025-06-29", "close": 102.0},
        {"date": "2025-06-30", "close": 103.0},  # trade_date — must be dropped
        {"date": "2025-07-01", "close": 104.0},  # strictly future — must be dropped
    ]

    def fake_call_akshare(function_name, payload, **kwargs):
        return {
            "available": True,
            "provider": "akshare",
            "records": list(fake_records),
            "columns": ["date", "close"],
            "row_count": len(fake_records),
        }

    with patch.object(akshare_adapter, "_call_akshare", side_effect=fake_call_akshare):
        result = akshare_adapter.stock_us_daily(
            {"symbol": "AAPL", "trade_date": "2025-06-30"}
        )

    kept_dates = [r["date"] for r in result["records"]]
    assert kept_dates == ["2025-06-27", "2025-06-28", "2025-06-29"], kept_dates
    assert result["row_count"] == 3


# ---------------------------------------------------------------------------
# 6. News filter must treat trade_date itself as future information
# ---------------------------------------------------------------------------

def test_news_filter_excludes_same_day_articles() -> None:
    from darwintrade.integrations.llm.real_data_provider import _is_future_article

    trade_date = "2025-06-30"
    # ISO published_utc string
    assert _is_future_article("2025-06-30T09:15:00Z", trade_date) is True
    assert _is_future_article("2025-06-30T04:00:00Z", trade_date) is True
    assert _is_future_article("2025-06-29T20:00:00Z", trade_date) is False
    assert _is_future_article("2025-07-01T00:00:00Z", trade_date) is True


# ---------------------------------------------------------------------------
# 8. Outcome feedback reaches the evolution loops one bar after it is measurable
# ---------------------------------------------------------------------------

def test_capsule_commit_lags_one_bar() -> None:
    # A return measured from the day-t open is the fill reference for day-t
    # orders, so it may only reach the capsules on bar t+1.
    from darwintrade.pipeline import DarwinTradePipeline

    pipe = DarwinTradePipeline.__new__(DarwinTradePipeline)
    pipe.analyst_memory = MagicMock()
    pipe.analyst_memory.commit_realized.return_value = 1
    pipe._pending_capsule_commit = None
    pipe._prev_bar_trade_date = "2025-06-27"
    pipe._prev_bar_prices = {"AAPL": 100.0}

    DarwinTradePipeline._stage_capsule_commit(pipe, {"AAPL": 101.0})
    pipe.analyst_memory.commit_realized.assert_not_called()
    assert pipe._pending_capsule_commit is not None
    assert pipe._pending_capsule_commit[0] == "2025-06-27"

    DarwinTradePipeline._flush_capsule_commit(pipe)
    pipe.analyst_memory.commit_realized.assert_called_once()
    assert pipe.analyst_memory.commit_realized.call_args.kwargs[
        "prediction_trade_date"
    ] == "2025-06-27"
    assert pipe._pending_capsule_commit is None


def test_strategy_releases_outcome_one_bar_late() -> None:
    # The (nav, alpha) pair spanning (t-1 open, t open) must not be visible to
    # the loops that shape the day-t order book.
    from backtest.stockbench.strategies.darwintrade import DarwinTradeStrategy

    strat = DarwinTradeStrategy.__new__(DarwinTradeStrategy)
    strat._prev_nav = None
    strat._prev_open_map = None
    strat._lagged_outcome = None

    def step(nav: float, px: float) -> tuple[float | None, float | None]:
        # Mirrors on_bar: release, then roll the previous-bar references forward.
        released = DarwinTradeStrategy._release_outcome(strat, nav, {"AAPL": px})
        strat._prev_nav = nav
        strat._prev_open_map = {"AAPL": px}
        return released

    assert step(100_000.0, 100.0) == (None, None)
    assert step(101_000.0, 110.0) == (None, None), "the (t-1, t) pair is withheld"

    nav, alpha = step(102_000.0, 121.0)
    assert nav == 101_000.0, "bar t sees the outcome measured through t-1"
    assert alpha == pytest.approx(0.01 - 0.10)
