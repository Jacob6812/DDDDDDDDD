"""
Tests for the live decision layer (`darwintrade/live/`).

Coverage:
- symbol normalization and request caps
- trade-date resolution (explicit vs latest trading day)
- market-context assembly: prices, benchmark trend, returns history
- look-ahead safety: benchmark and returns history stay strictly before the bar
- mark-forward equity, so the evolution loops learn from a real return
- outcome release timing (one bar behind, matching the backtest)
- session persistence and the forward-only replay guard
- API contract: health, session lifecycle, error codes

The LLM is stubbed by the autouse fixture in conftest.py. Tests that build a
market context need cached bars, so they skip when the cache is absent rather
than failing on a fresh clone.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from darwintrade.live.config import MAX_SYMBOLS_PER_REQUEST, LiveSettings
from darwintrade.live.context import (
    RETURNS_LOOKBACK_DAYS,
    MarketContext,
    MarketDataUnavailable,
    resolve_trade_date,
)
from darwintrade.live.session import (
    SessionError,
    SessionState,
    SessionStore,
    _equal_weight_return,
    _normalize_symbols,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "storage" / "cache"

# A date inside the paper's cached window, with the symbols known to be cached.
CACHED_DATE = "2025-06-25"
CACHED_SYMBOLS = ["AAPL", "MSFT", "JPM"]


def _has_cache() -> bool:
    return (CACHE_ROOT / "parquet" / "SPY" / "day").is_dir()


needs_cache = pytest.mark.skipif(
    not _has_cache(), reason="requires the pre-cached parquet bars"
)


@pytest.fixture
def settings(tmp_path: Path) -> LiveSettings:
    """Offline settings writing all state under tmp_path."""
    return dataclasses.replace(
        LiveSettings.from_env(),
        data_mode="offline_only",
        cache_root=CACHE_ROOT,
        session_root=tmp_path / "sessions",
    )


@pytest.fixture
def offline_data(settings: LiveSettings):
    """Point the data layer at the cache without touching third-party APIs."""
    from backtest.stockbench.core import data_hub
    from darwintrade.integrations.llm import real_data_provider

    prev_mode = data_hub._DATA_MODE
    prev_root = data_hub.get_cache_root()
    data_hub.set_cache_root(str(settings.cache_root))
    data_hub.set_data_mode("offline_only")
    real_data_provider.set_cache_root(settings.cache_root)
    real_data_provider.set_bars_data_mode("offline_only")
    yield
    data_hub.set_cache_root(prev_root)
    data_hub.set_data_mode(prev_mode)
    real_data_provider.set_cache_root(None)
    real_data_provider.set_bars_data_mode("offline_only")


# ---------------------------------------------------------------------------
# symbol handling
# ---------------------------------------------------------------------------

def test_normalize_symbols_splits_and_dedupes() -> None:
    assert _normalize_symbols(["aapl, msft", " jpm "]) == ["AAPL", "MSFT", "JPM"]
    assert _normalize_symbols(["AAPL", "aapl"]) == ["AAPL"]


def test_normalize_symbols_rejects_empty() -> None:
    with pytest.raises(SessionError):
        _normalize_symbols([])
    with pytest.raises(SessionError):
        _normalize_symbols(["   ", ","])


def test_normalize_symbols_enforces_cap() -> None:
    too_many = [f"SYM{i}" for i in range(MAX_SYMBOLS_PER_REQUEST + 1)]
    with pytest.raises(SessionError, match="Too many symbols"):
        _normalize_symbols(too_many)


# ---------------------------------------------------------------------------
# trade-date resolution
# ---------------------------------------------------------------------------

def test_explicit_trade_date_is_honoured(settings: LiveSettings) -> None:
    resolved, inferred = resolve_trade_date("2025-06-25", settings)
    assert resolved == "2025-06-25"
    assert inferred is False


def test_inferred_trade_date_is_a_trading_day(settings: LiveSettings) -> None:
    from backtest.stockbench.core.data_hub import is_trading_day
    import pandas as pd

    resolved, inferred = resolve_trade_date(None, settings)
    assert inferred is True
    assert is_trading_day(pd.Timestamp(resolved))


# ---------------------------------------------------------------------------
# market context
# ---------------------------------------------------------------------------

@needs_cache
def test_context_resolves_prices_and_history(
    settings: LiveSettings, offline_data
) -> None:
    ctx = MarketContext.build(
        trade_date=CACHED_DATE, symbols=CACHED_SYMBOLS, settings=settings
    )
    assert set(ctx.usable_symbols) == set(CACHED_SYMBOLS)
    assert all(px > 0 for px in ctx.open_map.values())
    # asset_data carries the decision-time price the analysts see
    for sym in CACHED_SYMBOLS:
        assert ctx.asset_data[sym]["price"] == ctx.open_map[sym]
    # the allocator's optimizers need >= 20 samples to be usable
    for rets in ctx.returns_history.values():
        assert 20 <= len(rets) <= RETURNS_LOOKBACK_DAYS


@needs_cache
def test_context_reports_unknown_symbols_without_failing(
    settings: LiveSettings, offline_data
) -> None:
    ctx = MarketContext.build(
        trade_date=CACHED_DATE,
        symbols=["AAPL", "NOTATICKER"],
        settings=settings,
    )
    assert ctx.usable_symbols == ["AAPL"]
    assert "NOTATICKER" in ctx.skipped_symbols


@needs_cache
def test_context_raises_when_nothing_is_usable(
    settings: LiveSettings, offline_data
) -> None:
    with pytest.raises(MarketDataUnavailable):
        MarketContext.build(
            trade_date=CACHED_DATE, symbols=["NOPE1", "NOPE2"], settings=settings
        )


@needs_cache
def test_benchmark_snapshot_has_trend_backbone(
    settings: LiveSettings, offline_data
) -> None:
    ctx = MarketContext.build(
        trade_date=CACHED_DATE, symbols=["AAPL"], settings=settings
    )
    # A single day's change flips the regime label on ~70% of days; the 5/20-day
    # trends are what give the classifier a stable backbone.
    assert ctx.benchmark["trend_5d"] is not None
    assert ctx.benchmark["trend_20d"] is not None
    assert ctx.market_data["spy_trend_20d_pct"] == ctx.benchmark["trend_20d"]


@needs_cache
def test_returns_history_excludes_the_trade_date(
    settings: LiveSettings, offline_data
) -> None:
    """Look-ahead guard: history must end strictly before the decision bar."""
    import math

    from backtest.stockbench.core.data_hub import get_bars

    ctx = MarketContext.build(
        trade_date=CACHED_DATE, symbols=["AAPL"], settings=settings
    )
    bars = get_bars(
        "AAPL", "2025-06-20", CACHED_DATE, 1, "day", True,
        cfg=settings.data_hub_cfg(),
    )
    closes = [float(c) for c in bars["close"].tolist() if c == c and float(c) > 0]
    # the return spanning into CACHED_DATE must not appear in the history
    leaked = round(math.log(closes[-1] / closes[-2]), 10)
    history = [round(r, 10) for r in ctx.returns_history["AAPL"]]
    assert leaked not in history


# ---------------------------------------------------------------------------
# outcome accounting
# ---------------------------------------------------------------------------

def test_equal_weight_return_ignores_missing_prices() -> None:
    assert _equal_weight_return({}, {"A": 10.0}) is None
    # B is missing today, C is missing yesterday: only A contributes
    result = _equal_weight_return({"A": 100.0, "B": 50.0}, {"A": 110.0, "C": 20.0})
    assert result == pytest.approx(0.10)


def test_mark_forward_moves_equity_with_the_previous_book(
    settings: LiveSettings,
) -> None:
    """The evolution loops must see a real return.

    Without marking the previous target book forward, portfolio_return is a
    constant 0.0 while the market moves, and every memory layer trains on a
    signal that does not exist.
    """
    store = SessionStore(settings)
    session = store.create(capital=100_000.0)
    session.state.prev_open_map = {"AAPL": 100.0, "MSFT": 200.0}
    session.state.prev_target_weights = {"AAPL": 0.5, "MSFT": -0.25}

    # AAPL +10% on a 0.5 long, MSFT +20% on a 0.25 short => 5% - 5% = 0%
    flat = session._mark_forward({"AAPL": 110.0, "MSFT": 240.0})
    assert flat == pytest.approx(100_000.0)

    # AAPL +10% long only => +5%
    session.state.prev_target_weights = {"AAPL": 0.5}
    up = session._mark_forward({"AAPL": 110.0, "MSFT": 200.0})
    assert up == pytest.approx(105_000.0)


def test_mark_forward_is_flat_without_a_previous_book(
    settings: LiveSettings,
) -> None:
    store = SessionStore(settings)
    session = store.create(capital=75_000.0)
    assert session._mark_forward({"AAPL": 110.0}) == pytest.approx(75_000.0)


def test_outcome_is_released_one_bar_late(settings: LiveSettings) -> None:
    """Matches the backtest: the (t-1, t) pair is only measurable once bar t's
    auction has printed, so it is released on the following bar."""
    store = SessionStore(settings)
    session = store.create(capital=100_000.0)

    # first bar: nothing to release, nothing to compute
    assert session._release_outcome(100_000.0, {"AAPL": 100.0}) == (None, None)

    # second bar: computes a pair but still releases nothing
    session.state.prev_nav = 100_000.0
    session.state.prev_open_map = {"AAPL": 100.0}
    assert session._release_outcome(101_000.0, {"AAPL": 105.0}) == (None, None)
    assert session.state.lagged_outcome is not None

    # third bar: the earlier pair is now released
    nav, alpha = session._release_outcome(101_000.0, {"AAPL": 105.0})
    assert nav == pytest.approx(101_000.0)
    assert alpha is not None


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def test_session_round_trips_through_disk(settings: LiveSettings) -> None:
    store = SessionStore(settings)
    session = store.create(capital=42_000.0, label="round trip")
    session.state.decisions = 3
    session.state.last_trade_date = "2025-06-25"
    session.save()

    # a fresh store must reload it from disk, not from memory
    reloaded = SessionStore(settings).get(session.session_id)
    assert reloaded.state.capital == pytest.approx(42_000.0)
    assert reloaded.state.label == "round trip"
    assert reloaded.state.decisions == 3
    assert reloaded.state.last_trade_date == "2025-06-25"


def test_unknown_session_raises(settings: LiveSettings) -> None:
    with pytest.raises(SessionError):
        SessionStore(settings).get("does-not-exist")


def test_session_state_ignores_unknown_fields() -> None:
    state = SessionState.from_dict(
        {"session_id": "abc", "created_at": "now", "capital": 1.0, "bogus": 9}
    )
    assert state.session_id == "abc"
    assert not hasattr(state, "bogus")


def test_session_listing_reports_saved_sessions(settings: LiveSettings) -> None:
    store = SessionStore(settings)
    store.create(capital=1000.0, label="one")
    store.create(capital=2000.0, label="two")
    labels = {row["label"] for row in store.list()}
    assert {"one", "two"} <= labels


@needs_cache
def test_session_refuses_to_go_backwards(
    settings: LiveSettings, offline_data
) -> None:
    """Memory evolves forward only; replaying an earlier bar would corrupt it."""
    store = SessionStore(settings)
    session = store.create()
    session.state.last_trade_date = "2025-06-25"
    with pytest.raises(SessionError, match="forward only"):
        session.decide(symbols=["AAPL"], trade_date="2025-06-02")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@pytest.fixture
def client(settings: LiveSettings):
    from fastapi.testclient import TestClient

    from darwintrade.live.api import create_app

    return TestClient(create_app(settings))


def test_health_reports_configuration(client) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["max_symbols"] == MAX_SYMBOLS_PER_REQUEST
    assert len(payload["default_universe"]) == 20


def test_session_endpoints(client) -> None:
    created = client.post("/api/sessions", json={"capital": 5000, "label": "api"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    assert client.get(f"/api/sessions/{session_id}").status_code == 200
    assert client.get("/api/sessions/missing").status_code == 404
    assert any(
        row["session_id"] == session_id
        for row in client.get("/api/sessions").json()["sessions"]
    )


def test_decide_requires_credentials(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an LLM key the endpoint must say so, not fail deep in the graph."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    response = client.post("/api/decide", json={"symbols": ["AAPL"]})
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]


def test_front_end_is_served(client) -> None:
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/sample-report.json").json()["sample"] is True


def test_analyst_rationale_reaches_signal():
    from types import SimpleNamespace
    from darwintrade.agents.market import _thesis_from_packet
    packet = SimpleNamespace(asset_view=SimpleNamespace(rationale="Evidence from the analyst team"))
    assert _thesis_from_packet(packet) == "Evidence from the analyst team"


def test_same_day_does_not_advance_memory(settings, monkeypatch):
    session = SessionStore(settings).create()
    session.state.last_trade_date = CACHED_DATE
    def unexpected(**kwargs):
        pytest.fail("Repeated dates must fail before fetching data or running analysts")
    monkeypatch.setattr(MarketContext, "build", unexpected)
    with pytest.raises(SessionError, match="forward only"):
        session.decide(symbols=["AAPL"], trade_date=CACHED_DATE)


@pytest.mark.parametrize("symbols", [["../secret"], ["AAPL/../../x"], ["<script>"]])
def test_rejects_unsafe_tickers(symbols):
    with pytest.raises(SessionError, match="Invalid symbol"):
        _normalize_symbols(symbols)


@pytest.mark.parametrize("trade_date", ["invalid", "2025-02-30", "2025-06-28", "2999-01-01"])
def test_invalid_date_is_actionable(settings, trade_date):
    with pytest.raises(MarketDataUnavailable):
        resolve_trade_date(trade_date, settings)


def test_invalid_request_does_not_create_session(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-only")
    before = client.get("/api/sessions").json()
    response = client.post("/api/decide", json={"symbols": []})
    assert response.status_code == 400
    assert client.get("/api/sessions").json() == before


def test_api_decision_and_continuation_without_private_cache(client, monkeypatch):
    """Exercise HTTP, pipeline allocation, report serialization and disk reload.

    Only market inputs and analyst signals are fixtures; the session and
    allocator are real. This runs on a fresh clone without provider access.
    """
    from darwintrade.agents.market import MarketAgent
    from darwintrade.core.contracts import AssetSignal

    monkeypatch.setenv("LLM_API_KEY", "test-only")
    def context(**kwargs):
        price = 100.0 if kwargs["trade_date"] == "2025-06-25" else 110.0
        return MarketContext(
            trade_date=kwargs["trade_date"], open_map={"AAPL": price},
            market_data={}, asset_data={"AAPL": {"price": price}},
            returns_history={}, usable_symbols=["AAPL"],
        )
    def signals(self, **kwargs):
        return [AssetSignal(symbol="AAPL", trade_date=kwargs["trade_date"], direction="long", confidence=0.8, thesis="Fixture evidence")]
    monkeypatch.setattr(MarketContext, "build", context)
    monkeypatch.setattr(MarketAgent, "analyze", signals)
    first = client.post("/api/decide", json={"symbols": ["AAPL"], "trade_date": "2025-06-25", "capital": 10000})
    assert first.status_code == 200, first.text
    report = first.json()
    assert report["positions"][0]["symbol"] == "AAPL"
    weight = report["positions"][0]["target_weight"]
    second = client.post("/api/decide", json={"symbols": ["AAPL"], "trade_date": "2025-06-26", "session_id": report["session_id"]})
    assert second.status_code == 200, second.text
    assert second.json()["capital"] == pytest.approx(10000 * (1 + weight * 0.1), abs=0.01)
    assert second.json()["decision_index"] == 2
    restored = SessionStore(client.app.state.settings).get(report["session_id"])
    assert restored.state.decisions == 2
    assert len(restored.state.history) == 2
