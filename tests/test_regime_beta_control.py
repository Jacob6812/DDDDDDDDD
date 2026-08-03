"""Regression tests for the DarwinTrade regime classification fix.

The regime classifier used to key off a single prior-day SPY return with VIX
hardcoded to 0, flipping the label ~70% of days. The heuristic now uses a
multi-day trend backbone (VIX removed); `volatile` means the 5d and 20d trends
disagree in sign (a turning point), not high realized vol.

NOTE: the allocator-side beta/gross exposure mechanisms that this file also
used to cover were removed when the exposure logic was reverted to the simple
"full gross + long/short confidence split" design (which backtested materially
better across the full window). Only the regime-classification fixes remain.
"""
from __future__ import annotations

from darwintrade.agents.regime import RegimeAgent


def _agent() -> RegimeAgent:
    return RegimeAgent.__new__(RegimeAgent)


def test_heuristic_trend_overrides_noisy_up_day():
    """A single green day against a strong downtrend must not flip to bull —
    this is exactly the noise that produced the 70% daily flip rate."""
    r = _agent()._heuristic_regime(
        "2025-03-13",
        {"spy_change_pct": 0.53, "spy_trend_20d_pct": -4.2,
         "spy_trend_5d_pct": -2.1, "vix": 0.0},
    )
    assert r.regime == "bear"


def test_heuristic_trend_overrides_noisy_down_day():
    r = _agent()._heuristic_regime(
        "2025-03-13",
        {"spy_change_pct": -0.2, "spy_trend_20d_pct": 3.5, "vix": 15.0},
    )
    assert r.regime == "bull"


def test_heuristic_trend_disagreement_is_volatile():
    """VIX removed: `volatile` now means the 5d and 20d trends disagree in sign
    (a turning point), NOT high realized vol. A sharp 5d bounce against a 20d
    downtrend is a reversal → volatile."""
    r = _agent()._heuristic_regime(
        "2025-03-13",
        {"spy_change_pct": 1.0, "spy_trend_5d_pct": 3.0, "spy_trend_20d_pct": -5.0},
    )
    assert r.regime == "volatile"


def test_heuristic_high_vol_no_longer_forces_volatile():
    """A clean uptrend must classify bull even in what used to be a high-VIX
    environment — VIX is gone and no longer overrides direction."""
    r = _agent()._heuristic_regime(
        "2025-03-13",
        {"spy_change_pct": 0.1, "spy_trend_5d_pct": 4.0, "spy_trend_20d_pct": 3.5},
    )
    assert r.regime == "bull"


def test_heuristic_falls_back_to_single_day_when_no_trend():
    r = _agent()._heuristic_regime(
        "2025-03-13", {"spy_change_pct": 0.1},
    )
    assert r.regime == "sideways"


def test_heuristic_aligned_trends_follow_backbone():
    """When 5d and 20d agree in sign, follow the trend — a mild 5d dip inside a
    strong 20d downtrend (same sign) stays bear, not volatile."""
    r = _agent()._heuristic_regime(
        "2025-03-13",
        {"spy_change_pct": 1.0, "spy_trend_5d_pct": -0.5, "spy_trend_20d_pct": -5.0},
    )
    assert r.regime == "bear"
