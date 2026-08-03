"""Tests for the IC-shrinkage analyst aggregator and its underlying
AnalystCapsule. The capsule API changed: it no longer exposes
beta/bias/precision; the aggregator now uses
max(0, IC) * shrink(n) as the role weight on a unit-stable
implied-return mapping. Negative-IC capsules contribute zero (drop, never
invert)."""
from __future__ import annotations

import math
import random
import tempfile

import pytest

from darwintrade.memory.analyst_capsules import (
    AnalystCapsule,
    CAPSULE_WARMUP_N,
    PRIOR_REALIZED_SD,
    SCORE_SD_FLOOR,
)
from darwintrade.memory.analyst_memory import AnalystMemory
from darwintrade.agents.bayesian_aggregator import aggregate


# ---------------------------------------------------------------------------
# Capsule math
# ---------------------------------------------------------------------------

class TestAnalystCapsule:
    def test_cold_start_capsule_has_zero_skill_weight(self):
        c = AnalystCapsule("bull", "market_analyst")
        assert c.n == 0
        assert c.is_warm is False
        assert c.ic == 0.0
        assert c.skill_weight == 0.0
        # Score sd floored, realized sd at prior so unit conversion is safe.
        assert c.score_sd == SCORE_SD_FLOOR
        assert c.realized_sd == PRIOR_REALIZED_SD
        assert c.implied_return(0.5) == 0.0

    def test_observe_clips_extreme_realized_returns(self):
        c = AnalystCapsule("bull", "market_analyst")
        c.observe(predicted_score=0.5, predicted_conf=0.8, realized_return=10.0,
                  trade_date="2025-01-02")
        assert len(c._samples) == 1
        _pred, _conf, realized, _date = c._samples[0]
        assert realized == 0.20

    def test_warmup_threshold(self):
        c = AnalystCapsule("bull", "news_analyst")
        for i in range(CAPSULE_WARMUP_N - 1):
            # use distinct trade_dates so demean does not collapse the sample
            c.observe(0.1, 0.5, 0.001, trade_date=f"2025-{i:03d}")
        assert c.is_warm is False
        c.observe(0.1, 0.5, 0.001, trade_date="2025-final")
        assert c.is_warm is True

    def test_skilled_capsule_has_positive_ic(self):
        # "Good" analyst tracks realized direction with bounded noise.
        # IC requires daily cross-sectional dispersion, so feed multiple
        # samples per trade_date.
        good = AnalystCapsule("bull", "good")
        bad = AnalystCapsule("bull", "bad")
        rng = random.Random(0)
        for day in range(40):
            for sym in range(4):
                real = rng.choice([-0.012, -0.005, 0.0, 0.006, 0.012, 0.018])
                good_score = max(-1.0, min(1.0, 30 * real + rng.gauss(0, 0.05)))
                bad_score = rng.gauss(0, 0.4)
                good.observe(predicted_score=good_score,
                             predicted_conf=0.7, realized_return=real,
                             trade_date=f"2025-{day:03d}")
                bad.observe(predicted_score=bad_score,
                            predicted_conf=0.7, realized_return=real,
                            trade_date=f"2025-{day:03d}")
        assert good.ic > bad.ic
        assert good.skill_weight > bad.skill_weight
        # A skilled analyst's mapped implied return shares the sign of
        # a same-sign new score.
        assert good.implied_return(0.5) > 0
        assert good.implied_return(-0.5) < 0

    def test_negative_ic_capsule_contributes_zero(self):
        # Anti-skilled analyst: prediction always opposite of realized.
        c = AnalystCapsule("bear", "contrarian")
        rng = random.Random(11)
        for day in range(30):
            for _ in range(3):
                real = rng.choice([-0.01, 0.0, 0.01])
                c.observe(predicted_score=-30 * real,  # inverted
                          predicted_conf=0.7,
                          realized_return=real,
                          trade_date=f"2025-{day:03d}")
        # IC is negative but skill_weight is clipped to 0 (down-weight, do
        # NOT invert — sign-flipping noise has historically been worse).
        assert c.ic < 0
        assert c.skill_weight == 0.0
        # implied_return for a negative-IC capsule is also clamped to 0.
        assert c.implied_return(0.7) == 0.0
        assert c.implied_return(-0.7) == 0.0

    def test_daily_cross_sectional_demean(self):
        # Two different "days" each with high common drift: every analyst
        # says +0.5 and every realized is +0.02, but no cross-sectional
        # variation on day 1; on day 2 every prediction is -0.5 and every
        # realized is -0.02. After demean both days contribute zero
        # cross-sectional variance, so IC should be 0 (no skill detectable).
        c = AnalystCapsule("bull", "drift_only")
        for sym in range(4):
            c.observe(0.5, 0.7, 0.02, trade_date="2025-001")
        for sym in range(4):
            c.observe(-0.5, 0.7, -0.02, trade_date="2025-002")
        # Demean leaves both days with zero variance → IC=0.
        assert c.ic == 0.0
        assert c.skill_weight == 0.0

    def test_serialization_roundtrip(self):
        c = AnalystCapsule("bull", "fundamentals_analyst")
        for i in range(10):
            c.observe(
                0.1 * (1 if i % 2 else -1),
                0.5,
                0.005 * (1 if i % 2 else -1),
                trade_date=f"2025-{i:03d}",
            )
        d = c.to_dict()
        restored = AnalystCapsule.from_dict(d)
        assert restored.regime == "bull"
        assert restored.role == "fundamentals_analyst"
        assert restored.n == 10
        assert restored.ic == pytest.approx(c.ic, rel=1e-6, abs=1e-9)
        assert restored.skill_weight == pytest.approx(c.skill_weight, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# Memory: pending → commit
# ---------------------------------------------------------------------------

class TestAnalystMemory:
    def test_stage_and_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = AnalystMemory(tmp)
            mem.stage_prediction(
                trade_date="2025-04-01", regime="bull", role="market_analyst",
                symbol="AAPL", predicted_score=0.4, predicted_confidence=0.8,
            )
            mem.stage_prediction(
                trade_date="2025-04-01", regime="bull", role="news_analyst",
                symbol="AAPL", predicted_score=-0.2, predicted_confidence=0.6,
            )
            committed = mem.commit_realized(
                prediction_trade_date="2025-04-01",
                realized_returns={"AAPL": 0.012},
            )
            assert committed == 2
            cap = mem.get_or_create("bull", "market_analyst")
            assert cap.n == 1
            # trade_date is propagated into the sample tuple
            _p, _c, _r, date = cap._samples[0]
            assert date == "2025-04-01"

    def test_unmatched_symbols_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = AnalystMemory(tmp)
            mem.stage_prediction(
                trade_date="2025-04-01", regime="bull", role="market_analyst",
                symbol="AAPL", predicted_score=0.4, predicted_confidence=0.8,
            )
            committed = mem.commit_realized(
                prediction_trade_date="2025-04-01",
                realized_returns={"MSFT": 0.012},  # different symbol
            )
            assert committed == 0

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = AnalystMemory(tmp)
            mem.stage_prediction(
                trade_date="2025-04-01", regime="bull", role="market_analyst",
                symbol="AAPL", predicted_score=0.4, predicted_confidence=0.8,
            )
            mem.commit_realized(
                prediction_trade_date="2025-04-01",
                realized_returns={"AAPL": 0.012},
            )
            mem2 = AnalystMemory(tmp)
            cap = mem2.get_or_create("bull", "market_analyst")
            assert cap.n == 1


# ---------------------------------------------------------------------------
# IC-shrinkage aggregator
# ---------------------------------------------------------------------------

def _make_skilled_capsule(regime: str, role: str, *, slope: float = 30.0,
                          noise: float = 0.03, n_days: int = 40,
                          per_day: int = 4, seed: int = 42) -> AnalystCapsule:
    """Build a calibrated capsule whose IC > 0 by simulating
    score = slope * realized + small noise, with daily cross-sectional
    variation over multiple symbols per day."""
    c = AnalystCapsule(regime, role)
    rng = random.Random(seed)
    for day in range(n_days):
        for _ in range(per_day):
            real = rng.choice([-0.012, -0.005, 0.0, 0.006, 0.012, 0.018])
            score = max(-1.0, min(1.0, slope * real + rng.gauss(0, noise)))
            c.observe(predicted_score=score, predicted_conf=0.7,
                      realized_return=real, trade_date=f"2025-{day:03d}")
    return c


class TestICShrinkageAggregator:
    def _lookup_factory(self, capsules: dict[tuple[str, str], AnalystCapsule]):
        def _lookup(regime: str, role: str) -> AnalystCapsule:
            key = (regime, role)
            if key not in capsules:
                capsules[key] = AnalystCapsule(regime, role)
            return capsules[key]
        return _lookup

    def test_empty_payloads_returns_hold(self):
        out = aggregate(role_payloads={}, regime="bull",
                        capsule_lookup=self._lookup_factory({}))
        assert out["direction"] == "HOLD"
        assert out["aggregate_confidence"] == 0.0
        assert out["aggregator"] == "ic_shrinkage_capsule"

    def test_cold_start_falls_back_to_confidence_weighted(self):
        # Pre-warmup capsules now fall back to confidence-weighted aggregation
        # rather than returning HOLD. With all-positive scores and good
        # confidence the result should be BUY.
        out = aggregate(
            role_payloads={
                "market_analyst":       {"score": 0.6, "confidence": 0.8},
                "news_analyst":         {"score": 0.5, "confidence": 0.7},
                "fundamentals_analyst": {"score": 0.55, "confidence": 0.75},
                "social_analyst":       {"score": 0.4, "confidence": 0.6},
            },
            regime="bull",
            capsule_lookup=self._lookup_factory({}),
        )
        # All 4 cold-start roles contribute via confidence-weighted fallback.
        assert out["n_contributing_roles"] == 4
        assert out["direction"] == "BUY"
        assert out["posterior_mean"] > 0

    def test_unanimous_buy_with_calibrated_capsules(self):
        capsules = {}
        for role in ("market_analyst", "news_analyst",
                     "fundamentals_analyst", "social_analyst"):
            capsules[("bull", role)] = _make_skilled_capsule(
                "bull", role, seed=hash(role) & 0xFFFF
            )
        out = aggregate(
            role_payloads={
                "market_analyst":       {"score": 0.6, "confidence": 0.8},
                "news_analyst":         {"score": 0.5, "confidence": 0.7},
                "fundamentals_analyst": {"score": 0.55, "confidence": 0.75},
                "social_analyst":       {"score": 0.4, "confidence": 0.6},
            },
            regime="bull",
            capsule_lookup=self._lookup_factory(capsules),
        )
        assert out["direction"] == "BUY"
        assert out["posterior_mean"] > 0
        assert out["aggregate_confidence"] > 0.0

    def test_low_confidence_role_is_filtered(self):
        capsules = {}
        capsules[("bull", "market_analyst")] = _make_skilled_capsule(
            "bull", "market_analyst", seed=1)
        capsules[("bull", "news_analyst")] = _make_skilled_capsule(
            "bull", "news_analyst", seed=2)
        out = aggregate(
            role_payloads={
                "market_analyst": {"score":  0.5, "confidence": 0.7},
                "news_analyst":   {"score": -0.5, "confidence": 0.02},  # abstain
            },
            regime="bull",
            capsule_lookup=self._lookup_factory(capsules),
        )
        assert out["n_contributing_roles"] == 1
        assert out["direction"] in ("BUY", "HOLD")
        # The remaining role had positive score → posterior_mean > 0
        assert out["posterior_mean"] > 0

    def test_negative_ic_role_drops_out_does_not_invert(self):
        # One skilled capsule and one negatively-skilled capsule. The
        # negatively-skilled one must drop out (skill_weight=0), not invert
        # the score. So the aggregate follows the skilled role.
        capsules = {}
        skilled = _make_skilled_capsule("bear", "market_analyst", seed=7)
        capsules[("bear", "market_analyst")] = skilled

        bad = AnalystCapsule("bear", "fundamentals_analyst")
        rng = random.Random(13)
        for day in range(30):
            for _ in range(3):
                real = rng.choice([-0.01, 0.0, 0.01])
                bad.observe(predicted_score=-30 * real,  # anti-correlated
                            predicted_conf=0.7,
                            realized_return=real,
                            trade_date=f"2025-{day:03d}")
        capsules[("bear", "fundamentals_analyst")] = bad
        assert bad.ic < 0
        assert bad.skill_weight == 0.0

        out = aggregate(
            role_payloads={
                "market_analyst":       {"score":  0.6, "confidence": 0.7},  # BUY
                "fundamentals_analyst": {"score": -0.6, "confidence": 0.7},  # would invert under old code
            },
            regime="bear",
            capsule_lookup=self._lookup_factory(capsules),
        )
        # Only the skilled role contributes; the anti-skilled one drops out.
        assert out["n_contributing_roles"] == 1
        assert out["posterior_mean"] > 0  # follows skilled BUY signal

    def test_warm_capsule_dominates_cold_capsule(self):
        # A warm calibrated capsule has skill_weight >> COLD_START_WEIGHT so
        # its direction should dominate the cold role's opposite signal.
        capsules = {}
        capsules[("bull", "fundamentals_analyst")] = _make_skilled_capsule(
            "bull", "fundamentals_analyst", seed=3)
        out = aggregate(
            role_payloads={
                "fundamentals_analyst": {"score":  0.4, "confidence": 0.7},  # warm, BUY
                "social_analyst":       {"score": -0.9, "confidence": 0.9},  # cold, strong SELL
            },
            regime="bull",
            capsule_lookup=self._lookup_factory(capsules),
        )
        # Both contribute, but the warm role's skill_weight >> COLD_START_WEIGHT.
        assert out["n_contributing_roles"] == 2
        # The warm skilled BUY should dominate the cold SELL.
        assert out["posterior_mean"] > 0

    def test_disagreement_among_skilled_yields_softer_posterior(self):
        # Two equally-skilled capsules giving opposite signals → posterior
        # near zero, HOLD.
        capsules = {}
        capsules[("bull", "market_analyst")] = _make_skilled_capsule(
            "bull", "market_analyst", seed=21)
        capsules[("bull", "news_analyst")] = _make_skilled_capsule(
            "bull", "news_analyst", seed=22)
        out = aggregate(
            role_payloads={
                "market_analyst": {"score":  0.6, "confidence": 0.7},
                "news_analyst":   {"score": -0.6, "confidence": 0.7},
            },
            regime="bull",
            capsule_lookup=self._lookup_factory(capsules),
        )
        assert abs(out["posterior_mean"]) < 0.05
        # Direction is HOLD or close to it given the cancellation.
        assert out["direction"] == "HOLD"

    def test_verification_fail_raises_threshold(self):
        # Build a calibrated capsule and put a single moderate signal through.
        # PASS allows |IR| >= 1.0 → BUY/SELL; FAIL requires |IR| >= 1.5.
        capsules = {}
        capsules[("bull", "market_analyst")] = _make_skilled_capsule(
            "bull", "market_analyst", seed=31)
        payloads = {"market_analyst": {"score": 0.3, "confidence": 0.5}}
        cap = self._lookup_factory(capsules)
        with_pass = aggregate(role_payloads=payloads, regime="bull",
                              capsule_lookup=cap, verification_verdict="PASS")
        with_fail = aggregate(role_payloads=payloads, regime="bull",
                              capsule_lookup=cap, verification_verdict="FAIL")
        # IR is identical across the two calls (same posterior); only the
        # decision threshold differs.
        assert with_pass["information_ratio"] == pytest.approx(
            with_fail["information_ratio"], rel=1e-6
        )
        if 1.0 <= with_pass["information_ratio"] < 1.5:
            assert with_pass["direction"] == "BUY"
            assert with_fail["direction"] == "HOLD"

    def test_information_ratio_corresponds_to_decision_confidence(self):
        capsules = {}
        for role in ("market_analyst", "news_analyst",
                     "fundamentals_analyst", "social_analyst"):
            capsules[("bull", role)] = _make_skilled_capsule(
                "bull", role, seed=hash(role) & 0xFFFF)
        out = aggregate(
            role_payloads={
                "market_analyst":       {"score": 0.5, "confidence": 0.7},
                "news_analyst":         {"score": 0.5, "confidence": 0.7},
                "fundamentals_analyst": {"score": 0.5, "confidence": 0.7},
                "social_analyst":       {"score": 0.5, "confidence": 0.7},
            },
            regime="bull",
            capsule_lookup=self._lookup_factory(capsules),
        )
        assert out["information_ratio"] > 0
        assert out["aggregate_confidence"] > 0


__all__ = []
