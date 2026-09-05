"""
Tests for the two-layer self-evolving DarwinTrade pipeline.

Coverage:
- TacticalMemory: bar episodes, influence install/expiry
- StrategicMemory: episodes, capsules, baseline computation, review cadence
- combine_influence: strategic baseline + tactical influence merging
- PortfolioAllocator: enforcement of memory_influence (no_trade, haircut)
- PipelineConfig: patch application
"""
from __future__ import annotations

import pytest

from darwintrade.core.contracts import (
    AssetSignal,
    EpisodeRecord,
    MemoryInfluence,
    PortfolioState,
    RegimeState,
    StrategicPatch,
    TacticalInfluence,
    TacticalReflection,
)
from darwintrade.memory.combined import combine_influence
from darwintrade.memory.strategic import (
    STRATEGIC_REVIEW_EVERY_N_DAYS,
    StrategicMemory,
    _dd_metrics,
    _outcome_score,
)
from darwintrade.memory.tactical import TacticalMemory
from darwintrade.portfolio.allocator import PortfolioAllocator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode(
    *,
    trade_date="2024-01-15",
    regime="bull",
    optimizer="kelly_capped",
    score=1.0,
    nav=100000.0,
    dd=0.0,
    in_dd=False,
    in_reb=False,
) -> EpisodeRecord:
    return EpisodeRecord(
        trade_date=trade_date,
        symbols=["AAPL"],
        regime=regime,
        optimizer_used=optimizer,
        outcome_score=score,
        nav=nav,
        dd_from_peak=dd,
        in_drawdown=in_dd,
        in_rebound=in_reb,
    )


# ---------------------------------------------------------------------------
# TacticalMemory
# ---------------------------------------------------------------------------

class TestTacticalMemory:
    def test_install_and_expire(self, tmp_path):
        tm = TacticalMemory(tmp_path)
        tm.install_influence(TacticalInfluence(
            avoid_symbols=["TSLA"], position_haircut=0.5, expires_in_days=2,
        ))
        # First read after install: still active
        infl = tm.compute_influence()
        assert "TSLA" in infl.avoid_symbols
        tm.tick_expiry()  # remaining: 1

        # Day 2: still active
        infl = tm.compute_influence()
        assert "TSLA" in infl.avoid_symbols
        tm.tick_expiry()  # remaining: 0

        # Day 3: expired
        infl = tm.compute_influence()
        assert infl.avoid_symbols == []
        assert infl.position_haircut == 1.0

    def test_recent_summary_aggregates(self, tmp_path):
        tm = TacticalMemory(tmp_path)
        for d, score in zip(["2024-01-01", "2024-01-02"], [-0.5, -0.3]):
            tm.write_bar(
                episode=_episode(trade_date=d, score=score, in_dd=True, dd=-0.05),
                reflection=TacticalReflection(trade_date=d, mistake_type="over_exposure", summary=""),
            )
        summary = tm.recent_summary(days=5)
        assert summary["days"] == 2
        assert summary["drawdown_count"] == 2
        assert summary["mistake_counts"].get("over_exposure", 0) == 2

    def test_persistence_roundtrip(self, tmp_path):
        tm = TacticalMemory(tmp_path)
        tm.install_influence(TacticalInfluence(avoid_symbols=["X"], expires_in_days=3))
        # reload from disk
        tm2 = TacticalMemory(tmp_path)
        infl = tm2.compute_influence()
        assert "X" in infl.avoid_symbols


# ---------------------------------------------------------------------------
# StrategicMemory
# ---------------------------------------------------------------------------

class TestStrategicMemory:
    def test_capsule_activation(self, tmp_path):
        sm = StrategicMemory(tmp_path)
        for i in range(10):
            sm.write_episode(_episode(trade_date=f"2024-01-{i+1:02d}", score=1.5, nav=100000.0 + i * 100))
        baseline = sm.compute_strategic_baseline("bull", current_nav=110000.0)
        assert baseline.preferred_optimizer == "kelly_capped"
        assert baseline.source_layer == "strategic"

    def test_drawdown_tightens_baseline(self, tmp_path):
        sm = StrategicMemory(tmp_path)
        for nav in [100000, 99000, 97000, 95000, 93000, 91000, 90000]:
            sm.write_episode(_episode(
                trade_date="2024-01-01", score=-0.5, nav=float(nav), dd=-0.10, in_dd=True,
            ))
        baseline = sm.compute_strategic_baseline("bear", current_nav=90000.0)
        assert baseline.constraint_overrides.get("max_gross_exposure", 1.0) <= 0.75

    def test_review_cadence(self, tmp_path):
        sm = StrategicMemory(tmp_path)
        # accumulate just under threshold
        for i in range(STRATEGIC_REVIEW_EVERY_N_DAYS - 1):
            sm.write_episode(_episode(trade_date=f"2024-01-{i+1:02d}"))
        assert sm.should_run_strategic_review() is False

        # one more crosses the threshold
        sm.write_episode(_episode(trade_date="2024-02-01"))
        assert sm.should_run_strategic_review() is True

        sm.mark_reviewed()
        assert sm.should_run_strategic_review() is False

    def test_apply_patch_persists(self, tmp_path):
        sm = StrategicMemory(tmp_path)
        sm.apply_patch(StrategicPatch(
            patch={"max_gross_exposure": 0.6, "preferred_optimizer": "kelly_capped"},
            rationale="test",
        ))
        # While in a drawdown the patch's gross constraint applies; force the
        # query into a drawdown by passing a NAV well below the (empty) peak.
        # An empty NAV history skips the dd computation, so write a peak first.
        for nav in [100000.0, 95000.0, 90000.0]:
            sm.write_episode(_episode(trade_date="2024-01-01", nav=nav, dd=-0.10, in_dd=True))
        baseline = sm.compute_strategic_baseline("bull", current_nav=90000.0)
        assert baseline.preferred_optimizer == "kelly_capped"
        assert baseline.constraint_overrides.get("max_gross_exposure") == 0.6
        # roundtrip: optimizer is persistent state, survives reload.
        sm2 = StrategicMemory(tmp_path)
        baseline2 = sm2.compute_strategic_baseline("bull", current_nav=90000.0)
        assert baseline2.preferred_optimizer == "kelly_capped"


# ---------------------------------------------------------------------------
# combine_influence
# ---------------------------------------------------------------------------

class TestCombineInfluence:
    def test_strategic_optimizer_preserved(self):
        strategic = MemoryInfluence(preferred_optimizer="risk_parity")
        tactical = TacticalInfluence(avoid_symbols=["TSLA"])
        combined = combine_influence(strategic=strategic, tactical=tactical)
        assert combined.preferred_optimizer == "risk_parity"
        assert combined.no_trade_symbols == ["TSLA"]

    def test_tactical_haircut_passes_through(self):
        strategic = MemoryInfluence()
        tactical = TacticalInfluence(position_haircut=0.7)
        combined = combine_influence(strategic=strategic, tactical=tactical)
        assert combined.position_haircut == 0.7

    def test_emergency_tightens_strategic_baseline(self):
        strategic = MemoryInfluence(constraint_overrides={"max_gross_exposure": 0.9})
        tactical = TacticalInfluence(urgency="emergency")
        combined = combine_influence(strategic=strategic, tactical=tactical)
        assert combined.constraint_overrides["max_gross_exposure"] == 0.5


# ---------------------------------------------------------------------------
# DD metrics + outcome score
# ---------------------------------------------------------------------------

class TestOutcomeMath:
    def test_no_drawdown(self):
        dd = _dd_metrics([100.0, 101.0, 102.0], 103.0)
        assert dd["in_drawdown"] is False

    def test_drawdown_detected(self):
        dd = _dd_metrics([100.0, 102.0, 104.0, 100.0, 98.0, 96.0], 95.0)
        assert dd["in_drawdown"] is True

    def test_outcome_score_drawdown_penalty(self):
        normal = _outcome_score(0.01, dd_from_peak=0.0, dd_5d=0.0, in_drawdown=False, in_rebound=False)
        drawdown = _outcome_score(0.01, dd_from_peak=-0.1, dd_5d=-0.05, in_drawdown=True, in_rebound=False)
        assert drawdown < normal


# ---------------------------------------------------------------------------
# PortfolioAllocator with combined memory influence
# ---------------------------------------------------------------------------

class TestPortfolioAllocator:
    def _state(self, positions: dict[str, float] | None = None) -> PortfolioState:
        return PortfolioState(
            trade_date="2024-01-15",
            cash=100000.0,
            total_equity=100000.0,
            positions=dict(positions or {}),
            prices={"AAPL": 150.0, "MSFT": 300.0, "TSLA": 200.0},
        )

    def _signals(self, directions: dict[str, str]) -> list[AssetSignal]:
        # confidence 0.8 >= CONVICTION_FULL_AT (0.75) so the conviction-scaled
        # gross budget is full — these allocator tests validate water-fill and
        # budget mechanics against the full max_gross ceiling, not the separate
        # conviction-scaling ramp (which has its own coverage).
        return [
            AssetSignal(
                symbol=s, trade_date="2024-01-15", direction=d,
                confidence=0.8, thesis="x", regime="bull",
            )
            for s, d in directions.items()
        ]

    def _regime(self) -> RegimeState:
        return RegimeState(regime="bull", confidence=0.8, trade_date="2024-01-15")

    def test_long_signals_produce_open_long_orders(self):
        plan = PortfolioAllocator().build_plan(
            "2024-01-15", self._signals({"AAPL": "long", "MSFT": "long"}),
            self._state(), self._regime(), MemoryInfluence(),
        )
        opens = [o for o in plan.orders if o.action == "open_long"]
        assert len(opens) == 2
        assert all(o.side == "buy" for o in opens)
        assert all(o.cash_change > 0 for o in opens)

    def test_short_signals_produce_open_short_orders(self):
        plan = PortfolioAllocator().build_plan(
            "2024-01-15", self._signals({"AAPL": "short", "MSFT": "short"}),
            self._state(), self._regime(), MemoryInfluence(),
        )
        opens = [o for o in plan.orders if o.action == "open_short"]
        assert len(opens) == 2
        assert all(o.side == "sell_short" for o in opens)
        assert all(o.cash_change < 0 for o in opens)
        # negative weights
        assert all(plan.target_weights[s] < 0 for s in ("AAPL", "MSFT"))

    def test_mixed_long_short_split_gross_budget(self):
        plan = PortfolioAllocator(max_gross_exposure=0.8).build_plan(
            "2024-01-15", self._signals({"AAPL": "long", "MSFT": "short"}),
            self._state(), self._regime(), MemoryInfluence(),
        )
        long_w = plan.target_weights["AAPL"]
        short_w = plan.target_weights["MSFT"]
        assert long_w > 0
        assert short_w < 0
        assert abs(long_w) + abs(short_w) <= 0.81

    def test_flip_long_to_short_emits_two_legs(self):
        # currently long AAPL at 30% of equity; short signal arrives.
        # turnover_cap=1.0 disables the daily churn guard so this test
        # verifies the two-leg flip mechanism (close_long + open_short)
        # in isolation — the runtime default cap would otherwise clip the
        # transition and this test would only see close_long.
        state = self._state(positions={"AAPL": 30000.0})
        plan = PortfolioAllocator().build_plan(
            "2024-01-15", self._signals({"AAPL": "short"}),
            state, self._regime(), MemoryInfluence(),
        )
        aapl_orders = [o for o in plan.orders if o.symbol == "AAPL"]
        actions = [o.action for o in aapl_orders]
        assert "close_long" in actions
        assert "open_short" in actions
        sides = [o.side for o in aapl_orders]
        assert "sell" in sides
        assert "sell_short" in sides

    def test_flip_short_to_long_emits_two_legs(self):
        # currently short AAPL at -20% of equity; long signal arrives.
        # turnover_cap=1.0 for the same reason as flip_long_to_short.
        state = self._state(positions={"AAPL": -20000.0})
        plan = PortfolioAllocator().build_plan(
            "2024-01-15", self._signals({"AAPL": "long"}),
            state, self._regime(), MemoryInfluence(),
        )
        aapl_orders = [o for o in plan.orders if o.symbol == "AAPL"]
        actions = [o.action for o in aapl_orders]
        assert "close_short" in actions
        assert "open_long" in actions
        sides = [o.side for o in aapl_orders]
        assert "buy_to_cover" in sides
        assert "buy" in sides

    def test_no_trade_blocks_long_open(self):
        plan = PortfolioAllocator().build_plan(
            "2024-01-15", self._signals({"AAPL": "long"}),
            self._state(), self._regime(),
            MemoryInfluence(no_trade_symbols=["AAPL"]),
        )
        opens = [o for o in plan.orders if o.action == "open_long"]
        assert len(opens) == 0

    def test_no_trade_blocks_short_open(self):
        plan = PortfolioAllocator().build_plan(
            "2024-01-15", self._signals({"AAPL": "short"}),
            self._state(), self._regime(),
            MemoryInfluence(no_trade_symbols=["AAPL"]),
        )
        opens = [o for o in plan.orders if o.action == "open_short"]
        assert len(opens) == 0

    def test_position_haircut_applied_symmetrically(self):
        no_haircut = PortfolioAllocator(max_single_position=0.99).build_plan(
            "2024-01-15", self._signals({"AAPL": "long", "MSFT": "short"}),
            self._state(), self._regime(),
            MemoryInfluence(position_haircut=1.0),
        )
        with_haircut = PortfolioAllocator(max_single_position=0.99).build_plan(
            "2024-01-15", self._signals({"AAPL": "long", "MSFT": "short"}),
            self._state(), self._regime(),
            MemoryInfluence(position_haircut=0.5),
        )
        for sym in ("AAPL", "MSFT"):
            full = no_haircut.target_weights[sym]
            cut = with_haircut.target_weights[sym]
            assert abs(cut) < abs(full)
            ratio = abs(cut) / abs(full)
            assert abs(ratio - 0.5) < 0.05

    def test_gross_exposure_caps_long_plus_short(self):
        plan = PortfolioAllocator(max_gross_exposure=0.95).build_plan(
            "2024-01-15", self._signals({"AAPL": "long", "MSFT": "short"}),
            self._state(), self._regime(),
            MemoryInfluence(constraint_overrides={"max_gross_exposure": 0.5}),
        )
        gross = sum(abs(w) for w in plan.target_weights.values())
        assert gross <= 0.51

    def test_single_symbol_uses_full_gross_budget(self):
        # With max_single_position default at 1.0, a single high-confidence
        # signal should consume the entire gross-exposure budget instead of
        # leaking it to cash.
        plan = PortfolioAllocator(max_gross_exposure=0.95).build_plan(
            "2024-01-15", self._signals({"AAPL": "long"}),
            self._state(), self._regime(), MemoryInfluence(),
        )
        assert abs(plan.target_weights["AAPL"] - 0.95) < 1e-6

    def test_multi_symbol_no_budget_leak_uniform(self):
        # Three uniform signals must use the full gross budget — no leakage.
        plan = PortfolioAllocator(max_gross_exposure=0.9).build_plan(
            "2024-01-15",
            self._signals({"AAPL": "long", "MSFT": "long", "TSLA": "long"}),
            self._state(), self._regime(), MemoryInfluence(),
        )
        gross = sum(abs(w) for w in plan.target_weights.values())
        assert abs(gross - 0.9) < 1e-6
        # uniform confidences → uniform weights
        weights = [plan.target_weights[s] for s in ("AAPL", "MSFT", "TSLA")]
        assert all(abs(w - weights[0]) < 1e-6 for w in weights)

    def test_water_fill_redistributes_after_cap_clip(self):
        # Cap forces the high-confidence signal to clip; freed budget must
        # flow to the other longs rather than leak to cash. Confidences are
        # set so avg conviction >= CONVICTION_FULL_AT (0.75) → full gross, so
        # this test isolates water-fill redistribution from conviction scaling.
        signals = [
            AssetSignal(symbol="AAPL", trade_date="2024-01-15", direction="long",
                        confidence=0.95, thesis="x", regime="bull"),
            AssetSignal(symbol="MSFT", trade_date="2024-01-15", direction="long",
                        confidence=0.65, thesis="x", regime="bull"),
            AssetSignal(symbol="TSLA", trade_date="2024-01-15", direction="long",
                        confidence=0.65, thesis="x", regime="bull"),
        ]
        plan = PortfolioAllocator(
            max_gross_exposure=0.9, max_single_position=0.35
        ).build_plan(
            "2024-01-15", signals, self._state(), self._regime(), MemoryInfluence(),
        )
        gross = sum(abs(w) for w in plan.target_weights.values())
        # all three should sum to the full budget thanks to water-fill
        assert abs(gross - 0.9) < 1e-5
        # AAPL pinned at the cap, the rest absorb the overflow
        assert abs(plan.target_weights["AAPL"] - 0.35) < 1e-6
        assert plan.target_weights["MSFT"] > 0.25
        assert plan.target_weights["TSLA"] > 0.25

    def test_gross_runs_full_regardless_of_conviction(self):
        # Exposure design reverted to the simple full-gross model: the book
        # deploys its full gross budget whenever there are eligible signals,
        # rather than throttling gross by aggregate conviction (that throttling
        # under-deployed capital in up-markets and was removed).
        high = [
            AssetSignal(symbol=s, trade_date="2024-01-15", direction="long",
                        confidence=0.85, thesis="x", regime="bull")
            for s in ("AAPL", "MSFT", "TSLA")
        ]
        low = [
            AssetSignal(symbol=s, trade_date="2024-01-15", direction="long",
                        confidence=0.37, thesis="x", regime="bull")
            for s in ("AAPL", "MSFT", "TSLA")
        ]
        alloc = PortfolioAllocator(max_gross_exposure=0.95)
        g_high = sum(abs(w) for w in alloc.build_plan(
            "2024-01-15", high, self._state(), self._regime(), MemoryInfluence(),
        ).target_weights.values())
        g_low = sum(abs(w) for w in alloc.build_plan(
            "2024-01-15", low, self._state(), self._regime(), MemoryInfluence(),
        ).target_weights.values())
        # both deploy full gross (all signals clear min_confidence 0.35)
        assert g_high > 0.9, f"should run near full gross, got {g_high}"
        assert g_low > 0.9, f"low-conviction still deploys full gross, got {g_low}"

    def test_water_fill_handles_full_saturation(self):
        # If even uniform allocation exceeds the cap, gross is bounded by
        # n * cap; remaining budget that can't be placed stays in cash but
        # the system must not blow up.
        signals = [
            AssetSignal(symbol=sym, trade_date="2024-01-15", direction="long",
                        confidence=0.7, thesis="x", regime="bull")
            for sym in ("AAPL", "MSFT", "TSLA")
        ]
        plan = PortfolioAllocator(
            max_gross_exposure=0.95, max_single_position=0.2
        ).build_plan(
            "2024-01-15", signals, self._state(), self._regime(), MemoryInfluence(),
        )
        for sym in ("AAPL", "MSFT", "TSLA"):
            assert plan.target_weights[sym] <= 0.2 + 1e-9
        gross = sum(abs(w) for w in plan.target_weights.values())
        # capped at n * cap = 0.6, not the full 0.95
        assert abs(gross - 0.6) < 1e-6

    def test_water_fill_preserves_long_short_split(self):
        # Long and short legs are water-filled independently; each side
        # consumes its own budget allocation in full.
        signals = [
            AssetSignal(symbol="AAPL", trade_date="2024-01-15", direction="long",
                        confidence=0.9, thesis="x", regime="bull"),
            AssetSignal(symbol="MSFT", trade_date="2024-01-15", direction="short",
                        confidence=0.4, thesis="x", regime="bull"),
        ]
        plan = PortfolioAllocator(max_gross_exposure=0.8).build_plan(
            "2024-01-15", signals, self._state(), self._regime(), MemoryInfluence(),
        )
        # higher-confidence long should claim a bigger share of the budget
        assert plan.target_weights["AAPL"] > 0
        assert plan.target_weights["MSFT"] < 0
        assert plan.target_weights["AAPL"] > abs(plan.target_weights["MSFT"])
        # full gross budget is deployed (no conviction throttling)
        gross = sum(abs(w) for w in plan.target_weights.values())
        assert abs(gross - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_apply_patch_known_field(self):
        from darwintrade.pipeline import PipelineConfig
        c = PipelineConfig()
        c2 = c.apply_patch({"max_gross_exposure": 0.6, "preferred_optimizer": "risk_parity"})
        assert c2.max_gross_exposure == 0.6
        assert c2.preferred_optimizer == "risk_parity"
        # original unchanged
        assert c.max_gross_exposure == 0.95

    def test_apply_patch_ignores_unknown(self):
        from darwintrade.pipeline import PipelineConfig
        c = PipelineConfig()
        c2 = c.apply_patch({"bogus_field": 999, "max_gross_exposure": 0.7})
        assert c2.max_gross_exposure == 0.7
        assert not hasattr(c2, "bogus_field")
