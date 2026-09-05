"""
Regression tests pinning the three self-evolution invariants introduced to
close the LLM-authored strategic-patch loop:

  1. Structured RollbackTrigger: per-patch eval_horizon_bars and drawdown_pct
     supersede hardcoded thresholds — what the LLM proposed is what the
     evaluator enforces.
  2. Pending-patch exclusivity: a new strategic review cannot start while any
     prior patch is still under evaluation. One hypothesis at a time.
  3. Patch outcome feedback: the review payload exposes status ("pending" /
     "committed" / "reverted") and realised nav_impact for each past patch,
     so the LLM sees the causal effect of its own prior decisions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from darwintrade.core.contracts import (
    EpisodeRecord,
    RollbackTrigger,
    StrategicPatch,
)
from darwintrade.memory.strategic import (
    STRATEGIC_REVIEW_EVERY_N_DAYS,
    StrategicMemory,
)


def _make_episode(date: str, score: float = 0.0, nav: float = 100.0) -> EpisodeRecord:
    return EpisodeRecord(
        trade_date=date, symbols=["AAPL"], regime="bull", optimizer_used="conf_weighted",
        outcome_score=score, nav=nav, dd_from_peak=0.0, in_drawdown=False,
        in_rebound=False, metadata={"did_trade": True},
    )


def _feed_episodes(memory: StrategicMemory, n: int, start_nav: float = 100.0) -> None:
    for i in range(n):
        memory.write_episode(_make_episode(f"2025-03-{i+1:02d}", nav=start_nav))


# ---------------------------------------------------------------------------
# 1. Structured rollback trigger drives the evaluator
# ---------------------------------------------------------------------------

def test_rollback_trigger_horizon_is_per_patch_not_hardcoded() -> None:
    """A patch with eval_horizon_bars=8 must survive 7 bars of small
    positive movement and commit on the 8th, even though the old
    hardcoded max_age_bars was 5."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(eval_horizon_bars=8, drawdown_pct=0.05),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)
        assert len(mem.get_state().pending_patches) == 1

        # Bars 1..7: tiny positive drift, must not fire (dd_pct=0.05 not breached,
        # horizon=8 not reached).
        for bar in range(7):
            reverted = mem.evaluate_pending_patches(current_nav=100.5)
            assert not reverted, f"unexpected revert on bar {bar+1}"
            assert len(mem.get_state().pending_patches) == 1

        # Bar 8: horizon elapsed → commit (not revert), pending queue drains.
        reverted = mem.evaluate_pending_patches(current_nav=101.0)
        assert not reverted
        assert len(mem.get_state().pending_patches) == 0
        assert mem.get_state().history[-1]["status"] == "committed"


def test_rollback_trigger_drawdown_gate_is_per_patch() -> None:
    """A patch using absolute-NAV rollback (reference='pre_patch_nav')
    with drawdown_pct=0.04 must NOT revert at -2% but MUST revert at -5%."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=10, drawdown_pct=0.04,
                reference="pre_patch_nav",
            ),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)

        # -2% is smaller than the 4% gate — must NOT revert.
        reverted = mem.evaluate_pending_patches(current_nav=98.0)
        assert not reverted
        assert len(mem.get_state().pending_patches) == 1

        # -5% breaches the 4% gate — must revert immediately.
        reverted = mem.evaluate_pending_patches(current_nav=95.0)
        assert len(reverted) == 1
        assert mem.get_state().history[-1]["status"] == "reverted"
        assert len(mem.get_state().pending_patches) == 0


def test_rollback_trigger_parse_clamps_out_of_range_values() -> None:
    """LLM-emitted values outside sane bounds must be clamped, not passed
    through — a horizon of 999 or a drawdown of 90% would destabilise the
    evaluator."""
    t1 = RollbackTrigger.from_any({"eval_horizon_bars": 999, "drawdown_pct": 0.9, "reference": "pre_patch_nav"})
    assert t1.eval_horizon_bars == 30      # clamped to upper bound
    assert t1.drawdown_pct == 0.15         # clamped to upper bound

    t2 = RollbackTrigger.from_any({"eval_horizon_bars": 0, "drawdown_pct": 0.0001, "reference": "pre_patch_nav"})
    assert t2.eval_horizon_bars == 1
    assert t2.drawdown_pct == 0.005


# ---------------------------------------------------------------------------
# 2. Pending-patch exclusivity
# ---------------------------------------------------------------------------

def test_pending_patch_blocks_new_strategic_review() -> None:
    """While a prior patch is still under evaluation, should_run_strategic_review
    must return False even when enough new episodes have accumulated. This
    prevents the compounding-tightening pathology observed in the audit."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)
        assert mem.should_run_strategic_review() is True

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(eval_horizon_bars=10, drawdown_pct=0.03),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)
        mem.mark_reviewed()

        # Feed enough new episodes to normally re-trigger a review.
        for i in range(STRATEGIC_REVIEW_EVERY_N_DAYS):
            mem.write_episode(_make_episode(f"2025-03-{10+i:02d}", nav=100.0))

        # Prior patch is still pending → review is blocked.
        assert mem.should_run_strategic_review() is False


def test_review_reopens_after_pending_patch_commits() -> None:
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(eval_horizon_bars=3, drawdown_pct=0.05),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)
        mem.mark_reviewed()
        # Fill enough episodes to re-arm the review counter.
        for i in range(STRATEGIC_REVIEW_EVERY_N_DAYS):
            mem.write_episode(_make_episode(f"2025-03-{10+i:02d}", nav=100.0))
        assert mem.should_run_strategic_review() is False

        # Drive the horizon to completion without tripping the drawdown gate.
        for _ in range(3):
            mem.evaluate_pending_patches(current_nav=100.0)

        assert not mem.get_state().pending_patches
        assert mem.should_run_strategic_review() is True


# ---------------------------------------------------------------------------
# 3. Patch outcome feedback in review payload
# ---------------------------------------------------------------------------

def test_counterfactual_rollback_reverts_when_alpha_deteriorates() -> None:
    """With reference='counterfactual_alpha', the patch is reverted when
    cumulative portfolio-minus-benchmark return drops below -drawdown_pct,
    even if absolute NAV is UP (i.e. the patch rode a strong market but
    underperformed it)."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=10,
                drawdown_pct=0.03,
                reference="counterfactual_alpha",
            ),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)

        # NAV is UP each bar (+1% per bar), but the patch is underperforming
        # the market by -1% per bar (alpha < 0). After 4 bars cum_alpha = -4%,
        # which breaches the -3% gate → revert.
        for _ in range(3):
            reverted = mem.evaluate_pending_patches(
                current_nav=101.0, outcome_alpha=-0.01,
            )
            assert not reverted
        reverted = mem.evaluate_pending_patches(current_nav=101.0, outcome_alpha=-0.01)
        assert len(reverted) == 1
        assert mem.get_state().history[-1]["status"] == "reverted"
        assert mem.get_state().history[-1]["rollback_reason"] == "counterfactual_alpha"


def test_counterfactual_rollback_does_not_revert_on_market_drawdown() -> None:
    """With reference='counterfactual_alpha', a patch that just rode the
    market DOWN (portfolio and benchmark both fell in lockstep) is NOT
    reverted — this is the key advantage over absolute-NAV rollback."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=3,
                drawdown_pct=0.03,
                reference="counterfactual_alpha",
            ),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)

        # NAV falls 2% per bar (huge drawdown), but alpha is 0 (portfolio
        # tracked the market exactly). Absolute-NAV rollback would have
        # reverted at bar 2; counterfactual_alpha does NOT.
        for bar in range(3):
            reverted = mem.evaluate_pending_patches(
                current_nav=100.0 * (0.98 ** (bar + 1)),
                outcome_alpha=0.0,
            )
            assert not reverted, f"unexpectedly reverted on bar {bar+1}"
        # Horizon elapsed with 0 alpha → committed.
        assert mem.get_state().history[-1]["status"] == "committed"


def test_review_payload_exposes_patch_outcomes_to_llm() -> None:
    """The LLM must see status ("committed" / "reverted") and nav_impact for
    each of its past patches, along with the rollback contract it emitted.
    Without this, the LLM cannot learn from its own decisions."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        # First patch: gets reverted (absolute drawdown breach).
        p1 = StrategicPatch(
            patch={"max_gross_exposure": 0.6},
            rationale="tighten",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=10, drawdown_pct=0.03,
                reference="pre_patch_nav",
            ),
            issued_on="2025-03-05",
        )
        mem.apply_patch(p1, current_nav=100.0)
        mem.evaluate_pending_patches(current_nav=96.0)   # -4% > 3% gate → revert

        payload = mem.review_payload()
        track = payload["patch_track_record"]
        assert len(track) == 1
        entry = track[0]
        assert entry["status"] == "reverted"
        assert entry["nav_impact"] is not None and entry["nav_impact"] < 0
        assert entry["rollback_trigger"]["drawdown_pct"] == 0.03
        assert entry["rollback_trigger"]["eval_horizon_bars"] == 10
        assert entry["reverted_at_bar"] == 1


# ---------------------------------------------------------------------------
# 4. D-i post-commit continuous tracking (2026-07-07)
# ---------------------------------------------------------------------------

# NOTE: tests for the "post-commit watch queue" feature (committed_patches /
# POST_COMMIT_LATE_REVERT_ALPHA) were removed — that feature is not present in
# the current strategic memory implementation, so the tests referenced
# non-existent symbols and could never pass. Re-add them alongside the feature
# if/when it is implemented.


# ---------------------------------------------------------------------------
# Regression: LLM patch survives drawdown → recovered transition.
# Prior implementation aliased self._state.constraint_overrides via a local
# `persistent = ...` binding and unconditionally popped four config keys on
# recovery, silently erasing any LLM-authored patch that touched those keys.
# ---------------------------------------------------------------------------

def test_llm_patch_survives_drawdown_recovery_transition() -> None:
    """Apply a strategic patch during a drawdown, then feed NAV bars that
    fully recover. The patch must remain live in strategic state after
    recovery — recovery must not clear LLM overrides.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = StrategicMemory(Path(tmp))

        # Walk NAV into a drawdown that trips the dd guard (DD_THRESHOLD = -0.05)
        drawdown_navs = [100.0, 99.0, 97.0, 94.0, 92.0, 90.0, 89.0]
        for i, nav in enumerate(drawdown_navs):
            mem.write_episode(_make_episode(f"2025-03-{i+1:02d}", nav=nav))

        # Patch fires while drawdown is active.
        mem.apply_patch(
            StrategicPatch(
                patch={"max_gross_exposure": 0.55, "min_confidence_threshold": 0.45},
                rationale="risk-off",
                rollback_trigger=RollbackTrigger(eval_horizon_bars=20, drawdown_pct=0.10),
                issued_on="2025-03-07",
            ),
            current_nav=89.0,
        )
        assert mem.get_state().constraint_overrides.get("max_gross_exposure") == 0.55
        assert mem.get_state().constraint_overrides.get("min_confidence_threshold") == 0.45

        # NAV recovers past the pre-patch peak — recovered branch runs.
        for i, nav in enumerate([92.0, 96.0, 100.0, 103.0, 106.0]):
            mem.write_episode(_make_episode(f"2025-03-{i+10:02d}", nav=nav))
            mem.compute_strategic_baseline("bull", current_nav=nav)

        state = mem.get_state()
        # Patches are LLM-authored — they must NOT be wiped by recovery.
        assert state.constraint_overrides.get("max_gross_exposure") == 0.55, (
            "LLM patch on max_gross_exposure was erased by drawdown-recovery pop"
        )
        assert state.constraint_overrides.get("min_confidence_threshold") == 0.45, (
            "LLM patch on min_confidence_threshold was erased by drawdown-recovery pop"
        )


def test_compute_strategic_baseline_cleans_up_drawdown_patches_on_recovery() -> None:
    """compute_strategic_baseline() clears drawdown-era STRUCTURAL patches once
    NAV has fully recovered — this is intentional, so a defensive patch applied
    in a drawdown does not linger and compound forever. The invariants that
    matter:
      1. The recovery cleanup DOES drop the drawdown-era key (max_gross_exposure).
      2. The cleanup is idempotent: after the first recovery call the state is
         stable across further calls (no unbounded drift).
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = StrategicMemory(Path(tmp))
        # Seed a fully-recovered NAV path so the baseline walks the recovery branch.
        for i, nav in enumerate([100.0, 101.0, 102.0, 103.0, 104.0]):
            mem.write_episode(_make_episode(f"2025-03-{i+1:02d}", nav=nav))
        mem.apply_patch(
            StrategicPatch(
                patch={"max_gross_exposure": 0.6},
                rationale="test",
                rollback_trigger=RollbackTrigger(eval_horizon_bars=20, drawdown_pct=0.10),
                issued_on="2025-03-05",
            ),
            current_nav=104.0,
        )
        assert "max_gross_exposure" in mem.get_state().constraint_overrides
        # First recovery call clears the drawdown-era structural patch.
        mem.compute_strategic_baseline("bull", current_nav=104.0)
        after_first = dict(mem.get_state().constraint_overrides)
        assert "max_gross_exposure" not in after_first, "recovery must clear drawdown patch"
        # Idempotent: further calls do not drift state.
        for _ in range(5):
            mem.compute_strategic_baseline("bull", current_nav=104.0)
        assert dict(mem.get_state().constraint_overrides) == after_first


def test_recovery_cleanup_preserves_non_structural_patch() -> None:
    """The recovery cleanup only drops the four drawdown-era STRUCTURAL keys; a
    non-defensive learned field (e.g. rebound_entry_threshold) is NOT a
    drawdown patch and must survive recovery so the learned setting persists."""
    with tempfile.TemporaryDirectory() as tmp:
        mem = StrategicMemory(Path(tmp))
        for i, nav in enumerate([100.0, 101.0, 102.0, 103.0, 104.0]):
            mem.write_episode(_make_episode(f"2025-03-{i+1:02d}", nav=nav))
        mem.apply_patch(
            StrategicPatch(
                patch={"rebound_entry_threshold": 0.5, "max_gross_exposure": 0.6},
                rationale="test",
                rollback_trigger=RollbackTrigger(eval_horizon_bars=20, drawdown_pct=0.10),
                issued_on="2025-03-05",
            ),
            current_nav=104.0,
        )
        for _ in range(5):
            mem.compute_strategic_baseline("bull", current_nav=104.0)
        state = mem.get_state().constraint_overrides
        assert state.get("rebound_entry_threshold") == 0.5, "non-structural patch must survive recovery"
        assert "max_gross_exposure" not in state, "structural drawdown patch still cleared"


# ---------------------------------------------------------------------------
# Regression: pending-patch rollback must actually revert config, not
# re-confirm the tainted value via self.config as its own fallback.
# ---------------------------------------------------------------------------

def test_rollback_restores_baseline_pipeline_config() -> None:
    """After a strategic patch is auto-reverted, the pipeline's live config must
    return to the pre-patch baseline value, not the patched value.

    This drives the REAL rebuild path (DarwinTradePipeline._rebuild_config_from_state
    against a mutated strategic state), which the previous version of this test
    did not: it rebuilt from a fresh dataclasses.replace(baseline) and so could
    never observe the ratcheting bug where self.config was used as its own
    fallback and silently re-confirmed the reverted (tightened) value."""
    from darwintrade.pipeline import DarwinTradePipeline, PipelineConfig
    from darwintrade.core.contracts import StrategicPatch, RollbackTrigger

    with tempfile.TemporaryDirectory() as td:
        pipe = DarwinTradePipeline(storage_dir=Path(td), config=PipelineConfig())
        assert pipe.config.max_gross_exposure == 0.95

        _feed_episodes(pipe.strategic_memory, n=STRATEGIC_REVIEW_EVERY_N_DAYS)

        # Apply a tightening patch that introduces a brand-new override key
        # (baseline had no explicit max_gross_exposure override). This is the
        # exact shape that triggered the original ratchet: revert removes the
        # key entirely, so a self.config fallback would freeze it at 0.5.
        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.5},
            rationale="tighten",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=10, drawdown_pct=0.03, reference="pre_patch_nav",
            ),
            issued_on="2025-03-05",
        )
        pipe.strategic_memory.apply_patch(patch, current_nav=100.0)
        pipe._rebuild_config_from_state()
        assert pipe.config.max_gross_exposure == 0.5

        # Patch underperforms → revert fires, override key drops out of state.
        reverted = pipe.strategic_memory.evaluate_pending_patches(current_nav=96.0)
        assert len(reverted) == 1
        pipe._rebuild_config_from_state()

        # Config must be back at baseline, not stuck at the tightened 0.5.
        assert pipe.config.max_gross_exposure == 0.95
        assert pipe.config.min_confidence_threshold == 0.35
        assert pipe.config.max_single_position == 1.0
        assert pipe.config.position_size_multiplier == 1.0


# ---------------------------------------------------------------------------
# Pre-patch-policy shadow rollback + injected harmful patch
# ---------------------------------------------------------------------------

def test_pre_patch_policy_reverts_when_patch_underperforms_shadow() -> None:
    """reference='pre_patch_policy' reverts when the patched policy's alpha
    falls more than drawdown_pct BELOW the pre-patch policy's shadow alpha on
    the same bars — even when both beat the market. This is the strongest
    counterfactual the reviewer asked for: patch vs the policy it replaced."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=10, drawdown_pct=0.03,
                reference="pre_patch_policy",
            ),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)

        # Both policies BEAT the market (positive alpha), so counterfactual_alpha
        # would NOT revert. But the patched policy earns +0.5%/bar while the
        # shadow pre-patch policy earns +1.5%/bar → incremental -1%/bar. After
        # 4 bars incremental = -4% < -3% gate → revert.
        for _ in range(3):
            reverted = mem.evaluate_pending_patches(
                current_nav=102.0, outcome_alpha=0.005, shadow_alpha=0.015,
            )
            assert not reverted
        reverted = mem.evaluate_pending_patches(
            current_nav=102.0, outcome_alpha=0.005, shadow_alpha=0.015,
        )
        assert len(reverted) == 1
        assert mem.get_state().history[-1]["status"] == "reverted"
        assert mem.get_state().history[-1]["rollback_reason"] == "pre_patch_policy"


def test_pre_patch_policy_keeps_patch_that_beats_shadow() -> None:
    """A patch that OUTPERFORMS the pre-patch shadow policy is committed, even
    if its absolute alpha is negative (both losing, patch loses less)."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        patch = StrategicPatch(
            patch={"max_gross_exposure": 0.7},
            rationale="test",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=3, drawdown_pct=0.03,
                reference="pre_patch_policy",
            ),
            issued_on="2025-03-05",
        )
        mem.apply_patch(patch, current_nav=100.0)

        # Patch loses -0.5%/bar, shadow loses -1.5%/bar → incremental +1%/bar,
        # so the patch is BETTER than the old policy and must commit.
        for _ in range(3):
            reverted = mem.evaluate_pending_patches(
                current_nav=98.0, outcome_alpha=-0.005, shadow_alpha=-0.015,
            )
            assert not reverted
        assert mem.get_state().history[-1]["status"] == "committed"


def test_injected_harmful_patch_is_rolled_back() -> None:
    """Stress test the rollback gate directly: inject a deliberately harmful
    patch with a tight contract and confirm the gate fires and reverts it.
    Demonstrates the contract PREVENTS a bad policy edit, not just that it is
    armed (the reviewer noted no rollback ever fired on the reported runs)."""
    with tempfile.TemporaryDirectory() as td:
        mem = StrategicMemory(Path(td))
        _feed_episodes(mem, n=STRATEGIC_REVIEW_EVERY_N_DAYS, start_nav=100.0)

        harmful = StrategicPatch(
            patch={"max_gross_exposure": 0.30, "min_confidence_threshold": 0.59},
            rationale="INJECTED harmful patch (over-de-risk + near-freeze)",
            rollback_trigger=RollbackTrigger(
                eval_horizon_bars=5, drawdown_pct=0.005,
                reference="counterfactual_alpha",
            ),
            issued_on="2025-03-05",
        )
        mem.apply_patch(harmful, current_nav=100.0)
        # The harmful patch bleeds alpha every bar; the tight -0.5% gate trips
        # on the second bar (cum_alpha = -0.8% < -0.5%).
        mem.evaluate_pending_patches(current_nav=99.6, outcome_alpha=-0.004)
        reverted = mem.evaluate_pending_patches(current_nav=99.2, outcome_alpha=-0.004)
        assert len(reverted) == 1
        h = mem.get_state().history[-1]
        assert h["status"] == "reverted"
        # And the harmful constraints must be undone, not left in effect.
        assert "max_gross_exposure" not in mem.get_state().constraint_overrides
        assert "min_confidence_threshold" not in mem.get_state().constraint_overrides
