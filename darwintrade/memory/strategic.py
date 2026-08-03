"""
StrategicMemory: long-window, multi-day pattern storage.

Paper correspondence (darwintrade_aaai2027.tex):
  Implements the strategic self-evolution loop of Section "Strategic
  Evolution" and the periodic review of Alg 2 ("DarwinTrade periodic strategic
  review, every N=3 episodes"). It holds the strategic policy pi_t (the
  StrategicState below) that parameterizes the deterministic allocator, and
  drives the commit-or-rollback loop of Eq. strategic-evolving
      pi_{j+1} = f_strategic-evolving(H^S_j, pi_j, Omega).
  Key mappings:
    - StrategicState              → the strategic policy pi_j (allocator mode +
                                     bounded scalar override fields, i.e. Omega)
    - should_run_strategic_review → the N=3-episode review cadence gate
                                     (STRATEGIC_REVIEW_EVERY_N_DAYS = 3) plus the
                                     "one patch at a time" exclusivity of Alg 2
    - apply_patch                 → installs a proposed patch (Delta-pi, H,
                                     epsilon, m) as the single pending patch p,
                                     anchored at t_p
    - evaluate_pending_patches    → Alg 2 PHASE "Evaluate the patch under trial":
                                     Revert when the monitored performance
                                     g_{m}(t_p, t) < -epsilon, else Commit once
                                     t >= t_p + H
  The 20-day review window is STRATEGIC_REVIEW_LOOKBACK = 20 (H^S_j; Table
  "Fixed hyperparameters": strategic lookback 20 days). The reflect/diagnose
  and policy-author LLM steps (f_str-review, f_str-policy) live in
  reflection.py + agents/evolution.py; this module owns the policy state and
  the logged rollback trigger.

Persists the full episode history, semantic capsules (regime × optimizer),
and active strategic patches. Drives long-lived config changes.

The strategic layer reads from this every N episodes to reflect on cross-day
patterns and propose strategy patches (N = STRATEGIC_REVIEW_EVERY_N_DAYS = 3,
matching the paper's reported strategic period).
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from ..core.contracts import EpisodeRecord, MemoryInfluence, StrategicPatch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODE_CAP = 200            # full historical episodes kept
NAV_HISTORY_CAP = 250        # ~1 year of daily NAV
DD_THRESHOLD = -0.05         # in_drawdown if NAV < peak * 0.95. Calibrated for a
                             # diversified multi-symbol book: a 30-name portfolio
                             # routinely wiggles -3% on normal market noise, so a
                             # tighter threshold would fire the guard on routine
                             # noise and chronically suppress max_gross in up
                             # markets. -0.05 keeps the guard live for genuine
                             # deeper drawdowns (~10% of days).
REBOUND_THRESHOLD = 0.02     # rebound when 5d gain >= 2%
ALPHA_NORMALIZER = 30.0
DD_PENALTY = 6.0
REBOUND_BONUS = 3.0

# capsule activation
CAPSULE_MIN_SAMPLES_FOR_ACTIVE = 8
CAPSULE_MIN_CONFIDENCE_FOR_ACTIVE = 0.55
CAPSULE_MIN_SAMPLES_FOR_INFLUENCE = 8

# strategic review cadence. A short cadence keeps the strategic layer
# responsive to sustained patterns; a longer cadence lags the market and
# under-produces patches.
# Paper: strategic review period N = 3 completed episodes (Alg 2 header
# "every N=3 episodes"; Table "Fixed hyperparameters": strategic period N).
STRATEGIC_REVIEW_EVERY_N_DAYS = 3

# Paper: strategic lookback = 20 days — the twenty-day history window H^S_j of
# Eq. strategic-evolving (Table "Fixed hyperparameters": strategic lookback).
STRATEGIC_REVIEW_LOOKBACK = 20


def _safe_finite(v: Any, default: float = 0.0) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except (TypeError, ValueError):
        return default


def _dd_metrics(
    nav_history: list[float],
    current_nav: float,
    *,
    dd_threshold: float = DD_THRESHOLD,
    rebound_threshold: float = REBOUND_THRESHOLD,
) -> dict[str, float]:
    navs = [n for n in nav_history if n > 0] + ([current_nav] if current_nav > 0 else [])
    if not navs:
        return {"nav_peak": 0.0, "dd_from_peak": 0.0, "dd_5d": 0.0, "in_drawdown": False, "in_rebound": False}
    peak = max(navs)
    last = navs[-1]
    dd_from_peak = (last - peak) / peak if peak > 0 else 0.0
    dd_5d = (last - navs[-6]) / navs[-6] if len(navs) >= 6 and navs[-6] > 0 else 0.0
    prior_dd = (navs[-6] - peak) / peak if len(navs) >= 6 and peak > 0 else 0.0
    in_drawdown = dd_from_peak < dd_threshold
    in_rebound = (not in_drawdown) and dd_5d >= rebound_threshold and prior_dd < dd_threshold
    return {
        "nav_peak": peak,
        "dd_from_peak": dd_from_peak,
        "dd_5d": dd_5d,
        "in_drawdown": in_drawdown,
        "in_rebound": in_rebound,
    }


def _outcome_score(alpha: float, dd_from_peak: float, dd_5d: float, in_drawdown: bool, in_rebound: bool) -> float:
    base = alpha * ALPHA_NORMALIZER
    pain = max(0.0, -dd_from_peak) if in_drawdown else 0.0
    # dd_5d is positive during recovery; reward actual progress out of drawdown
    bonus = max(0.0, dd_5d) if in_rebound else 0.0
    return base - DD_PENALTY * pain + REBOUND_BONUS * bonus


# ---------------------------------------------------------------------------
# Semantic capsule — distilled (regime, optimizer) pattern
# ---------------------------------------------------------------------------

class SemanticCapsule:
    def __init__(self, regime: str, optimizer: str) -> None:
        self.regime = regime
        self.optimizer = optimizer
        self.scores: list[float] = []
        self.status: str = "candidate"   # candidate / active / deprecated

    @property
    def sample_count(self) -> int:
        return len(self.scores)

    @property
    def mean_score(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0

    @property
    def confidence(self) -> float:
        if self.sample_count < 2:
            return 0.0
        std = statistics.stdev(self.scores)
        sem = std / math.sqrt(self.sample_count)
        return max(0.0, min(1.0, 1.0 - sem / (abs(self.mean_score) + 1e-6)))

    def add_score(self, score: float) -> None:
        self.scores.append(score)
        self._update_status()

    def _update_status(self) -> None:
        if (
            self.sample_count >= CAPSULE_MIN_SAMPLES_FOR_ACTIVE
            and self.confidence >= CAPSULE_MIN_CONFIDENCE_FOR_ACTIVE
        ):
            if self.mean_score < -1.0:
                self.status = "deprecated"
            else:
                self.status = "active"
        elif self.sample_count < 2:
            self.status = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "optimizer": self.optimizer,
            "sample_count": self.sample_count,
            "mean_score": round(self.mean_score, 4),
            "confidence": round(self.confidence, 4),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticCapsule:
        c = cls(d["regime"], d["optimizer"])
        c.scores = [_safe_finite(s) for s in d.get("scores", [])]
        c.status = d.get("status", "candidate")
        return c


# ---------------------------------------------------------------------------
# Active strategic patch (the current "live" config override stack)
# ---------------------------------------------------------------------------

class StrategicState:
    """The current effective strategy config delta produced by past patches.

    Paper: the strategic policy pi_j of Section "Strategic Evolution".
    `preferred_optimizer` selects one of the nine allocator modes (m in the
    patch tuple), and `constraint_overrides` holds the bounded scalar policy
    fields; together they are the allowed parameter set Omega. `pending_patches`
    holds the single patch p currently under trial (Alg 2 exclusivity)."""

    def __init__(self) -> None:
        self.preferred_optimizer: str = ""
        self.constraint_overrides: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []   # list of patches applied
        # patches that may still be rolled back if NAV deteriorates
        self.pending_patches: list[dict[str, Any]] = []

    def apply_patch(self, patch: StrategicPatch, *, current_nav: float = 0.0) -> None:
        # Paper: install the proposed patch (Delta-pi, H, epsilon, m) as the
        # pending patch p with anchor t_p (Alg 2 "install pi^+ as pending patch
        # p with (H, epsilon, m), anchor t_p"). The pre-patch values captured
        # below make the patch fully invertible, so Revert restores exactly
        # pi_j (Eq. strategic-evolving's rollback branch).
        update = dict(patch.patch or {})
        # Record the pre-patch state so the patch is fully invertible.
        prev_optimizer = self.preferred_optimizer
        prev_constraints = {
            k: self.constraint_overrides.get(k) for k in update.keys()
            if k != "preferred_optimizer"
        }

        if "preferred_optimizer" in update:
            self.preferred_optimizer = str(update.pop("preferred_optimizer", ""))
        self.constraint_overrides.update(update)
        # Structured rollback contract emitted by the policy-author LLM. The
        # evaluator reads eval_horizon_bars / drawdown_pct from here — not
        # from hardcoded constants — so what the LLM proposed and what the
        # code enforces are the same thing.
        trigger = patch.rollback_trigger.to_dict()
        record = {
            "issued_on": patch.issued_on,
            "patch": dict(patch.patch or {}),
            "rationale": patch.rationale,
            "rollback_trigger": trigger,
            "anchor_nav": float(current_nav) if current_nav and current_nav > 0 else 0.0,
            "status": "pending",   # pending → committed | reverted (set by evaluator)
        }
        self.history.append(record)
        self.pending_patches.append({
            "issued_on": patch.issued_on,
            "patch": dict(patch.patch or {}),
            "prev_optimizer": prev_optimizer,
            "prev_constraints": prev_constraints,
            "anchor_nav": float(current_nav) if current_nav and current_nav > 0 else 0.0,
            "bars_since": 0,
            "history_idx": len(self.history) - 1,
            "rollback_trigger": trigger,
            # Cumulative counterfactual signal. Each bar the evaluator adds
            # the per-bar outcome_alpha (portfolio_return - benchmark_return)
            # here, so the number represents cumulative alpha delta since
            # patch. When rollback_trigger.reference == "counterfactual_alpha"
            # the evaluator compares this against drawdown_pct.
            "cum_alpha_since_patch": 0.0,
            # Cumulative alpha the PRE-PATCH policy would have earned on the
            # same bars (shadow replay). Only advanced when the caller supplies
            # shadow_alpha; used by reference == "pre_patch_policy" to test the
            # patch against the policy it replaced, not just against the market.
            "cum_shadow_alpha_since_patch": 0.0,
        })

    def revert_patch(self, pending_entry: dict[str, Any]) -> None:
        """Undo a pending patch by restoring its captured pre-patch values.

        Paper: the Revert(p) operation of Alg 2 — restores pi_j when the
        rollback trigger fires (g_{m}(t_p, t) < -epsilon)."""
        if "preferred_optimizer" in pending_entry.get("patch", {}):
            self.preferred_optimizer = str(pending_entry.get("prev_optimizer", ""))
        for k, v in (pending_entry.get("prev_constraints", {}) or {}).items():
            if v is None:
                self.constraint_overrides.pop(k, None)
            else:
                self.constraint_overrides[k] = v

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_optimizer": self.preferred_optimizer,
            "constraint_overrides": dict(self.constraint_overrides),
            "history_size": len(self.history),
            "pending_patches": list(self.pending_patches),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategicState:
        s = cls()
        s.preferred_optimizer = str(d.get("preferred_optimizer", ""))
        s.constraint_overrides = dict(d.get("constraint_overrides", {}) or {})
        s.history = list(d.get("history", []) or [])
        # pending_patches must persist across process restarts (resume mode):
        # otherwise a resumed run would silently allow new reviews to layer on
        # top of an un-evaluated patch, defeating pending-patch exclusivity.
        s.pending_patches = list(d.get("pending_patches", []) or [])
        return s


# ---------------------------------------------------------------------------
# Strategic memory store
# ---------------------------------------------------------------------------

class StrategicMemory:
    """
    Long-term memory store. One per pipeline instance.

    Layout on disk:
      <storage_dir>/strategic_episodes.jsonl   (full history, capped at 200)
      <storage_dir>/strategic_capsules.json    (regime × optimizer capsules)
      <storage_dir>/strategic_nav.json         (NAV history)
      <storage_dir>/strategic_state.json       (active patches and config delta)
    """

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = self.storage_dir / "strategic_episodes.jsonl"
        self._capsules_path = self.storage_dir / "strategic_capsules.json"
        self._nav_path = self.storage_dir / "strategic_nav.json"
        self._state_path = self.storage_dir / "strategic_state.json"

        self._episodes: list[EpisodeRecord] = []
        self._capsules: dict[str, SemanticCapsule] = {}
        self._nav_history: list[float] = []
        self._state: StrategicState = StrategicState()
        self._last_review_idx: int = -1
        # drawdown thresholds — patchable via update_thresholds().
        # dd_threshold is stored as a negative number (matches dd_from_peak sign).
        self._dd_threshold: float = DD_THRESHOLD
        self._rebound_threshold: float = REBOUND_THRESHOLD
        self._load()

    def update_thresholds(self, *, drawdown_guard: float | None = None, rebound_entry: float | None = None) -> None:
        """Reconfigure dd/rebound thresholds. drawdown_guard is the positive
        magnitude (e.g. 0.03 == 3% drawdown triggers guard); stored as negative."""
        if drawdown_guard is not None:
            mag = max(0.0, min(1.0, abs(float(drawdown_guard))))
            self._dd_threshold = -mag
        if rebound_entry is not None:
            self._rebound_threshold = max(0.0, min(1.0, float(rebound_entry)))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_episode(self, episode: EpisodeRecord) -> None:
        if episode.nav > 0:
            self._nav_history.append(episode.nav)
            if len(self._nav_history) > NAV_HISTORY_CAP:
                self._nav_history = self._nav_history[-NAV_HISTORY_CAP:]
        self._episodes.append(episode)
        if len(self._episodes) > EPISODE_CAP:
            self._episodes = self._episodes[-EPISODE_CAP:]

        # update capsule — skip no_trade days so they don't pollute optimizer attribution
        if episode.optimizer_used != "no_trade":
            key = f"{episode.regime}:{episode.optimizer_used or 'default'}"
            if key not in self._capsules:
                self._capsules[key] = SemanticCapsule(episode.regime, episode.optimizer_used or "default")
            self._capsules[key].add_score(episode.outcome_score)
        self._persist()

    def apply_patch(self, patch: StrategicPatch, *, current_nav: float = 0.0) -> None:
        if not current_nav:
            current_nav = self._nav_history[-1] if self._nav_history else 0.0
        self._state.apply_patch(patch, current_nav=current_nav)
        self._persist()

    def evaluate_pending_patches(self, *, current_nav: float,
                                  outcome_alpha: float = 0.0,
                                  shadow_alpha: float = 0.0,
                                  **legacy_kwargs: Any) -> list[dict[str, Any]]:
        """Auto-rollback each pending patch using ITS OWN rollback contract.

        Paper: Alg 2 PHASE "Evaluate the patch under trial before proposing a
        new one" and the validation half of Eq. strategic-evolving. The
        per-patch RollbackTrigger supplies epsilon (drawdown_pct) and H
        (eval_horizon_bars); this method restores pi_j at the first bar where
        the monitored performance g_{m}(t_p, t) < -epsilon (the `fired` branch)
        and commits the patch once bars_since reaches H without a breach. The
        `reference` field selects which g is monitored (counterfactual alpha vs
        absolute NAV vs pre-patch-policy shadow).

        Each pending entry carries the RollbackTrigger the policy-author LLM
        emitted with the patch itself. The evaluator reads
        eval_horizon_bars, drawdown_pct, and reference from that per-patch
        contract — no hardcoded thresholds, no hardcoded reference. This
        closes the loop between the stated hypothesis and the code that
        enforces it.

        Reference modes (chosen by the LLM per patch):

          reference = "counterfactual_alpha" (default, recommended):
            The evaluator accumulates per-bar outcome_alpha
            (= portfolio_return - benchmark_return) since apply_patch, and
            reverts when cum_alpha_since_patch < -drawdown_pct. This is a
            causal rollback: it separates "patch was bad" from "market
            fell". A patch that just rode the market down is NOT reverted;
            only one that materially underperformed the market is.

          reference = "pre_patch_nav" (legacy):
            Reverts when absolute NAV drops more than drawdown_pct below
            the pre-patch anchor. Kept for backward compatibility with
            older LLM-authored patches or persisted state.

        In either mode, when bars_since reaches eval_horizon_bars without
        tripping, the patch is committed and drops out of the queue.

        legacy_kwargs is accepted for callers that still pass the old
        drawdown_trigger / max_age_bars arguments; ignored.
        """
        _ = legacy_kwargs
        if not self._state.pending_patches:
            return []
        reverted: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        for entry in self._state.pending_patches:
            entry["bars_since"] = int(entry.get("bars_since", 0)) + 1
            anchor = float(entry.get("anchor_nav", 0.0) or 0.0)
            history_idx = entry.get("history_idx")

            trig = entry.get("rollback_trigger") or {}
            horizon = int(trig.get("eval_horizon_bars", 5) or 5)
            dd_gate = float(trig.get("drawdown_pct", 0.02) or 0.02)
            reference = str(trig.get("reference", "counterfactual_alpha") or "counterfactual_alpha")

            # Cumulative counterfactual alpha since patch. Always maintained
            # so it's available for both diagnostics and history records,
            # regardless of which reference mode is active.
            cum_alpha = float(entry.get("cum_alpha_since_patch", 0.0) or 0.0) + float(outcome_alpha or 0.0)
            entry["cum_alpha_since_patch"] = cum_alpha

            # Cumulative alpha the pre-patch policy would have earned on the
            # same bars (only advanced when the caller replays the shadow).
            cum_shadow = float(entry.get("cum_shadow_alpha_since_patch", 0.0) or 0.0) + float(shadow_alpha or 0.0)
            entry["cum_shadow_alpha_since_patch"] = cum_shadow

            nav_pct_change = (
                (current_nav - anchor) / anchor
                if anchor > 0 and current_nav > 0 else 0.0
            )

            if reference == "counterfactual_alpha":
                trigger_signal = cum_alpha
                fired = trigger_signal < -abs(dd_gate)
                impact = cum_alpha
            elif reference == "pre_patch_policy":
                # Strongest counterfactual: revert when the patched policy has
                # fallen more than dd_gate BELOW the pre-patch policy's shadow
                # alpha on the same bars. This isolates the patch's incremental
                # effect from both market moves and the old policy's own drift.
                incremental = cum_alpha - cum_shadow
                trigger_signal = incremental
                fired = incremental < -abs(dd_gate)
                impact = incremental
            else:  # pre_patch_nav
                trigger_signal = nav_pct_change
                fired = (
                    anchor > 0 and current_nav > 0
                    and nav_pct_change < -abs(dd_gate)
                )
                impact = nav_pct_change

            if fired:
                # Hypothesis falsified before the horizon expired.
                self._state.revert_patch(entry)
                reverted.append(entry)
                if history_idx is not None and 0 <= history_idx < len(self._state.history):
                    h = self._state.history[history_idx]
                    h["status"] = "reverted"
                    h["post_patch_return"] = round(nav_pct_change, 6)
                    h["cum_alpha_since_patch"] = round(cum_alpha, 6)
                    h["rollback_reason"] = reference
                    h["reverted_at_bar"] = entry["bars_since"]
                continue

            if entry["bars_since"] < horizon:
                kept.append(entry)
                continue

            # Horizon elapsed without tripping — commit the patch.
            if history_idx is not None and 0 <= history_idx < len(self._state.history):
                h = self._state.history[history_idx]
                h["status"] = "committed"
                h["post_patch_return"] = round(nav_pct_change, 6)
                h["cum_alpha_since_patch"] = round(cum_alpha, 6)
                h["committed_at_bar"] = entry["bars_since"]
        self._state.pending_patches = kept
        if reverted:
            self._persist()
        return reverted

    def mark_reviewed(self) -> None:
        self._last_review_idx = len(self._episodes) - 1
        self._persist()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def should_run_strategic_review(self) -> bool:
        """True when N new episodes have been written since the last review
        AND no earlier patch is still pending evaluation.

        Paper: the review cadence gate of Alg 2 (fires every N=3 completed
        episodes) combined with the "Only one proposal is evaluated at a time"
        exclusivity (Section "Strategic Evolution"; Table "Fixed
        hyperparameters": pending patches = 1).

        Pending-patch exclusivity: a previous patch that has not yet committed
        or reverted blocks a new review. This enforces "one hypothesis under
        test at a time" and prevents the compounding-tightening pathology (each
        review layering another adjustment on an unvalidated prior patch)."""
        last = self._last_review_idx + 1 if self._last_review_idx >= 0 else 0
        episodes_ready = (len(self._episodes) - last) >= STRATEGIC_REVIEW_EVERY_N_DAYS
        if not episodes_ready:
            return False
        return not self._state.pending_patches

    def review_payload(self, lookback_days: int = STRATEGIC_REVIEW_LOOKBACK) -> dict[str, Any]:
        """Build the input payload the strategic reflection LLM will see.

        Includes objective metrics + a feedback loop on past patches so the LLM
        can learn from its own track record (R6': self-evolution via reflection
        on outcomes, not via rules)."""
        recent = self._episodes[-lookback_days:]

        # ── objective system-state metrics ────────────────────────────────────
        nav_window = self._nav_history[-lookback_days:] if self._nav_history else []
        cur_nav = nav_window[-1] if nav_window else 0.0
        peak_nav = max(nav_window) if nav_window else 0.0
        dd_from_peak = (cur_nav - peak_nav) / peak_nav if peak_nav > 0 else 0.0
        # rolling 10-bar return
        recent10 = self._nav_history[-10:]
        ret_10d = (
            (recent10[-1] - recent10[0]) / recent10[0]
            if len(recent10) >= 2 and recent10[0] > 0 else 0.0
        )
        # trade-engagement ratio in last 10 bars (alpha-learning signal)
        recent_eps = self._episodes[-10:]
        traded = sum(
            1 for e in recent_eps if (e.metadata or {}).get("did_trade", False)
        )
        trade_ratio = traded / len(recent_eps) if recent_eps else 0.0

        objective_metrics = {
            "dd_from_peak": round(dd_from_peak, 4),
            "rolling_10d_return": round(ret_10d, 4),
            "trade_engagement_ratio_10d": round(trade_ratio, 3),
            "current_nav": round(cur_nav, 2),
            "peak_nav": round(peak_nav, 2),
        }

        # ── patch outcome history ────────────────────────────────────────────
        # Feed the LLM a causal record of its own past patches: what it
        # proposed, what rollback contract it attached, and — critically —
        # whether the patch was committed (survived its horizon) or
        # reverted (tripped its own drawdown gate), plus the realised NAV
        # impact. This is what makes the loop self-evolving: the LLM sees
        # the consequences of prior hypotheses and can revise direction
        # when its tightening patches have been consistently reverted.
        patch_track_record = []
        for h in self._state.history[-10:]:
            patch_track_record.append({
                "issued_on": h.get("issued_on"),
                "patch": h.get("patch"),
                "rationale": (h.get("rationale") or "")[:200],
                "rollback_trigger": h.get("rollback_trigger"),
                "anchor_nav": h.get("anchor_nav"),
                "status": h.get("status", "unknown"),
                # nav_impact = absolute NAV return since patch (rides market).
                # cum_alpha_since_patch = causal alpha vs benchmark (the
                # counterfactual rollback signal). The LLM should attend to
                # cum_alpha_since_patch when deciding whether prior patches
                # actually added alpha, rather than nav_impact which is
                # dominated by market drift.
                "nav_impact": h.get("post_patch_return"),
                "cum_alpha_since_patch": h.get("cum_alpha_since_patch"),
                "rollback_reason": h.get("rollback_reason"),
                "reverted_at_bar": h.get("reverted_at_bar"),
                "committed_at_bar": h.get("committed_at_bar"),
            })

        return {
            "lookback_days": lookback_days,
            "episodes": [e.to_dict() for e in recent],
            "capsule_summary": [c.to_dict() for c in sorted(self._capsules.values(), key=lambda c: -c.mean_score)],
            "nav_curve": list(self._nav_history[-lookback_days:]),
            "regime_history": [e.regime for e in recent],
            "active_strategy_state": self._state.to_dict(),
            "objective_metrics": objective_metrics,
            "patch_track_record": patch_track_record,
        }


    def compute_strategic_baseline(self, regime: str, current_nav: float = 0.0) -> MemoryInfluence:
        """
        Strategic baseline = (active patch state) merged with capsule-derived
        preferred optimizer.

        The pipeline combines this with TacticalInfluence later.
        """
        dd = _dd_metrics(
            self._nav_history, current_nav,
            dd_threshold=self._dd_threshold,
            rebound_threshold=self._rebound_threshold,
        )
        in_drawdown = bool(dd["in_drawdown"])
        in_rebound = bool(dd["in_rebound"])

        regime_capsules = [
            c for c in self._capsules.values()
            if c.regime == regime and c.status == "active"
        ]
        preferred_optimizer = self._state.preferred_optimizer
        confidence = 0.0
        sample_count = sum(c.sample_count for c in regime_capsules)
        if not preferred_optimizer and regime_capsules:
            best = max(regime_capsules, key=lambda c: c.mean_score)
            if (
                best.sample_count >= CAPSULE_MIN_SAMPLES_FOR_INFLUENCE
                and best.confidence >= CAPSULE_MIN_CONFIDENCE_FOR_ACTIVE
            ):
                preferred_optimizer = best.optimizer
                confidence = best.confidence

        # constraint overrides: explicit strategic patch first, then drawdown-aware defaults.
        # Recovery clears reach into self._state.constraint_overrides so patches
        # don't accumulate indefinitely. Otherwise a patch that tightens
        # min_confidence_threshold or max_single_position would live forever
        # and the LLM would see them as "already tightened, tighten more".
        overrides: dict[str, Any] = dict(self._state.constraint_overrides)
        persistent = self._state.constraint_overrides
        if in_drawdown:
            # De-risk by shrinking the TOTAL exposure budget, not the per-name
            # concentration cap. max_single_position is a diversification limit:
            # in a multi-symbol portfolio it spreads risk, but with a single
            # eligible symbol it collapses to a hard "go (near) flat" order,
            # which whipsaws the book out of high-beta names right before they
            # rebound. max_gross_exposure is the correct lever for de-risking
            # because it scales with how many names are actually held.
            overrides["max_gross_exposure"] = min(overrides.get("max_gross_exposure", 0.7), 0.7)
        elif in_rebound:
            overrides["max_gross_exposure"] = max(overrides.get("max_gross_exposure", 0.9), 0.9)
            # remove single-position cap if it was set by a prior drawdown patch
            overrides.pop("max_single_position", None)
            persistent.pop("max_single_position", None)
        else:
            # fully recovered: clear ALL drawdown-era patches from BOTH the
            # local overrides dict and the persistent state. Without writing
            # back to persistent state, patches stay forever and compound.
            persistent_changed = False
            for key in (
                "max_gross_exposure",
                "max_single_position",
                "min_confidence_threshold",
                "position_size_multiplier",
            ):
                overrides.pop(key, None)
                if key in persistent:
                    persistent.pop(key)
                    persistent_changed = True
            if persistent_changed:
                self._persist()

        rationale_parts = []
        if preferred_optimizer:
            rationale_parts.append(f"strategic_optimizer={preferred_optimizer}")
        if self._state.constraint_overrides:
            rationale_parts.append(f"active_patch_keys={list(self._state.constraint_overrides)}")
        if in_drawdown:
            rationale_parts.append(f"drawdown={dd['dd_from_peak']:.1%}")
        if in_rebound:
            rationale_parts.append(f"rebound={dd['dd_5d']:.1%}")

        return MemoryInfluence(
            preferred_optimizer=preferred_optimizer,
            no_trade_symbols=[],
            reduce_only_symbols=[],
            constraint_overrides=overrides,
            position_haircut=1.0,
            rationale="; ".join(rationale_parts),
            confidence=confidence,
            sample_count=sample_count,
            source_layer="strategic",
        )

    def get_recent_episodes(self, n: int = 20) -> list[EpisodeRecord]:
        return list(self._episodes[-n:])

    def get_capsule_summary(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in sorted(self._capsules.values(), key=lambda c: -c.mean_score)]

    def get_nav_curve(self, n: int = 60) -> list[float]:
        return list(self._nav_history[-n:])

    def get_state(self) -> StrategicState:
        return self._state

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    @property
    def nav_history(self) -> list[float]:
        return list(self._nav_history)

    @property
    def dd_threshold(self) -> float:
        return self._dd_threshold

    @property
    def rebound_threshold(self) -> float:
        return self._rebound_threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        with self._episodes_path.open("w", encoding="utf-8") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=True) + "\n")
        capsule_data = {k: {**c.to_dict(), "scores": c.scores} for k, c in self._capsules.items()}
        self._capsules_path.write_text(
            json.dumps({"capsules": capsule_data}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._nav_path.write_text(json.dumps(self._nav_history, ensure_ascii=True), encoding="utf-8")
        self._state_path.write_text(
            json.dumps({
                "state": self._state.to_dict(),
                "history": self._state.history,
                "last_review_idx": self._last_review_idx,
            }, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if self._episodes_path.exists():
            for line in self._episodes_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._episodes.append(EpisodeRecord(
                        trade_date=d.get("trade_date", ""),
                        symbols=list(d.get("symbols", [])),
                        regime=d.get("regime", ""),
                        optimizer_used=d.get("optimizer_used", ""),
                        outcome_score=_safe_finite(d.get("outcome_score")),
                        nav=_safe_finite(d.get("nav")),
                        dd_from_peak=_safe_finite(d.get("dd_from_peak")),
                        in_drawdown=bool(d.get("in_drawdown", False)),
                        in_rebound=bool(d.get("in_rebound", False)),
                        tactical_reflection=dict(d.get("tactical_reflection", {}) or {}),
                        metadata=dict(d.get("metadata", {}) or {}),
                    ))
                except Exception:
                    continue
        if self._capsules_path.exists():
            try:
                data = json.loads(self._capsules_path.read_text(encoding="utf-8"))
                for k, v in data.get("capsules", {}).items():
                    self._capsules[k] = SemanticCapsule.from_dict(v)
            except Exception:
                pass
        if self._nav_path.exists():
            try:
                self._nav_history = [_safe_finite(x) for x in json.loads(self._nav_path.read_text(encoding="utf-8"))]
            except Exception:
                pass
        if self._state_path.exists():
            try:
                payload = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._state = StrategicState.from_dict({
                    **(payload.get("state", {}) or {}),
                    "history": payload.get("history", []) or [],
                })
                self._last_review_idx = int(payload.get("last_review_idx", -1) or -1)
            except Exception:
                pass


class NullStrategicMemory:
    def write_episode(self, episode: EpisodeRecord) -> None:
        pass

    def apply_patch(self, patch: StrategicPatch, *, current_nav: float = 0.0) -> None:
        pass

    def evaluate_pending_patches(self, *, current_nav: float,
                                  outcome_alpha: float = 0.0,
                                  **_legacy_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def update_thresholds(self, *, drawdown_guard: float | None = None, rebound_entry: float | None = None) -> None:
        pass

    def mark_reviewed(self) -> None:
        pass

    def should_run_strategic_review(self) -> bool:
        return False

    def review_payload(self, lookback_days: int = STRATEGIC_REVIEW_LOOKBACK) -> dict[str, Any]:
        return {"lookback_days": lookback_days, "episodes": [], "capsule_summary": [], "nav_curve": [], "regime_history": [], "active_strategy_state": {}}

    def compute_strategic_baseline(self, regime: str, current_nav: float = 0.0) -> MemoryInfluence:
        return MemoryInfluence(source_layer="strategic")

    def get_recent_episodes(self, n: int = 20) -> list[EpisodeRecord]:
        return []

    def get_capsule_summary(self) -> list[dict[str, Any]]:
        return []

    def get_nav_curve(self, n: int = 60) -> list[float]:
        return []

    def get_state(self) -> StrategicState:
        return StrategicState()

    @property
    def total_episodes(self) -> int:
        return 0

    @property
    def nav_history(self) -> list[float]:
        return []

    @property
    def dd_threshold(self) -> float:
        return DD_THRESHOLD

    @property
    def rebound_threshold(self) -> float:
        return REBOUND_THRESHOLD


__all__ = [
    "StrategicMemory",
    "NullStrategicMemory",
    "SemanticCapsule",
    "StrategicState",
    "STRATEGIC_REVIEW_EVERY_N_DAYS",
    "STRATEGIC_REVIEW_LOOKBACK",
    "_dd_metrics",
    "_outcome_score",
]
