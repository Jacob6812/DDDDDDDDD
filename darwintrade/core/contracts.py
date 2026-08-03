from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


_VALID_ROLLBACK_REFS = ("counterfactual_alpha", "pre_patch_nav", "pre_patch_policy")


def _rollback_reference_override() -> str | None:
    """Force all rollback contracts to a given reference mode, overriding
    whatever the policy-author LLM emitted. Set DARWIN_ROLLBACK_REFERENCE to one
    of the valid modes to enable; unset/invalid leaves per-patch behaviour."""
    ref = os.getenv("DARWIN_ROLLBACK_REFERENCE", "").strip()
    return ref if ref in _VALID_ROLLBACK_REFS else None



def _dd(instance: Any, **overrides: Any) -> dict[str, Any]:
    data = asdict(instance)
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Market / regime
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegimeState:
    """Current market regime belief."""
    regime: str                          # bull / bear / sideways / volatile
    confidence: float                    # 0-1
    trade_date: str
    evidence: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


# ---------------------------------------------------------------------------
# Asset research
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AssetSignal:
    """LLM-produced signal for a single asset."""
    symbol: str
    trade_date: str
    direction: str                       # long / short / hold
    confidence: float                    # 0-1
    thesis: str
    regime: str = ""
    risk_score: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PortfolioState:
    trade_date: str
    cash: float
    total_equity: float
    positions: dict[str, float] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


@dataclass(slots=True)
class AllocationTarget:
    symbol: str
    target_weight: float                 # signed: positive=long, negative=short
    direction: str                       # long / short / flat
    confidence: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


@dataclass(slots=True)
class ExecutionOrder:
    symbol: str
    action: str                          # open_long / close_long / open_short / close_short / flip_to_long / flip_to_short / hold
    side: str                            # buy / sell / sell_short / buy_to_cover
    target_cash_amount: float            # signed: positive=long exposure, negative=short exposure
    cash_change: float                   # signed delta to current signed exposure
    shares: float                        # absolute share count for THIS leg
    reference_price: float
    confidence: float
    execution_intent: str = ""
    rationale: str = ""
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


@dataclass(slots=True)
class ExecutionPlan:
    trade_date: str
    orders: list[ExecutionOrder] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    no_trade_reason: str = ""
    feasible: bool = True
    audit_trail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


# ---------------------------------------------------------------------------
# Tactical layer (daily, short-term)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TacticalInfluence:
    """Short-lived tactical guidance produced after each bar.
    Only adjusts the next day's behavior — never modifies strategy config."""
    avoid_symbols: list[str] = field(default_factory=list)
    reduce_only_symbols: list[str] = field(default_factory=list)
    position_haircut: float = 1.0        # multiply target weights by this; 1.0 = no haircut
    expires_in_days: int = 1
    urgency: str = "normal"              # normal / elevated / emergency
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


@dataclass(slots=True)
class TacticalReflection:
    """Output of the daily tactical reflection LLM."""
    trade_date: str
    mistake_type: str                    # over_exposure / wrong_direction / missed_signal / execution_friction / risk_event / none
    immediate_lessons: list[str] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


# ---------------------------------------------------------------------------
# Strategic layer (multi-day, long-term)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EpisodeRecord:
    """One completed trade-day episode persisted in strategic memory."""
    trade_date: str
    symbols: list[str]
    regime: str
    optimizer_used: str
    outcome_score: float                 # alpha-adjusted score
    nav: float
    dd_from_peak: float
    in_drawdown: bool
    in_rebound: bool
    tactical_reflection: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


@dataclass(slots=True)
class StrategicReflection:
    """Cross-day pattern analysis."""
    review_date: str
    lookback_days: int
    dominant_pattern: str                # alpha_decay / regime_shift / capsule_unstable / consistent_winner / drawdown_structural / none
    persistent_winners: list[str] = field(default_factory=list)
    persistent_losers: list[str] = field(default_factory=list)
    regime_observation: str = ""
    lessons: list[str] = field(default_factory=list)
    strategic_signal_strength: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


@dataclass(slots=True)
class RollbackTrigger:
    """Machine-executable rollback contract attached to every StrategicPatch.

    The strategic policy-author LLM emits this alongside the patch itself. The
    strategic memory later reads these fields (not hardcoded constants) when
    deciding whether an applied patch has hurt performance and should be
    reverted. This closes the loop between the LLM's stated hypothesis
    ("this patch should be reverted if X") and the code that verifies it.

    Fields:
      eval_horizon_bars : number of bars to wait before the patch is either
                          committed (kept) or reverted. Must be >= 1.
      drawdown_pct      : positive fraction. Interpretation depends on
                          `reference` (see below).
      reference         : which signal the evaluator compares against the
                          drawdown_pct gate:
        - "counterfactual_alpha" (default, recommended): revert when the
          cumulative alpha (portfolio return minus benchmark return)
          since the patch was applied drops more than drawdown_pct below
          zero. This is a causal-inference rollback: it asks "did this
          patch add or subtract alpha vs the market I trade against?".
          Robust to co-moving market drawdowns — a patch that just rode
          the market down is NOT reverted; only one that materially
          underperformed the market is.
        - "pre_patch_nav" (legacy): revert when absolute NAV drops more
          than drawdown_pct below the pre-patch anchor. Provided for
          backward compatibility; prone to false positives in market-
          wide drawdowns and false negatives in market-wide rallies.
        - "pre_patch_policy" (strongest counterfactual): revert when the
          patched policy's cumulative alpha falls more than drawdown_pct
          below the alpha the PRE-PATCH policy would have earned on the same
          bars (a shadow replay). This answers "is the patch better than the
          policy it replaced?" rather than "is it better than the market?".
          Requires the caller to supply shadow_alpha each bar.
    """
    eval_horizon_bars: int = 5
    drawdown_pct: float = 0.02
    reference: str = "counterfactual_alpha"

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)

    @classmethod
    def from_any(cls, value: Any) -> "RollbackTrigger":
        """Parse from a dict or return the default. Free-text strings (legacy
        payloads) are parsed on a best-effort basis and any missing fields
        fall back to defaults. Fields outside safe bounds are clamped."""
        if isinstance(value, RollbackTrigger):
            return value
        if isinstance(value, dict):
            raw_h = value.get("eval_horizon_bars", 5)
            try:
                horizon = int(raw_h) if raw_h is not None else 5
            except (TypeError, ValueError):
                horizon = 5
            raw_dd = value.get("drawdown_pct", 0.02)
            try:
                dd = float(raw_dd) if raw_dd is not None else 0.02
            except (TypeError, ValueError):
                dd = 0.02
            raw_ref = str(value.get("reference", "counterfactual_alpha") or "counterfactual_alpha")
            _valid_refs = _VALID_ROLLBACK_REFS
            ref = raw_ref if raw_ref in _valid_refs else "counterfactual_alpha"
            # The env override wins over the LLM-emitted reference when set.
            ref = _rollback_reference_override() or ref
            # Bounds: horizon in [1, 30] bars, dd in [0.005, 0.15]. Outside
            # these ranges either strips the trigger of meaning (huge dd
            # never fires) or destabilizes evaluation (dd < 0.5% fires on
            # normal noise).
            horizon = max(1, min(30, horizon))
            dd = max(0.005, min(0.15, dd))
            return cls(eval_horizon_bars=horizon, drawdown_pct=dd, reference=ref)
        return cls(reference=_rollback_reference_override() or "counterfactual_alpha")


@dataclass(slots=True)
class StrategicPatch:
    """Long-lived strategy config patch."""
    patch: dict[str, Any]
    rationale: str
    expected_effects: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    # Structured, machine-executable rollback contract. See RollbackTrigger.
    # LLM output is validated + clamped by RollbackTrigger.from_any at parse
    # time so out-of-range values cannot destabilize the evaluator.
    rollback_trigger: RollbackTrigger = field(default_factory=RollbackTrigger)
    issued_on: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _dd(self, rollback_trigger=self.rollback_trigger.to_dict())


# ---------------------------------------------------------------------------
# Combined memory influence (what allocator/market_agent actually consume)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MemoryInfluence:
    """Combined view of strategic baseline + active tactical hints.
    The allocator and market agent read this single object."""
    preferred_optimizer: str = ""
    no_trade_symbols: list[str] = field(default_factory=list)
    reduce_only_symbols: list[str] = field(default_factory=list)
    constraint_overrides: dict[str, Any] = field(default_factory=dict)
    position_haircut: float = 1.0
    rationale: str = ""
    confidence: float = 0.0
    sample_count: int = 0
    source_layer: str = "combined"       # tactical / strategic / combined

    def to_dict(self) -> dict[str, Any]:
        return _dd(self)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineResult:
    trade_date: str
    universe: list[str]
    regime: RegimeState
    signals: list[AssetSignal]
    execution_plan: ExecutionPlan
    memory_influence: MemoryInfluence
    benchmark_payload: list[dict[str, Any]] = field(default_factory=list)
    tactical_reflection: TacticalReflection | None = None
    tactical_influence: TacticalInfluence | None = None
    strategic_reflection: StrategicReflection | None = None
    strategic_patch: StrategicPatch | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dd(
            self,
            regime=self.regime.to_dict(),
            signals=[s.to_dict() for s in self.signals],
            execution_plan=self.execution_plan.to_dict(),
            memory_influence=self.memory_influence.to_dict(),
            tactical_reflection=self.tactical_reflection.to_dict() if self.tactical_reflection else None,
            tactical_influence=self.tactical_influence.to_dict() if self.tactical_influence else None,
            strategic_reflection=self.strategic_reflection.to_dict() if self.strategic_reflection else None,
            strategic_patch=self.strategic_patch.to_dict() if self.strategic_patch else None,
        )


__all__ = [
    "RegimeState",
    "AssetSignal",
    "PortfolioState",
    "AllocationTarget",
    "ExecutionOrder",
    "ExecutionPlan",
    "TacticalInfluence",
    "TacticalReflection",
    "EpisodeRecord",
    "StrategicReflection",
    "StrategicPatch",
    "RollbackTrigger",
    "MemoryInfluence",
    "PipelineResult",
]
