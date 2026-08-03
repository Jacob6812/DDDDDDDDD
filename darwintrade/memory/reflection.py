"""
Reflection agents — two layers.

  TacticalReflectionAgent  — runs every bar (daily). Looks at last day only.
                             Emits TacticalReflection (mistake type + lessons).
                             Used to feed TacticalEvolutionAgent.

  StrategicReflectionAgent — runs every N days. Looks across long lookback.
                             Emits StrategicReflection (cross-day pattern).
                             Used to feed StrategicDiagnosisAgent.

Paper correspondence (darwintrade_aaai2027.tex):
  These are the "reflect / diagnose" LLM steps that precede each self-evolution
  update.
  - TacticalReflectionAgent = f_tac-reflect of Alg 1 (q^T <- f_tac-reflect(
    tau_{t-1}, y^pf_t, U^T_t)): it composes a reflection over the previous
    decision trace and the realized next-open portfolio return, the first half
    of f_tactical-evolving (Eq. tactical-evolving).
  - StrategicReflectionAgent = f_str-review of Alg 2 (z^S <- f_str-review(
    H^S_j, Z^S_j, pi_j)): it diagnoses the multi-day pattern over the 20-day
    history H^S_j; its strategic_signal_strength is the z^S.signal compared
    against the tau=0.5 gate (Section "Strategic Evolution").
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..core.contracts import (
    EpisodeRecord,
    MemoryInfluence,
    StrategicReflection,
    TacticalReflection,
)
from ..core.llm import LLMClient
from ..core.skills import skill_section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TACTICAL reflection
# ---------------------------------------------------------------------------

_TACTICAL_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "mistake_type": {
            "type": "string",
            "enum": ["over_exposure", "wrong_direction", "missed_signal",
                     "execution_friction", "risk_event", "none"],
        },
        "immediate_lessons": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["mistake_type", "summary"],
    "additionalProperties": False,
}

_TACTICAL_REFLECTION_BASE = """You are DarwinTrade's tactical reflection agent.
Look at YESTERDAY's trading day in isolation. Identify the dominant mistake (or
say 'none'), and extract 1-3 lessons for tomorrow.
Do NOT propose strategy parameter changes — that is the strategic layer's job.

CRITICAL — distinguish "did not trade" from "traded badly":
- If did_trade is False and orders_placed is empty, the system held its prior
  state. alpha=0 in that case is EXPECTED, not a wrong_direction signal.
  Use mistake_type="none" or "missed_signal" (only if a clear opportunity was
  ignored), never "wrong_direction" or "over_exposure" on a no-trade day.
- "wrong_direction" requires actual orders to have been placed AND alpha to be
  meaningfully negative.
- "over_exposure" requires real positions AND a drawdown driven by them.

Return valid JSON only."""


def _tactical_reflection_system() -> str:
    return _TACTICAL_REFLECTION_BASE + skill_section("tactical-reflection")


class TacticalReflectionAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def reflect(
        self,
        *,
        trade_date: str,
        episode: EpisodeRecord,
        execution_summary: dict[str, Any],
        memory_influence: MemoryInfluence,
        regime: str,
    ) -> TacticalReflection:
        meta = episode.metadata or {}
        prompt = json.dumps({
            "trade_date": trade_date,
            "regime": regime,
            "last_day_outcome": {
                "outcome_score": round(episode.outcome_score, 4),
                "alpha": round(meta.get("alpha", 0.0) or 0.0, 4),
                "nav": round(episode.nav, 2),
                "dd_from_peak": round(episode.dd_from_peak, 4),
                "in_drawdown": episode.in_drawdown,
                "in_rebound": episode.in_rebound,
                "did_trade": bool(meta.get("did_trade", False)),
                "orders_placed": list(meta.get("orders", []) or []),
                "holdings_at_open": list(meta.get("holdings", []) or []),
                "target_weights": dict(meta.get("target_weights", {}) or {}),
                "universe_analysed": list(episode.symbols),
            },
            "last_day_decisions": dict(execution_summary or {}),
            "active_memory_influence": memory_influence.to_dict(),
            "task": (
                "Pick mistake_type from the enum. Write up to 3 immediate_lessons. "
                "Write a one-sentence summary. Stay tactical — do not propose "
                "strategy config changes. If did_trade is False, prefer 'none' or "
                "'missed_signal' over wrong_direction/over_exposure."
            ),
        }, ensure_ascii=True)

        try:
            payload = self.llm.invoke_json(
                _tactical_reflection_system(), prompt,
                caller_tag="tactical_reflection",
                schema_name="TacticalReflection",
                schema=_TACTICAL_REFLECTION_SCHEMA,
            )
        except Exception as exc:
            logger.warning("TacticalReflectionAgent failed: %s", exc)
            payload = {"mistake_type": "none", "summary": "reflection unavailable", "immediate_lessons": []}

        return TacticalReflection(
            trade_date=trade_date,
            mistake_type=str(payload.get("mistake_type", "none")),
            immediate_lessons=list(payload.get("immediate_lessons", [])),
            summary=str(payload.get("summary", "")),
        )


# ---------------------------------------------------------------------------
# STRATEGIC reflection
# ---------------------------------------------------------------------------

_STRATEGIC_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_pattern": {
            "type": "string",
            "enum": ["alpha_decay", "regime_shift", "capsule_unstable",
                     "consistent_winner", "drawdown_structural", "none"],
        },
        "persistent_winners": {"type": "array", "items": {"type": "string"}},
        "persistent_losers": {"type": "array", "items": {"type": "string"}},
        "regime_observation": {"type": "string"},
        "lessons": {"type": "array", "items": {"type": "string"}},
        "strategic_signal_strength": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["dominant_pattern", "strategic_signal_strength", "summary"],
    "additionalProperties": False,
}

_STRATEGIC_REFLECTION_BASE = """You are DarwinTrade's strategic reflection agent.
Look at the multi-day pattern across the lookback window. Identify ONE dominant
cross-day pattern. Output strategic_signal_strength 0-1 — high only if a strategy
patch is genuinely warranted. Be conservative.

The input includes two critical feedback channels — USE THEM:

1. `objective_metrics`: dd_from_peak, rolling_10d_return,
   trade_engagement_ratio_10d, current_nav, peak_nav.
   • A patch is rarely warranted when dd_from_peak > -0.02 AND
     rolling_10d_return > -0.01 — the system is healthy, leave it alone.
   • Low trade_engagement_ratio_10d (< 0.3) signals under-deployment;
     diagnosis should consider RELAXING constraints, not tightening.

2. `patch_track_record`: every recent patch you (or a prior reviewer) issued,
   with status (`pending`, `committed`, `reverted`) and `post_patch_return`.
   • Read your own history before recommending another patch.
   • If a recent patch was `reverted` (auto-rolled-back due to NAV loss),
     do NOT propose the same change again — it failed.
   • If a recent patch was `committed` with positive post_patch_return, the
     direction worked — do not reverse it without strong reason.
   • If you've patched in the last 5-10 bars, consider whether the system
     has had time to express the change before adding more.

CRITICAL — read each episode's metadata.did_trade and metadata.orders:
- If most/all recent episodes have did_trade=False (flat NAV, no orders), the
  problem is NOT alpha decay or wrong direction — it's that the system is
  failing to ENTER. The dominant pattern in that case is closer to
  "alpha_decay" only if you also see deteriorating signal quality; otherwise
  use "none" with low signal_strength.
- When the system is already not trading, do NOT recommend tightening ANY of:
  min_confidence_threshold, max_single_position, max_gross_exposure,
  position_size_multiplier. Tightening these makes a no-trade system trade
  even less. The correct response is to RELAX or do nothing.
- A patch that pushes any of these fields toward the floor (max_single_position
  below 0.1, max_gross_exposure below 0.5, position_size_multiplier below 1.0)
  is suspect and requires very strong evidence in the episode log of actual
  over-exposure / over-trading — not just losses or flat NAV.

Return valid JSON only."""


def _strategic_reflection_system() -> str:
    return _STRATEGIC_REFLECTION_BASE + skill_section("strategic-reflection")


class StrategicReflectionAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def reflect(
        self,
        *,
        review_date: str,
        review_payload: dict[str, Any],
    ) -> StrategicReflection:
        prompt = json.dumps({
            "review_date": review_date,
            **review_payload,
            "task": (
                "Identify the dominant cross-day pattern. List persistent_winners "
                "and persistent_losers from data. Note any regime observation. "
                "Set strategic_signal_strength 0-1 — high only if a config patch "
                "is warranted, low if tactical adjustment suffices."
            ),
        }, ensure_ascii=True)

        try:
            payload = self.llm.invoke_json(
                _strategic_reflection_system(), prompt,
                caller_tag="strategic_reflection",
                schema_name="StrategicReflection",
                schema=_STRATEGIC_REFLECTION_SCHEMA,
            )
        except Exception as exc:
            logger.warning("StrategicReflectionAgent failed: %s", exc)
            payload = {"dominant_pattern": "none", "strategic_signal_strength": 0.0, "summary": "reflection unavailable"}

        return StrategicReflection(
            review_date=review_date,
            lookback_days=int(review_payload.get("lookback_days", 0) or 0),
            dominant_pattern=str(payload.get("dominant_pattern", "none")),
            persistent_winners=list(payload.get("persistent_winners", [])),
            persistent_losers=list(payload.get("persistent_losers", [])),
            regime_observation=str(payload.get("regime_observation", "")),
            lessons=list(payload.get("lessons", [])),
            strategic_signal_strength=float(payload.get("strategic_signal_strength", 0.0) or 0.0),
            summary=str(payload.get("summary", "")),
        )


__all__ = ["TacticalReflectionAgent", "StrategicReflectionAgent"]
