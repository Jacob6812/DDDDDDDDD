"""
Evolution agents — two layers.

  TacticalEvolutionAgent  — every bar. Reads tactical reflection + recent 5d
                            summary. Outputs TacticalInfluence (1-3d TTL).
                            Never modifies strategy config.

  StrategicEvolutionAgent — every N days. Reads strategic reflection + capsule
                            summary. Runs diagnose → policy_author. Outputs
                            StrategicPatch that updates strategy config.

Paper correspondence (darwintrade_aaai2027.tex):
  These are the LLM producers of the tactical and strategic evolution states.
  - TacticalEvolutionAgent.evolve is the install-with-TTL producer inside
    f_tactical-evolving (Eq. tactical-evolving; Alg 1 f_tac-evolve). It emits a
    guard action (avoid / reduce-only / haircut) or an empty influence.
  - StrategicEvolutionAgent.maybe_evolve is f_str-policy of Alg 2 ("Author a
    minimal patch with a rollback trigger"): it proposes the patch tuple
    (Delta-pi, H, epsilon, m) over the allowed parameter set Omega and attaches
    the RollbackTrigger (epsilon, H, reference) — the validated candidate of
    Eq. strategic-evolving. The commit/rollback decision itself lives in
    StrategicMemory.evaluate_pending_patches.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from ..core.contracts import (
    RollbackTrigger,
    StrategicPatch,
    StrategicReflection,
    TacticalInfluence,
    TacticalReflection,
)
from ..core.llm import LLMClient
from ..core.skills import skill_section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TACTICAL evolution (per bar)
# ---------------------------------------------------------------------------

_TACTICAL_INFLUENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "tactical_influence": {
            "type": "object",
            "properties": {
                "avoid_symbols": {"type": "array", "items": {"type": "string"}},
                "reduce_only_symbols": {"type": "array", "items": {"type": "string"}},
                "position_haircut": {"type": "number"},
                "expires_in_days": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
        "urgency": {"type": "string", "enum": ["normal", "elevated", "emergency"]},
    },
    "required": ["tactical_influence", "rationale"],
    "additionalProperties": False,
}

_TACTICAL_EVOLUTION_BASE = """You are DarwinTrade's tactical evolution agent.
Given yesterday's reflection, recent 5-day summary, AND your own past
tactical-influence track record, decide whether to install short-term
constraints for tomorrow. You are NOT limited to "tighten" — you can also
return an empty influence (no adjustment) when the right call is to do nothing.

The input includes `track_record`: each prior install_log entry with
post_install_return (NAV % change while the influence was active or just after).
USE IT:
- If your last few haircut/avoid installs had NEGATIVE post_install_return,
  they HURT the system. Be much more conservative — prefer no influence over
  installing one that is likely to hurt again.
- If your last installs had POSITIVE post_install_return, the direction
  worked; you can install similar adjustments with confidence.
- Repeated `wrong_direction` reflections do NOT automatically warrant a
  haircut. The data is in your track record — check whether haircuts have
  actually helped recently before installing another one.
- An influence with avoid_symbols can lock the system out of a name during
  a reversal — only avoid a symbol when reflection has direct, specific
  evidence (not just "lost money on it once").

Do NOT change strategy config. Return valid JSON only."""


def _tactical_evolution_system() -> str:
    return _TACTICAL_EVOLUTION_BASE + skill_section("tactical-evolution")


# Field names the LLM has been observed echoing into avoid/reduce_only lists.
# These are not tickers — drop them silently.
_INFLUENCE_FIELD_NOISE = {
    "POSITION_HAIRCUT", "AVOID_SYMBOLS", "REDUCE_ONLY_SYMBOLS",
    "EXPIRES_IN_DAYS", "URGENCY", "RATIONALE",
}


def _clean_symbol_list(
    raw: Any,
    universe: set[str],
    *,
    exclude: set[str] | None = None,
    cap: int = 10,
) -> list[str]:
    """Normalize, dedup, drop field-name noise, and (if a universe is provided)
    intersect with the tradable set. Order-preserving."""
    if not raw:
        return []
    excl = exclude or set()
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        sym = str(item).strip().upper()
        if not sym or sym in seen or sym in _INFLUENCE_FIELD_NOISE or sym in excl:
            continue
        if universe and sym not in universe:
            continue
        seen.add(sym)
        out.append(sym)
        if len(out) >= cap:
            break
    return out


def _make_tactical_validator(universe: set[str]) -> Callable[[dict[str, Any]], None]:
    """Return a validator that raises (triggers retry) when the LLM emits
    obviously-broken tactical influence — invalid symbols, field-name noise,
    or duplicate ticker spam — instead of letting us silently sanitize and
    losing the chance to teach the LLM."""
    def validate(payload: dict[str, Any]) -> None:
        inf = payload.get("tactical_influence") or {}
        for field_key in ("avoid_symbols", "reduce_only_symbols"):
            raw = inf.get(field_key) or []
            if not isinstance(raw, list):
                raise ValueError(
                    f"tactical_influence.{field_key} must be a list, got {type(raw).__name__}"
                )
            seen: set[str] = set()
            for item in raw:
                sym = str(item).strip().upper()
                if not sym:
                    continue
                if sym in _INFLUENCE_FIELD_NOISE:
                    raise ValueError(
                        f"tactical_influence.{field_key} contains schema field name "
                        f"{sym!r} — this is a list of TICKERS, not field names"
                    )
                if universe and sym not in universe:
                    raise ValueError(
                        f"tactical_influence.{field_key} contains {sym!r} which is "
                        f"not in tradable_universe — only use tickers from the "
                        f"provided universe list"
                    )
                if sym in seen:
                    raise ValueError(
                        f"tactical_influence.{field_key} contains duplicate {sym!r} — "
                        f"each symbol must appear at most once"
                    )
                seen.add(sym)
        haircut = inf.get("position_haircut")
        if haircut is not None:
            try:
                hv = float(haircut)
            except (TypeError, ValueError):
                raise ValueError("position_haircut must be a number")
            if not 0.4 <= hv <= 1.0:
                raise ValueError(
                    f"position_haircut={hv} out of range — must be in [0.4, 1.0]"
                )
        expires = inf.get("expires_in_days")
        if expires is not None:
            try:
                ev = int(expires)
            except (TypeError, ValueError):
                raise ValueError("expires_in_days must be an integer")
            if not 1 <= ev <= 3:
                raise ValueError(f"expires_in_days={ev} out of range — must be in [1, 3]")
    return validate


def _validate_strategic_patch_payload(payload: dict[str, Any]) -> None:
    """Reject strategic patches the runtime would silently drop or clamp.
    Raising here triggers the LLM retry loop with the error message embedded
    so the LLM can self-correct rather than us absorbing the malformed output."""
    patch = payload.get("patch") or {}
    if not isinstance(patch, dict):
        raise ValueError(f"patch must be an object, got {type(patch).__name__}")
    unknown = [k for k in patch if k not in ALLOWED_PATCH_FIELDS]
    if unknown:
        raise ValueError(
            f"patch contains unknown fields {unknown} — allowed fields are "
            f"{sorted(ALLOWED_PATCH_FIELDS)}"
        )
    opt = patch.get("preferred_optimizer")
    if opt is not None:
        opt_norm = str(opt).strip().lower()
        if opt_norm not in ALLOWED_OPTIMIZERS:
            raise ValueError(
                f"preferred_optimizer={opt!r} is not supported — pick one of "
                f"{sorted(ALLOWED_OPTIMIZERS)}"
            )
    for k, lo, hi in (
        ("min_confidence_threshold", 0.0, 0.6),
        ("max_single_position", 0.05, 1.0),
        ("max_gross_exposure", 0.3, 1.0),
        ("position_size_multiplier", 0.5, 3.0),
        ("drawdown_guard_threshold", 0.0, 1.0),
        ("rebound_entry_threshold", 0.0, 1.0),
    ):
        if k in patch:
            try:
                v = float(patch[k])
            except (TypeError, ValueError):
                raise ValueError(f"{k} must be a number, got {patch[k]!r}")
            if not lo <= v <= hi:
                raise ValueError(
                    f"{k}={v} out of range — must be in [{lo}, {hi}]"
                )


class TacticalEvolutionAgent:
    """Reads tactical reflection, emits short-lived TacticalInfluence.

    Paper: the guard-action producer inside f_tactical-evolving (Eq.
    tactical-evolving; Alg 1 f_tac-evolve). It "produces a guard action that
    may install an avoid, reduce-only, or haircut constraint, or remain empty"
    (Section "Tactical Evolution"). The InstallExpire/TTL is applied by
    TacticalMemory, not here."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def evolve(
        self,
        *,
        reflection: TacticalReflection,
        recent_summary: dict[str, Any],
        active_strategy_state: dict[str, Any],
        track_record: list[dict[str, Any]] | None = None,
        universe: list[str] | None = None,
    ) -> TacticalInfluence:
        # Cheap shortcut: if everything looks fine, skip LLM
        if reflection.mistake_type == "none":
            mistake_counts = recent_summary.get("mistake_counts", {})
            if not mistake_counts:
                return TacticalInfluence(rationale="no tactical adjustment needed")

        universe_set = {str(s).upper() for s in (universe or []) if s}
        prompt = json.dumps({
            "tactical_reflection": reflection.to_dict(),
            "recent_5d_summary": recent_summary,
            "active_strategy_state": active_strategy_state,
            "track_record": track_record or [],
            "tradable_universe": sorted(universe_set),
            "task": (
                "Decide whether to install a tactical influence for tomorrow. "
                "Read your track_record FIRST — if recent installs hurt NAV "
                "(negative post_install_return), prefer no influence this time. "
                "Only install constraints with concrete evidence from yesterday's "
                "reflection. Empty influence (no adjustment) is a valid output. "
                "avoid_symbols and reduce_only_symbols MUST be drawn from "
                "tradable_universe; do NOT invent tickers or echo field names. "
                "position_haircut in [0.4, 1.0]. expires_in_days in [1, 3]. "
                "Set urgency=emergency only for risk_event."
            ),
        }, ensure_ascii=True)

        try:
            payload = self.llm.invoke_json(
                _tactical_evolution_system(), prompt,
                caller_tag="tactical_evolution",
                schema_name="TacticalEvolution",
                schema=_TACTICAL_INFLUENCE_SCHEMA,
                domain_validator=_make_tactical_validator(universe_set),
            )
        except Exception as exc:
            logger.warning("TacticalEvolutionAgent failed: %s", exc)
            return TacticalInfluence(rationale="evolution unavailable")

        inf_data = dict(payload.get("tactical_influence", {}) or {})
        urgency = str(payload.get("urgency", "normal"))
        if urgency not in ("normal", "elevated", "emergency"):
            urgency = "normal"

        haircut = float(inf_data.get("position_haircut", 1.0) or 1.0)
        haircut = max(0.4, min(1.0, haircut))
        # Treat anything >= 0.95 as "no haircut intended" — LLM rounding noise
        # should not install a persistent drag. Only install if clearly < 0.95.
        if haircut >= 0.95:
            haircut = 1.0
        expires = int(inf_data.get("expires_in_days", 1) or 1)
        expires = max(1, min(3, expires))

        avoid = _clean_symbol_list(inf_data.get("avoid_symbols"), universe_set)
        reduce_only = _clean_symbol_list(
            inf_data.get("reduce_only_symbols"), universe_set, exclude=set(avoid),
        )

        return TacticalInfluence(
            avoid_symbols=avoid,
            reduce_only_symbols=reduce_only,
            position_haircut=haircut,
            expires_in_days=expires,
            urgency=urgency,
            rationale=str(payload.get("rationale", "")),
        )


# ---------------------------------------------------------------------------
# STRATEGIC evolution (every N days)
# ---------------------------------------------------------------------------

_STRATEGIC_DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_failure_mode": {"type": "string"},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
        "regime_hypotheses": {"type": "array", "items": {"type": "string"}},
        "patch_goals": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["dominant_failure_mode", "patch_goals", "confidence"],
    "additionalProperties": False,
}

_STRATEGIC_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "patch": {"type": "object", "additionalProperties": True},
        "rationale": {"type": "string"},
        "expected_effects": {"type": "array", "items": {"type": "string"}},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        # Structured, machine-executable rollback contract. The evaluator
        # reads these fields directly — this closes the loop between the
        # LLM's stated hypothesis and the code that verifies it.
        "rollback_trigger": {
            "type": "object",
            "properties": {
                "eval_horizon_bars": {"type": "integer", "minimum": 1, "maximum": 30},
                "drawdown_pct": {"type": "number", "minimum": 0.005, "maximum": 0.15},
                "reference": {
                    "type": "string",
                    "enum": ["counterfactual_alpha", "pre_patch_nav"],
                },
            },
            "required": ["eval_horizon_bars", "drawdown_pct", "reference"],
            "additionalProperties": False,
        },
    },
    "required": ["patch", "rationale", "rollback_trigger"],
    "additionalProperties": False,
}

_DIAGNOSIS_BASE = """You are DarwinTrade's strategic diagnostician.
Given a strategic reflection that has flagged a non-trivial pattern, identify
the root cause and frame patch goals. Be concrete and evidence-based.
Return valid JSON only."""

_PATCH_AUTHOR_BASE = """You are DarwinTrade's strategic policy author.
Given a diagnosis, propose the MINIMAL config patch needed. Use only allowed
fields. Smaller is better.

You MUST include a machine-executable rollback_trigger with three fields:
  - eval_horizon_bars: integer in [1, 30]. How many bars the patch is held
    before it is either committed (kept) or reverted.
  - drawdown_pct: fraction in [0.005, 0.15]. Interpretation depends on
    reference (below).
  - reference: which signal the evaluator compares against drawdown_pct.
    * "counterfactual_alpha" (STRONGLY PREFERRED default): revert when
      cumulative alpha (portfolio return minus benchmark return) since
      the patch was applied drops more than drawdown_pct below zero.
      This is a causal rollback — it separates "patch was bad" from
      "market went down". Use this unless you have a specific reason
      not to.
    * "pre_patch_nav" (legacy, use ONLY when explicitly comparing to a
      fixed target NAV level): revert when absolute NAV drops more than
      drawdown_pct below the pre-patch anchor. In backtests dominated by
      a rising market this rarely fires even for harmful patches; in
      falling markets it fires too aggressively even for neutral patches.

The evaluator reads these fields directly. Free-text descriptions are
ignored — every rollback condition you state must be encoded in this object.

You also see the outcome of your own PAST patches in patch_track_record —
each entry has an explicit `status` ("pending" / "committed" / "reverted")
and a `nav_impact` fraction so you can learn whether your prior hypotheses
actually helped. If your recent patches have been reverted, propose a
different direction; blindly tightening again is not self-evolution.

Return valid JSON only."""


def _diagnosis_system() -> str:
    return _DIAGNOSIS_BASE + skill_section("strategic-diagnosis")


def _patch_system() -> str:
    return _PATCH_AUTHOR_BASE + skill_section("strategic-policy-author")


# Allowed strategic patch fields (synced with PipelineConfig)
# Paper: together with ALLOWED_OPTIMIZERS below, this is the allowed parameter
# set Omega of Eq. strategic-evolving — "nine allocator modes plus six
# bounded scalar fields" (Section "Strategic Evolution"; the six scalar
# fields are the six non-optimizer entries listed here). A patch is validated
# against Omega before install (Alg 2 "Validate(...) reject invalid").
ALLOWED_PATCH_FIELDS = {
    "max_gross_exposure",
    "max_single_position",
    "min_confidence_threshold",
    "preferred_optimizer",
    "position_size_multiplier",
    "drawdown_guard_threshold",
    "rebound_entry_threshold",
}

# Optimizers the allocator actually understands. Anything else corrupts
# strategic state without changing behavior, so we reject it. Synced with
# darwintrade.portfolio.allocator.SUPPORTED_OPTIMIZERS.
ALLOWED_OPTIMIZERS = {
    "",                          # empty == fall through to default
    # Synthetic (no price history needed)
    "conf_weighted",             # weight ∝ confidence × risk-haircut (default)
    "equal_weight",              # flat across eligible signals
    "confidence_concentrated",   # top-K by confidence, equal among them
    "risk_parity",               # weight ∝ 1 / (risk_score + ε), conf-gated
    "inverse_risk_weighted",     # weight ∝ confidence × (1 - risk_score)^2
    "kelly_capped",              # weight ∝ confidence^2 × (1 - risk_score)
    # Quant — require returns_history; auto-downgrade to conf_weighted otherwise
    "min_variance",              # w ∝ Σ⁻¹ 1, shrunk covariance
    "max_sharpe",                # tangency: w ∝ Σ⁻¹ μ, μ blends history + conf
    "hrp",                       # hierarchical risk parity (no matrix inversion)
}

# Strategic signal threshold below which we don't even attempt diagnosis
# Paper: the signal gate tau = 0.5 of Alg 2 (PHASE "Diagnose the multi-day
# pattern": "if z^S.signal < tau_gate return pi_j"; Table "Fixed
# hyperparameters": signal gate tau = 0.5). Below it the pattern is deemed
# healthy and no patch is authored.
STRATEGIC_SIGNAL_GATE = 0.5


def _validate_patch(patch: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for k, v in patch.items():
        if k not in ALLOWED_PATCH_FIELDS:
            continue
        if k == "min_confidence_threshold":
            # Cap at 0.6: above that the entire signal stream gets filtered
            # and the system locks into permanent no-trade.
            try:
                clean[k] = max(0.0, min(0.6, float(v)))
            except (TypeError, ValueError):
                pass
        elif k == "max_single_position":
            # Floor at 0.05 (5%): below that even single-share lots collapse
            # to zero on a typical $100k portfolio at $300+ stocks.
            try:
                clean[k] = max(0.05, min(1.0, float(v)))
            except (TypeError, ValueError):
                pass
        elif k == "max_gross_exposure":
            # Floor at 0.3 (30%): below that the system can't meaningfully
            # express conviction across a multi-name universe.
            try:
                clean[k] = max(0.3, min(1.0, float(v)))
            except (TypeError, ValueError):
                pass
        elif k in ("drawdown_guard_threshold", "rebound_entry_threshold"):
            try:
                clean[k] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                pass
        elif k == "position_size_multiplier":
            # Floor at 0.5: below that compounds with per-name caps and
            # gross caps to effectively zero out positions.
            try:
                clean[k] = max(0.5, min(3.0, float(v)))
            except (TypeError, ValueError):
                pass
        elif k == "preferred_optimizer":
            opt = str(v).strip().lower()
            if opt in ALLOWED_OPTIMIZERS:
                clean[k] = opt
            else:
                logger.info("rejecting unknown preferred_optimizer=%r", v)
    # cap at 3 fields total
    if len(clean) > 3:
        clean = dict(list(clean.items())[:3])
    return clean


class StrategicEvolutionAgent:
    """diagnose → policy_author. Returns StrategicPatch or None.

    Paper: f_str-policy of Alg 2 (PHASE "Author a minimal patch with a rollback
    trigger"). Runs only after the signal gate (tau=0.5) passes, then proposes
    the patch tuple (Delta-pi, H, epsilon, m) over Omega with its
    RollbackTrigger."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def maybe_evolve(
        self,
        *,
        review_date: str,
        reflection: StrategicReflection,
        capsule_summary: list[dict[str, Any]],
        active_strategy_state: dict[str, Any],
        objective_metrics: dict[str, Any] | None = None,
        patch_track_record: list[dict[str, Any]] | None = None,
    ) -> StrategicPatch | None:
        # gate by reflection signal strength
        # Paper: Alg 2 signal gate — "if z^S.signal < tau_gate return pi_j"
        # (tau_gate = STRATEGIC_SIGNAL_GATE = 0.5). A healthy multi-day pattern
        # produces no patch, so the long-lived policy is left unchanged.
        if reflection.dominant_pattern == "none":
            return None
        if reflection.strategic_signal_strength < STRATEGIC_SIGNAL_GATE:
            return None

        # diagnose
        try:
            diagnosis = self.llm.invoke_json(
                _diagnosis_system(),
                json.dumps({
                    "strategic_reflection": reflection.to_dict(),
                    "capsule_summary": capsule_summary[:20],
                    "active_strategy_state": active_strategy_state,
                    "objective_metrics": objective_metrics or {},
                    "patch_track_record": patch_track_record or [],
                }, ensure_ascii=True),
                caller_tag="strategic_diagnosis",
                schema_name="StrategicDiagnosis",
                schema=_STRATEGIC_DIAGNOSIS_SCHEMA,
            )
        except Exception as exc:
            logger.warning("StrategicEvolutionAgent diagnosis failed: %s", exc)
            return None

        diag_confidence = float(diagnosis.get("confidence", 0.0) or 0.0)
        if diag_confidence < 0.5:
            logger.info("Strategic diagnosis confidence %.2f below 0.5, skipping patch", diag_confidence)
            return None

        # policy author
        try:
            patch_payload = self.llm.invoke_json(
                _patch_system(),
                json.dumps({
                    "strategic_diagnosis": diagnosis,
                    "active_strategy_state": active_strategy_state,
                    "allowed_patch_fields": sorted(ALLOWED_PATCH_FIELDS),
                    "allowed_preferred_optimizer_values": sorted(ALLOWED_OPTIMIZERS),
                }, ensure_ascii=True),
                caller_tag="strategic_policy_author",
                schema_name="StrategicPatch",
                schema=_STRATEGIC_PATCH_SCHEMA,
                domain_validator=_validate_strategic_patch_payload,
            )
        except Exception as exc:
            logger.warning("StrategicEvolutionAgent policy author failed: %s", exc)
            return None

        clean_patch = _validate_patch(dict(patch_payload.get("patch", {}) or {}))
        if not clean_patch:
            logger.info("Strategic patch was empty after validation, skipping")
            return None

        return StrategicPatch(
            patch=clean_patch,
            rationale=str(patch_payload.get("rationale", "")),
            expected_effects=list(patch_payload.get("expected_effects", [])),
            risk_notes=list(patch_payload.get("risk_notes", [])),
            rollback_trigger=RollbackTrigger.from_any(patch_payload.get("rollback_trigger")),
            issued_on=review_date,
            confidence=diag_confidence,
        )


__all__ = [
    "TacticalEvolutionAgent",
    "StrategicEvolutionAgent",
    "ALLOWED_PATCH_FIELDS",
    "STRATEGIC_SIGNAL_GATE",
]
