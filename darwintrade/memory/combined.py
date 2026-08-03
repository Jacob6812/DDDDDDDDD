"""
Combine strategic baseline + tactical influence into a single MemoryInfluence
that the allocator and market agent consume.

Paper correspondence (darwintrade_aaai2027.tex):
  Merges the two disjoint policy states consumed on day t — the long-lived
  strategic policy pi_t and the short-lived tactical guard state U^T_t (Method,
  overview of the three state-disjoint loops). The resulting MemoryInfluence is
  what turns the raw signals S_t into the guarded signals S~_t = ApplyGuards(
  S_t, U^T_t) that the deterministic allocator consumes (Alg 1 PHASE "Apply
  tactical guards, then allocate"; Eq. strategic-allocation). Strategic fields
  set allocator direction (mode m and bounded scalars in pi_t); tactical fields
  (avoid / reduce-only / haircut) are the guards, and an emergency guard may
  additionally tighten the gross budget G_t.

Priority:
- Strategic = direction (preferred_optimizer, baseline constraint_overrides)
- Tactical  = immediate guardrails (avoid_symbols, reduce_only, position_haircut)
            ; tactical urgency=emergency CAN tighten strategic baseline further
"""
from __future__ import annotations

from typing import Any

from ..core.contracts import MemoryInfluence, TacticalInfluence


def combine_influence(
    *,
    strategic: MemoryInfluence,
    tactical: TacticalInfluence,
) -> MemoryInfluence:
    """Merge strategic and tactical guidance with strategic taking precedence
    on optimizer choice and tactical taking precedence on near-term constraints.
    """
    overrides = dict(strategic.constraint_overrides)

    # Tactical haircut never overrides strategic exposure cap; instead applied
    # downstream by the allocator. We pass it through on the combined object.
    haircut = float(tactical.position_haircut or 1.0)

    # Emergency tactical can shrink max_gross further than strategic baseline
    if tactical.urgency == "emergency":
        cur_gross = float(overrides.get("max_gross_exposure", 0.95) or 0.95)
        overrides["max_gross_exposure"] = min(cur_gross, 0.5)

    rationale_parts = []
    if strategic.rationale:
        rationale_parts.append(f"strategic: {strategic.rationale}")
    if tactical.rationale:
        rationale_parts.append(f"tactical: {tactical.rationale}")

    return MemoryInfluence(
        preferred_optimizer=strategic.preferred_optimizer,
        no_trade_symbols=list(tactical.avoid_symbols),
        reduce_only_symbols=list(tactical.reduce_only_symbols),
        constraint_overrides=overrides,
        position_haircut=haircut,
        rationale=" | ".join(rationale_parts),
        confidence=strategic.confidence,
        sample_count=strategic.sample_count,
        source_layer="combined",
    )


__all__ = ["combine_influence"]
