"""
IC-shrinkage analyst aggregator: combines heterogeneous analyst predictions
into a Normal posterior over the next-bar log return, using each (regime,
role) capsule's information coefficient as the skill measure.

Paper correspondence (darwintrade_aaai2027.tex):
  Implements the regime-conditioned role fusion of Eq. analyst-fusion in
  Section "Analyst Evolution", Forward Process, and Alg 1 PHASE
  "Regime-conditioned fusion":
      alpha_{t,i,k} = u_{rho_t,k,t} * c_{t,i,k}
      s_{t,i}       = sum_k alpha_{t,i,k} x_{t,i,k} / sum_k alpha_{t,i,k}
  Here x_{t,i,k} is the role's directional score in [-1,1] ("score"),
  c_{t,i,k} its self-reported confidence in [0,1] ("confidence"), and the
  regime-conditioned reliability u_{rho_t,k,t} is the capsule skill weight
  max(0, IC) * n/(n+20) (Eq. analyst-evolving; see analyst_capsules.py). The
  weighted mean `mu_score` below is exactly s_{t,i} in score space.

  The c^2 conviction weighting (confidence entering sizing quadratically) is
  the canonical default described in Appendix "Implementation and
  Hyperparameters"; the linear-c variant is the linear-confidence ablation
  (Table "Sensitivity to Design Alternatives"). Non-positive-IC capsules are
  dropped rather than sign-flipped, matching the Method statement "a warm role
  with negative IC is removed instead of reversed" and the signed-IC row of the
  design-choice table.

This replaces the earlier `tau = beta**2 / sigma**2` formulation, which
conflated score-vs-return unit conversion (beta) with skill. That conflation
made any analyst with high score variance dominate the posterior regardless
of true predictive power, and a sign-flipped beta_hat from a small noisy
regression would invert the analyst's vote — anti-information.

Aggregation
-----------
For each contributing role i:
  - skill weight w_i = max(0, IC_i) * shrink(n_i),  shrink = n / (n + 20)
    Negative-IC analysts contribute zero (down-weight, never invert).
  - implied return r_hat_i = IC_i * (sd_realized_i / sd_score_i) * x_i
    Unit-stable mapping; bounded by clipped raw score.
  - effective weight w_eff_i = w_i * conf_i
    Self-reported confidence multiplicatively gates the contribution.

Posterior:
  mu_post = sum(w_eff_i * r_hat_i) / sum(w_eff_i)        (precision-weighted mean)
  sigma_post is derived from a heteroskedastic Normal model:
    each contribution carries irreducible per-bar noise sigma_role = realized_sd_i,
    and the posterior precision is sum(w_eff_i / sigma_role_i**2).
  Information ratio IR = mu_post / sigma_post.

When no role contributes (all negative-IC or zero confidence), the
aggregator reports HOLD with zero confidence — a clean cold-start
signal for downstream consumers.

Direction discretization
------------------------
  IR >= IR_BUY_Z  → BUY
  IR <= -IR_BUY_Z → SELL
  else            → HOLD

A FAIL verification raises the bar to IR_HARD_Z; PASS uses IR_BUY_Z.
"""
from __future__ import annotations

import math
import os
from typing import Any

from ..memory.analyst_capsules import AnalystCapsule, PRIOR_REALIZED_SD


# Posterior information-ratio thresholds used to discretize direction.
# IR=1 means the posterior puts ~84% probability the true return has the
# same sign as mu_post — a textbook 1-sigma trade-the-signal threshold.
IR_BUY_Z = 1.0
IR_HARD_Z = 1.5    # high-conviction band when verification fails

# Cap |posterior mean| before reporting expected_return_view. Real next-bar
# log-returns rarely exceed 12% even on event-driven names; clamping
# prevents an outlier-driven posterior from over-stating expected return.
EXPECTED_RETURN_CLIP = 0.12

# Confidence floor — roles below this are abstained from aggregation.
ABSTAIN_CONFIDENCE = 0.05

# Cold-start weight used when a capsule has not yet warmed up (n < WARMUP_N).
# Deliberately small so that a single warm role can quickly dominate once
# enough data accumulates. Matches the prior precision order of magnitude
# from the original beta=1, sigma2=0.0025 aggregator: tau_prior ≈ 400,
# vs a warm skilled role's tau ≈ SCORE_RESCALE^2 * IC^2 / realized_sd^2
# ≈ 2500*0.09/0.0004 ≈ 56k. At 0.01 the cold role stays ~1/5600 relative
# to a fully-warm skilled peer, so warm capsules dominate as intended.
COLD_START_WEIGHT = 0.01

# Whether self-reported confidence enters sizing conviction twice.
# Default (True) reproduces the shipped design: effective weight w_eff = skill*conf,
# and the conviction step re-weights each role's confidence by w_eff, so confidence
# enters as c^2 (∝ Σ skill·c²). Set DARWIN_CONFIDENCE_SQUARED=0 to weight the
# confidence average by skill only (linear c), which is the calibration-ablation
# baseline. Read at call time so tests/backtests can toggle
# it per run without reimporting the module.
def _confidence_squared() -> bool:
    return os.getenv("DARWIN_CONFIDENCE_SQUARED", "1").strip().lower() not in (
        "0", "false", "no",
    )


# Score-scale rescaling factor for the [-1,1] aggregate_score field that
# downstream consumers expect. Posterior means live on the return scale
# (typically <= 0.05); this maps them back to the score band.
SCORE_RESCALE = 50.0

# Max fractional uplift to sizing conviction when the contributing capsules are
# fully proven (mean skill_weight → 1). At 0.5 a book of proven-IC analysts is
# sized up to 1.5× a zero-skill consensus with the same raw scores/confidence,
# so learned skill amplifies alpha in position size. Cold-start capsules have
# skill_weight≈0, so this is a no-op during warmup.
SKILL_CONVICTION_GAIN = 0.5


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _phi(x: float) -> float:
    """Standard normal CDF — used for confidence calibration."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def aggregate(
    *,
    role_payloads: dict[str, dict[str, Any]],
    regime: str,
    capsule_lookup,                 # callable(regime, role) -> AnalystCapsule
    verification_verdict: str = "PARTIAL",
    risk_uncertainty: float = 0.0,
) -> dict[str, Any]:
    """
    IC-shrinkage posterior over the next-bar log return.

    Each role payload must contain at minimum:
      - "score":      float in [-1, 1]   (analyst's directional view)
      - "confidence": float in [0, 1]    (analyst's self-reported certainty)

    Roles with confidence below ABSTAIN_CONFIDENCE are dropped. Roles whose
    capsule has non-positive IC contribute zero weight (down-weighted, not
    inverted).
    """
    # tuple: (role, implied_return, weight_eff, raw_score, raw_conf,
    #         is_warm, ic, score_sd, realized_sd, skill_weight)
    contributions: list[tuple[str, float, float, float, float, bool, float, float, float, float]] = []
    # Warm-but-unskilled roles (IC <= 0). We never invert them, but if NO role
    # has positive skill we fall back to their raw directional view rather than
    # going permanently silent (see fallback block below).
    unskilled_warm: list[tuple[str, float, float, bool, float, float, float, float]] = []

    for role, payload in role_payloads.items():
        if not isinstance(payload, dict):
            continue
        x_raw = _clip(float(payload.get("score", 0.0) or 0.0), -1.0, 1.0)
        conf = _clip(float(payload.get("confidence", 0.0) or 0.0), 0.0, 1.0)
        if conf < ABSTAIN_CONFIDENCE:
            continue
        capsule: AnalystCapsule = capsule_lookup(regime, role)
        ic = capsule.ic
        skill_w = capsule.skill_weight  # max(0, ic) * shrink(n)

        if skill_w > 0.0:
            # Paper: warm capsule with positive IC. skill_w is the reliability
            # u_{rho,k} of Eq. analyst-evolving, and weight_eff = skill_w * conf
            # is the fusion weight alpha_{t,i,k} = u_{rho_t,k,t} * c_{t,i,k} of
            # Eq. analyst-fusion. `implied_r` is the unit-stable return mapping
            # of the raw score x_{t,i,k}.
            implied_r = capsule.implied_return(x_raw)
            weight_eff = skill_w * conf
        elif not capsule.is_warm:
            # Cold-start fallback: treat raw score as implied return directly
            # (equivalent to beta=1, scale-agnostic prior from the original
            # aggregator). Weight is confidence × a prior precision that
            # matches the original tau = beta²/sigma² = 1/0.0025 = 400
            # modulated by conf. This restores pre-warmup BUY/SELL behaviour
            # so the portfolio trades during the learning period.
            implied_r = float(x_raw)
            weight_eff = COLD_START_WEIGHT * conf
        else:
            # Warm capsule but IC <= 0: never invert. Stash as a fallback
            # candidate instead of dropping outright — a book where every role
            # is warm+unskilled must still express the analysts' raw view, not
            # freeze into permanent HOLD.
            unskilled_warm.append((
                role, float(x_raw), conf, capsule.is_warm, ic,
                capsule.score_sd, capsule.realized_sd, skill_w,
            ))
            continue

        if weight_eff <= 0.0:
            continue
        contributions.append((
            role,
            implied_r,
            weight_eff,
            x_raw,
            conf,
            capsule.is_warm,
            ic,
            capsule.score_sd,
            capsule.realized_sd,
            skill_w,
        ))

    # No skilled or cold-start role contributed, but warm-unskilled roles have a
    # directional view. Fall back to a cold-start-style pass-through on them so
    # the aggregator keeps trading. Weight is the same small COLD_START_WEIGHT
    # so a single skilled role would still dominate once one warms up.
    if not contributions and unskilled_warm:
        for role, x_raw, conf, is_warm, ic, ssd, rsd, skill_w in unskilled_warm:
            weight_eff = COLD_START_WEIGHT * conf
            if weight_eff <= 0.0:
                continue
            contributions.append((
                role, float(x_raw), weight_eff, x_raw, conf,
                is_warm, ic, ssd, rsd, skill_w,
            ))

    if not contributions:
        return _empty_summary(regime, verification_verdict, risk_uncertainty)

    weight_sum = sum(c[2] for c in contributions)
    if weight_sum <= 0:
        return _empty_summary(regime, verification_verdict, risk_uncertainty)

    # Posterior in score-space: IC-skill-weighted mean of raw scores.
    # Paper: this is s_{t,i} of Eq. analyst-fusion — the alpha-weighted mean of
    # the role scores x_{t,i,k}, with alpha_{t,i,k} = weight c[2] = skill*conf.
    # Downstream the +/-0.15 buy/sell threshold on s_{t,i} (Method, Forward
    # Process; Alg 1 line "buy if > 0.15, sell if < -0.15") is applied by the
    # allocator/market agent, not here.
    mu_score = sum(c[3] * c[2] for c in contributions) / weight_sum

    # Disagreement: weighted std of raw scores, in [0,1].
    weighted_var_score = sum(c[2] * (c[3] - mu_score) ** 2 for c in contributions) / weight_sum
    disagreement = _clip(math.sqrt(max(weighted_var_score, 0.0)), 0.0, 1.0)

    # decision_confidence: skill-weighted mean of self-reported confidence
    # shrunk by disagreement. By default the weight c[2] already contains one
    # confidence factor (w_eff = skill*conf), so multiplying by c[4]=conf makes
    # confidence enter quadratically (Σ skill·c²). The linear-conf ablation weights by
    # skill only (c[2]/conf), giving a linear-confidence average.
    if _confidence_squared():
        raw_conf_avg = sum(c[4] * c[2] for c in contributions) / weight_sum
    else:
        skill_only = [(c[2] / c[4] if c[4] > 1e-9 else 0.0) for c in contributions]
        skill_sum = sum(skill_only)
        raw_conf_avg = (
            sum(c[4] * sw for c, sw in zip(contributions, skill_only)) / skill_sum
            if skill_sum > 1e-12 else 0.0
        )

    agreement_factor = _clip(1.0 - disagreement, 0.0, 1.0)
    decision_confidence = _clip(raw_conf_avg * agreement_factor, 0.0, 1.0)

    # Skill-conviction amplification. The step above only uses skill in the
    # RELATIVE weighting between roles, so a consensus of proven-IC analysts and
    # a consensus of zero-IC analysts with identical raw scores/confidence
    # produced identical decision_confidence — i.e. learned skill never reached
    # position size, and memory did not amplify alpha in sizing (verified).
    # Here we scale sizing conviction UP with the aggregate proven skill of the
    # contributing capsules. Cold-start / zero-skill (skill_weight≈0) gives
    # conviction≈0 → factor 1.0 → NO change (no regression to warmup trading);
    # a book of proven winners is sized up to (1 + SKILL_CONVICTION_GAIN)×,
    # bounded to [0,1]. Direction logic (mu_score) is untouched, so this does
    # not reintroduce the scale/skill conflation the IC-shrinkage rewrite removed.
    skill_conviction = _clip(sum(c[9] for c in contributions) / len(contributions), 0.0, 1.0)
    decision_confidence = _clip(
        decision_confidence * (1.0 + SKILL_CONVICTION_GAIN * skill_conviction),
        0.0, 1.0,
    )

    # IR: reported for diagnostics. mu_score normalized by sigma = max of
    # disagreement and a floor based on aggregate weight.
    sigma_score = max(disagreement, 1.0 / math.sqrt(max(weight_sum, 1e-6)))
    information_ratio = mu_score / sigma_score if sigma_score > 1e-12 else 0.0

    # Direction: driven directly by mu_score magnitude, not IR. This decouples
    # direction from the sigma calculation and gives consistent BUY/SELL/HOLD
    # across cold-start and warm capsule regimes.
    # Threshold: PARTIAL verdict requires |mu| >= 0.15; FAIL raises it to 0.25.
    mu_threshold = 0.15 if str(verification_verdict).upper() != "FAIL" else 0.25
    if mu_score >= mu_threshold:
        direction = "BUY"
    elif mu_score <= -mu_threshold:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Map score-space mean to an expected return estimate for diagnostics.
    mean_rsd = sum(c[8] for c in contributions) / len(contributions)
    mean_ssd = sum(c[7] for c in contributions) / len(contributions)
    verification_shrink = {"PASS": 1.0, "PARTIAL": 0.85, "FAIL": 0.65}.get(
        str(verification_verdict).upper(), 0.85,
    )
    expected_return_view = round(
        _clip(mu_score * (mean_rsd / max(mean_ssd, 1e-6)) * verification_shrink,
              -EXPECTED_RETURN_CLIP, EXPECTED_RETURN_CLIP),
        6,
    )

    # aggregate_score stays in [-1,1] score-space (no *50 rescale needed).
    aggregate_score = _clip(mu_score, -1.0, 1.0)
    # tau_total and sigma_post reported in score-space for diagnostics.
    tau_total = 1.0 / max(sigma_score ** 2, 1e-12)
    sigma_post = sigma_score

    # Analyst-derived risk. The research roles do not emit a dedicated risk
    # field (their schema is score/confidence/summary/regime), and the
    # research-node writes risk_uncertainty into shared_state only AFTER this
    # aggregate runs — so the externally-supplied `risk_uncertainty` arg is
    # effectively always 0. That silently zeroed AssetSignal.risk_score
    # downstream, degenerating risk_parity / inverse_risk_weighted /
    # kelly_capped into pure-confidence optimizers and disabling the allocator
    # risk-haircut entirely. When no external risk is supplied, derive it from
    # the aggregate's own uncertainty signals: cross-analyst disagreement, the
    # downside skew, and the confidence deficit. Bounded to [0,1].
    external_risk = _clip(float(risk_uncertainty or 0.0), 0.0, 1.0)
    downside_risk_val = _clip(max(0.0, -mu_score) + 0.5 * disagreement, 0.0, 1.0)
    derived_risk = _clip(
        0.5 * disagreement
        + 0.3 * downside_risk_val
        + 0.2 * (1.0 - decision_confidence),
        0.0, 1.0,
    )
    effective_risk = external_risk if external_risk > 0.0 else derived_risk

    role_score_map = {role: round(raw, 6) for role, _ir, _w, raw, _c, _wm, _ic, _ssd, _rsd, _sk in contributions}
    role_confidence_map = {role: round(conf, 6) for role, _ir, _w, _raw, conf, _wm, _ic, _ssd, _rsd, _sk in contributions}
    role_skill_weight_map = {role: round(sk, 6) for role, _ir, _w, _raw, _c, _wm, _ic, _ssd, _rsd, sk in contributions}
    role_ic_map = {role: round(ic, 4) for role, _ir, _w, _raw, _c, _wm, ic, _ssd, _rsd, _sk in contributions}
    role_warm_map = {role: bool(warm) for role, _ir, _w, _raw, _c, warm, _ic, _ssd, _rsd, _sk in contributions}
    role_implied_return_map = {role: round(ir, 6) for role, ir, _w, _raw, _c, _wm, _ic, _ssd, _rsd, _sk in contributions}
    # role_precision_map is read by downstream consumers (e.g. allocator audit
    # trail). It reports the IC-based effective weight.
    role_precision_map = role_skill_weight_map

    return {
        "direction": direction,
        "aggregate_score": round(aggregate_score, 6),
        "aggregate_confidence": round(decision_confidence, 6),
        "decision_confidence": round(decision_confidence, 6),
        "information_ratio": round(information_ratio, 6),
        "posterior_mean": round(mu_score, 6),
        "posterior_sigma": round(sigma_post, 6),
        "tau_total": round(tau_total, 6),
        "disagreement": round(disagreement, 6),
        "downside_risk": round(downside_risk_val, 6),
        "expected_return_view": expected_return_view,
        "verification_verdict": str(verification_verdict).upper(),
        "risk_uncertainty": round(effective_risk, 6),
        "score_dispersion": round(disagreement, 6),
        "size_multiplier": round(decision_confidence, 6),
        "role_score_map": role_score_map,
        "role_confidence_map": role_confidence_map,
        "role_precision_map": role_precision_map,
        "role_skill_weight_map": role_skill_weight_map,
        "role_ic_map": role_ic_map,
        "role_warm_map": role_warm_map,
        "role_implied_return_map": role_implied_return_map,
        "n_contributing_roles": len(contributions),
        "aggregator": "ic_shrinkage_capsule",
    }


def _empty_summary(regime: str, verification_verdict: str, risk_uncertainty: float) -> dict[str, Any]:
    return {
        "direction": "HOLD",
        "aggregate_score": 0.0,
        "aggregate_confidence": 0.0,
        "decision_confidence": 0.0,
        "information_ratio": 0.0,
        "posterior_mean": 0.0,
        "posterior_sigma": 0.0,
        "tau_total": 0.0,
        "disagreement": 0.0,
        "downside_risk": 0.0,
        "expected_return_view": 0.0,
        "verification_verdict": str(verification_verdict).upper(),
        "risk_uncertainty": round(_clip(float(risk_uncertainty or 0.0), 0.0, 1.0), 6),
        "score_dispersion": 0.0,
        "size_multiplier": 0.0,
        "role_score_map": {},
        "role_confidence_map": {},
        "role_precision_map": {},
        "role_skill_weight_map": {},
        "role_ic_map": {},
        "role_warm_map": {},
        "role_implied_return_map": {},
        "n_contributing_roles": 0,
        "aggregator": "ic_shrinkage_capsule",
    }


__all__ = ["aggregate", "IR_BUY_Z", "IR_HARD_Z", "EXPECTED_RETURN_CLIP"]
