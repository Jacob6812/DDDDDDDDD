"""
AnalystCapsule: per-(regime, role) online estimator of an analyst's
information coefficient (IC) and supporting scale statistics, used by the
Bayesian aggregator under an IC-shrinkage formulation.

Paper correspondence (darwintrade_aaai2027.tex):
  This is the evaluation capsule C_{rho,k,t} of Section "Analyst Evolution",
  Self-Evolution Loop, and the concrete implementation of the role-skill update
  f_analyst-evolving in Eq. analyst-evolving:
      u_{rho,k,t+1} = u_0                       if n < 20   (cold start)
                    = max(0, IC) * n/(n+20)      if n >= 20
  Mapping to this file:
    - one capsule per (regime, role) pair (rho, k), storing up to the latest
      200 tuples (x_{t,i,k}, c_{t,i,k}, y_{t+1,i}, t)  → CAPSULE_HISTORY_CAP=200
    - `ic` computes IC_{rho,k,t} as the demeaned Pearson correlation between
      predictions and realized returns (Method: "after removing the
      cross-sectional mean of each trade date from both scores and returns")
    - `skill_weight` returns u_{rho,k,t+1} = max(0, IC) * shrink(n) with
      shrink(n) = n/(n+20) (the sample-size shrinkage) and the warmup at n<20
      handled by the cold-start weight in the aggregator (COLD_START_WEIGHT=u_0)
  Constants: CAPSULE_WARMUP_N=20 is the warmup N and shrink denominator add of
  Table "Fixed hyperparameters" (analyst-evolution block). Because IC is
  truncated at zero (default "zero_floor" mode), a warm role with negative IC
  is dropped rather than sign-flipped (Method: "removed instead of reversed").
  The signed-IC variant is a design-choice ablation.

Design (replacing the prior beta**2 / sigma**2 formulation)
-----------------------------------------------------------
The earlier capsule fit `x_score = beta * r + bias + eps` and exposed the
implied-return precision `tau = beta**2 / sigma**2`. That formulation
conflated two unrelated quantities:

  1. SCALE: beta is sd(score)/sd(return) * IC, dominated by the unit
     conversion between [-1,1] scores and ~1% daily returns.
  2. SKILL: only IC (correlation between prediction and realized return)
     is true skill; sign and magnitude both matter.

Because tau scales with beta**2, an analyst's vote magnitude exploded with
score variance even when IC was zero, and a sign-flipped beta_hat from a
small noisy regression literally inverted the analyst's direction. Both
behaviors are anti-information.

The new capsule estimates only:

  - IC (Pearson correlation of prediction vs realized return), on
    daily-cross-sectionally demeaned series so cross-symbol level offsets
    don't leak into the OLS estimate.
  - sd(score) and sd(realized) over the demeaned series, used purely as
    unit-conversion factors when mapping a fresh score to an implied return.
  - sample size n and a warm flag.

The aggregator combines analysts with weights ~ max(0, IC) * shrink(n),
where shrink(n) = n / (n + WARMUP_N). A negative-IC analyst is dropped
(weight 0) rather than inverted, eliminating the small-sample sign-flip
hazard. The implied return reported by an analyst with score x in regime
g is `(sd_realized / sd_score) * IC * x_demeaned`, which is unit-correct
and bounded.

Daily cross-sectional demean
----------------------------
At each trade_date, score_i and realized_i are demeaned by that day's
cross-sectional mean over all symbols. This removes "everyone said
positive on a positive day" effects from the IC estimate so the metric
captures true cross-sectional skill, not market-wide drift.
"""
from __future__ import annotations

import math
import os
import statistics
from collections import defaultdict, deque
from typing import Any


# ---------------------------------------------------------------------------
# Hyper-priors
# ---------------------------------------------------------------------------

# Below 20 samples the IC estimate is dominated by sample noise; the
# shrinkage factor n/(n + WARMUP_N) blends the data-driven IC toward zero.
CAPSULE_WARMUP_N = 20

# Sample window cap. Older history is dropped so capsule statistics adapt
# to regime drift rather than averaging across structurally different
# market periods.
CAPSULE_HISTORY_CAP = 200

# Clip realized returns at this magnitude (20%) to prevent single limit-up
# / limit-down days from dominating the variance estimate.
REALIZED_RETURN_CLIP = 0.20

# Floor for the score standard deviation used during unit-conversion;
# guards against degenerate cases where every observed prediction is
# identical (avoid divide-by-zero in the implied-return mapping).
SCORE_SD_FLOOR = 0.05

# Default scale factor used pre-warmup when sd(realized) cannot yet be
# estimated from data: ~2% per bar, matching diversified-equity daily vol.
PRIOR_REALIZED_SD = 0.02

# Exported for import paths in other modules; not used by the capsule itself.
PRIOR_ALPHA = 2.0
PRIOR_BETA_VAR = 0.0025
PRIOR_BETA_SLOPE = 1.0


# ---------------------------------------------------------------------------
# Sign-handling ablation, read at call time so a backtest can toggle it per run.
#
#   DARWIN_IC_SIGN_MODE = "zero_floor" (default) | "signed"
#       "zero_floor": weight = max(0, IC) * shrink; negative IC dropped.
#       "signed"    : weight = |IC| * shrink and the implied return uses IC with
#                     its sign, so a stable negative-IC role is used as a
#                     contrarian signal instead of discarded.
# ---------------------------------------------------------------------------


def _ic_sign_mode() -> str:
    return os.getenv("DARWIN_IC_SIGN_MODE", "zero_floor").strip().lower()



def _safe_finite(v: Any, default: float = 0.0) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except (TypeError, ValueError):
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class AnalystCapsule:
    """
    Online estimator for a single (regime, role) analyst.

    Tracks (predicted_score, predicted_confidence, realized_return,
    trade_date) tuples and exposes:

      - n               : number of observations (warm when >= WARMUP_N)
      - ic              : Pearson correlation on daily-demeaned series
      - shrink          : n / (n + WARMUP_N)            in [0, 1]
      - skill_weight    : max(0, ic) * shrink           in [0, 1]
      - score_sd        : sd of demeaned predictions    >= SCORE_SD_FLOOR
      - realized_sd     : sd of demeaned realizeds      pre-warmup falls back
                          to PRIOR_REALIZED_SD
      - implied_return(x_score)  : unit-correct mapping of a new score
                                   to an estimated next-bar return

    The aggregator uses (skill_weight, implied_return) only. Negative-IC
    analysts are not inverted; they are simply down-weighted to zero.
    """

    def __init__(self, regime: str, role: str) -> None:
        self.regime = regime
        self.role = role
        # (pred_score, pred_conf, realized_return, trade_date)
        self._samples: deque[tuple[float, float, float, str]] = deque(
            maxlen=CAPSULE_HISTORY_CAP
        )

    # ---- raw counts --------------------------------------------------

    @property
    def n(self) -> int:
        return len(self._samples)

    @property
    def is_warm(self) -> bool:
        return self.n >= CAPSULE_WARMUP_N

    @property
    def n_days(self) -> int:
        """Distinct trade dates observed. Reported alongside `n` because
        same-day cross-sectional observations are strongly correlated, so the
        tuple count overstates how much independent evidence backs the IC."""
        return len({d for _p, _c, _r, d in self._samples if d})

    @property
    def shrink(self) -> float:
        return self.n / (self.n + CAPSULE_WARMUP_N) if self.n > 0 else 0.0


    # ---- demeaned series --------------------------------------------

    def _demeaned_series(self) -> tuple[list[float], list[float]]:
        """Return demeaned (prediction, realized) series for the IC estimate.

        Preferred mode is cross-sectional: subtract each trade_date's mean over
        all symbols so market-wide drift does not leak into the correlation.
        This requires >= 2 symbols sharing a trade_date.

        Fallback mode is time-series: on a single-symbol universe every date
        has exactly one observation, so cross-sectional demean collapses to an
        empty series and IC would be a constant 0 no matter how skilled the
        analyst is (verified: a perfect-IC analyst reported ic=0 over 60 bars).
        When cross-sectional demean yields too few points, subtract the global
        mean across all samples instead. Time-series correlation of prediction
        vs realized is a valid single-name skill metric — it just measures
        temporal rather than cross-sectional skill.

        Multi-symbol runs are unaffected: they always clear MIN_XS_POINTS and
        keep the cross-sectional path.
        """
        if not self._samples:
            return [], []
        by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for pred, _conf, real, date in self._samples:
            by_date[date].append((pred, real))
        preds: list[float] = []
        reals: list[float] = []
        for date, entries in by_date.items():
            if len(entries) < 2:
                # single-sample days carry no cross-sectional information
                continue
            mean_p = sum(p for p, _ in entries) / len(entries)
            mean_r = sum(r for _, r in entries) / len(entries)
            for p, r in entries:
                preds.append(p - mean_p)
                reals.append(r - mean_r)

        # Enough cross-sectional structure — use it.
        MIN_XS_POINTS = 5
        if len(preds) >= MIN_XS_POINTS:
            return preds, reals

        # Fallback: time-series demean over the full sample set (global mean).
        all_p = [p for p, _c, _r, _d in self._samples]
        all_r = [r for _p, _c, r, _d in self._samples]
        if len(all_p) < 2:
            return [], []
        mean_p = sum(all_p) / len(all_p)
        mean_r = sum(all_r) / len(all_r)
        return [p - mean_p for p in all_p], [r - mean_r for r in all_r]

    # ---- statistics --------------------------------------------------

    @property
    def ic(self) -> float:
        """Pearson correlation between demeaned predictions and demeaned
        realized returns. Pure cross-sectional skill metric in [-1, 1].
        Returns 0 when n is too small to estimate.

        Paper: IC_{rho,k,t} in Eq. analyst-evolving — the demeaned Pearson
        correlation between {x_{t,i,k}} and {y_{t+1,i}}. The len<5 guard is the
        "min cross-section pts = 5" hyperparameter (Table "Fixed
        hyperparameters"): below it IC is skipped (returns 0)."""
        preds, reals = self._demeaned_series()
        if len(preds) < 5:
            return 0.0
        try:
            sd_p = statistics.pstdev(preds)
            sd_r = statistics.pstdev(reals)
            if sd_p <= 1e-12 or sd_r <= 1e-12:
                return 0.0
            mean_p = sum(preds) / len(preds)
            mean_r = sum(reals) / len(reals)
            cov = sum(
                (p - mean_p) * (r - mean_r) for p, r in zip(preds, reals)
            ) / len(preds)
            return _clip(cov / (sd_p * sd_r), -1.0, 1.0)
        except statistics.StatisticsError:
            return 0.0

    @property
    def information_coefficient(self) -> float:
        """Alias for ic, retained for serialization compatibility."""
        return self.ic

    @property
    def skill_weight(self) -> float:
        """Aggregator weight.

        Paper: this is the evolved role reliability u_{rho,k,t+1} of Eq.
        analyst-evolving, consumed as the fusion reliability u_{rho_t,k,t} in
        Eq. analyst-fusion (see bayesian_aggregator.py).

        Default ("zero_floor"): max(0, IC) * shrink(n) — matches the n>=20 case
        of Eq. analyst-evolving; negative-IC analysts contribute 0 weight
        (down-weight, never invert), i.e. "removed instead of reversed".

        Signed mode: |IC| * shrink(n) — the design-choice ablation where a
        stable negative-IC role still earns weight and is used contrarian via
        implied_return's sign.

        The shrink factor n/(n+20) is the sample-size shrinkage of Eq.
        analyst-evolving.
        """
        if _ic_sign_mode() == "signed":
            return abs(self.ic) * self.shrink
        return max(0.0, self.ic) * self.shrink


    @property
    def score_sd(self) -> float:
        """Standard deviation of demeaned predictions, floored to
        SCORE_SD_FLOOR. Used as the score-side unit when mapping to
        implied returns."""
        preds, _ = self._demeaned_series()
        if len(preds) < 2:
            return SCORE_SD_FLOOR
        sd = statistics.pstdev(preds)
        return max(sd, SCORE_SD_FLOOR)

    @property
    def realized_sd(self) -> float:
        """Standard deviation of demeaned realized returns. Pre-warmup
        falls back to PRIOR_REALIZED_SD so the implied-return scale is
        sane during cold-start."""
        _, reals = self._demeaned_series()
        if len(reals) < CAPSULE_WARMUP_N:
            return PRIOR_REALIZED_SD
        sd = statistics.pstdev(reals)
        return max(sd, 1e-6)

    @property
    def hit_rate(self) -> float:
        """Fraction of samples where sign(prediction) == sign(realized)."""
        if self.n == 0:
            return 0.0
        hits = sum(1 for p, _c, r, _d in self._samples if p * r > 0)
        return hits / self.n

    # ---- implied-return mapping -------------------------------------

    def implied_return(self, score: float) -> float:
        """Map a new score to an estimated next-bar return using the
        unit-conversion factor sd_realized / sd_score, scaled by IC.

        Default ("zero_floor"): returns 0 when IC is non-positive (negative
        skill treated as no information). Signed mode: uses IC with its sign, so
        a negative-IC role flips the score's direction (contrarian use).
        """
        ic = self.ic
        if _ic_sign_mode() != "signed" and ic <= 0.0:
            return 0.0
        score_clipped = _clip(_safe_finite(score), -1.0, 1.0)
        # Interpret raw score as already centered: capsule learns the
        # cross-sectional structure, and a fresh score is by convention
        # interpreted relative to "typical" (no per-day mean is yet
        # known for the new prediction). This keeps the mapping simple
        # and unit-stable.
        return ic * (self.realized_sd / self.score_sd) * score_clipped


    # ---- updates -----------------------------------------------------

    def observe(
        self,
        predicted_score: float,
        predicted_conf: float,
        realized_return: float,
        trade_date: str = "",
    ) -> None:
        pred = _clip(_safe_finite(predicted_score), -1.0, 1.0)
        conf = _clip(_safe_finite(predicted_conf), 0.0, 1.0)
        real = _clip(
            _safe_finite(realized_return), -REALIZED_RETURN_CLIP, REALIZED_RETURN_CLIP
        )
        self._samples.append((pred, conf, real, str(trade_date or "")))

    # ---- serialization -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "role": self.role,
            "n": self.n,
            "is_warm": self.is_warm,
            "ic": round(self.ic, 4),
            "information_coefficient": round(self.ic, 4),
            "skill_weight": round(self.skill_weight, 4),
            "score_sd": round(self.score_sd, 6),
            "realized_sd": round(self.realized_sd, 6),
            "hit_rate": round(self.hit_rate, 4),
            "samples": [
                [round(p, 6), round(c, 6), round(r, 6), str(d)]
                for p, c, r, d in self._samples
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalystCapsule":
        c = cls(str(d.get("regime", "")), str(d.get("role", "")))
        for entry in d.get("samples", []) or []:
            try:
                # Dumps may store 3-tuples without a date.
                if len(entry) >= 4:
                    p, conf, r, date = entry[0], entry[1], entry[2], entry[3]
                else:
                    p, conf, r = entry[0], entry[1], entry[2]
                    date = ""
                c._samples.append((
                    _safe_finite(p),
                    _safe_finite(conf),
                    _safe_finite(r),
                    str(date or ""),
                ))
            except (KeyError, IndexError, TypeError):
                continue
        return c


__all__ = [
    "AnalystCapsule",
    "CAPSULE_WARMUP_N",
    "CAPSULE_HISTORY_CAP",
    "REALIZED_RETURN_CLIP",
    "SCORE_SD_FLOOR",
    "PRIOR_REALIZED_SD",
    # Exported for import compatibility; not used internally.
    "PRIOR_ALPHA",
    "PRIOR_BETA_VAR",
    "PRIOR_BETA_SLOPE",
]
