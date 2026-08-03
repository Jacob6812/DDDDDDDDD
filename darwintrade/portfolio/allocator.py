"""
Portfolio allocator: converts AssetSignals + MemoryInfluence into signed target
weights (long & short), then builds an ExecutionPlan.

Paper correspondence (darwintrade_aaai2027.tex):
  This is the DETERMINISTIC executor / Allocate(.) of Eq. strategic-allocation
  (Section "Strategic Evolution", Forward Process) and Appendix "Deterministic
  Allocator". It is a pure function of the guarded signals S~_t, the current
  portfolio P_t, and historical returns R_{<t}, parameterized only by the
  strategic policy pi_t; it contains NO LLM call and NO learned parameters
  beyond the strategic policy fields (Method: "the LLM never edits" the
  executor). It enforces the two allocation constraints of Eq.
  strategic-allocation: per-name cap |w_{t,i}| <= w_max,t and gross budget
  sum_i |w_{t,i}| <= G_t. SUPPORTED_OPTIMIZERS is the set of nine allocator
  modes of Table "The nine allocator modes selectable by the strategic policy".

Long & short are first-class. There is no long-only fallback. A signal with
direction == "long" produces a positive weight; "short" produces a negative
weight. Crossing zero (e.g. long → short) is split into a closing leg and an
opening leg so the engine sees explicit sell / sell_short events.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ..core.contracts import (
    AllocationTarget,
    AssetSignal,
    ExecutionOrder,
    ExecutionPlan,
    MemoryInfluence,
    PortfolioState,
    RegimeState,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_GROSS_EXPOSURE = 0.95     # |long| + |short| upper bound
DEFAULT_MAX_SINGLE_POSITION = 1.0     # per-symbol safety net; 1.0 == no active cap
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_MIN_TRADE_WEIGHT = 0.005

# All optimizers operate on (confidence, risk_score, direction) which are the
# only per-asset quantities the upstream contract carries. Each is a distinct
# allocation philosophy — the LLM picks which to favor based on regime, recent
# results in capsules, and patch track record.
#
# The first six are "synthetic" — they only use the LLM-supplied
# (confidence, risk_score) per asset. The last three are "quant" — they use
# returns_history (last N daily returns per symbol) to estimate covariance
# and run real Markowitz/HRP. Quant optimizers fall back to conf_weighted
# silently when returns_history is missing or has too few samples.
# Paper: the nine allocator modes m of Table "The nine allocator modes
# selectable by the strategic policy" (Appendix "Deterministic Allocator"),
# spanning classical mean-variance, active-allocation, and hierarchical-risk-
# parity families. Rows below correspond 1:1 to that table's weight rules; the
# three QUANT_OPTIMIZERS are the covariance modes that "require >= 20 per-asset
# returns and otherwise fall back to conf_weighted".
SUPPORTED_OPTIMIZERS = {
    # synthetic (no price history needed) — paper "Needs history: no"
    "conf_weighted",            # weight ∝ confidence × risk-haircut (default)
    "equal_weight",              # flat across eligible signals
    "confidence_concentrated",   # top-3 by confidence, equal among them
    "risk_parity",               # weight ∝ 1 / (risk_score + ε), confidence-gated
    "inverse_risk_weighted",     # weight ∝ confidence × (1 - risk_score)^2
    "kelly_capped",              # weight ∝ confidence^2 × (1 - risk_score)
    # quant (require returns_history) — paper "Needs history: yes"
    "min_variance",              # w ∝ Σ⁻¹ 1, shrunk covariance
    "max_sharpe",                # tangency: w ∝ Σ⁻¹ μ, μ blended with confidence
    "hrp",                       # hierarchical risk parity, no matrix inversion
}
KELLY_TOPK = 3
QUANT_OPTIMIZERS = {"min_variance", "max_sharpe", "hrp"}
RETURNS_MIN_SAMPLES = 20         # below this, fall back to conf_weighted
COV_SHRINKAGE = 0.10             # Ledoit-Wolf-lite shrinkage toward diagonal


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except (TypeError, ValueError):
        return default


def _haircut_for_risk(risk_score: float) -> float:
    """risk=0 keeps full size, risk=1 cuts to 40%."""
    r = max(0.0, min(1.0, risk_score))
    return max(0.4, 1.0 - 0.6 * r)


def _water_fill(
    proposed: dict[str, float],
    *,
    budget: float,
    cap: float,
) -> dict[str, float]:
    """
    Allocate `budget` across keys proportional to `proposed` magnitudes,
    enforcing `|w| <= cap` per key. Any overflow from clipped keys is
    redistributed to the still-unsaturated keys until either the budget is
    fully placed or every key has hit `cap`.

    Inputs are non-negative magnitudes. Sign handling is the caller's job.
    """
    if budget <= 0 or cap <= 0 or not proposed:
        return {k: 0.0 for k in proposed}
    demand = {k: max(0.0, float(v)) for k, v in proposed.items()}
    if sum(demand.values()) <= 0:
        return {k: 0.0 for k in proposed}

    out: dict[str, float] = {k: 0.0 for k in proposed}
    saturated: set[str] = set()
    # Hard ceiling: budget can never exceed n_keys * cap
    available = min(budget, cap * len(proposed))

    # Each iteration either saturates at least one key or places the rest;
    # so at most len(proposed) iterations are needed.
    for _ in range(len(proposed) + 1):
        active = {
            k: v for k, v in demand.items()
            if k not in saturated and v > 0
        }
        if not active or available <= 1e-12:
            break
        active_total = sum(active.values())
        scale = available / active_total
        newly_saturated: list[str] = []
        for k, v in active.items():
            if v * scale >= cap - 1e-12:
                newly_saturated.append(k)
        if newly_saturated:
            for k in newly_saturated:
                out[k] = cap
                available -= cap
                saturated.add(k)
            continue
        # Nothing else gets clipped — place all active proportionally and stop
        for k, v in active.items():
            out[k] = v * scale
        available = 0.0
        break

    return out


# ---------------------------------------------------------------------------
# Signals → signed AllocationTargets
# ---------------------------------------------------------------------------

def _resolve_optimizer(
    preferred: str,
    total_conf: float,
    *,
    returns_history: dict[str, list[float]] | None = None,
    needed_symbols: list[str] | None = None,
) -> str:
    """Map LLM-supplied name to a usable allocator.

    Paper: the covariance-mode fallback of Appendix "Deterministic Allocator"
    — "The three covariance-based modes require at least 20 historical returns
    per asset and otherwise downgrade to confidence weighting" (Table
    "min_variance / max_sharpe / hrp ... fall back to conf_weighted").
    RETURNS_MIN_SAMPLES = 20 is that per-asset history requirement; it keeps
    the allocator well-defined early in each window.
    """
    name = (preferred or "").strip().lower()
    if total_conf <= 0:
        return "equal_weight"
    if name not in SUPPORTED_OPTIMIZERS:
        return "conf_weighted"
    if name in QUANT_OPTIMIZERS:
        if not returns_history or not needed_symbols:
            logger.info("optimizer=%s downgraded to conf_weighted (no returns_history)", name)
            return "conf_weighted"
        for sym in needed_symbols:
            r = returns_history.get(sym)
            if r is None or len(r) < RETURNS_MIN_SAMPLES:
                logger.info(
                    "optimizer=%s downgraded to conf_weighted (sym=%s has %d returns < %d)",
                    name, sym, len(r) if r is not None else 0, RETURNS_MIN_SAMPLES,
                )
                return "conf_weighted"
    return name


def _shrunk_covariance(returns_matrix: np.ndarray, shrinkage: float = COV_SHRINKAGE) -> np.ndarray:
    """Sample covariance shrunk toward its diagonal — keeps Σ invertible when
    sample count is small relative to asset count. shrinkage=1.0 collapses to
    a diagonal (variance-only) matrix; shrinkage=0.0 is the raw sample cov."""
    raw = np.cov(returns_matrix, rowvar=False, ddof=1)
    if raw.ndim == 0:
        raw = raw.reshape(1, 1)
    n = raw.shape[0]
    diag = np.diag(np.diag(raw))
    return (1.0 - shrinkage) * raw + shrinkage * diag


def _min_variance_weights(returns_matrix: np.ndarray) -> np.ndarray:
    """w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1). Returns non-negative magnitudes (negatives
    clipped to 0 then renormalized) so downstream water-fill respects the
    side-budget invariant."""
    n = returns_matrix.shape[1]
    if n == 1:
        return np.array([1.0])
    sigma = _shrunk_covariance(returns_matrix)
    try:
        inv = np.linalg.pinv(sigma)
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)
    ones = np.ones(n)
    raw = inv @ ones
    raw = np.clip(raw, 0.0, None)
    s = raw.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)
    return raw / s


def _max_sharpe_weights(
    returns_matrix: np.ndarray, expected_returns: np.ndarray,
) -> np.ndarray:
    """Tangency portfolio against the supplied expected return vector.
    expected_returns can blend historical mean with confidence — the
    typical signal-driven Markowitz setup."""
    n = returns_matrix.shape[1]
    if n == 1:
        return np.array([1.0])
    sigma = _shrunk_covariance(returns_matrix)
    try:
        inv = np.linalg.pinv(sigma)
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)
    raw = inv @ expected_returns
    raw = np.clip(raw, 0.0, None)
    s = raw.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)
    return raw / s


def _hrp_weights(returns_matrix: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado, 2016) without scipy.
    Cluster by correlation distance, then recursively bisect the inverse-
    variance allocation along the quasi-diagonal ordering. Robust on small
    samples — no matrix inversion."""
    n = returns_matrix.shape[1]
    if n == 1:
        return np.array([1.0])
    sigma = _shrunk_covariance(returns_matrix)
    var = np.diag(sigma)
    std = np.sqrt(np.maximum(var, 1e-12))
    corr = sigma / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))

    # Single-linkage clustering produces the asset ordering. Bisection then
    # operates on contiguous slices of that ordering — that's the whole
    # point of HRP, you split similar assets into the same sub-cluster.
    order = _quasi_diagonal_order(dist)

    weights = np.ones(n)
    # Each "cluster" is a contiguous slice of the quasi-diagonal ordering,
    # represented by ORIGINAL asset indices in cluster-traversal order.
    clusters: list[list[int]] = [order]
    while True:
        next_clusters: list[list[int]] = []
        any_split = False
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            any_split = True
            half = len(cluster) // 2
            left = cluster[:half]
            right = cluster[half:]
            v_left = _cluster_var(sigma, left)
            v_right = _cluster_var(sigma, right)
            alpha = 1.0 - v_left / (v_left + v_right + 1e-12)
            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)
            next_clusters.append(left)
            next_clusters.append(right)
        if not any_split:
            break
        clusters = next_clusters

    s = weights.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)
    return weights / s


def _cluster_var(sigma: np.ndarray, idx: list[int]) -> float:
    """Inverse-variance-weighted portfolio variance for a sub-cluster."""
    sub = sigma[np.ix_(idx, idx)]
    inv_var = 1.0 / np.maximum(np.diag(sub), 1e-12)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def _quasi_diagonal_order(dist: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor traversal — produces the same ordering as
    single-linkage hierarchical clustering for purposes of HRP bisection."""
    n = dist.shape[0]
    if n <= 1:
        return list(range(n))
    visited = {0}
    order = [0]
    while len(order) < n:
        last = order[-1]
        best = None
        best_d = math.inf
        for j in range(n):
            if j in visited:
                continue
            if dist[last, j] < best_d:
                best_d = dist[last, j]
                best = j
        if best is None:
            break
        order.append(best)
        visited.add(best)
    return order


def _build_proposal(
    signals: list[AssetSignal],
    optimizer: str,
    *,
    returns_history: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    """Compute per-asset proposal magnitudes (non-negative). The water-fill
    step then maps proposals to a budget. Different optimizers produce
    materially different shapes — concentrated vs flat vs anti-risk.

    Quant optimizers receive returns_history; if missing or short, the
    resolver should have downgraded the optimizer name before reaching here.
    """
    if not signals:
        return {}
    eps = 1e-3
    if optimizer == "equal_weight":
        return {s.symbol: 1.0 for s in signals}
    if optimizer == "confidence_concentrated":
        ranked = sorted(signals, key=lambda s: s.confidence, reverse=True)
        keep = {s.symbol for s in ranked[:KELLY_TOPK]}
        return {s.symbol: (1.0 if s.symbol in keep else 0.0) for s in signals}
    if optimizer == "risk_parity":
        return {
            s.symbol: max(0.0, s.confidence) * (1.0 / (max(0.0, s.risk_score) + eps))
            for s in signals
        }
    if optimizer == "inverse_risk_weighted":
        return {
            s.symbol: max(0.0, s.confidence) * (max(0.0, 1.0 - s.risk_score) ** 2)
            for s in signals
        }
    if optimizer == "kelly_capped":
        return {
            s.symbol: (max(0.0, s.confidence) ** 2) * max(0.0, 1.0 - s.risk_score)
            for s in signals
        }
    if optimizer in QUANT_OPTIMIZERS and returns_history:
        symbols = [s.symbol for s in signals]
        # build returns matrix [T, N], aligning on the shortest series
        min_len = min(len(returns_history[sym]) for sym in symbols)
        if min_len < RETURNS_MIN_SAMPLES:
            return _conf_weighted_proposal(signals)
        cols = [np.asarray(returns_history[sym][-min_len:], dtype=float) for sym in symbols]
        R = np.column_stack(cols)
        if optimizer == "min_variance":
            w = _min_variance_weights(R)
        elif optimizer == "max_sharpe":
            # Blend historical mean with LLM confidence — pure historical mu
            # is dominated by recent noise; confidence injects forward-looking
            # signal. Risk-haircut on the confidence side.
            mu_hist = R.mean(axis=0)
            mu_conf = np.array([
                max(0.0, s.confidence) * _haircut_for_risk(s.risk_score)
                for s in signals
            ])
            mu = 0.5 * mu_hist + 0.5 * (mu_conf / (mu_conf.sum() + 1e-12))
            w = _max_sharpe_weights(R, mu)
        else:  # hrp
            w = _hrp_weights(R)
        return {sym: float(w[i]) for i, sym in enumerate(symbols)}
    # conf_weighted (default fallback)
    return _conf_weighted_proposal(signals)


def _conf_weighted_proposal(signals: list[AssetSignal]) -> dict[str, float]:
    return {
        s.symbol: max(0.0, s.confidence) * _haircut_for_risk(s.risk_score)
        for s in signals
    }


def _signals_to_targets(
    signals: list[AssetSignal],
    *,
    max_gross_exposure: float,
    max_single_position: float,
    min_confidence: float,
    no_trade_symbols: set[str],
    reduce_only_symbols: set[str],
    current_weights: dict[str, float],
    preferred_optimizer: str,
    long_only: bool = False,
    returns_history: dict[str, list[float]] | None = None,
) -> list[AllocationTarget]:
    """
    Convert signals into signed allocation targets.
      direction == "long"  → positive weight
      direction == "short" → negative weight (suppressed when long_only=True)
      direction == "hold"  → flat (target = 0 unless reduce_only protects current)
    Gross exposure budget is split between long and short legs proportionally
    to the sum of confidence weights on each side.
    When long_only=True, short signals are dropped before budget allocation so
    the full gross budget flows to long signals rather than being post-hoc clamped.
    """
    eligible: list[AssetSignal] = []
    for s in signals:
        if s.direction not in ("long", "short"):
            continue
        if long_only and s.direction == "short":
            continue
        if s.confidence < min_confidence:
            continue
        if s.symbol in no_trade_symbols:
            continue
        eligible.append(s)

    long_signals = [s for s in eligible if s.direction == "long"]
    short_signals = [s for s in eligible if s.direction == "short"]

    long_conf = sum(s.confidence for s in long_signals)
    short_conf = sum(s.confidence for s in short_signals)
    total_conf = long_conf + short_conf

    targets: dict[str, AllocationTarget] = {}

    needed_symbols = [s.symbol for s in eligible]
    optimizer = _resolve_optimizer(
        preferred_optimizer, total_conf,
        returns_history=returns_history, needed_symbols=needed_symbols,
    )
    rationale_tag = optimizer

    long_proposal = _build_proposal(long_signals, optimizer, returns_history=returns_history)
    short_proposal = _build_proposal(short_signals, optimizer, returns_history=returns_history)

    # Side budgets — for equal/k-of-N flavors, share gross by sign-count;
    # for confidence-driven flavors, share by confidence mass.
    # Paper: "Net exposure is not a free parameter: the gross budget is split
    # between the long and short legs strictly by the analyst confidence mass on
    # each side" (Table "The nine allocator modes ...", caption). The allocator
    # holds no directional prior; net exposure is a consequence of cross-sectional
    # selection.
    if optimizer in ("equal_weight", "confidence_concentrated"):
        n_total = max(len(long_signals) + len(short_signals), 1)
        long_budget = max_gross_exposure * (len(long_signals) / n_total)
        short_budget = max_gross_exposure * (len(short_signals) / n_total)
    else:
        # For ALL other optimizers (synthetic AND quant), the side budget is
        # set by raw confidence mass, NOT by post-water-fill proposal mass.
        # Quant optimizers (min_variance/max_sharpe/hrp) return per-side
        # weights normalized to ~1.0 each, so using their proposal mass for
        # cross-leg budgeting always yields a 50/50 long/short split — which
        # erases directional alpha. Confidence mass preserves the LLM's
        # signal-dispersion bias (e.g. 12 longs vs 4 shorts → 75/25 budget).
        if total_conf > 0:
            long_budget = max_gross_exposure * (long_conf / total_conf)
            short_budget = max_gross_exposure * (short_conf / total_conf)
        else:
            long_budget = 0.0
            short_budget = 0.0

    # Per-name concentration cap only makes sense as a *diversification* limit.
    # When a side has a single eligible signal, that one name IS the whole side,
    # so max_single_position must not throttle it below the side budget — else a
    # single-symbol universe collapses to "hold at most max_single_position",
    # forcing the book (near) flat regardless of conviction. Relax the cap to the
    # side budget in that case; multi-name sides keep the real cap.
    long_cap = max_single_position if len(long_signals) > 1 else max(max_single_position, long_budget)
    short_cap = max_single_position if len(short_signals) > 1 else max(max_single_position, short_budget)

    long_alloc = _water_fill(long_proposal, budget=long_budget, cap=long_cap)
    short_alloc = _water_fill(short_proposal, budget=short_budget, cap=short_cap)

    for s in long_signals:
        w = long_alloc.get(s.symbol, 0.0)
        targets[s.symbol] = AllocationTarget(
            symbol=s.symbol, target_weight=w, direction="long",
            confidence=s.confidence,
            rationale=f"{rationale_tag}_long conf={s.confidence:.2f} risk={s.risk_score:.2f}",
        )
    for s in short_signals:
        w = short_alloc.get(s.symbol, 0.0)
        targets[s.symbol] = AllocationTarget(
            symbol=s.symbol, target_weight=-w, direction="short",
            confidence=s.confidence,
            rationale=f"{rationale_tag}_short conf={s.confidence:.2f} risk={s.risk_score:.2f}",
        )

    # reduce-only: cannot grow exposure on either side; cap |target| at |current|.
    # no_trade is a hard stop and takes precedence — those symbols are forced flat
    # even if also in reduce_only_symbols.
    for s in signals:
        if s.symbol not in reduce_only_symbols:
            continue
        if s.symbol in no_trade_symbols:
            continue
        cur = current_weights.get(s.symbol, 0.0)
        cur_abs = abs(cur)
        if s.symbol in targets:
            t = targets[s.symbol]
            new_abs = min(abs(t.target_weight), cur_abs)
            sign = 1.0 if t.target_weight >= 0 else -1.0
            # if current is on the opposite side, force flat (cannot flip under reduce_only)
            if cur != 0.0 and (cur > 0) != (t.target_weight >= 0):
                new_abs = 0.0
            targets[s.symbol] = AllocationTarget(
                symbol=t.symbol,
                target_weight=sign * new_abs if new_abs > 0 else 0.0,
                direction=t.direction if new_abs > 0 else "flat",
                confidence=t.confidence,
                rationale=t.rationale + " [reduce_only]",
            )
        else:
            # no fresh signal → keep current exposure but no growth
            targets[s.symbol] = AllocationTarget(
                symbol=s.symbol,
                target_weight=cur,
                direction="long" if cur > 0 else ("short" if cur < 0 else "flat"),
                confidence=0.0,
                rationale="reduce_only_hold",
            )

    # fill in flat targets for held names without a fresh signal (so they get
    # closed if the analyst stops covering them and they aren't reduce_only)
    for sym, cur in current_weights.items():
        if sym in targets:
            continue
        targets[sym] = AllocationTarget(
            symbol=sym,
            target_weight=0.0,
            direction="flat",
            confidence=0.0,
            rationale="no_signal_close",
        )

    # symbols that had a signal but were filtered (low confidence etc.) → flat
    for s in signals:
        if s.symbol in targets:
            continue
        targets[s.symbol] = AllocationTarget(
            symbol=s.symbol,
            target_weight=0.0,
            direction="flat",
            confidence=s.confidence,
            rationale="signal_filtered",
        )

    return list(targets.values())


# ---------------------------------------------------------------------------
# Targets → execution orders (handles long/short/flip)
# ---------------------------------------------------------------------------

def _emit_leg(
    *,
    symbol: str,
    side: str,
    action: str,
    cash_change: float,
    target_cash: float,
    price: float,
    confidence: float,
    rationale: str,
    audit: dict[str, Any],
) -> ExecutionOrder:
    shares = round(abs(cash_change) / price, 4) if price > 0 else 0.0
    return ExecutionOrder(
        symbol=symbol,
        action=action,
        side=side,
        target_cash_amount=round(target_cash, 2),
        cash_change=round(cash_change, 2),
        shares=shares,
        reference_price=price,
        confidence=confidence,
        execution_intent=action,
        rationale=rationale,
        audit=audit,
    )


def _targets_to_orders(
    targets: list[AllocationTarget],
    portfolio_state: PortfolioState,
    min_trade_weight: float,
) -> list[ExecutionOrder]:
    equity = portfolio_state.total_equity
    if equity <= 0:
        return []

    orders: list[ExecutionOrder] = []
    for t in targets:
        price = portfolio_state.prices.get(t.symbol, 0.0)
        current_value = portfolio_state.positions.get(t.symbol, 0.0)  # signed
        target_value = equity * t.target_weight                       # signed
        delta_value = target_value - current_value                    # signed
        delta_weight = delta_value / equity if equity > 0 else 0.0

        audit = {
            "target_weight": round(t.target_weight, 6),
            "current_weight": round(current_value / equity, 6),
            "delta_weight": round(delta_weight, 6),
        }

        # No-op band — guard against constant churn on rounding noise
        if abs(delta_weight) < min_trade_weight:
            orders.append(_emit_leg(
                symbol=t.symbol,
                side="buy",  # nominal; action=hold suppresses execution
                action="hold",
                cash_change=0.0,
                target_cash=current_value,
                price=price,
                confidence=t.confidence,
                rationale=t.rationale + " [hold_within_band]",
                audit=audit,
            ))
            continue

        crosses_zero = (current_value > 0 and target_value < 0) or (current_value < 0 and target_value > 0)

        if not crosses_zero:
            # single-leg adjustment
            if current_value >= 0 and target_value >= 0:
                # long-side movement
                if delta_value > 0:
                    side, action = "buy", ("open_long" if current_value == 0 else "increase_long")
                else:
                    side, action = "sell", ("close_long" if target_value == 0 else "reduce_long")
            else:
                # short-side movement (current_value <= 0 and target_value <= 0, with at least one < 0)
                if delta_value < 0:
                    # extending short = sell more
                    side, action = "sell_short", ("open_short" if current_value == 0 else "increase_short")
                else:
                    # covering short
                    side, action = "buy_to_cover", ("close_short" if target_value == 0 else "reduce_short")

            orders.append(_emit_leg(
                symbol=t.symbol, side=side, action=action,
                cash_change=delta_value, target_cash=target_value,
                price=price, confidence=t.confidence,
                rationale=t.rationale, audit=audit,
            ))
            continue

        # Crossing zero → emit two legs: close existing, then open opposite
        if current_value > 0:
            # close long, then open short
            close_change = -current_value
            orders.append(_emit_leg(
                symbol=t.symbol, side="sell", action="close_long",
                cash_change=close_change, target_cash=0.0,
                price=price, confidence=t.confidence,
                rationale=t.rationale + " [flip_close_long]",
                audit={**audit, "leg": "close_long"},
            ))
            open_change = target_value  # negative
            orders.append(_emit_leg(
                symbol=t.symbol, side="sell_short", action="open_short",
                cash_change=open_change, target_cash=target_value,
                price=price, confidence=t.confidence,
                rationale=t.rationale + " [flip_open_short]",
                audit={**audit, "leg": "open_short"},
            ))
        else:
            # close short, then open long
            close_change = -current_value  # positive
            orders.append(_emit_leg(
                symbol=t.symbol, side="buy_to_cover", action="close_short",
                cash_change=close_change, target_cash=0.0,
                price=price, confidence=t.confidence,
                rationale=t.rationale + " [flip_close_short]",
                audit={**audit, "leg": "close_short"},
            ))
            open_change = target_value  # positive
            orders.append(_emit_leg(
                symbol=t.symbol, side="buy", action="open_long",
                cash_change=open_change, target_cash=target_value,
                price=price, confidence=t.confidence,
                rationale=t.rationale + " [flip_open_long]",
                audit={**audit, "leg": "open_long"},
            ))

    return orders


# ---------------------------------------------------------------------------
# Main allocator
# ---------------------------------------------------------------------------

class PortfolioAllocator:
    """
    Converts signals + memory influence into a long/short ExecutionPlan.
    Gross exposure constraint applies to |long| + |short|; per-symbol cap
    applies to |target_weight|.
    """

    def __init__(
        self,
        *,
        max_gross_exposure: float = DEFAULT_MAX_GROSS_EXPOSURE,
        max_single_position: float = DEFAULT_MAX_SINGLE_POSITION,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_trade_weight: float = DEFAULT_MIN_TRADE_WEIGHT,
        long_only: bool = False,
        position_size_multiplier: float = 1.0,
    ) -> None:
        self.max_gross_exposure = max_gross_exposure
        self.max_single_position = max_single_position
        self.min_confidence = min_confidence
        self.min_trade_weight = min_trade_weight
        self.long_only = long_only
        self.position_size_multiplier = max(0.1, min(3.0, float(position_size_multiplier)))

    def build_plan(
        self,
        trade_date: str,
        signals: list[AssetSignal],
        portfolio_state: PortfolioState,
        regime: RegimeState,
        memory_influence: MemoryInfluence,
        returns_history: dict[str, list[float]] | None = None,
    ) -> ExecutionPlan:
        overrides = memory_influence.constraint_overrides
        max_gross = _safe_float(overrides.get("max_gross_exposure"), self.max_gross_exposure)
        max_single = _safe_float(overrides.get("max_single_position"), self.max_single_position)
        min_conf = _safe_float(overrides.get("min_confidence_threshold"), self.min_confidence)

        equity = portfolio_state.total_equity
        current_weights = {
            sym: (val / equity) if equity > 0 else 0.0
            for sym, val in portfolio_state.positions.items()
        }

        # Pre-resolve the optimizer here (eligible signals only) so we can
        # report the *resolved* name in the audit trail and the strategic
        # capsule indexes the optimizer that actually ran. Without this, a
        # quant optimizer that auto-downgraded to conf_weighted would be
        # credited as if it executed, biasing the LLM toward unusable choices.
        eligible_for_resolve = [
            s for s in signals
            if s.direction in ("long", "short")
            and not (self.long_only and s.direction == "short")
            and s.confidence >= min_conf
            and s.symbol not in set(memory_influence.no_trade_symbols)
        ]
        total_conf_for_resolve = sum(s.confidence for s in eligible_for_resolve)
        resolved_optimizer = (
            "no_trade"
            if not eligible_for_resolve
            else _resolve_optimizer(
                memory_influence.preferred_optimizer, total_conf_for_resolve,
                returns_history=returns_history,
                needed_symbols=[s.symbol for s in eligible_for_resolve],
            )
        )

        targets = _signals_to_targets(
            signals,
            max_gross_exposure=max_gross,
            max_single_position=max_single,
            min_confidence=min_conf,
            no_trade_symbols=set(memory_influence.no_trade_symbols),
            reduce_only_symbols=set(memory_influence.reduce_only_symbols),
            current_weights=current_weights,
            preferred_optimizer=memory_influence.preferred_optimizer,
            long_only=self.long_only,
            returns_history=returns_history,
        )

        # tactical haircut — symmetric on long and short legs (preserves sign)
        haircut = max(0.0, min(1.0, _safe_float(memory_influence.position_haircut, 1.0)))
        if haircut < 1.0:
            targets = [
                AllocationTarget(
                    symbol=t.symbol,
                    target_weight=t.target_weight * haircut,
                    direction=t.direction,
                    confidence=t.confidence,
                    rationale=t.rationale + f" [tactical_haircut={haircut:.2f}]",
                )
                for t in targets
            ]

        # strategic position-size multiplier — scales gross size up or down.
        # Distinct from haircut: multiplier is a long-lived strategic decision
        # set by StrategicPatch; haircut is a 1-3 day tactical adjustment.
        multiplier = self.position_size_multiplier
        if multiplier != 1.0:
            targets = [
                AllocationTarget(
                    symbol=t.symbol,
                    target_weight=t.target_weight * multiplier,
                    direction=t.direction,
                    confidence=t.confidence,
                    rationale=t.rationale + f" [size_mult={multiplier:.2f}]",
                )
                for t in targets
            ]

        # gross exposure enforcement — if |long|+|short| > max_gross, scale down
        # Paper: enforces the gross-budget constraint sum_i |w_{t,i}| <= G_t of
        # Eq. strategic-allocation, where G_t = max_gross is the strategic
        # policy's gross budget (default G_0 = 0.95; Table "Fixed
        # hyperparameters"). The per-name cap |w_{t,i}| <= w_max,t is enforced
        # earlier by the water-fill `cap` argument.
        gross = sum(abs(t.target_weight) for t in targets)
        if gross > max_gross > 0:
            scale = max_gross / gross
            targets = [
                AllocationTarget(
                    symbol=t.symbol,
                    target_weight=t.target_weight * scale,
                    direction=t.direction,
                    confidence=t.confidence,
                    rationale=t.rationale + f" [gross_scale={scale:.3f}]",
                )
                for t in targets
            ]

        orders = _targets_to_orders(targets, portfolio_state, self.min_trade_weight)
        target_weights = {t.symbol: round(t.target_weight, 6) for t in targets}

        long_gross = sum(t.target_weight for t in targets if t.target_weight > 0)
        short_gross = sum(-t.target_weight for t in targets if t.target_weight < 0)

        return ExecutionPlan(
            trade_date=trade_date,
            orders=orders,
            target_weights=target_weights,
            no_trade_reason="" if any(o.action != "hold" for o in orders) else "no_actionable_targets",
            feasible=True,
            audit_trail={
                "regime": regime.regime,
                "regime_confidence": regime.confidence,
                "memory_influence": memory_influence.to_dict(),
                "preferred_optimizer": memory_influence.preferred_optimizer or "",
                "resolved_optimizer": resolved_optimizer,
                "constraints": {
                    "max_gross_exposure": max_gross,
                    "max_single_position": max_single,
                    "min_confidence": min_conf,
                    "tactical_haircut": haircut,
                },
                "exposure": {
                    "long_gross": round(long_gross, 6),
                    "short_gross": round(short_gross, 6),
                    "gross": round(long_gross + short_gross, 6),
                    "net": round(long_gross - short_gross, 6),
                },
            },
        )


__all__ = ["PortfolioAllocator"]
