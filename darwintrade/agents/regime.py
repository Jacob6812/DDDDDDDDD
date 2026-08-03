"""
RegimeAgent: determines current market regime using LLM + market data signals.
Regime drives optimizer selection and constraint tightening.

Paper correspondence (darwintrade_aaai2027.tex):
  Implements the regime classifier f_regime of Section "Analyst Evolution",
  Forward Process, and Alg 1 PHASE "Classify regime (evidence dated < t)".
  The paper defines f_regime over SPY's 5- and 20-bar trailing returns dated
  before t and labels the market bull / bear / volatile, where "volatile" is a
  trend-DISAGREEMENT state (short- and long-horizon trends carry opposite
  signs), NOT a high-realized-vol state. When the LLM output is malformed the
  paper falls back to a deterministic trend rule (Alg 1: "if malformed then use
  deterministic trend rule"), implemented here as _heuristic_regime.

  Note: the code exposes a fourth label "sideways" (range-bound, low
  conviction) as an operational refinement of the paper's three-way scheme;
  the paper text reports the bull/bear/volatile taxonomy.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..core.llm import LLMClient, build_json_schema_response_format
from ..core.contracts import RegimeState
from ..core.skills import skill_section

logger = logging.getLogger(__name__)

REGIMES = ["bull", "bear", "sideways", "volatile"]

_SCHEMA = {
    "type": "object",
    "properties": {
        "regime": {"type": "string", "enum": REGIMES},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "raw_scores": {
            "type": "object",
            "properties": {r: {"type": "number"} for r in REGIMES},
            "additionalProperties": False,
        },
        "reasoning": {"type": "string"},
    },
    "required": ["regime", "confidence", "evidence", "raw_scores"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT_BASE = """You are DarwinTrade's market regime analyst.
Classify the current market into exactly one regime: bull / bear / sideways / volatile.
- bull: broad uptrend, risk-on, positive momentum
- bear: broad downtrend, risk-off, negative momentum
- sideways: range-bound, low directional conviction
- volatile: conflicting trends — the 5-day and 20-day trends disagree in sign
  (e.g. a sharp reversal), so directional conviction is genuinely low.

Classify from the DIRECTION and CONSISTENCY of the multi-day trend:
- spy_trend_20d_pct is the backbone; spy_trend_5d_pct is the recent tilt.
- Both clearly positive → bull. Both clearly negative → bear.
- Both near zero → sideways.
- Opposite signs (5d vs 20d) → volatile (a turning point; do not assume the old
  trend continues, but also do not over-react — a reversal can be the START of a
  new bull leg).
A single day's spy_change_pct is noise — never flip the regime on one day against
a persistent trend.

Return valid JSON only. Be decisive — pick the most likely regime."""


def _system_prompt() -> str:
    return _SYSTEM_PROMPT_BASE + skill_section("market-regime-analyst")


def _build_user_prompt(trade_date: str, market_data: dict[str, Any], macro_signals: dict[str, Any]) -> str:
    return json.dumps({
        "trade_date": trade_date,
        "market_data": market_data,
        "macro_signals": macro_signals,
        "task": (
            "1. Score each regime 0-1 in raw_scores (must sum to ~1). "
            "2. Pick the highest-scoring regime. "
            "3. Set confidence = raw_scores[regime]. "
            "4. List 2-4 evidence strings supporting your choice."
        ),
    }, ensure_ascii=True)


class RegimeAgent:
    """
    Classifies market regime from price/macro signals.
    Falls back to heuristic if LLM unavailable.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    @staticmethod
    def _rule_only() -> bool:
        # Rule-regime ablation: force the deterministic trend rule, skip the LLM.
        # Set DARWIN_REGIME_RULE_ONLY=1 to test whether the LLM regime classifier
        # adds anything over the SPY 5d/20d trend rule it approximates.
        return os.getenv("DARWIN_REGIME_RULE_ONLY", "0").strip().lower() in (
            "1", "true", "yes",
        )

    @staticmethod
    def _hmm_mode() -> bool:
        # Reviewer baseline: replace the trend-disagreement regime rule with a
        # Gaussian HMM fit on SPY returns dated strictly before t. Set
        # DARWIN_REGIME_HMM=1 to run the pipeline with HMM-based regime
        # conditioning instead of the trend rule (isolates the regime module).
        return os.getenv("DARWIN_REGIME_HMM", "0").strip().lower() in (
            "1", "true", "yes",
        )

    def classify(
        self,
        trade_date: str,
        market_data: dict[str, Any],
        macro_signals: dict[str, Any] | None = None,
    ) -> RegimeState:
        # Paper: this is f_regime (Section "Analyst Evolution", Forward Process;
        # Alg 1 PHASE "Classify regime"). Primary path is the LLM classifier;
        # both env-gated branches below are reviewer/ablation baselines, and the
        # except-clause realizes the paper's "if malformed then use deterministic
        # trend rule" fallback.
        if self._rule_only():
            return self._heuristic_regime(trade_date, market_data)
        if self._hmm_mode():
            hmm = self._hmm_regime(trade_date, market_data)
            if hmm is not None:
                return hmm
            return self._heuristic_regime(trade_date, market_data)
        try:
            payload = self.llm.invoke_json(
                _system_prompt(),
                _build_user_prompt(trade_date, market_data, macro_signals or {}),
                caller_tag="regime_agent",
                schema_name="RegimeClassification",
                schema=_SCHEMA,
            )
            regime = str(payload.get("regime", "sideways"))
            if regime not in REGIMES:
                regime = "sideways"
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))
            evidence = [str(e) for e in payload.get("evidence", [])]
            raw_scores = {r: float(payload.get("raw_scores", {}).get(r, 0.0)) for r in REGIMES}
            return RegimeState(
                regime=regime,
                confidence=confidence,
                trade_date=trade_date,
                evidence=evidence,
                raw_scores=raw_scores,
                metadata={"reasoning": payload.get("reasoning", "")},
            )
        except Exception as exc:
            logger.warning("RegimeAgent LLM failed, using heuristic: %s", exc)
            return self._heuristic_regime(trade_date, market_data)

    def _hmm_regime(self, trade_date: str, market_data: dict[str, Any]) -> RegimeState | None:
        """Gaussian-HMM regime baseline on SPY daily log-returns dated < t.

        Fits a 3-state HMM on the trailing return series (all closes strictly
        before trade_date, so no look-ahead), decodes the most recent state, and
        maps states to regime labels by their fitted mean/variance: highest-mean
        state -> bull, lowest-mean -> bear, and among the rest the higher-variance
        -> volatile else sideways. Returns None (caller falls back) if hmmlearn is
        unavailable or the series is too short.
        """
        try:
            import numpy as np
            import pandas as pd
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:
            logger.warning("HMM regime unavailable (%s); falling back", exc)
            return None
        try:
            cfg = market_data.get("cfg") or {}
            sym = "SPY"
            from backtest.stockbench.core.data_hub import get_bars
            end = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
            start = end - pd.Timedelta(days=400)
            df = get_bars(sym, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                          1, "day", True, cfg=cfg)
            if df is None or len(df) < 60:
                return None
            col = "close" if "close" in df.columns else "adjusted_close"
            closes = np.asarray([c for c in df[col].dropna().tolist() if c > 0], dtype=float)
            if closes.size < 60:
                return None
            rets = np.diff(np.log(closes)).reshape(-1, 1)
            n_states = 3
            model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=50, random_state=42)
            model.fit(rets)
            states = model.predict(rets)
            cur = int(states[-1])
            means = model.means_.flatten()
            variances = model.covars_.flatten()
            order_by_mean = list(np.argsort(means))       # low->high mean
            bear_s, mid_s, bull_s = order_by_mean[0], order_by_mean[1], order_by_mean[2]
            if cur == bull_s:
                regime = "bull"
            elif cur == bear_s:
                regime = "bear"
            else:
                # middle-mean state: call it volatile if it is the higher-variance
                # of the non-extreme states, else sideways.
                regime = "volatile" if variances[mid_s] >= np.median(variances) else "sideways"
            post = model.predict_proba(rets)[-1]
            confidence = float(max(0.0, min(1.0, post[cur])))
            return RegimeState(
                regime=regime,
                confidence=confidence,
                trade_date=trade_date,
                evidence=[f"hmm3: state={cur} mean={means[cur]:.4f} var={variances[cur]:.2e}"],
                raw_scores={r: (confidence if r == regime else (1 - confidence) / 3)
                            for r in REGIMES},
                metadata={"source": "gaussian_hmm", "n_states": n_states},
            )
        except Exception as exc:
            logger.warning("HMM regime failed (%s); falling back", exc)
            return None

    def _heuristic_regime(self, trade_date: str, market_data: dict[str, Any]) -> RegimeState:
        """Trend-based heuristic fallback.

        Paper: the "deterministic trend rule" of Alg 1 PHASE "Classify regime"
        used when the LLM f_regime output is malformed. Directly realizes the
        Method definition of f_regime over SPY's 5- and 20-bar trailing returns:
        the `disagree` test (trend_5d * trend_20d < 0) is the paper's
        "trend-disagreement" volatile state where short- and long-horizon trends
        carry opposite signs.

        Keys off the multi-day trend rather than a single prior-day return: one
        noisy daily print used to flip the label every bar (~70% flip rate),
        fragmenting the regime-conditional memory. The 5d/20d trend gives a
        persistent regime backbone; the single-day change is only a tiebreaker
        when no trend is available.
        """
        spy_change = float(market_data.get("spy_change_pct", 0.0) or 0.0)
        trend_5d = market_data.get("spy_trend_5d_pct")
        trend_20d = market_data.get("spy_trend_20d_pct")
        # 20d trend is the backbone; 5d is the recent tilt. No VIX: regime is
        # purely directional. `volatile` means the 5d and 20d trends disagree in
        # sign (a turning point / reversal), not "high vol" — a lagging realized-
        # vol proxy would lock the label into volatile through entire rebounds.
        backbone = None
        for cand in (trend_20d, trend_5d):
            if cand is not None:
                backbone = float(cand)
                break
        signal = backbone if backbone is not None else spy_change

        disagree = (
            trend_5d is not None and trend_20d is not None
            and float(trend_5d) * float(trend_20d) < 0
            and abs(float(trend_5d)) > 1.0
        )

        if disagree:
            regime, confidence = "volatile", 0.55
        elif signal > 1.0:
            regime, confidence = "bull", 0.6
        elif signal < -1.0:
            regime, confidence = "bear", 0.6
        else:
            regime, confidence = "sideways", 0.5

        return RegimeState(
            regime=regime,
            confidence=confidence,
            trade_date=trade_date,
            evidence=[
                f"heuristic: trend_20d={trend_20d} trend_5d={trend_5d} "
                f"spy_change={spy_change:.2f}%"
            ],
            raw_scores={r: (confidence if r == regime else (1 - confidence) / 3) for r in REGIMES},
            metadata={"source": "heuristic_fallback"},
        )


__all__ = ["RegimeAgent", "REGIMES"]
