"""
AnalystMemory: registry of (regime, role) capsules with disk persistence.

Paper correspondence (darwintrade_aaai2027.tex):
  Holds the collection of evaluation capsules C_{rho,k,t} (one per regime-role
  pair) of Section "Analyst Evolution", Self-Evolution Loop. The individual
  capsule math (IC, shrinkage, u_{rho,k,t+1}) lives in analyst_capsules.py;
  this class stores them keyed by (regime, role) and implements the STRICT
  ONE-BAR DELAY of the update: a prediction made on day t is staged and only
  folded into its capsule after day-(t+1)'s realized return is known
  (Method: "Since y_{t+1,i} is unavailable at decision time t, the update is
  strictly one bar delayed"; Alg 1: "append (x_{t-1,i,k}, c_{t-1,i,k}, y_{t,i})
  to capsule (rho_{t-1}, k)"). This staging is what guarantees no future
  information leaks into the day-t decision.

This is the third memory layer alongside TacticalMemory and StrategicMemory.
While the other two layers learn from outcome NAV, AnalystMemory learns from
per-analyst (predicted_score, realized_return) pairs — i.e. it teaches the
system which role to trust under which regime.

Pending pairs
-------------
At bar t, analyst i predicts on each symbol. The realized log-return for
symbol s is only known at bar t+1. So each bar's predictions are queued in
`pending_pairs` keyed by (trade_date, regime, role, symbol, predicted_score,
predicted_confidence). When the next bar arrives, the close-of-bar logic
calls `commit_realized(prev_trade_date, realized_returns)` which drains the
queue into the appropriate capsules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .analyst_capsules import AnalystCapsule, CAPSULE_WARMUP_N

logger = logging.getLogger(__name__)


class AnalystMemory:
    """Registry of AnalystCapsules keyed by (regime, role)."""

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._capsules_path = self.storage_dir / "analyst_capsules.json"
        self._pending_path = self.storage_dir / "analyst_pending.json"

        self._capsules: dict[tuple[str, str], AnalystCapsule] = {}
        # pending[trade_date] = list of {regime, role, symbol, score, conf}
        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._load()

    # ---- capsule lookup ---------------------------------------------

    def get_or_create(self, regime: str, role: str) -> AnalystCapsule:
        key = (regime, role)
        if key not in self._capsules:
            self._capsules[key] = AnalystCapsule(regime, role)
        return self._capsules[key]

    def get_all(self) -> list[AnalystCapsule]:
        return list(self._capsules.values())

    def get_by_role(self, role: str) -> list[AnalystCapsule]:
        return [c for c in self._capsules.values() if c.role == role]

    def skill_weight_for(self, regime: str, role: str) -> float:
        return self.get_or_create(regime, role).skill_weight

    def ic_for(self, regime: str, role: str) -> float:
        return self.get_or_create(regime, role).ic

    def is_warm(self, regime: str, role: str) -> bool:
        return self.get_or_create(regime, role).is_warm

    # ---- pending queue ----------------------------------------------

    def stage_prediction(
        self, *,
        trade_date: str,
        regime: str,
        role: str,
        symbol: str,
        predicted_score: float,
        predicted_confidence: float,
    ) -> None:
        """Queue a per-symbol prediction for later realization."""
        bucket = self._pending.setdefault(trade_date, [])
        bucket.append({
            "regime": regime,
            "role": role,
            "symbol": symbol,
            "score": float(predicted_score),
            "conf": float(predicted_confidence),
        })

    def commit_realized(
        self, *,
        prediction_trade_date: str,
        realized_returns: dict[str, float],
    ) -> int:
        """Drain pending predictions for `prediction_trade_date` into the
        capsules using the supplied per-symbol realized log-returns.

        Paper: this is the one-bar-delayed capsule write of Eq. analyst-evolving
        (Alg 1: "append (x_{t-1,i,k}, c_{t-1,i,k}, y_{t,i}) to capsule
        (rho_{t-1}, k)"). It is invoked only at the NEXT bar, so the realized
        outcome y_{t+1,i} can never reach the day-t decision that produced the
        prediction.

        Predictions for symbols missing from `realized_returns` are dropped
        (we never had ground truth).

        Returns the number of (prediction, realized) pairs committed.
        """
        bucket = self._pending.pop(prediction_trade_date, []) or []
        committed = 0
        for entry in bucket:
            sym = entry.get("symbol")
            if sym is None or sym not in realized_returns:
                continue
            cap = self.get_or_create(str(entry.get("regime", "")), str(entry.get("role", "")))
            cap.observe(
                predicted_score=float(entry.get("score", 0.0) or 0.0),
                predicted_conf=float(entry.get("conf", 0.0) or 0.0),
                realized_return=float(realized_returns.get(sym, 0.0) or 0.0),
                trade_date=str(prediction_trade_date),
            )
            committed += 1
        if committed > 0:
            self._persist()
        return committed

    def drop_pending_older_than(self, *, keep_dates: set[str]) -> int:
        """House-keeping: drop pending buckets that no longer have a chance
        of being committed (e.g. data gap). Returns number of buckets pruned."""
        before = len(self._pending)
        self._pending = {k: v for k, v in self._pending.items() if k in keep_dates}
        pruned = before - len(self._pending)
        if pruned > 0:
            self._persist()
        return pruned

    # ---- diagnostics ------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """Compact view for logging/reports/the paper. One row per capsule."""
        return [c.to_dict() for c in sorted(
            self._capsules.values(),
            key=lambda c: (c.regime, c.role),
        )]

    # ---- persistence ------------------------------------------------

    def _persist(self) -> None:
        try:
            payload = {
                "capsules": [c.to_dict() for c in self._capsules.values()],
            }
            self._capsules_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
            self._pending_path.write_text(json.dumps(self._pending, ensure_ascii=True), encoding="utf-8")
        except Exception as exc:
            logger.warning("AnalystMemory persist failed: %s", exc)

    def _load(self) -> None:
        if self._capsules_path.exists():
            try:
                payload = json.loads(self._capsules_path.read_text(encoding="utf-8"))
                for entry in payload.get("capsules", []) or []:
                    cap = AnalystCapsule.from_dict(entry)
                    self._capsules[(cap.regime, cap.role)] = cap
            except Exception as exc:
                logger.warning("AnalystMemory capsule load failed: %s", exc)
        if self._pending_path.exists():
            try:
                self._pending = dict(json.loads(self._pending_path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("AnalystMemory pending load failed: %s", exc)


class NullAnalystMemory:
    """No-op stand-in used when memory_enabled=False."""

    def get_or_create(self, regime: str, role: str) -> AnalystCapsule:
        return AnalystCapsule(regime, role)

    def get_all(self) -> list[AnalystCapsule]:
        return []

    def get_by_role(self, role: str) -> list[AnalystCapsule]:
        return []

    def skill_weight_for(self, regime: str, role: str) -> float:
        return 0.0

    def ic_for(self, regime: str, role: str) -> float:
        return 0.0

    def is_warm(self, regime: str, role: str) -> bool:
        return False

    def stage_prediction(self, **_kwargs: Any) -> None:
        pass

    def commit_realized(self, **_kwargs: Any) -> int:
        return 0

    def drop_pending_older_than(self, **_kwargs: Any) -> int:
        return 0

    def summary(self) -> list[dict[str, Any]]:
        return []


__all__ = ["AnalystMemory", "NullAnalystMemory"]
