"""
TacticalMemory: short-window, day-to-day reflection storage.

Paper correspondence (darwintrade_aaai2027.tex):
  Holds the tactical guard state U^T_t of Section "Tactical Evolution". The
  paper defines U^T_t as an avoid set, a reduce-only set, a portfolio haircut
  in [0.4, 1.0], and a time-to-live; every guard expires after 1-3 trading days
  (Table "Fixed hyperparameters": guard TTL 1-3 bars, haircut range [0.4,1.0]).
  - install_influence  = the InstallExpire step of Alg 1 (install a guard with
    its 1-3 bar TTL and anchor NAV)
  - tick_expiry        = the auto-expiry that makes a tactical response
    constrain only the next 1-3 decisions "without silently becoming permanent
    policy"
  - the install-log anchor NAV / post_install_return is the "logs its anchor NAV
    and post-install return at expiry, so a hurting guard family cannot keep
    re-arming" mechanism.
  The reflect-then-evolve producer of this state (f_tactical-evolving, Eq.
  tactical-evolving) lives in reflection.py + agents/evolution.py; this module
  is the state container and its TTL lifecycle.

Holds the last N days of bar-level outcomes and tactical reflections.
Produces TacticalInfluence — short-lived (1-3 day) constraints that
shape next bar's behavior. Never modifies strategy config.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from ..core.contracts import (
    EpisodeRecord,
    TacticalInfluence,
    TacticalReflection,
)


TACTICAL_WINDOW = 5          # how many recent days to consider for tactical influence
TACTICAL_RING_CAP = 20       # ring buffer cap


def _safe_finite(v: Any, default: float = 0.0) -> float:
    try:
        r = float(v)
        return r if r == r and abs(r) != float("inf") else default
    except (TypeError, ValueError):
        return default


class TacticalMemory:
    """
    Stores recent tactical reflections + active tactical influence with TTL.

    Layout on disk:
      <storage_dir>/tactical_episodes.jsonl   (ring of last 20 day-episodes)
      <storage_dir>/tactical_active.json      (currently-active influence + remaining days)
    """

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = self.storage_dir / "tactical_episodes.jsonl"
        self._active_path = self.storage_dir / "tactical_active.json"
        self._install_log_path = self.storage_dir / "tactical_install_log.jsonl"

        self._recent: deque[dict[str, Any]] = deque(maxlen=TACTICAL_RING_CAP)
        self._active_influence: TacticalInfluence | None = None
        self._active_remaining_days: int = 0
        # Track every install with anchor NAV and a per-bar NAV trajectory.
        # When the influence expires, post_install_return is computed.
        # Used by the LLM as "did my last tactical adjustment help or hurt?".
        self._install_log: list[dict[str, Any]] = []
        self._active_install_idx: int | None = None
        self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_bar(
        self,
        *,
        episode: EpisodeRecord,
        reflection: TacticalReflection | None,
    ) -> None:
        """Persist a bar-level snapshot for tactical lookback."""
        record = {
            "trade_date": episode.trade_date,
            "regime": episode.regime,
            "outcome_score": _safe_finite(episode.outcome_score),
            "nav": _safe_finite(episode.nav),
            "dd_from_peak": _safe_finite(episode.dd_from_peak),
            "in_drawdown": bool(episode.in_drawdown),
            "in_rebound": bool(episode.in_rebound),
            "symbols": list(episode.symbols),
            "reflection": reflection.to_dict() if reflection else None,
        }
        self._recent.append(record)
        self._persist_episodes()

    def install_influence(self, influence: TacticalInfluence, *, current_nav: float = 0.0) -> None:
        """Activate a tactical influence with its expiry.

        Paper: the InstallExpire step, U^T_t <- InstallExpire(U^T_t, eta^T)
        (Alg 1; Section "Tactical Evolution"). The guard is stored with its 1-3
        bar TTL (expires_in_days) and its anchor NAV, so it can only shape the
        next few decisions. The urgency gate below prevents the every-bar LLM
        from re-installing before the TTL drains (which would make a guard
        permanent, contradicting the paper's expire-guarantee).

        If an influence is already active and its urgency is not higher than
        the existing one, do nothing — letting the existing TTL drain naturally.
        Without this gate the LLM, called every bar, would reinstall on every
        bar and the TTL would never reach zero (constraints become permanent).

        Urgency precedence: emergency > elevated > normal.
        """
        urgency_rank = {"normal": 0, "elevated": 1, "emergency": 2}
        new_rank = urgency_rank.get(influence.urgency, 0)
        if self._active_influence is not None and self._active_remaining_days > 0:
            existing_rank = urgency_rank.get(self._active_influence.urgency, 0)
            if new_rank <= existing_rank:
                return
        self._active_influence = influence
        self._active_remaining_days = max(1, int(influence.expires_in_days or 1))
        # Open a new install_log entry; closes when the influence expires.
        self._install_log.append({
            "installed_on_nav": float(current_nav) if current_nav > 0 else 0.0,
            "anchor_nav": float(current_nav) if current_nav > 0 else 0.0,
            "influence": influence.to_dict(),
            "expires_in_days": self._active_remaining_days,
            "status": "active",
            "post_install_return": None,
            "bars_active": 0,
        })
        self._active_install_idx = len(self._install_log) - 1
        self._persist_active()
        self._persist_install_log()

    def tick_expiry(self, *, current_nav: float = 0.0) -> None:
        """Call once per bar after compute_influence to age out the active influence.

        Paper: the auto-expire half of InstallExpire — decrements the guard's
        1-3 bar TTL and, at expiry, records post_install_return against the
        anchor NAV (Section "Tactical Evolution": "logs its anchor NAV and
        post-install return at expiry, so a hurting guard family cannot keep
        re-arming"). This is what guarantees a tactical guard "expires
        automatically" rather than becoming permanent policy."""
        if self._active_influence is None:
            return
        self._active_remaining_days -= 1
        # update bars_active on the live install_log entry
        if self._active_install_idx is not None and 0 <= self._active_install_idx < len(self._install_log):
            self._install_log[self._active_install_idx]["bars_active"] = (
                int(self._install_log[self._active_install_idx].get("bars_active", 0)) + 1
            )
        if self._active_remaining_days <= 0:
            # close out the install log entry with post_install_return
            if self._active_install_idx is not None and 0 <= self._active_install_idx < len(self._install_log):
                entry = self._install_log[self._active_install_idx]
                anchor = float(entry.get("anchor_nav", 0.0) or 0.0)
                if anchor > 0 and current_nav > 0:
                    # Bar-to-bar NAV moves are typically 1-100 bps; round(4) collapses
                    # them all to 0.0 and breaks the LLM's self-feedback loop —
                    # every install looks "neutral" so it keeps installing more.
                    # round(6) preserves enough resolution for the LLM to tell
                    # +5bp from -30bp.
                    entry["post_install_return"] = round((current_nav - anchor) / anchor, 6)
                entry["status"] = "expired"
                self._persist_install_log()
            self._active_influence = None
            self._active_remaining_days = 0
            self._active_install_idx = None
        self._persist_active()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def compute_influence(self) -> TacticalInfluence:
        """Return the currently-active tactical influence, or empty if none."""
        if self._active_influence is None or self._active_remaining_days <= 0:
            return TacticalInfluence()
        return self._active_influence

    def recent_summary(self, days: int = TACTICAL_WINDOW) -> dict[str, Any]:
        """Compact summary of recent days for tactical reasoning."""
        records = list(self._recent)[-days:]
        if not records:
            return {"days": 0, "avg_score": 0.0, "drawdown_count": 0, "mistake_counts": {}}
        scores = [r["outcome_score"] for r in records]
        mistakes: dict[str, int] = {}
        for r in records:
            ref = r.get("reflection") or {}
            kind = str(ref.get("mistake_type", "")).strip()
            if kind and kind != "none":
                mistakes[kind] = mistakes.get(kind, 0) + 1
        return {
            "days": len(records),
            "avg_score": sum(scores) / len(scores),
            "drawdown_count": sum(1 for r in records if r["in_drawdown"]),
            "mistake_counts": mistakes,
            "records": records,
        }

    def track_record(self, n: int = 8) -> list[dict[str, Any]]:
        """Recent install history with post_install_return. Used by the LLM
        to learn from its own past tactical adjustments — did the haircut/
        avoid help or hurt? Self-evolution feedback channel for Layer 1.
        """
        out = []
        for entry in self._install_log[-n:]:
            inf = entry.get("influence", {}) or {}
            out.append({
                "anchor_nav": entry.get("anchor_nav"),
                "haircut": inf.get("position_haircut"),
                "avoid_symbols": inf.get("avoid_symbols"),
                "reduce_only_symbols": inf.get("reduce_only_symbols"),
                "urgency": inf.get("urgency"),
                "expires_in_days": entry.get("expires_in_days"),
                "status": entry.get("status"),
                "bars_active": entry.get("bars_active"),
                "post_install_return": entry.get("post_install_return"),
            })
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_episodes(self) -> None:
        with self._episodes_path.open("w", encoding="utf-8") as f:
            for r in self._recent:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

    def _persist_active(self) -> None:
        payload = {
            "influence": self._active_influence.to_dict() if self._active_influence else None,
            "remaining_days": self._active_remaining_days,
            "active_install_idx": self._active_install_idx,
        }
        self._active_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _persist_install_log(self) -> None:
        with self._install_log_path.open("w", encoding="utf-8") as f:
            for entry in self._install_log:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _load(self) -> None:
        if self._episodes_path.exists():
            for line in self._episodes_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._recent.append(json.loads(line))
                except Exception:
                    continue
        if self._active_path.exists():
            try:
                data = json.loads(self._active_path.read_text(encoding="utf-8"))
                inf = data.get("influence")
                if inf:
                    self._active_influence = TacticalInfluence(
                        avoid_symbols=list(inf.get("avoid_symbols", [])),
                        reduce_only_symbols=list(inf.get("reduce_only_symbols", [])),
                        position_haircut=_safe_finite(inf.get("position_haircut"), 1.0),
                        expires_in_days=int(inf.get("expires_in_days", 1) or 1),
                        urgency=str(inf.get("urgency", "normal")),
                        rationale=str(inf.get("rationale", "")),
                    )
                self._active_remaining_days = int(data.get("remaining_days", 0) or 0)
                idx = data.get("active_install_idx")
                self._active_install_idx = int(idx) if idx is not None else None
            except Exception:
                pass
        if self._install_log_path.exists():
            for line in self._install_log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._install_log.append(json.loads(line))
                except Exception:
                    continue


class NullTacticalMemory:
    def write_bar(self, *, episode: EpisodeRecord, reflection: TacticalReflection | None) -> None:
        pass

    def install_influence(self, influence: TacticalInfluence, *, current_nav: float = 0.0) -> None:
        pass

    def tick_expiry(self, *, current_nav: float = 0.0) -> None:
        pass

    def compute_influence(self) -> TacticalInfluence:
        return TacticalInfluence()

    def recent_summary(self, days: int = TACTICAL_WINDOW) -> dict[str, Any]:
        return {"days": 0, "avg_score": 0.0, "drawdown_count": 0, "mistake_counts": {}}

    def track_record(self, n: int = 8) -> list[dict[str, Any]]:
        return []


__all__ = ["TacticalMemory", "NullTacticalMemory", "TACTICAL_WINDOW"]
