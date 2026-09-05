"""
Live sessions — persistent memory across decisions.

A session is the live analogue of one backtest run: it owns a memory directory,
so calling `decide()` on successive trading days lets the three memory layers
learn exactly as they do in a backtest. That is the point of persisting them —
a stateless tool would run the analyst team but never show the self-evolution
the system is built around.

Outcome feedback follows the backtest's timing rule. The evolution loops read
an outcome spanning (t-2 open, t-1 open), because the (t-1, t) pair is only
measurable once bar t's auction — the fill event for bar t-1's orders — has
printed. So an outcome is computed on the bar after the one it describes, and
released on the bar after that.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from darwintrade.core.contracts import PortfolioState
from darwintrade.pipeline import DarwinTradePipeline, PipelineConfig
from darwintrade.serialization import safe_path_component, to_jsonable

from .config import MAX_SYMBOLS_PER_REQUEST, LiveSettings
from .context import MarketContext, resolve_trade_date

logger = logging.getLogger(__name__)

_STATE_FILE = "session.json"


class SessionError(RuntimeError):
    """Invalid session usage (unknown id, bad symbols, replayed date)."""


@dataclass
class SessionState:
    """Cross-decision state. Mirrors the backtest adapter's bar-to-bar fields."""

    session_id: str
    created_at: str
    capital: float
    label: str = ""
    decisions: int = 0
    last_trade_date: str = ""
    # equity and open prices from the previous decision, for the outcome pair
    prev_nav: float | None = None
    prev_open_map: dict[str, float] = field(default_factory=dict)
    # the book the previous decision recommended, marked forward to get the
    # realized return that the evolution loops learn from
    prev_target_weights: dict[str, float] = field(default_factory=dict)
    # (nav, alpha) over (t-2 open, t-1 open), released on the next decision
    lagged_outcome: list[float] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


class LiveSession:
    """One persistent DarwinTrade instance with its own memory directory."""

    def __init__(
        self,
        *,
        session_id: str,
        settings: LiveSettings,
        state: SessionState | None = None,
    ) -> None:
        self.settings = settings
        self.session_id = session_id
        self.root = settings.session_root / safe_path_component(session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = state or SessionState(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            capital=settings.default_capital,
        )
        self._lock = threading.Lock()
        self._pipeline: DarwinTradePipeline | None = None

    # -- lifecycle ------------------------------------------------------

    @property
    def pipeline(self) -> DarwinTradePipeline:
        """Built lazily so creating a session never needs LLM credentials."""
        if self._pipeline is None:
            self._pipeline = DarwinTradePipeline(
                storage_dir=self.root / "memory",
                config=PipelineConfig(
                    max_gross_exposure=self.settings.max_gross_exposure,
                    max_single_position=self.settings.max_single_position,
                    min_confidence_threshold=self.settings.min_confidence_threshold,
                ),
            )
        return self._pipeline

    def save(self) -> None:
        path = self.root / _STATE_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(to_jsonable(self.state.to_dict()), indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    # -- decisions ------------------------------------------------------

    def decide(
        self,
        *,
        symbols: list[str],
        trade_date: str | None = None,
        capital: float | None = None,
    ) -> dict[str, Any]:
        """Produce one day's long/short book.

        Serialized per session: the memory layers and the bar-to-bar state are
        not safe under concurrent writes, and two decisions for one session are
        never usefully parallel anyway.
        """
        with self._lock:
            return self._decide_locked(
                symbols=symbols, trade_date=trade_date, capital=capital
            )

    def _decide_locked(
        self,
        *,
        symbols: list[str],
        trade_date: str | None,
        capital: float | None,
    ) -> dict[str, Any]:
        universe = _normalize_symbols(symbols)
        resolved_date, inferred = resolve_trade_date(trade_date, self.settings)

        if self.state.last_trade_date and resolved_date <= self.state.last_trade_date:
            raise SessionError(
                f"This session already decided through {self.state.last_trade_date}; "
                f"{resolved_date} is not later. Memory evolves forward only — start a "
                "new session to analyse the same or an earlier date."
            )

        if capital is not None:
            _validate_capital(capital)
            if self.state.decisions and capital != self.state.capital:
                raise SessionError("Capital carries forward in a continued session. Omit capital or start a new session to change it.")

        context = MarketContext.build(
            trade_date=resolved_date, symbols=universe, settings=self.settings
        )

        if capital is not None and not self.state.decisions:
            self.state.capital = float(capital)

        equity = self._mark_forward(context.open_map)
        outcome_nav, outcome_alpha = self._release_outcome(equity, context.open_map)

        portfolio_state = PortfolioState(
            trade_date=resolved_date,
            cash=equity,
            total_equity=equity,
            # The tool reports a target book rather than tracking fills, so each
            # decision starts flat and target weights are the answer.
            positions={},
            prices=dict(context.open_map),
        )

        result = self.pipeline.run(
            trade_date=resolved_date,
            symbols=context.usable_symbols,
            portfolio_state=portfolio_state,
            market_data=context.market_data,
            asset_data=context.asset_data,
            outcome_nav=outcome_nav,
            outcome_alpha=outcome_alpha,
            returns_history=context.returns_history,
        )

        self.state.prev_nav = equity
        self.state.prev_open_map = dict(context.open_map)
        self.state.prev_target_weights = {
            symbol: round(float(weight), 6)
            for symbol, weight in result.execution_plan.target_weights.items()
            if abs(float(weight)) > 1e-6
        }
        # Compound the marked-forward equity so the session tracks a running
        # NAV rather than restarting from the original capital each bar.
        self.state.capital = equity
        self.state.decisions += 1
        self.state.last_trade_date = resolved_date

        payload = _render(
            result=result,
            context=context,
            equity=equity,
            session=self,
            date_inferred=inferred,
            outcome_released=outcome_nav is not None,
        )
        self.state.history.append(
            {
                "trade_date": resolved_date,
                "regime": payload["regime"]["label"],
                "decided_at": payload["decided_at"],
                "positions": len(payload["positions"]),
                "gross": payload["exposure"].get("gross"),
            }
        )
        self.state.history = self.state.history[-60:]
        self.save()
        _write_artifact(self.root / "decisions", resolved_date, payload)
        return payload

    def _mark_forward(self, open_map: dict[str, float]) -> float:
        """Mark the previously recommended book forward to today's prices.

        The tool reports a target book rather than tracking fills, so there is no
        position ledger to value. But the evolution loops must learn from a real
        return: if equity were simply carried over, `portfolio_return` would be a
        constant 0.0 while the market moved, and every memory layer would be
        trained on a signal that does not exist.

        So the previous decision's signed target weights are marked forward over
        open-to-open returns, which is exactly the return that book would have
        earned. Symbols missing a price in either bar contribute nothing.
        """
        capital = float(self.state.capital)
        weights = self.state.prev_target_weights
        prev_prices = self.state.prev_open_map
        if not weights or not prev_prices:
            return capital

        weighted_return = 0.0
        for symbol, weight in weights.items():
            before = prev_prices.get(symbol)
            now = open_map.get(symbol)
            if before and before > 0 and now and now > 0:
                weighted_return += float(weight) * (now / before - 1.0)
        return capital * (1.0 + weighted_return)

    def _release_outcome(
        self, equity: float, open_map: dict[str, float]
    ) -> tuple[float | None, float | None]:
        """Return the outcome pair the evolution loops may consume on this bar.

        Computes the (t-1, t) pair now and holds it until the next call, matching
        the backtest's one-bar release delay.
        """
        released = tuple(self.state.lagged_outcome or (None, None))

        prev_nav = self.state.prev_nav
        if prev_nav and prev_nav > 0 and equity > 0:
            portfolio_return = (equity - prev_nav) / prev_nav
            market_return = _equal_weight_return(self.state.prev_open_map, open_map)
            # Strategic memory reads EXCESS return: raw NAV return on a
            # multi-name book is dominated by market beta, and the strategic
            # layer would read ordinary market moves as lost alpha and de-risk
            # straight through an up market.
            alpha = (
                portfolio_return - market_return
                if market_return is not None
                else portfolio_return
            )
            self.state.lagged_outcome = [equity, alpha]

        nav, alpha = (released + (None, None))[:2]
        return nav, alpha


class SessionStore:
    """Session registry backed by one directory per session."""

    def __init__(self, settings: LiveSettings) -> None:
        self.settings = settings
        self.root = settings.session_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._live: dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    def create(self, *, capital: float | None = None, label: str = "") -> LiveSession:
        if capital is not None:
            _validate_capital(capital)
        session_id = uuid.uuid4().hex[:12]
        with self._lock:
            session = LiveSession(session_id=session_id, settings=self.settings)
            if capital is not None and capital > 0:
                session.state.capital = float(capital)
            session.state.label = label.strip()[:120]
            session.save()
            self._live[session_id] = session
            return session

    def get(self, session_id: str) -> LiveSession:
        key = safe_path_component(session_id)
        with self._lock:
            cached = self._live.get(key)
            if cached is not None:
                return cached
            path = self.root / key / _STATE_FILE
            if not path.exists():
                raise SessionError(f"Unknown session: {session_id}")
            try:
                state = SessionState.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise SessionError(f"Corrupt session state for {session_id}: {exc}")
            session = LiveSession(
                session_id=state.session_id, settings=self.settings, state=state
            )
            self._live[key] = session
            return session

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.root.glob(f"*/{_STATE_FILE}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(
                {
                    "session_id": payload.get("session_id"),
                    "label": payload.get("label", ""),
                    "created_at": payload.get("created_at"),
                    "decisions": payload.get("decisions", 0),
                    "last_trade_date": payload.get("last_trade_date", ""),
                    "capital": payload.get("capital"),
                }
            )
        out.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        return out


def decide(
    *,
    symbols: list[str],
    trade_date: str | None = None,
    capital: float | None = None,
    settings: LiveSettings | None = None,
) -> dict[str, Any]:
    """One-shot decision in a throwaway session.

    Convenient for scripting, but the memory layers start empty every call, so
    repeated use will not self-evolve. Use a `SessionStore` session for that.
    """
    resolved = settings or LiveSettings.from_env()
    store = SessionStore(resolved)
    session = store.create(capital=capital, label="one-shot")
    return session.decide(symbols=symbols, trade_date=trade_date, capital=capital)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normalize_symbols(symbols: list[str]) -> list[str]:
    seen: list[str] = []
    for raw in symbols or []:
        for piece in str(raw).replace(",", " ").split():
            token = piece.strip().upper()
            if token and not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", token):
                raise SessionError(f"Invalid symbol: {token}. Use US ticker symbols such as AAPL or BRK.B.")
            if token and token not in seen:
                seen.append(token)
    if not seen:
        raise SessionError("No symbols supplied.")
    if len(seen) > MAX_SYMBOLS_PER_REQUEST:
        raise SessionError(
            f"Too many symbols ({len(seen)}); the cap is "
            f"{MAX_SYMBOLS_PER_REQUEST} per request because each one runs a "
            "four-role analyst team."
        )
    return seen


def _validate_capital(capital: float) -> None:
    if not math.isfinite(capital) or capital <= 0:
        raise SessionError("Capital must be a finite positive number.")


def _equal_weight_return(
    prev_open_map: dict[str, float], open_map: dict[str, float]
) -> float | None:
    """Equal-weight open-to-open return, the beta benchmark for alpha."""
    if not prev_open_map:
        return None
    rets: list[float] = []
    for symbol, now in open_map.items():
        before = prev_open_map.get(symbol)
        if before and before > 0 and now and now > 0:
            rets.append(now / before - 1.0)
    return sum(rets) / len(rets) if rets else None


def _write_artifact(root: Path, trade_date: str, payload: dict[str, Any]) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{safe_path_component(trade_date)}.json"
        path.write_text(
            json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("could not persist decision artifact: %s", exc)


def _render(
    *,
    result: Any,
    context: MarketContext,
    equity: float,
    session: LiveSession,
    date_inferred: bool,
    outcome_released: bool,
) -> dict[str, Any]:
    """Shape a PipelineResult into the tool's response payload."""
    plan = result.execution_plan
    audit = plan.audit_trail or {}
    signals = {s.symbol: s for s in result.signals}

    positions: list[dict[str, Any]] = []
    for symbol, weight in sorted(
        plan.target_weights.items(), key=lambda kv: -abs(float(kv[1]))
    ):
        weight = float(weight)
        if abs(weight) < 1e-6:
            continue
        signal = signals.get(symbol)
        positions.append(
            {
                "symbol": symbol,
                "direction": "long" if weight > 0 else "short",
                "target_weight": round(weight, 6),
                "target_notional": round(weight * equity, 2),
                "reference_price": context.open_map.get(symbol),
                "confidence": round(float(signal.confidence), 4) if signal else None,
                "thesis": signal.thesis if signal else "",
                "risk_flags": list(signal.risk_flags) if signal else [],
            }
        )

    held = {p["symbol"] for p in positions}
    watchlist = [
        {
            "symbol": s.symbol,
            "direction": s.direction,
            "confidence": round(float(s.confidence), 4),
            "thesis": s.thesis,
            "reason": (
                "analyst team unavailable; no reliable signal"
                if s.metadata.get("source") == "fallback"
                else "not selected by portfolio constraints or confidence threshold"
                if s.direction in {"long", "short"}
                else "no directional signal"
            ),
        }
        for s in result.signals
        if s.symbol not in held
    ]

    memory = result.memory_influence
    return {
        "session_id": session.session_id,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trade_date": result.trade_date,
        "trade_date_inferred": date_inferred,
        "capital": round(equity, 2),
        "decision_index": session.state.decisions,
        "regime": {
            "label": result.regime.regime,
            "confidence": round(float(result.regime.confidence), 4),
            "evidence": list(result.regime.evidence),
            "benchmark": context.benchmark,
        },
        "positions": positions,
        "watchlist": watchlist,
        "exposure": audit.get("exposure", {}),
        "constraints": audit.get("constraints", {}),
        "optimizer": audit.get("resolved_optimizer", ""),
        "no_trade_reason": plan.no_trade_reason,
        "memory": {
            "source_layer": memory.source_layer,
            "rationale": memory.rationale,
            "position_haircut": memory.position_haircut,
            "avoid_symbols": list(memory.no_trade_symbols),
            "reduce_only_symbols": list(memory.reduce_only_symbols),
            "preferred_optimizer": memory.preferred_optimizer,
            "sample_count": memory.sample_count,
            "outcome_released": outcome_released,
            "capsules": session.pipeline.analyst_memory.summary(),
            "tactical_reflection": (
                result.tactical_reflection.to_dict()
                if result.tactical_reflection
                else None
            ),
            "strategic_patch": (
                result.strategic_patch.to_dict() if result.strategic_patch else None
            ),
            "total_episodes": result.metadata.get("total_episodes", 0),
        },
        "data": {
            "warnings": [
                f"{s.symbol}: analyst team unavailable; the hold signal is a fallback."
                for s in result.signals if s.metadata.get("source") == "fallback"
            ],
            "skipped_symbols": context.skipped_symbols,
            "analysed": context.usable_symbols,
        },
        "disclaimer": (
            "Research output from an experimental multi-agent system. Not "
            "investment advice."
        ),
    }


__all__ = ["LiveSession", "SessionError", "SessionState", "SessionStore", "decide"]
