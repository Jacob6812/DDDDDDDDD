"""
MarketAgent: wraps DarwinTradeAgenticAnalystBackend.

Each symbol is researched by market, news, social and fundamentals analysts.
The deterministic aggregator combines their signals before portfolio allocation.

Results are converted to AssetSignal for the portfolio allocator.
"""
from __future__ import annotations

import logging
from typing import Any

from ..core.contracts import AssetSignal, MemoryInfluence, RegimeState

logger = logging.getLogger(__name__)


def _direction_from_signal(signal: str) -> str:
    s = str(signal or "").upper()
    if s in ("BUY", "STRONG_BUY", "LONG"):
        return "long"
    if s in ("SELL", "STRONG_SELL", "SHORT"):
        return "short"
    return "hold"


def _confidence_from_packet(packet: Any) -> float:
    """Extract confidence from AssetResearchPacket."""
    try:
        view = packet.asset_view
        conf = float(getattr(view, "confidence", 0.5) or 0.5)
        return max(0.0, min(1.0, conf))
    except Exception:
        return 0.5


def _thesis_from_packet(packet: Any) -> str:
    try:
        view = packet.asset_view
        return str(getattr(view, "summary", "") or getattr(view, "thesis", "") or getattr(view, "rationale", "") or "")[:4000]
    except Exception:
        return ""


def _evidence_refs_from_packet(packet: Any) -> list[str]:
    try:
        meta = dict(packet.metadata or {})
        tool_traces = list(meta.get("tool_traces", []) or [])
        return [str(t.get("tool_id", "")) for t in tool_traces if isinstance(t, dict)][:8]
    except Exception:
        return []


def _risk_score_from_packet(packet: Any) -> float:
    """Extract risk_uncertainty from research aggregation. Higher = more risky."""
    try:
        meta = dict(getattr(packet, "metadata", None) or {})
        shared = dict(meta.get("shared_state", {}) or {})
        # set by the Bayesian aggregator's research node from role payloads
        risk_uncertainty = float(shared.get("risk_uncertainty", 0.0) or 0.0)
        return max(0.0, min(1.0, risk_uncertainty))
    except Exception:
        return 0.0


def _risk_flags_from_packet(packet: Any) -> list[str]:
    try:
        view = packet.asset_view
        flags = list(getattr(view, "risk_flags", []) or [])
        return [str(f) for f in flags][:5]
    except Exception:
        return []


class MarketAgent:
    """
    Runs the full LangGraph multi-role analyst team for each symbol concurrently.
    Falls back to neutral hold signals if the backend is unavailable.
    """

    def __init__(self, llm_client: Any = None) -> None:
        self._backend: Any = None
        self._llm_client = llm_client
        self._backend_initialized = False
        # Default to NullAnalystMemory so the Bayesian aggregator always has
        # a capsule lookup; pipeline overrides this with a real AnalystMemory
        # via attach_analyst_memory when memory_enabled=True.
        from darwintrade.memory.analyst_memory import NullAnalystMemory
        self._analyst_memory: Any = NullAnalystMemory()

    def attach_analyst_memory(self, analyst_memory: Any) -> None:
        """Bind the AnalystMemory used to drive the Bayesian aggregator.
        Idempotent; safe to call before or after the backend is lazy-loaded.
        Passing a NullAnalystMemory is the no-memory ablation: capsules stay
        at prior so the Bayesian posterior collapses to confidence-weighted."""
        self._analyst_memory = analyst_memory
        if self._backend_initialized and self._backend is not None:
            try:
                self._backend.set_analyst_memory(analyst_memory)
            except AttributeError:
                logger.debug("backend has no set_analyst_memory; aggregator disabled")

    def _get_backend(self) -> Any:
        if not self._backend_initialized:
            try:
                from darwintrade.agentic.analysts import create_analyzer_backend
                self._backend = create_analyzer_backend(
                    "darwintrade_agentic",
                    llm_client=self._llm_client,
                )
            except Exception as exc:
                logger.warning("MarketAgent: agentic backend unavailable: %s", exc)
                self._backend = None
            self._backend_initialized = True
            # Bind analyst memory now that the backend exists. The memory is
            # always non-None (NullAnalystMemory is the default) so the
            # Bayesian aggregator runs unconditionally.
            if self._backend is not None:
                try:
                    self._backend.set_analyst_memory(self._analyst_memory)
                except AttributeError:
                    logger.debug("backend lacks set_analyst_memory; aggregator inactive")
        return self._backend

    def analyze(
        self,
        trade_date: str,
        symbols: list[str],
        regime: RegimeState,
        asset_data: dict[str, Any],
        memory_influence: MemoryInfluence,
    ) -> list[AssetSignal]:
        if not symbols:
            return []

        backend = self._get_backend()
        if backend is None:
            return self._neutral_signals(symbols, trade_date, regime)

        # Inject portfolio-level regime into the aggregator's bypass channel.
        # This deliberately does NOT reach the analyst LLM prompts (those go
        # through sanitize_runtime_context, which strips memory/regime
        # signals to keep the research team uninfluenced by memory). The
        # bypass lets the deterministic aggregator below the LLM look up
        # the correct (regime, role) capsule for IC-shrinkage weighting.
        if hasattr(backend, "set_aggregator_runtime"):
            backend.set_aggregator_runtime({
                "regime": regime.regime,
                "regime_confidence": regime.confidence,
            })

        # build PortfolioState-like object for analyze_universe_packets
        portfolio_state = _make_portfolio_state(asset_data, symbols)

        try:
            packets = backend.analyze_universe_packets(symbols, trade_date, portfolio_state)
        except Exception as exc:
            logger.warning("MarketAgent: analyze_universe_packets failed: %s", exc)
            return self._neutral_signals(symbols, trade_date, regime)

        packets_by_symbol = {p.symbol: p for p in packets if hasattr(p, "symbol")}
        no_trade = set(memory_influence.no_trade_symbols)
        results: list[AssetSignal] = []

        for sym in symbols:
            packet = packets_by_symbol.get(sym)
            if packet is None:
                results.append(self._neutral_signal(sym, trade_date, regime))
                continue

            # extract signal from packet
            packet_meta = dict(getattr(packet, "metadata", None) or {})
            raw_signal = str(
                packet_meta.get("signal")
                or getattr(packet, "signal", "")
                or getattr(getattr(packet, "asset_view", None), "signal", "")
                or "HOLD"
            )
            direction = _direction_from_signal(raw_signal)

            # enforce memory no-trade constraint
            if sym in no_trade:
                direction = "hold"

            # Stage per-role predictions for this (regime, role, symbol) so the
            # next bar can commit them with realized log-returns. This is the
            # ground-truth feedback channel that drives capsule precision learning.
            # NullAnalystMemory.stage_prediction is a no-op, so this loop runs
            # safely under the no-memory ablation.
            role_payloads = dict(packet_meta.get("role_payloads", {}) or {})
            for role, payload in role_payloads.items():
                if not isinstance(payload, dict):
                    continue
                score = float(payload.get("score", 0.0) or 0.0)
                conf = float(payload.get("confidence", 0.0) or 0.0)
                self._analyst_memory.stage_prediction(
                    trade_date=trade_date,
                    regime=regime.regime,
                    role=str(role),
                    symbol=sym,
                    predicted_score=score,
                    predicted_confidence=conf,
                )

            results.append(AssetSignal(
                symbol=sym,
                trade_date=trade_date,
                direction=direction,
                confidence=_confidence_from_packet(packet),
                thesis=_thesis_from_packet(packet),
                regime=regime.regime,
                risk_score=_risk_score_from_packet(packet),
                evidence_refs=_evidence_refs_from_packet(packet),
                risk_flags=_risk_flags_from_packet(packet),
                metadata={"source": "agentic_analyst_team", "raw_signal": raw_signal},
            ))

        return results

    def _neutral_signals(self, symbols: list[str], trade_date: str, regime: RegimeState) -> list[AssetSignal]:
        return [self._neutral_signal(sym, trade_date, regime) for sym in symbols]

    def _neutral_signal(self, symbol: str, trade_date: str, regime: RegimeState) -> AssetSignal:
        return AssetSignal(
            symbol=symbol,
            trade_date=trade_date,
            direction="hold",
            confidence=0.3,
            thesis="analyst team unavailable",
            regime=regime.regime,
            metadata={"source": "fallback"},
        )


def _make_portfolio_state(asset_data: dict[str, Any], symbols: list[str]) -> Any:
    """
    Build a minimal PortfolioState-compatible object from asset_data.
    The agentic backend uses portfolio_state.to_dict() for context.
    """
    prices = {sym: float(asset_data.get(sym, {}).get("price", 0.0)) for sym in symbols}
    total_equity = sum(prices.values()) or 100000.0

    class _MinimalPortfolioState:
        def __init__(self) -> None:
            self.trade_date = ""
            self.cash = total_equity
            self.total_equity = total_equity
            self.positions: dict = {}
            self.prices = prices

        def to_dict(self) -> dict:
            return {
                "cash": self.cash,
                "total_equity": self.total_equity,
                "positions": self.positions,
                "prices": self.prices,
            }

    return _MinimalPortfolioState()


__all__ = ["MarketAgent"]
