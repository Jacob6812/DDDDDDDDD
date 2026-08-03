"""
DarwinTrade Pipeline — two-layer self-evolving entry point.

Paper correspondence (darwintrade_aaai2027.tex):
  This module is the concrete implementation of the DarwinTrade control loop
  described in Section "Method" and stated formally as pseudocode in
  Appendix "Control-Loop Pseudocode" (Algorithm 1, "DarwinTrade per-bar
  decision and delayed feedback", and Algorithm 2, the periodic strategic
  review). ``DarwinTradePipeline.run`` implements Algorithm 1; the periodic
  branch inside ``_close_previous_bar`` implements Algorithm 2. The three
  state-disjoint self-evolution loops of the Method section map onto:
    - analyst loop     → Section "Analyst Evolution"   (AnalystMemory capsules)
    - tactical loop    → Section "Tactical Evolution"  (TacticalMemory guards)
    - strategic loop   → Section "Strategic Evolution" (StrategicMemory policy)
  The deterministic executor that "the LLM never edits" is PortfolioAllocator
  (Appendix "Deterministic Allocator").

Per-bar (every trading day) — Algorithm 1:
  1. Build memory influence (strategic baseline + active tactical hints)
     → Alg 1 line "Apply tactical guards, then allocate" inputs
  2. Classify regime (LLM) → Alg 1 PHASE "Classify regime"; Method f_regime
  3. Generate asset signals via multi-role analyst team (LLM)
     → Alg 1 PHASE "Multi-analyst research"; Method f_k, x/c per role
  4. Allocate portfolio (deterministic, memory-constrained)
     → Alg 1 PHASE "Apply tactical guards, then allocate"; Eq. strategic-allocation
  5. Emit engine payload → Alg 1 PHASE "Freeze weights; open is fill price only"
  6. (close previous bar) Tactical reflection + tactical evolution → install
     fresh tactical influence for tomorrow → Alg 1 PHASE "Close previous bar";
     Eq. tactical-evolving
  7. Write episode to strategic memory (episode = one decision bar + next-open
     outcome, the unit for the strategic review cadence; see Method)

Periodic (every N completed episodes) — Algorithm 2:
  NOTE ON CADENCE: the paper reports a strategic review period of N=3 episodes
  (Table "Fixed hyperparameters", row "strategic period N"). The actual cadence
  is owned by StrategicMemory.should_run_strategic_review(), not by this
  docstring; verify the constant there matches the reported N=3.
  8. Strategic reflection → cross-day pattern
     → Alg 2 PHASE "Diagnose the multi-day pattern (20-day history)"; f_str-review
  9. If signal strong: strategic diagnosis → policy author → StrategicPatch
     → Alg 2 PHASE "Author a minimal patch with a rollback trigger"; f_str-policy,
       Eq. strategic-evolving, patch tuple (Delta-pi, H, epsilon, m)
 10. Apply patch to strategic state (long-lived), evaluated under a logged
     commit-or-rollback trigger → Alg 2 PHASE "Evaluate the patch under trial"
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agents.evolution import StrategicEvolutionAgent, TacticalEvolutionAgent
from .agents.market import MarketAgent
from .agents.regime import RegimeAgent
from .core.contracts import (
    EpisodeRecord,
    ExecutionPlan,
    MemoryInfluence,
    PipelineResult,
    PortfolioState,
    RegimeState,
    StrategicPatch,
    StrategicReflection,
    TacticalInfluence,
    TacticalReflection,
)
from .core.llm import LLMClient
from .memory.combined import combine_influence
from .memory.reflection import StrategicReflectionAgent, TacticalReflectionAgent
from .memory.strategic import (
    NullStrategicMemory,
    STRATEGIC_REVIEW_LOOKBACK,
    StrategicMemory,
    _dd_metrics,
    _outcome_score,
)
from .memory.tactical import NullTacticalMemory, TacticalMemory
from .memory.analyst_memory import AnalystMemory, NullAnalystMemory
from .portfolio.allocator import PortfolioAllocator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine payload builder
# ---------------------------------------------------------------------------

def _execution_plan_to_engine_payload(
    plan: ExecutionPlan,
    portfolio_state: PortfolioState,
) -> list[dict[str, Any]]:
    if not plan.feasible or not plan.orders:
        return [{
            "schema_version": "portfolio",
            "trade_date": plan.trade_date,
            "symbol": "__PORTFOLIO__",
            "action": "hold",
            "side": "buy",
            "semantic_action": "portfolio_no_trade",
            "non_trade_reason": plan.no_trade_reason or "no_orders",
            "target_cash_amount": round(portfolio_state.cash, 2),
            "cash_change": 0.0,
            "shares": 0.0,
            "reference_price": 0.0,
            "portfolio_total_equity": portfolio_state.total_equity,
            "audit_trail": plan.audit_trail,
        }]

    decisions = []
    for order in plan.orders:
        if order.action == "hold":
            decisions.append({
                "schema_version": "portfolio",
                "trade_date": plan.trade_date,
                "symbol": order.symbol,
                "action": "hold",
                "side": order.side,
                "semantic_action": "maintain_existing",
                "non_trade_reason": "below_min_trade_band",
                "target_cash_amount": order.target_cash_amount,
                "cash_change": 0.0,
                "shares": 0.0,
                "confidence": round(order.confidence, 4),
                "reference_price": order.reference_price,
                "execution_intent": order.execution_intent,
                "rationale": order.rationale,
                "audit": order.audit,
                "portfolio_total_equity": portfolio_state.total_equity,
                "audit_trail": plan.audit_trail,
            })
            continue

        # sim_order forces the executor to honor the exact side
        # (buy / sell / sell_short / buy_to_cover) instead of inferring direction
        # from the sign of cash_change.
        sim_order = {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.shares,
            "reference_price": order.reference_price,
            "schedule": {"slices": 1},
            "constraints": {},
            "order_type": "market",
        }
        decisions.append({
            "schema_version": "portfolio",
            "trade_date": plan.trade_date,
            "symbol": order.symbol,
            "action": order.action,
            "side": order.side,
            "semantic_action": "trade",
            "non_trade_reason": "",
            "target_cash_amount": order.target_cash_amount,
            "cash_change": order.cash_change,
            "shares": order.shares,
            "confidence": round(order.confidence, 4),
            "reference_price": order.reference_price,
            "execution_intent": order.execution_intent,
            "rationale": order.rationale,
            "audit": order.audit,
            "sim_order": sim_order,
            "portfolio_total_equity": portfolio_state.total_equity,
            "audit_trail": plan.audit_trail,
        })
    return decisions


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Strategy-level config. Modified ONLY by StrategicPatch."""
    max_gross_exposure: float = 0.95
    max_single_position: float = 1.0
    min_confidence_threshold: float = 0.35
    preferred_optimizer: str = ""
    drawdown_guard_threshold: float = 0.05
    rebound_entry_threshold: float = 0.02
    position_size_multiplier: float = 1.0
    flip_confidence_premium: float = 0.20

    tactical_enabled: bool = True
    strategic_enabled: bool = True
    memory_enabled: bool = True
    analyst_capsule_enabled: bool = True
    long_only: bool = False

    def apply_patch(self, patch: dict[str, Any]) -> "PipelineConfig":
        import dataclasses
        updates = {k: v for k, v in patch.items() if hasattr(self, k)}
        return dataclasses.replace(self, **updates)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        def _bool(key: str, default: bool) -> bool:
            v = os.getenv(key, "").strip().lower()
            if v in ("1", "true", "yes"):
                return True
            if v in ("0", "false", "no"):
                return False
            return default
        return cls(
            tactical_enabled=_bool("TACTICAL_ENABLED", True),
            strategic_enabled=_bool("STRATEGIC_ENABLED", True),
            memory_enabled=_bool("MEMORY_ENABLED", True),
            analyst_capsule_enabled=_bool("ANALYST_CAPSULE_ENABLED", True),
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class DarwinTradePipeline:
    """
    Two-layer self-evolving DarwinTrade pipeline.

    Layer 1 (tactical, every bar): TacticalReflection → TacticalEvolution →
            TacticalInfluence (1-3d TTL). Adjusts daily guardrails only.

    Layer 2 (strategic, every N days): StrategicReflection → diagnosis →
            policy_author → StrategicPatch. Modifies long-lived strategy config.
    """

    def __init__(
        self,
        *,
        storage_dir: str | Path | None = None,
        config: PipelineConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or PipelineConfig.from_env()
        # Immutable baseline — the un-patched strategy config. self.config is
        # always recomputed as (baseline + current strategic state overrides)
        # so that when a patch is reverted (and its keys drop out of the
        # strategic state), self.config truly returns to baseline instead of
        # ratcheting the tightened value forever.
        self._baseline_config = self.config
        self.llm = llm_client or LLMClient()

        # agents
        self.regime_agent = RegimeAgent(self.llm)
        self.market_agent = MarketAgent(self.llm)

        # memory
        if self.config.memory_enabled and storage_dir:
            self.tactical_memory = TacticalMemory(Path(storage_dir) / "tactical")
            self.strategic_memory = StrategicMemory(Path(storage_dir) / "strategic")
            if self.config.analyst_capsule_enabled:
                self.analyst_memory = AnalystMemory(Path(storage_dir) / "analysts")
            else:
                self.analyst_memory = NullAnalystMemory()
        else:
            self.tactical_memory = NullTacticalMemory()
            self.strategic_memory = NullStrategicMemory()
            self.analyst_memory = NullAnalystMemory()
        # push initial thresholds from config to memory
        self.strategic_memory.update_thresholds(
            drawdown_guard=self.config.drawdown_guard_threshold,
            rebound_entry=self.config.rebound_entry_threshold,
        )

        # Wire the analyst memory into the agentic backend. The Bayesian
        # aggregator runs in BOTH ablation modes:
        #   memory_enabled=True  → AnalystMemory: capsules learn online
        #   memory_enabled=False → NullAnalystMemory: capsules permanently at
        #                          prior (no updates), so the same Bayesian
        #                          posterior collapses to a confidence-weighted
        #                          aggregator. This is the clean ablation
        #                          control: identical aggregation math, only
        #                          the learning is toggled.
        self.market_agent.attach_analyst_memory(self.analyst_memory)

        # reflection + evolution agents
        self.tactical_reflection_agent = TacticalReflectionAgent(self.llm)
        self.tactical_evolution_agent = TacticalEvolutionAgent(self.llm)
        self.strategic_reflection_agent = StrategicReflectionAgent(self.llm)
        self.strategic_evolution_agent = StrategicEvolutionAgent(self.llm)

        # allocator (rebuilt when strategic patch updates config)
        self._allocator = self._build_allocator(self.config)

        # Track the previous bar's actual context so the reflection LLM can
        # reason about what the system actually did, not infer from alpha=0.
        # Distinguishing "traded but lost" from "didn't trade" is critical to
        # avoid false `wrong_direction` diagnoses on flat days.
        self._prev_bar_universe: list[str] = []         # symbols analysed
        self._prev_bar_orders: list[str] = []            # symbols with non-hold orders
        self._prev_bar_holdings: list[str] = []          # symbols actually held
        self._prev_bar_target_weights: dict[str, float] = {}
        self._prev_bar_regime: str = "unknown"
        self._prev_bar_optimizer: str = "default"
        # Prices at the previous bar's open — used to compute realized log-returns
        # for the analyst-capsule online learning loop. The realized return for
        # symbol s observed at bar t+1 is log(open_{t+1} / open_t).
        self._prev_bar_prices: dict[str, float] = {}
        self._prev_bar_trade_date: str = ""
        # (prediction_date, realized_returns) staged on the bar it becomes
        # measurable and committed to the capsules on the following bar.
        self._pending_capsule_commit: tuple[str, dict[str, float]] | None = None

    def _flush_capsule_commit(self) -> None:
        """Write the staged realized returns into the analyst capsules."""
        if self._pending_capsule_commit is None:
            return
        pred_date, staged = self._pending_capsule_commit
        self._pending_capsule_commit = None
        committed = self.analyst_memory.commit_realized(
            prediction_trade_date=pred_date,
            realized_returns=staged,
        )
        if committed > 0:
            logger.debug(
                "AnalystMemory committed %d realized pairs for %s", committed, pred_date,
            )

    def _stage_capsule_commit(self, today_prices: dict[str, float]) -> None:
        """Measure the previous bar's realized log-returns and hold them for the
        next bar's flush."""
        if not (self._prev_bar_trade_date and self._prev_bar_prices and today_prices):
            return
        realized: dict[str, float] = {}
        for sym, prev_px in self._prev_bar_prices.items():
            today_px = today_prices.get(sym, 0.0)
            if prev_px > 0 and today_px > 0:
                try:
                    realized[sym] = math.log(float(today_px) / float(prev_px))
                except (ValueError, ZeroDivisionError):
                    continue
        if realized:
            self._pending_capsule_commit = (self._prev_bar_trade_date, realized)

    def _rebuild_config_from_state(self) -> None:
        """Recompute self.config = baseline + current strategic-state overrides,
        then rebuild the allocator. Because the baseline is immutable, dropping
        a key from the strategic state (e.g. after a patch revert) restores that
        field to its baseline value instead of freezing the patched value."""
        state = self.strategic_memory.get_state()
        overrides = dict(state.constraint_overrides)
        if state.preferred_optimizer:
            overrides["preferred_optimizer"] = state.preferred_optimizer
        self.config = self._baseline_config.apply_patch(overrides)
        self._allocator = self._build_allocator(self.config)

    def _build_allocator(self, config: PipelineConfig) -> PortfolioAllocator:
        return PortfolioAllocator(
            max_gross_exposure=config.max_gross_exposure,
            max_single_position=config.max_single_position,
            min_confidence=config.min_confidence_threshold,
            long_only=config.long_only,
            position_size_multiplier=config.position_size_multiplier,
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        trade_date: str,
        symbols: list[str],
        portfolio_state: PortfolioState,
        market_data: dict[str, Any] | None = None,
        macro_signals: dict[str, Any] | None = None,
        asset_data: dict[str, Any] | None = None,
        outcome_nav: float | None = None,
        outcome_alpha: float | None = None,
        returns_history: dict[str, list[float]] | None = None,
    ) -> PipelineResult:
        market_data = dict(market_data or {})
        macro_signals = dict(macro_signals or {})
        asset_data = dict(asset_data or {})

        # === Step 0: Commit the realized returns staged on the previous bar, so
        # the AnalystMemory capsules learn from ground truth.
        #
        # Paper: Section "Analyst Evolution", Self-Evolution Loop. A realized
        # outcome y_{t,i} = log(open_t / open_{t-1}) is measurable only once the
        # day-t auction has printed, and that auction is where day-t orders fill.
        # Capsule commits therefore lag one bar, which keeps the reliabilities
        # u_{rho,k} that weight the day-t fusion a function of information
        # available strictly before the day-t open.
        self._flush_capsule_commit()
        self._stage_capsule_commit(portfolio_state.prices)

        # === Step 1: Close out previous bar (if outcome available) ===
        # Runs FIRST so today's decisions can use freshly-installed tactical hints.
        #
        # Paper: Alg 1 PHASE "Close previous bar: attach outcome, then evolve"
        # (the `if t > t_start` block). This is the single point where realized
        # feedback re-enters the analyst, tactical, and strategic evolution
        # states, matching the Method claim that "lines that write to an
        # evolution state ... are the only places where feedback re-enters".
        tactical_reflection: TacticalReflection | None = None
        tactical_influence_just_installed: TacticalInfluence | None = None
        strategic_reflection: StrategicReflection | None = None
        strategic_patch: StrategicPatch | None = None

        if outcome_nav is not None and outcome_alpha is not None:
            (
                tactical_reflection,
                tactical_influence_just_installed,
                strategic_reflection,
                strategic_patch,
            ) = self._close_previous_bar(
                trade_date=trade_date,
                outcome_nav=outcome_nav,
                outcome_alpha=outcome_alpha,
            )

        # === Step 2: Build combined memory influence ===
        # Strategic baseline (long-term) + active tactical (short-term)
        regime_for_baseline = "unknown"  # refined below after regime classification
        strategic_baseline = self.strategic_memory.compute_strategic_baseline(
            regime=regime_for_baseline,
            current_nav=portfolio_state.total_equity,
        )
        tactical_active = self.tactical_memory.compute_influence()
        # tick_expiry is called after _close_previous_bar so that the reflection
        # sees the same influence that governed today's trades (Bug 3 fix)

        # === Step 3: Regime classification ===
        # Paper: Alg 1 PHASE "Classify regime (evidence dated < t)"; Section
        # "Analyst Evolution", Forward Process — f_regime labels bull/bear/
        # volatile from SPY's 5- and 20-bar trailing returns dated before t,
        # with a deterministic trend rule fallback when the LLM output is
        # malformed (Alg 1: "if malformed then use deterministic trend rule").
        regime = self.regime_agent.classify(trade_date, market_data, macro_signals)

        # rebuild strategic baseline now that regime is known
        strategic_baseline = self.strategic_memory.compute_strategic_baseline(
            regime=regime.regime,
            current_nav=portfolio_state.total_equity,
        )
        # Paper: the combined influence carries the strategic policy pi_t
        # (long-lived, Section "Strategic Evolution") plus the active tactical
        # guard state U^T_t (short-lived, Section "Tactical Evolution"). These
        # are the two disjoint policy states consumed on day t.
        influence = combine_influence(strategic=strategic_baseline, tactical=tactical_active)

        # === Step 4: Apply current strategic config to allocator (in case patch landed) ===
        # Paper: pi_t parameterizes the deterministic allocator (Section
        # "Strategic Evolution", Forward Process; Eq. strategic-allocation).
        # The strategic loop can only reselect these validated policy fields;
        # it cannot edit the allocator code (Method: "the LLM never edits").
        active_config = self.config
        if influence.constraint_overrides:
            active_config = self.config.apply_patch(influence.constraint_overrides)
            self._allocator = self._build_allocator(active_config)

        # === Step 5: Asset signals ===
        # Paper: Alg 1 PHASE "Multi-analyst research, parallel per asset" and
        # PHASE "Regime-conditioned fusion (Eq. analyst-fusion)". The four
        # role-specific LLM modules f_k return directional score x_{t,i,k} in
        # [-1,1] and confidence c_{t,i,k} in [0,1]; the Bayesian aggregator
        # fuses them under regime-conditioned reliability u_{rho,k} to produce
        # s_{t,i} (Eq. analyst-fusion), with the +/-0.15 buy/sell thresholds.
        signals = self.market_agent.analyze(
            trade_date=trade_date,
            symbols=symbols,
            regime=regime,
            asset_data=asset_data,
            memory_influence=influence,
        )

        # === Step 6: Allocate ===
        # Paper: Alg 1 PHASE "Apply tactical guards, then allocate"; Eq.
        # strategic-allocation. The guarded signals (tactical avoid/reduce-only/
        # haircut already folded into `influence`) are converted to target
        # weights w_t by the deterministic allocator subject to the gross budget
        # G_t and per-name cap w_max,t.
        plan = self._allocator.build_plan(
            trade_date=trade_date,
            signals=signals,
            portfolio_state=portfolio_state,
            regime=regime,
            memory_influence=influence,
            returns_history=returns_history,
        )

        # Tick tactical TTL at end of bar so _close_previous_bar (above) and the
        # allocator both see the same influence that was active this bar.
        # current_nav lets tick_expiry compute post_install_return when the
        # influence reaches end-of-life — the LLM uses this to learn whether
        # past haircut/avoid decisions actually helped.
        self.tactical_memory.tick_expiry(current_nav=portfolio_state.total_equity)

        # Cache this bar's full execution context for next bar's reflection.
        # Tracking the traded universe and orders separately (rather than a
        # union of everything) avoids feeding the LLM ambiguous context that
        # triggers false wrong_direction signals.
        self._prev_bar_universe = sorted(set(symbols))
        self._prev_bar_orders = sorted(
            {o.symbol for o in plan.orders if getattr(o, "action", "hold") != "hold"}
        )
        self._prev_bar_holdings = sorted(portfolio_state.positions.keys())
        self._prev_bar_target_weights = {
            sym: round(float(w), 4) for sym, w in plan.target_weights.items() if abs(float(w)) > 1e-6
        }
        self._prev_bar_regime = regime.regime
        # Use the *resolved* optimizer (what the allocator actually ran), not the
        # LLM's preferred name — quant optimizers may have auto-downgraded due to
        # missing returns_history. Capsule attribution must reflect what actually
        # executed or the LLM will keep requesting unusable optimizers.
        resolved_opt = (
            (plan.audit_trail or {}).get("resolved_optimizer")
            or influence.preferred_optimizer
            or self.config.preferred_optimizer
            or "default"
        )
        self._prev_bar_optimizer = resolved_opt
        # Snapshot of today's open prices, used to compute realized log-returns
        # at the next bar's open for the analyst-capsule learning loop.
        self._prev_bar_prices = dict(portfolio_state.prices or {})
        self._prev_bar_trade_date = trade_date

        # === Step 7: Engine payload ===
        benchmark_payload = _execution_plan_to_engine_payload(plan, portfolio_state)

        return PipelineResult(
            trade_date=trade_date,
            universe=list(symbols),
            regime=regime,
            signals=signals,
            execution_plan=plan,
            memory_influence=influence,
            benchmark_payload=benchmark_payload,
            tactical_reflection=tactical_reflection,
            tactical_influence=tactical_influence_just_installed,
            strategic_reflection=strategic_reflection,
            strategic_patch=strategic_patch,
            metadata={
                "config": active_config.__dict__,
                "strategic_state": self.strategic_memory.get_state().to_dict(),
                "total_episodes": self.strategic_memory.total_episodes,
            },
        )

    # ------------------------------------------------------------------
    # Two-layer close-of-bar logic
    # ------------------------------------------------------------------

    def _close_previous_bar(
        self,
        *,
        trade_date: str,
        outcome_nav: float,
        outcome_alpha: float,
    ) -> tuple[
        TacticalReflection | None,
        TacticalInfluence | None,
        StrategicReflection | None,
        StrategicPatch | None,
    ]:
        # Build episode for the day we just closed
        nav_history = self.strategic_memory.nav_history
        dd = _dd_metrics(
            nav_history, outcome_nav,
            dd_threshold=self.strategic_memory.dd_threshold,
            rebound_threshold=self.strategic_memory.rebound_threshold,
        )
        score = _outcome_score(
            alpha=outcome_alpha,
            dd_from_peak=dd["dd_from_peak"],
            dd_5d=dd["dd_5d"],
            in_drawdown=bool(dd["in_drawdown"]),
            in_rebound=bool(dd["in_rebound"]),
        )
        # Use the previous bar's actual context — what the system saw and did
        # last bar. Falling back to strategic memory mostly returns the same
        # value but breaks down when memory_enabled=False.
        prev_regime = self._prev_bar_regime or "unknown"
        prev_optimizer = self._prev_bar_optimizer or "default"
        prev_universe = list(self._prev_bar_universe)
        prev_orders = list(self._prev_bar_orders)
        prev_holdings = list(self._prev_bar_holdings)
        prev_target_weights = dict(self._prev_bar_target_weights)
        did_trade = bool(prev_orders)

        episode = EpisodeRecord(
            trade_date=trade_date,
            symbols=prev_universe,
            regime=prev_regime,
            optimizer_used=prev_optimizer,
            outcome_score=score,
            nav=outcome_nav,
            dd_from_peak=dd["dd_from_peak"],
            in_drawdown=bool(dd["in_drawdown"]),
            in_rebound=bool(dd["in_rebound"]),
            metadata={
                "alpha": outcome_alpha,
                "did_trade": did_trade,
                "orders": prev_orders,
                "holdings": prev_holdings,
                "target_weights": prev_target_weights,
            },
        )

        # Auto-rollback any pending strategic patches that worsened NAV.
        # Two triggers: early (>= -1% in any bar) and late (>= -2% at 5 bars).
        # Early-revert exists for V-reversal scenarios where a tightening patch
        # at the bottom locks the system out of the bounce.
        #
        # Paper: Alg 2 PHASE "Evaluate the patch under trial before proposing a
        # new one". This is the logged commit-or-rollback trigger of Eq.
        # strategic-evolving: with the monitored performance g_{m_j}(t_j, t),
        # restore pi_j at the first bar where g < -epsilon_j (Revert), otherwise
        # commit once t >= t_j + H_j. Only one patch is under trial at a time
        # ("pending patches = 1" in Table "Fixed hyperparameters").
        try:
            # Rollback thresholds AND reference mode live on each patch's
            # own RollbackTrigger (emitted by the strategic policy-author
            # LLM and validated at parse time). We pass NAV + per-bar alpha;
            # the evaluator selects between absolute-NAV and counterfactual-
            # alpha rollback based on the patch's stated reference field.
            # NOTE: the "pre_patch_policy" reference (shadow-replay counterfactual)
            # additionally needs a per-bar shadow_alpha from replaying the
            # pre-patch policy on the same frozen signals; the live pipeline does
            # not yet run that second allocation, so shadow_alpha defaults to 0
            # and pre_patch_policy behaves as counterfactual_alpha here. The
            # shadow mode is exercised under controlled tests.
            reverted = self.strategic_memory.evaluate_pending_patches(
                current_nav=outcome_nav,
                outcome_alpha=outcome_alpha,
            )
            if reverted:
                logger.info(
                    "Strategic patches auto-reverted (%d): %s",
                    len(reverted), [r.get("patch") for r in reverted],
                )
                # Rebuild from the immutable baseline + whatever remains in the
                # strategic state after the revert. A patch that introduced a
                # brand-new override key leaves an empty state on revert, so
                # this restores baseline defaults instead of freezing the
                # tightened value.
                self._rebuild_config_from_state()
        except Exception as exc:
            logger.warning("Pending-patch rollback evaluation failed: %s", exc)

        # === LAYER 1: Tactical (every bar) ===
        # Paper: Section "Tactical Evolution", Self-Evolution Loop; Eq.
        # tactical-evolving U^T_{t+1} = f_tactical-evolving(tau_{t-1}, y^pf_t,
        # U^T_t). Implemented as the two-step reflect-then-evolve of Alg 1:
        #   q^T <- f_tac-reflect(tau_{t-1}, y^pf_t, U^T_t)   (reflect step below)
        #   eta^T <- f_tac-evolve(q^T, ...)                   (evolve step below)
        #   U^T_t <- InstallExpire(U^T_t, eta^T)              (guards live 1-3 bars)
        tactical_reflection: TacticalReflection | None = None
        tactical_influence: TacticalInfluence | None = None

        if self.config.tactical_enabled:
            try:
                influence_pre = combine_influence(
                    strategic=self.strategic_memory.compute_strategic_baseline(
                        regime=prev_regime, current_nav=outcome_nav,
                    ),
                    tactical=self.tactical_memory.compute_influence(),
                )
                tactical_reflection = self.tactical_reflection_agent.reflect(
                    trade_date=trade_date,
                    episode=episode,
                    execution_summary={
                        "alpha": outcome_alpha,
                        "nav": outcome_nav,
                        "did_trade": did_trade,
                        "orders_placed": prev_orders,
                        "holdings_at_open": prev_holdings,
                        "target_weights": prev_target_weights,
                        "universe": prev_universe,
                    },
                    memory_influence=influence_pre,
                    regime=prev_regime,
                )
                episode.tactical_reflection = tactical_reflection.to_dict()
            except Exception as exc:
                logger.warning("Tactical reflection failed: %s", exc)

            # Tactical evolution: emit short-term influence
            # Paper: f_tactical-evolving produces a guard action that may install
            # an avoid, reduce-only, or haircut constraint, or remain empty
            # (Section "Tactical Evolution"). install_influence corresponds to
            # the InstallExpire step; the guard carries a 1-3 bar TTL and logs
            # its anchor NAV so a hurting guard family cannot keep re-arming.
            if tactical_reflection is not None:
                try:
                    tactical_influence = self.tactical_evolution_agent.evolve(
                        reflection=tactical_reflection,
                        recent_summary=self.tactical_memory.recent_summary(),
                        active_strategy_state=self.strategic_memory.get_state().to_dict(),
                        track_record=self.tactical_memory.track_record(),
                        universe=list(self._prev_bar_universe),
                    )
                    if tactical_influence and (
                        tactical_influence.avoid_symbols
                        or tactical_influence.reduce_only_symbols
                        or tactical_influence.position_haircut < 1.0
                    ):
                        self.tactical_memory.install_influence(
                            tactical_influence, current_nav=outcome_nav or 0.0,
                        )
                except Exception as exc:
                    logger.warning("Tactical evolution failed: %s", exc)

        # Persist episode to BOTH memories
        self.tactical_memory.write_bar(episode=episode, reflection=tactical_reflection)
        self.strategic_memory.write_episode(episode)

        # === LAYER 2: Strategic (every N days) ===
        # Paper: Section "Strategic Evolution" + Algorithm 2 (periodic strategic
        # review, every N=3 completed episodes). should_run_strategic_review()
        # gates on the review cadence; the body implements Alg 2 PHASES
        # "Diagnose the multi-day pattern (20-day history)" (reflect) and
        # "Author a minimal patch with a rollback trigger" (maybe_evolve).
        strategic_reflection: StrategicReflection | None = None
        strategic_patch: StrategicPatch | None = None

        if (
            self.config.strategic_enabled
            and self.strategic_memory.should_run_strategic_review()
        ):
            try:
                payload = self.strategic_memory.review_payload(
                    lookback_days=STRATEGIC_REVIEW_LOOKBACK,
                )
                # Paper: f_str-review over the 20-day history H^S_j; its signal is
                # gated by tau_gate=0.5 (Alg 2: "if z^S.signal < tau_gate return").
                strategic_reflection = self.strategic_reflection_agent.reflect(
                    review_date=trade_date,
                    review_payload=payload,
                )
            except Exception as exc:
                logger.warning("Strategic reflection failed: %s", exc)

            if strategic_reflection is not None:
                try:
                    strategic_patch = self.strategic_evolution_agent.maybe_evolve(
                        review_date=trade_date,
                        reflection=strategic_reflection,
                        capsule_summary=self.strategic_memory.get_capsule_summary(),
                        active_strategy_state=self.strategic_memory.get_state().to_dict(),
                        objective_metrics=payload.get("objective_metrics"),
                        patch_track_record=payload.get("patch_track_record"),
                    )
                    if strategic_patch is not None:
                        logger.info("Strategic patch applied: %s", strategic_patch.patch)
                        self.strategic_memory.apply_patch(strategic_patch, current_nav=outcome_nav or 0.0)
                        # Rebuild config from baseline + updated strategic state
                        # (which now carries this patch's overrides). Keeps the
                        # apply/revert paths symmetric and invertible.
                        self._rebuild_config_from_state()
                        # push patched thresholds to memory so dd/rebound detection picks them up
                        self.strategic_memory.update_thresholds(
                            drawdown_guard=self.config.drawdown_guard_threshold,
                            rebound_entry=self.config.rebound_entry_threshold,
                        )
                except Exception as exc:
                    logger.warning("Strategic evolution failed: %s", exc)

            self.strategic_memory.mark_reviewed()

        return tactical_reflection, tactical_influence, strategic_reflection, strategic_patch


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_pipeline(
    *,
    storage_dir: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> DarwinTradePipeline:
    return DarwinTradePipeline(storage_dir=storage_dir, config=config)


__all__ = ["DarwinTradePipeline", "PipelineConfig", "PipelineResult", "build_pipeline"]
