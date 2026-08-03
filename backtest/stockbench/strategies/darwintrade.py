"""
DarwinTrade strategy adapter for the bundled backtest engine.

Bridges the long/short DarwinTradePipeline to the engine's `on_bar(ctx)`
interface. Short selling is first-class — long & short orders go through
`plan_orders_from_sim_order` so the executor sees explicit
`buy / sell / sell_short / buy_to_cover` sides.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from backtest.stockbench.core.executor import plan_orders

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env", override=False)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


def _cfg_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_artifact_root(darwintrade_cfg: Dict[str, Any]) -> Path:
    configured = str(darwintrade_cfg.get("artifact_root", "") or "").strip()
    if configured:
        return Path(configured)
    run_id = str(os.getenv("TA_RUN_ID", "") or "").strip()
    if run_id:
        safe = run_id.replace("/", "_").replace("\\", "_")
        return REPO_ROOT / "storage" / "reports" / "backtest" / safe
    return REPO_ROOT / "storage" / "reports" / "backtest" / "darwintrade_harness"


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class DarwinTradeStrategy:
    """
    Strategy adapter for DarwinTrade.

    Engine contract:
      __init__(cfg)
      on_bar(ctx) -> list[OrderDict]
      set_daily_output_dir(path)
      record_executed_decisions(executed_symbols, portfolio=None)
      after_bar(...) -> dict | None      [optional]
    """

    strategy_name = "darwintrade"

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        from darwintrade import DarwinTradePipeline
        from darwintrade.pipeline import PipelineConfig

        cfg = dict(cfg or {})
        self.cfg = cfg
        darwintrade_cfg = dict(cfg.get("darwintrade", cfg) or {})
        memory_cfg = dict(darwintrade_cfg.get("memory", {}) or {})

        artifact_root = _default_artifact_root(darwintrade_cfg)
        artifact_root.mkdir(parents=True, exist_ok=True)
        storage_dir = Path(
            memory_cfg.get("memory_root", "") or str(artifact_root / "memory")
        )

        pipeline_config = PipelineConfig(
            max_gross_exposure=_cfg_float(
                darwintrade_cfg.get("max_gross_exposure"), default=0.95
            ),
            max_single_position=_cfg_float(
                darwintrade_cfg.get("max_single_position"), default=1.0
            ),
            min_confidence_threshold=_cfg_float(
                darwintrade_cfg.get("min_confidence_threshold"), default=0.35
            ),
            memory_enabled=_cfg_bool(memory_cfg.get("enabled"), default=True),
            tactical_enabled=_cfg_bool(
                darwintrade_cfg.get("tactical_enabled"), default=True
            ),
            strategic_enabled=_cfg_bool(
                darwintrade_cfg.get("strategic_enabled"), default=True
            ),
            analyst_capsule_enabled=_cfg_bool(
                darwintrade_cfg.get("analyst_capsule_enabled"), default=True
            ),
            long_only=_cfg_bool(darwintrade_cfg.get("long_only"), default=False),
        )

        self.pipeline = DarwinTradePipeline(
            storage_dir=storage_dir,
            config=pipeline_config,
        )
        self.artifact_root = artifact_root
        self.daily_output_dir: Path | None = None

        # symbol universe — supplied per-bar from ctx, but cfg can pre-seed it
        self.symbols: List[str] = list(cfg.get("symbols") or [])

        # state for outcome tracking across bars
        self._prev_nav: float | None = None
        self._prev_open_map: dict[str, float] | None = None
        # (nav, alpha) over (t-2 open, t-1 open) — the outcome pair released to
        # the evolution loops on bar t.
        self._lagged_outcome: tuple[float, float] | None = None
        self._spy_change_cache_date: str | None = None
        self._spy_change_cache_val: Dict[str, Any] | None = None
        self._bar_results: list[dict[str, Any]] = []
        self._pending_decisions: dict[str, dict[str, Any]] = {}
        self._last_bar_summary: dict[str, Any] | None = None

        logger.info(
            "[darwintrade] strategy initialized memory=%s tactical=%s strategic=%s",
            pipeline_config.memory_enabled,
            pipeline_config.tactical_enabled,
            pipeline_config.strategic_enabled,
        )

    # ------------------------------------------------------------------
    # Engine plumbing
    # ------------------------------------------------------------------

    def set_daily_output_dir(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.daily_output_dir = out

    def on_bar(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        trade_date = self._extract_date_str(ctx)
        open_map = dict(ctx.get("open_map") or ctx.get("open_price_map") or {})
        portfolio = ctx.get("portfolio")
        symbols = list(ctx.get("symbols") or self.symbols)
        # filter to symbols that have a real open price for today
        usable_symbols = [s for s in symbols if float(open_map.get(s) or 0.0) > 0.0]
        self.symbols = symbols

        total_equity = float(
            ctx.get("equity_for_sizing")
            or getattr(portfolio, "total_equity", 0.0)
            or 0.0
        )
        if total_equity <= 0.0:
            total_equity = float(getattr(portfolio, "cash", 0.0) or 0.0)
        cash = float(getattr(portfolio, "cash", total_equity) or total_equity)

        # signed market values per symbol — short positions carry negative shares
        positions = self._signed_positions(portfolio, open_map)

        from darwintrade.core.contracts import PortfolioState

        portfolio_state = PortfolioState(
            trade_date=trade_date,
            cash=cash,
            total_equity=total_equity,
            positions=positions,
            prices={s: float(open_map.get(s) or 0.0) for s in usable_symbols},
        )

        outcome_nav, outcome_alpha = self._release_outcome(total_equity, open_map)

        market_data = self._build_market_data(trade_date, ctx, open_map)
        asset_data = self._build_asset_data(trade_date, usable_symbols, open_map)
        returns_history = self._build_returns_history(trade_date, usable_symbols, ctx)

        result = self.pipeline.run(
            trade_date=trade_date,
            symbols=usable_symbols,
            portfolio_state=portfolio_state,
            market_data=market_data,
            asset_data=asset_data,
            outcome_nav=outcome_nav,
            outcome_alpha=outcome_alpha,
            returns_history=returns_history,
        )

        self._prev_nav = total_equity
        self._prev_open_map = dict(open_map)

        # convert pipeline payload → engine order dicts
        orders: list[dict[str, Any]] = []
        portfolio_dict = self._portfolio_for_executor(portfolio, open_map, total_equity)
        self._pending_decisions = {}

        for decision in result.benchmark_payload:
            sym = str(decision.get("symbol") or "")
            if sym in {"", "__PORTFOLIO__"}:
                self._record_decision(trade_date, decision)
                continue
            if str(decision.get("action")) == "hold":
                self._record_decision(trade_date, decision)
                continue

            self._pending_decisions[sym] = decision
            planned = plan_orders(
                decision=decision,
                snapshot_price=float(open_map.get(sym) or 0.0),
                cfg=self.cfg,
                portfolio=portfolio_dict,
            )
            orders.extend(planned)
            self._record_decision(trade_date, decision)

        bar_summary = {
            "trade_date": trade_date,
            "regime": result.regime.regime,
            "regime_confidence": result.regime.confidence,
            "memory_influence": result.memory_influence.to_dict(),
            "tactical_reflection": (
                result.tactical_reflection.to_dict() if result.tactical_reflection else None
            ),
            "tactical_influence": (
                result.tactical_influence.to_dict() if result.tactical_influence else None
            ),
            "strategic_reflection": (
                result.strategic_reflection.to_dict() if result.strategic_reflection else None
            ),
            "strategic_patch": (
                result.strategic_patch.to_dict() if result.strategic_patch else None
            ),
            "order_count": len(result.execution_plan.orders),
            "exposure": result.execution_plan.audit_trail.get("exposure", {}),
        }
        self._bar_results.append(bar_summary)
        self._last_bar_summary = bar_summary
        self._write_bar_artifact(trade_date, result)

        return orders

    def record_executed_decisions(
        self, executed_symbols: List[str], portfolio: Any = None
    ) -> None:
        if self.daily_output_dir is not None:
            row = {
                "executed_symbols": list(executed_symbols),
                "pending_symbols": sorted(self._pending_decisions.keys()),
                "bar_summary": self._last_bar_summary,
            }
            self._append_jsonl(
                self.daily_output_dir / "darwintrade_execution_summary.daily.jsonl",
                row,
            )
        self._pending_decisions = {}

    def after_bar(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        # pipeline already does its own per-bar reflection inside run(); this
        # hook only surfaces the latest bar summary back to the engine for any
        # downstream telemetry the harness wants to attach.
        return self._last_bar_summary

    # ------------------------------------------------------------------
    # End of run summary (called by harnesses; engine itself doesn't use it)
    # ------------------------------------------------------------------

    def on_end(self, portfolio: Any = None) -> Dict[str, Any]:
        total_equity = float(getattr(portfolio, "total_equity", 0.0) or 0.0) if portfolio else 0.0
        start_nav = float(self._bar_results[0].get("nav", total_equity)) if self._bar_results else total_equity
        total_return = (total_equity - start_nav) / start_nav if start_nav > 0 else 0.0
        summary = {
            "total_return": round(total_return, 6),
            "end_nav": round(total_equity, 2),
            "bars_run": len(self._bar_results),
            "strategic_capsules": self.pipeline.strategic_memory.get_capsule_summary(),
            "strategic_state": self.pipeline.strategic_memory.get_state().to_dict(),
            "regime_distribution": self._regime_distribution(),
        }
        try:
            (self.artifact_root / "backtest_summary.json").write_text(
                json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("backtest_summary write failed: %s", exc)
        logger.info(
            "[darwintrade] backtest complete bars=%d total_return=%.2f%%",
            len(self._bar_results), total_return * 100,
        )
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_date_str(ctx: Dict[str, Any]) -> str:
        raw = ctx.get("date") or ctx.get("trade_date") or ctx.get("ts")
        if isinstance(raw, datetime):
            return raw.strftime("%Y-%m-%d")
        if isinstance(raw, pd.Timestamp):
            return raw.strftime("%Y-%m-%d")
        if raw is None:
            return ""
        # could be a date or a string already
        try:
            return pd.Timestamp(raw).strftime("%Y-%m-%d")
        except Exception:
            return str(raw)

    @staticmethod
    def _signed_positions(portfolio: Any, open_map: Dict[str, float]) -> Dict[str, float]:
        """
        Return {symbol: signed_market_value}. Negative for short positions.
        Portfolio.positions[symbol].shares is signed (negative for shorts), so we
        propagate the sign forward.
        """
        if portfolio is None:
            return {}
        positions: Dict[str, float] = {}
        raw = getattr(portfolio, "positions", {}) or {}
        for sym, pos in raw.items():
            if isinstance(pos, dict):
                shares = float(pos.get("shares", 0.0) or 0.0)
                price = float(open_map.get(sym) or pos.get("avg_price", 0.0) or 0.0)
            else:
                shares = float(getattr(pos, "shares", 0.0) or 0.0)
                price = float(open_map.get(sym) or getattr(pos, "avg_price", 0.0) or 0.0)
            if abs(shares) <= 0.0:
                continue
            positions[sym] = shares * price
        return positions

    @staticmethod
    def _portfolio_for_executor(
        portfolio: Any, open_map: Dict[str, float], total_equity: float
    ) -> Dict[str, Any]:
        if portfolio is None:
            return {"equity": total_equity, "positions": {}}
        positions: Dict[str, Dict[str, float]] = {}
        for sym, pos in (getattr(portfolio, "positions", {}) or {}).items():
            shares = float(getattr(pos, "shares", 0.0) or 0.0)
            ref_price = float(
                open_map.get(sym) or getattr(pos, "avg_price", 0.0) or 0.0
            )
            positions[sym] = {
                "shares": shares,
                "position_value": shares * ref_price,
            }
        return {"equity": total_equity, "positions": positions}

    def _release_outcome(
        self, total_equity: float, open_map: Dict[str, float]
    ) -> tuple[float | None, float | None]:
        """Return the outcome pair the evolution loops may consume on this bar.

        The loops read outcomes spanning (t-2 open, t-1 open). Only information
        available before the day-t open may shape the day-t order book, and the
        (t-1, t) pair is measurable only once the day-t auction — the fill event
        for those orders — has printed, so it is released on the following bar.
        """
        released = self._lagged_outcome or (None, None)

        if self._prev_nav is not None and self._prev_nav > 0 and total_equity > 0:
            portfolio_return = (total_equity - self._prev_nav) / self._prev_nav
            # Strategic memory reads EXCESS return, not raw NAV return. Raw NAV
            # return on a multi-name book is dominated by market beta, so the
            # strategic layer would read normal market wiggles as lost alpha and
            # de-risk straight through an up market. The benchmark NAV is built
            # post-hoc in metrics.py, so the same-interval market return is
            # reconstructed here as the equal-weight open-to-open return over the
            # symbols priced in both bars (mirrors
            # metrics.compute_simple_average_benchmark).
            market_return = self._equal_weight_market_return(open_map)
            self._lagged_outcome = (
                total_equity,
                portfolio_return - market_return
                if market_return is not None
                else portfolio_return,
            )
        return released

    def _equal_weight_market_return(self, open_map: Dict[str, float]) -> float | None:
        """Equal-weight open-to-open market return over the last bar.

        Used as the beta benchmark to turn raw NAV return into excess return for
        strategic memory. Returns None on the first bar (no prior prices) or when
        no symbol has a valid price in both bars.
        """
        prev = self._prev_open_map
        if not prev:
            return None
        rets: list[float] = []
        for sym, px_now in open_map.items():
            px_prev = prev.get(sym)
            if px_prev and px_prev > 0 and px_now and px_now > 0:
                rets.append(px_now / px_prev - 1.0)
        if not rets:
            return None
        return sum(rets) / len(rets)

    def _build_market_data(
        self, trade_date: str, ctx: Dict[str, Any], open_map: Dict[str, float]
    ) -> Dict[str, Any]:
        # The regime classifier is purely directional: it reads SPY's multi-day
        # trend. VIX was removed — the only VIX source here was a realized-vol
        # proxy, which is a LAGGING indicator that stayed pinned near 50 for
        # weeks after the April 2025 crash and locked the regime into `volatile`
        # straight through the rebound, holding the book at low exposure while
        # the market recovered. Direction (5d/20d trend) is the signal that
        # matters; realized vol added no forward information and hurt.
        market_ctx = dict(ctx.get("market_ctx") or {})
        benchmark = ctx.get("benchmark") or market_ctx.get("benchmark") or {}

        # Look-ahead-safe SPY snapshot: a single prior-day return alone is pure
        # noise (one +0.5% print flipped the regime label bull→bear daily; measured
        # 70% day-over-day flip rate), which fragments the whole regime-conditional
        # memory system. The multi-day trend gives a persistent regime backbone.
        snap = self._benchmark_snapshot(trade_date, ctx)
        spy_change = snap.get("change_pct")
        if spy_change is None:
            if benchmark and benchmark.get("change_pct") is not None:
                spy_change = float(benchmark.get("change_pct"))
            else:
                spy_change = float(market_ctx.get("benchmark_change_pct") or 0.0)

        return {
            "trade_date": trade_date,
            # canonical key read by RegimeAgent heuristic + injected into LLM prompt
            "spy_change_pct": round(spy_change, 4),
            # alias for other consumers
            "benchmark_change_pct": round(spy_change, 4),
            "spy_trend_5d_pct": snap.get("trend_5d"),
            "spy_trend_20d_pct": snap.get("trend_20d"),
            "prices_snapshot": {k: float(v) for k, v in list(open_map.items())[:20]},
            # passed through so the HMM regime baseline can fetch SPY bars with the
            # same data-mode/cfg the rest of the pipeline uses.
            "cfg": ctx.get("cfg") or self.cfg,
        }

    def _benchmark_snapshot(
        self, trade_date: str, ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Look-ahead-safe SPY (benchmark) snapshot AS OF the last CLOSED bar.

        All fields are derived from SPY closes strictly BEFORE trade_date: today's
        decisions run at bar T's open, so today's SPY close is future information.

        Returns:
          change_pct : last closed daily % change, (close_{T-1}/close_{T-2}-1)*100
          trend_5d   : 5-bar % change, the recent-direction tilt
          trend_20d  : 20-bar % change, the medium-term regime backbone

        The single daily change alone flips the regime classifier ~70% of days on
        market noise. The trend fields give the classifier a stable regime
        backbone. Cached per trade_date to avoid refetching.
        """
        if self._spy_change_cache_date == trade_date:
            return self._spy_change_cache_val
        snap: Dict[str, Any] = {
            "change_pct": None, "trend_5d": None, "trend_20d": None,
        }
        try:
            cfg = ctx.get("cfg") or {}
            bench_cfg = (cfg.get("backtest", {}) or {}).get("benchmark", {}) or {}
            sym = str(bench_cfg.get("symbol") or "SPY").upper()
            from backtest.stockbench.core.data_hub import get_bars
            import pandas as pd
            # end < trade_date so today's SPY close is excluded. 40 calendar days
            # covers ~21 trading bars for the 20-day trend plus weekends/holidays.
            end = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
            start = end - pd.Timedelta(days=40)
            df = get_bars(
                sym, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                1, "day", True, cfg=cfg,
            )
            if df is not None and len(df) >= 2:
                col = "close" if "close" in df.columns else (
                    "adjusted_close" if "adjusted_close" in df.columns else None
                )
                if col is not None:
                    closes = df[col].dropna().tolist()
                    if len(closes) >= 2 and closes[-2] > 0:
                        snap["change_pct"] = (closes[-1] / closes[-2] - 1.0) * 100.0
                    if len(closes) >= 6 and closes[-6] > 0:
                        snap["trend_5d"] = round((closes[-1] / closes[-6] - 1.0) * 100.0, 4)
                    if len(closes) >= 21 and closes[-21] > 0:
                        snap["trend_20d"] = round((closes[-1] / closes[-21] - 1.0) * 100.0, 4)
        except Exception as exc:
            logger.debug("benchmark snapshot unavailable for %s: %s", trade_date, exc)
        self._spy_change_cache_date = trade_date
        self._spy_change_cache_val = snap
        return snap

    @staticmethod
    def _build_asset_data(
        trade_date: str, symbols: List[str], open_map: Dict[str, float]
    ) -> Dict[str, Any]:
        return {
            sym: {
                "price": float(open_map.get(sym) or 0.0),
                "trade_date": trade_date,
            }
            for sym in symbols
        }

    @staticmethod
    def _build_returns_history(
        trade_date: str, symbols: List[str], ctx: Dict[str, Any],
        lookback_days: int = 60,
    ) -> Dict[str, list[float]]:
        """Pull last N days of close-to-close log returns per symbol from
        the engine's dataset accessor. The allocator's quant optimizers
        (min_variance / max_sharpe / hrp) need at least 20 samples to
        produce a meaningful covariance estimate."""
        datasets = ctx.get("datasets")
        if datasets is None or not symbols:
            return {}
        try:
            from datetime import date, timedelta
            end_d = date.fromisoformat(trade_date)
            # ~3 calendar months back to clear the 20-trading-day floor with margin.
            start_d = end_d - timedelta(days=lookback_days * 2 + 14)
            start_s = start_d.isoformat()
            end_s = (end_d - timedelta(days=1)).isoformat()
        except Exception as exc:
            logger.debug("returns_history date parse failed: %s", exc)
            return {}
        out: Dict[str, list[float]] = {}
        import math
        for sym in symbols:
            try:
                bars = datasets.get_day_bars(sym, start_s, end_s)
                if bars is None or len(bars) < 2:
                    continue
                closes = [float(c) for c in bars["close"].tolist() if c == c and c > 0]
                if len(closes) < 2:
                    continue
                rets = [
                    math.log(closes[i] / closes[i - 1])
                    for i in range(1, len(closes))
                    if closes[i - 1] > 0 and closes[i] > 0
                ]
                if rets:
                    out[sym] = rets[-lookback_days:]
            except Exception as exc:
                logger.debug("returns_history fetch failed for %s: %s", sym, exc)
        return out

    def _record_decision(self, trade_date: str, decision: Dict[str, Any]) -> None:
        if self.daily_output_dir is None:
            return
        row = {"date": trade_date, **decision}
        self._append_jsonl(
            self.daily_output_dir / "darwintrade_portfolio_decisions.daily.jsonl",
            row,
        )

    @staticmethod
    def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
        except Exception as exc:
            logger.debug("decision jsonl write failed: %s", exc)

    def _write_bar_artifact(self, trade_date: str, result: Any) -> None:
        try:
            day_dir = self.artifact_root / "bars" / trade_date.replace("-", "")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "result.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=True, default=str, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("bar artifact write failed: %s", exc)

    def _regime_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for bar in self._bar_results:
            r = str(bar.get("regime", "unknown"))
            dist[r] = dist.get(r, 0) + 1
        return dist


__all__ = ["DarwinTradeStrategy"]
