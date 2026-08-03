"""Classical cross-sectional quant baselines that feed the SAME deterministic
allocator DarwinTrade uses, but with non-LLM signals.

Reviewers asked us to isolate the value of the LLM analyst layer by plugging a
standard cross-sectional signal (momentum / value / quality composite) into the
identical portfolio-construction stack (HRP / min-variance / max-Sharpe) under the
same universe, windows, and cost model. This module does exactly that: it computes
point-in-time signals from cached prices and fundamentals, wraps them as the same
``AssetSignal`` objects the LLM pipeline emits, and calls the shared
``PortfolioAllocator``. No LLM, no news, no API calls.

Signals (all standardized cross-sectionally each bar to zero-mean/unit-std z-scores,
then averaged into a composite score in [-1, 1] via tanh):
  - momentum : 12-1 style trailing return (skip most recent 5 bars) using closes < t
  - value    : earnings yield (net income / market cap) + book/price, higher = cheaper
  - quality  : return on equity (net income / stockholders' equity) + low leverage
The composite drives direction (long top scores, short bottom) and confidence
(|score|); the allocator's optimizer then sizes positions from returns_history.

Fundamentals are read from the per-symbol / per-trade-date cache written by
``data_hub`` and are already point-in-time (records filed strictly before t).
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backtest.stockbench.core.executor import plan_orders

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
_FUND_BASE = REPO_ROOT / "storage" / "cache" / "fundamentals"


def _safe_nanmean(xs: list[float]) -> float:
    """nanmean that returns NaN (not a warning) when every entry is NaN."""
    arr = np.array([x for x in xs if x == x], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _zscore(vals: Dict[str, float]) -> Dict[str, float]:
    """Cross-sectional z-score; symbols with no value are dropped by the caller."""
    xs = np.array([v for v in vals.values() if v == v], dtype=float)
    if xs.size == 0:
        return {k: 0.0 for k in vals}
    mu = float(xs.mean())
    sd = float(xs.std(ddof=0))
    if sd <= 1e-12:
        return {k: 0.0 for k in vals}
    return {k: ((v - mu) / sd if v == v else 0.0) for k, v in vals.items()}


def _latest_fundamental_record(stmt: str, symbol: str, trade_date: str) -> dict | None:
    """Read the newest cached statement record dated on/before trade_date.

    The cache is storage/cache/fundamentals/<stmt>/<SYM>/<YYYY-MM-DD>.json where the
    file's trade_date is the as-of date and its records were filed strictly before
    it, so selecting the file with the largest date <= trade_date is point-in-time
    safe.
    """
    d = _FUND_BASE / stmt / symbol
    if not d.is_dir():
        return None
    best_path = None
    for name in os.listdir(d):
        if not name.endswith(".json"):
            continue
        day = name[:-5]
        if day <= trade_date:
            if best_path is None or day > best_path[0]:
                best_path = (day, d / name)
    if best_path is None:
        return None
    try:
        blob = json.loads(best_path[1].read_text(encoding="utf-8"))
    except Exception:
        return None
    section = stmt.replace("get_", "")
    recs = (blob.get("payload", {}) or {}).get(section, {}).get("records", []) or []
    return recs[0] if recs else None


def _num(rec: dict | None, *keys: str) -> float:
    if not rec:
        return float("nan")
    for k in keys:
        v = rec.get(k)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
    return float("nan")


class ClassicalQuantStrategy:
    """Cross-sectional momentum/value/quality composite into the shared allocator.

    Engine contract mirrors DarwinTradeStrategy: __init__(cfg), on_bar(ctx),
    set_daily_output_dir(path). The optimizer is selectable via
    cfg['classical_quant']['optimizer'] in {hrp, min_variance, max_sharpe,
    equal_weight, conf_weighted}; the signal blend weights and long/short fraction
    are also config-driven with sane defaults.
    """

    strategy_name = "classical_quant"

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        from darwintrade.portfolio.allocator import PortfolioAllocator

        cfg = dict(cfg or {})
        self.cfg = cfg
        qcfg = dict(cfg.get("classical_quant", {}) or {})
        self.optimizer = str(qcfg.get("optimizer", "hrp"))
        self.momentum_lookback = int(qcfg.get("momentum_lookback", 60))
        self.momentum_skip = int(qcfg.get("momentum_skip", 5))
        self.long_frac = float(qcfg.get("long_fraction", 0.3))
        self.w_mom = float(qcfg.get("w_momentum", 1.0))
        self.w_val = float(qcfg.get("w_value", 1.0))
        self.w_qual = float(qcfg.get("w_quality", 1.0))
        self.strategy_name = f"classical_quant_{self.optimizer}"

        self.allocator = PortfolioAllocator(
            max_gross_exposure=float(qcfg.get("max_gross_exposure", 0.95)),
            max_single_position=float(qcfg.get("max_single_position", 1.0)),
            min_confidence=float(qcfg.get("min_confidence", 0.0)),
            long_only=bool(qcfg.get("long_only", False)),
        )
        self.symbols: List[str] = list(cfg.get("symbols") or [])
        self.daily_output_dir: Path | None = None
        self._records: list[dict] = []
        # In-memory cache of each symbol's full (date, close) history. Populated
        # once on first use; every get_day_bars call from the data layer costs
        # ~1s even from the local Parquet cache, so re-reading it per bar would
        # dominate runtime (~40 reads/bar). We read once and slice in memory.
        self._close_cache: Dict[str, Any] = {}
        self._cache_loaded = False

    def _load_close_cache(self, ctx: Dict) -> None:
        if self._cache_loaded:
            return
        self._cache_loaded = True
        datasets = ctx.get("datasets")
        if datasets is None:
            return
        import pandas as pd
        syms = list(ctx.get("symbols") or self.symbols)
        # Pull a wide window that covers both evaluation windows plus lookback.
        for sym in syms:
            try:
                bars = datasets.get_day_bars(sym, "2024-09-01", "2026-06-30")
                if bars is None or bars.empty:
                    continue
                b = bars.copy()
                b["date"] = pd.to_datetime(b["date"]).dt.normalize()
                b = b[["date", "close"]].dropna().sort_values("date")
                self._close_cache[sym] = b.reset_index(drop=True)
            except Exception:
                continue

    def set_daily_output_dir(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.daily_output_dir = out

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _closes_before(self, ctx: Dict, symbol: str, trade_date: str) -> list[float]:
        self._load_close_cache(ctx)
        b = self._close_cache.get(symbol)
        if b is None or b.empty:
            return []
        import pandas as pd
        end_d = pd.Timestamp(trade_date).normalize()
        sub = b[b["date"] < end_d]
        return [float(c) for c in sub["close"].tolist() if c == c and c > 0]

    def _momentum(self, closes: list[float]) -> float:
        # 12-1 style: trailing return over the lookback, skipping the most recent
        # `momentum_skip` bars to avoid short-term reversal contamination.
        if len(closes) < self.momentum_skip + 10:
            return float("nan")
        end = closes[-(self.momentum_skip + 1)]
        start_idx = max(0, len(closes) - self.momentum_lookback)
        start = closes[start_idx]
        if start <= 0 or end <= 0:
            return float("nan")
        return math.log(end / start)

    def _value_quality(self, symbol: str, trade_date: str, last_close: float) -> tuple[float, float]:
        """Return (value_score, quality_score) raw levels; NaN when data missing.

        value = earnings yield + book/price (higher = cheaper).
        quality = ROE + ROA (higher = better), lightly de-levered.
        Market cap uses diluted shares * last close (point-in-time price).
        """
        inc = _latest_fundamental_record("get_income_statement", symbol, trade_date)
        bs = _latest_fundamental_record("get_balance_sheet", symbol, trade_date)
        ni = _num(inc, "Net Income", "Net income")
        shares = _num(inc, "Diluted (in shares)", "Weighted average diluted shares",
                      "Basic (in shares)")
        equity = _num(bs, "Stockholders Equity", "Total shareholders' equity")
        assets = _num(bs, "Total assets")

        mktcap = shares * last_close if (shares == shares and last_close > 0) else float("nan")
        earnings_yield = (ni / mktcap) if (ni == ni and mktcap == mktcap and mktcap > 0) else float("nan")
        book_to_price = (equity / mktcap) if (equity == equity and mktcap == mktcap and mktcap > 0) else float("nan")
        roe = (ni / equity) if (ni == ni and equity == equity and equity > 0) else float("nan")
        roa = (ni / assets) if (ni == ni and assets == assets and assets > 0) else float("nan")

        value = _safe_nanmean([earnings_yield, book_to_price])
        quality = _safe_nanmean([roe, roa])
        return float(value), float(quality)

    # ------------------------------------------------------------------
    # Engine entry
    # ------------------------------------------------------------------

    def _returns_history(self, ctx: Dict, symbols: List[str], trade_date: str,
                         lookback: int = 60) -> Dict[str, list[float]]:
        self._load_close_cache(ctx)
        import pandas as pd
        end_d = pd.Timestamp(trade_date).normalize()
        out: Dict[str, list[float]] = {}
        for sym in symbols:
            b = self._close_cache.get(sym)
            if b is None or b.empty:
                continue
            closes = [float(c) for c in b[b["date"] < end_d]["close"].tolist()
                      if c == c and c > 0]
            if len(closes) < 2:
                continue
            rets = [math.log(closes[i] / closes[i - 1])
                    for i in range(1, len(closes))
                    if closes[i - 1] > 0 and closes[i] > 0]
            if rets:
                out[sym] = rets[-lookback:]
        return out

    def on_bar(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        import pandas as pd
        from darwintrade.core.contracts import (
            AssetSignal, MemoryInfluence, PortfolioState, RegimeState,
        )
        from darwintrade.pipeline import _execution_plan_to_engine_payload

        raw_date = ctx.get("date") or ctx.get("trade_date")
        trade_date = pd.Timestamp(raw_date).strftime("%Y-%m-%d") if raw_date else ""
        open_map = dict(ctx.get("open_map") or ctx.get("open_price_map") or {})
        portfolio = ctx.get("portfolio")
        symbols = [s for s in (ctx.get("symbols") or self.symbols)
                   if float(open_map.get(s) or 0.0) > 0.0]

        total_equity = float(ctx.get("equity_for_sizing")
                             or getattr(portfolio, "total_equity", 0.0) or 0.0)
        if total_equity <= 0.0:
            total_equity = float(getattr(portfolio, "cash", 0.0) or 0.0)
        cash = float(getattr(portfolio, "cash", total_equity) or total_equity)

        positions: Dict[str, float] = {}
        for sym, pos in (getattr(portfolio, "positions", {}) or {}).items():
            shares = float(getattr(pos, "shares", 0.0) or 0.0)
            if abs(shares) <= 0.0:
                continue
            price = float(open_map.get(sym) or getattr(pos, "avg_price", 0.0) or 0.0)
            positions[sym] = shares * price

        # --- cross-sectional signals ---
        mom_raw, val_raw, qual_raw = {}, {}, {}
        for sym in symbols:
            closes = self._closes_before(ctx, sym, trade_date)
            mom_raw[sym] = self._momentum(closes)
            last_close = closes[-1] if closes else float(open_map.get(sym) or 0.0)
            v, q = self._value_quality(sym, trade_date, last_close)
            val_raw[sym], qual_raw[sym] = v, q

        z_mom, z_val, z_qual = _zscore(mom_raw), _zscore(val_raw), _zscore(qual_raw)
        composite: Dict[str, float] = {}
        for sym in symbols:
            score = (self.w_mom * z_mom.get(sym, 0.0)
                     + self.w_val * z_val.get(sym, 0.0)
                     + self.w_qual * z_qual.get(sym, 0.0))
            composite[sym] = math.tanh(score / 3.0)

        # --- pick long/short cross-section by rank ---
        ranked = sorted(symbols, key=lambda s: composite[s], reverse=True)
        n = len(ranked)
        k = max(1, int(round(self.long_frac * n))) if n else 0
        longs = set(ranked[:k])
        shorts = set() if self.allocator.long_only else set(ranked[n - k:])
        # a symbol cannot be both when k*2 > n; longs take precedence
        shorts -= longs

        signals: list[AssetSignal] = []
        for sym in symbols:
            sc = composite[sym]
            if sym in longs:
                direction = "long"
            elif sym in shorts:
                direction = "short"
            else:
                direction = "hold"
            signals.append(AssetSignal(
                symbol=sym,
                trade_date=trade_date,
                direction=direction,
                confidence=min(1.0, abs(sc)),
                thesis=f"mom={z_mom.get(sym,0):.2f} val={z_val.get(sym,0):.2f} "
                       f"qual={z_qual.get(sym,0):.2f} composite={sc:.3f}",
                risk_score=0.0,
                metadata={"optimizer": self.optimizer},
            ))

        portfolio_state = PortfolioState(
            trade_date=trade_date, cash=cash, total_equity=total_equity,
            positions=positions,
            prices={s: float(open_map.get(s) or 0.0) for s in symbols},
        )
        regime = RegimeState(regime="n/a", confidence=0.0, trade_date=trade_date)
        mem = MemoryInfluence(preferred_optimizer=self.optimizer, source_layer="none")
        returns_history = self._returns_history(ctx, symbols, trade_date)

        plan = self.allocator.build_plan(
            trade_date=trade_date,
            signals=signals,
            portfolio_state=portfolio_state,
            regime=regime,
            memory_influence=mem,
            returns_history=returns_history,
        )

        payload = _execution_plan_to_engine_payload(plan, portfolio_state)
        portfolio_dict = {"equity": total_equity, "positions": {
            sym: {"shares": float(getattr(pos, "shares", 0.0) or 0.0),
                  "position_value": float(getattr(pos, "shares", 0.0) or 0.0)
                  * float(open_map.get(sym) or getattr(pos, "avg_price", 0.0) or 0.0)}
            for sym, pos in (getattr(portfolio, "positions", {}) or {}).items()
        }}

        orders: list[dict] = []
        for decision in payload:
            sym = str(decision.get("symbol") or "")
            if sym in {"", "__PORTFOLIO__"} or str(decision.get("action")) == "hold":
                self._record(trade_date, decision)
                continue
            orders.extend(plan_orders(
                decision=decision,
                snapshot_price=float(open_map.get(sym) or 0.0),
                cfg=self.cfg,
                portfolio=portfolio_dict,
            ))
            self._record(trade_date, decision)
        return orders

    def _record(self, trade_date: str, decision: dict) -> None:
        if self.daily_output_dir is None:
            return
        path = self.daily_output_dir / f"{self.strategy_name}_portfolio_decisions.daily.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"date": trade_date, **decision}, ensure_ascii=True) + "\n")
        except Exception:
            pass


# Thin per-optimizer subclasses so the baseline runner (which calls
# strategy_cls(config)) can select the allocator without extra config plumbing.
def _with_optimizer(cfg: Dict[str, Any] | None, optimizer: str) -> Dict[str, Any]:
    cfg = dict(cfg or {})
    q = dict(cfg.get("classical_quant", {}) or {})
    q["optimizer"] = optimizer
    cfg["classical_quant"] = q
    return cfg


class ClassicalHRPStrategy(ClassicalQuantStrategy):
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        super().__init__(_with_optimizer(cfg, "hrp"))


class ClassicalMaxSharpeStrategy(ClassicalQuantStrategy):
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        super().__init__(_with_optimizer(cfg, "max_sharpe"))


class ClassicalEqualWeightStrategy(ClassicalQuantStrategy):
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        super().__init__(_with_optimizer(cfg, "equal_weight"))
