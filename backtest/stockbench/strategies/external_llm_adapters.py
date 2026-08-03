from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from threading import local
from typing import Any, Dict, List
import hashlib
import httpx
import importlib.util
import json
import math
import random
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

from backtest.stockbench.core.executor import plan_orders

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = ROOT / "external"
CACHE_ROOT = ROOT / "storage" / "cache" / "external_strategy_adapters"
FINAGENT_ROOT = EXTERNAL_ROOT / "FinAgent"


def _load_repo_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)
    api_key = str(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = str(os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL") or "").strip()
    model = str(os.getenv("LLM_MODEL") or os.getenv("MODEL") or "").strip()
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    if api_key and not os.getenv("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = api_key
    if api_key and not os.getenv("OA_OPENAI_KEY"):
        os.environ["OA_OPENAI_KEY"] = api_key
    if base_url and not os.getenv("BASE_URL"):
        os.environ["BASE_URL"] = base_url
    if base_url and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = base_url
    if base_url and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = base_url
    if model and not os.getenv("MODEL"):
        os.environ["MODEL"] = model


@contextmanager
def _push_import_path(path: Path):
    inserted = False
    as_str = str(path)
    if as_str not in sys.path:
        sys.path.insert(0, as_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(as_str)
            except ValueError:
                pass


@contextmanager
def _push_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass
class _Signal:
    symbol: str
    signal: str
    reason: str


FINAGENT_XML_RETRY_ATTEMPTS = 3

# bs4's deepcopy is not thread-safe when copying shared BeautifulSoup
# objects built from shared memory modules. Serialize those calls.
import threading as _threading
_FINAGENT_BS4_DEEPCOPY_LOCK = _threading.Lock()

# API-call retry policy: fixed 1-second interval, 100 attempts. Applied to
# FinAgent's OpenAIProvider.create_completion so the agent survives transient
# provider outages during long backtests. Interval kept tight so warm-up
# under high concurrency (100 workers) doesn't stall on transient 429s.
EXTERNAL_LLM_MAX_ATTEMPTS = 100
EXTERNAL_LLM_RETRY_INTERVAL_SEC = 1.0


def _call_with_retry(fn, *, label: str):
    """Fixed-interval retry loop.

    fn is a zero-arg callable that performs one API attempt and either
    returns the value or raises.  We swallow only transient exceptions;
    unexpected errors propagate on the final attempt.
    """
    import time as _time

    max_attempts = EXTERNAL_LLM_MAX_ATTEMPTS
    interval = EXTERNAL_LLM_RETRY_INTERVAL_SEC
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            try:
                import sys as _sys
                _sys.stderr.write(
                    f"[external-retry] {label} attempt {attempt}/{max_attempts} "
                    f"failed: {type(exc).__name__}: {exc}. Sleeping {interval}s.\n"
                )
                _sys.stderr.flush()
            except Exception:
                pass
            _time.sleep(interval)
    assert last_exc is not None
    raise last_exc


def _canonical_strategy_signal(action: Any) -> str | None:
    if isinstance(action, (int, float)) and not isinstance(action, bool):
        if action > 0:
            return "BUY"
        if action < 0:
            return "SELL"
        return "HOLD"
    text = str(action or "").strip().upper()
    if not text:
        return None
    decision_aliases = {
        "BUY": ["BUY", "BULLISH", "LONG", "OVERWEIGHT", "ACCUMULATE", "ADD", "INCREASE", "GO LONG"],
        "SELL": ["SELL", "BEARISH", "SHORT", "UNDERWEIGHT", "REDUCE", "DECREASE", "EXIT", "GO SHORT"],
        "HOLD": ["HOLD", "NEUTRAL", "NO ACTION", "WAIT", "STAY PUT", "MAINTAIN", "REFUSE", "ABSTAIN", "DECLINE", "PASS", "NO TRADE"],
    }
    for canonical, aliases in decision_aliases.items():
        if any(text == alias or alias in text for alias in aliases):
            return canonical
    return None


def _finagent_high_level_fallback_payload(response_text: Any) -> dict[str, str]:
    fallback_reasoning = str(response_text or "").strip() or "Fallback high-level reflection after XML parse failure."
    return {
        "reasoning": fallback_reasoning,
        "improvement": "",
        "summary": fallback_reasoning,
        "query": fallback_reasoning[:500],
    }


def _finagent_decision_fallback_payload(response_text: Any) -> dict[str, str]:
    normalized = str(response_text or "")
    action = _canonical_strategy_signal(normalized) or "HOLD"
    fallback_reasoning = normalized.strip() or f"Fallback parsed action {action} after XML parse failure."
    return {"action": action, "reasoning": fallback_reasoning}


def _finagent_prompt_fallback_payload(check_keys: list[str], response_text: Any) -> dict[str, str]:
    fallback_reasoning = str(response_text or "").strip() or "Fallback FinAgent prompt response after XML parse failure."
    payload: dict[str, str] = {}
    for key in check_keys:
        if key == "query":
            payload[key] = fallback_reasoning[:500]
        elif key == "improvement":
            payload[key] = ""
        elif key in {"action", "decision"}:
            payload[key] = _canonical_strategy_signal(response_text) or "HOLD"
        else:
            payload[key] = fallback_reasoning
    if "reasoning" in check_keys and "reasoning" not in payload:
        payload["reasoning"] = fallback_reasoning
    return payload


class _FinAgentDatasetShim:
    def __init__(self, *, prices: pd.DataFrame, news: pd.DataFrame, symbol: str) -> None:
        self.assets = [symbol]
        self.prices = {symbol: prices}
        self.news = {symbol: news}
        self.guidances = None
        self.sentiments = None
        self.economics = None


class _NoopFinAgentPlots:
    def plot_kline(self, state, info, save_dir, mode="train"):
        del state, info, save_dir, mode
        return None

    def plot_trading(self, records, info, save_dir):
        del records, info, save_dir
        return None


class _BaseExternalStrategy:
    # Native long-short action space? FinAgent's env.action_map = {SELL:-1, HOLD:0,
    # BUY:+1}, so it natively supports shorts. Subclasses set this to True to enable
    # long-short bridging (SELL → open short at equal weight); leave False for
    # long-only agents (e.g. TradingAgents).
    allow_short: bool = False

    def __init__(self, cfg: Dict) -> None:
        _load_repo_env()
        self.cfg = cfg or {}
        self.total_equity = float(
            (self.cfg or {}).get("portfolio", {}).get("total_cash", 100000.0)
        )
        self.records: list[dict[str, Any]] = []
        self.daily_output_dir: Path | None = None
        # Process-wide symbol cache. Was thread-local, which broke re-use
        # across bar-N ThreadPoolExecutor fanouts (each worker got a fresh
        # empty dict → warm-up ran every bar). Guarded by a lock for the
        # {check, populate} sequence.
        self._state = local()  # legacy attrs; symbol cache is separate
        self._symbol_cache: dict[str, Any] = {}
        self._symbol_cache_lock = _threading.Lock()
        self._record_index: dict[tuple[str, str, str], int] = {}
        self._signal_cache: dict[tuple[str, str], list[_Signal]] = {}
        self._artifact_root = CACHE_ROOT / self.__class__.__name__.lower()
        self._artifact_root.mkdir(parents=True, exist_ok=True)

    def _run_id(self, ctx: Dict[str, Any]) -> str:
        return str(ctx.get("run_id") or "default").replace("/", "_").replace("\\", "_")

    def _parallel_workers(self) -> int:
        """Symbol-level thread pool size.

        Explicit knob so a single bar can dispatch 20 concurrent LLM
        interactions for a 20-symbol universe. Overridable per class
        via cfg[<class>]['parallel_workers'] or env var
        EXTERNAL_STRATEGY_PARALLEL_WORKERS. Default = 20 to match the
        typical stockbench universe.
        """
        cls_key = self.__class__.__name__.lower()
        cfg_bucket = (self.cfg or {}).get(cls_key, {}) if isinstance(self.cfg, dict) else {}
        configured = None
        if isinstance(cfg_bucket, dict):
            configured = cfg_bucket.get("parallel_workers")
        if configured is None:
            configured = os.getenv("EXTERNAL_STRATEGY_PARALLEL_WORKERS")
        if configured is None:
            configured = 20
        try:
            return max(int(configured), 1)
        except (TypeError, ValueError):
            return 20

    def _symbol_state(self) -> dict[str, Any]:
        # Process-wide (see __init__). Not thread-local: with a fresh
        # ThreadPoolExecutor per bar, thread-local storage would give each
        # worker an empty dict and force warm-up to re-run every bar.
        return self._symbol_cache

    def _acquire_symbol_lock(self, key: str) -> "_threading.Lock":
        """Per-key lock ensuring only one thread warms up a given symbol.

        Bar-1 fanout submits 20 concurrent _initialize_symbol calls, one per
        symbol. Without a per-symbol lock the first bar would race safely
        (each key unique), but any bar where two calls target the same key
        (e.g. resume path) would double-train. Cheap insurance either way.
        """
        with self._symbol_cache_lock:
            per_key = self._symbol_cache.setdefault("__locks__", {})
            lock = per_key.get(key)
            if lock is None:
                lock = _threading.Lock()
                per_key[key] = lock
            return lock

    def set_daily_output_dir(self, output_dir: str | Path) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.daily_output_dir = out_dir

    def _append_jsonl_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_jsonl_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _dependency_issues(self) -> list[str]:
        return []

    def preflight_readiness(self) -> dict[str, Any]:
        issues: list[str] = []
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_API_KEY"):
            issues.append("llm_api_key_missing")
        if not os.getenv("MODEL") and not os.getenv("LLM_MODEL"):
            issues.append("llm_model_missing")
        issues.extend(self._dependency_issues())
        return {
            "ready": not issues,
            "issues": issues,
            "strategy": self.__class__.__name__,
        }

    def _price_series(self, bars: pd.DataFrame) -> pd.Series:
        if bars.empty:
            return pd.Series(dtype=float)
        data = bars.copy()
        if "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"])
        elif "date" in data.columns:
            ts = pd.to_datetime(data["date"])
        else:
            ts = pd.to_datetime(data.index)
        ts_index = pd.DatetimeIndex(ts)
        price_col = "adj_close" if "adj_close" in data.columns else "close"
        return pd.Series(data[price_col].astype(float).values, index=ts_index.date)

    def _max_positions(self, symbol_count: int) -> int:
        configured = (
            (self.cfg or {}).get("backtest", {}).get("max_positions")
            or (self.cfg or {}).get("risk", {}).get("max_positions")
            or symbol_count
            or 1
        )
        return max(int(configured), 1)

    def _estimated_buy_fee_rate(self) -> float:
        backtest_cfg = (self.cfg or {}).get("backtest", {}) or {}
        commission_bps = float(backtest_cfg.get("commission_bps", 0.0) or 0.0)
        slippage_bps = float(backtest_cfg.get("slippage_bps", 0.0) or 0.0)
        return max(commission_bps + slippage_bps, 0.0) / 10_000.0

    def _quantize_buy_notional(self, cash_budget: float, price: float) -> float:
        if cash_budget <= 0 or price <= 0:
            return 0.0
        raw_shares = Decimal(str(cash_budget / price))
        shares = raw_shares.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if shares <= 0:
            return 0.0
        notional = shares * Decimal(str(price))
        return float(notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN))

    def _target_cash_amounts(
        self,
        *,
        total_equity: float,
        symbols: list[str],
        signals: list[_Signal],
        open_map: dict[str, float],
        portfolio: Any,
    ) -> list[dict[str, Any]]:
        max_positions = self._max_positions(len(symbols))
        held_symbols = {
            ticker
            for ticker, pos in ((portfolio.positions.items() if portfolio else []))
            if float(getattr(pos, "shares", 0.0) or 0.0) > 0
        }
        current_held_count = len(held_symbols)
        remaining_new_slots = max(max_positions - current_held_count, 0)
        eligible_buy_symbols: set[str] = set()
        for signal in signals:
            if signal.signal != "BUY":
                continue
            if signal.symbol in held_symbols:
                eligible_buy_symbols.add(signal.symbol)
                continue
            if remaining_new_slots > 0:
                eligible_buy_symbols.add(signal.symbol)
                remaining_new_slots -= 1
        buy_slots = max(len(eligible_buy_symbols), 1)
        fee_rate = self._estimated_buy_fee_rate()
        current_total_value = sum(
            float(getattr(pos, "shares", 0.0) or 0.0) * float(open_map.get(ticker) or getattr(pos, "avg_price", 0.0) or 0.0)
            for ticker, pos in (portfolio.positions.items() if portfolio else [])
        )
        available_cash = float(getattr(portfolio, "cash", max(total_equity - current_total_value, 0.0)) or 0.0)
        per_buy_amount = available_cash / float(buy_slots)
        if fee_rate > 0:
            per_buy_amount = per_buy_amount / (1.0 + fee_rate)
        decisions: list[dict[str, Any]] = []
        for signal in signals:
            current_position = portfolio.positions.get(signal.symbol) if portfolio else None
            current_value = (
                float(getattr(current_position, "shares", 0.0) or 0.0)
                * float(open_map.get(signal.symbol) or 0.0)
            ) if current_position else 0.0
            if signal.signal == "BUY" and signal.symbol in eligible_buy_symbols:
                action = "increase"
                cash_change = self._quantize_buy_notional(
                    per_buy_amount,
                    float(open_map.get(signal.symbol) or 0.0),
                )
                target_cash_amount = cash_change
            elif signal.signal == "BUY":
                action = "hold"
                target_cash_amount = current_value
                cash_change = 0.0
            elif signal.signal == "SELL":
                action = "close"
                target_cash_amount = 0.0
                cash_change = -current_value if current_value > 0 else 0.0
            else:
                action = "hold"
                target_cash_amount = current_value
                cash_change = 0.0
            decisions.append({
                "symbol": signal.symbol,
                "action": action,
                "target_cash_amount": round(float(target_cash_amount), 2),
                "cash_change": round(float(cash_change), 2),
                "confidence": 0.6,
                "reason": signal.reason,
                "signal": signal.signal,
            })
        return decisions

    def _long_short_decisions(
        self,
        *,
        total_equity: float,
        symbols: list[str],
        signals: list["_Signal"],
        open_map: dict[str, float],
        portfolio: Any,
    ) -> list[dict[str, Any]]:
        """Equal-weight long-short sleeve bridge.

        Each symbol's single-name signal maps to one leg:
        BUY  → hold or open a long-weight position (share of equity / N)
        SELL → hold or open a short-weight position of the same size
        HOLD → close any existing exposure
        Position changes are emitted as explicit sim_orders so the engine
        sees buy / sell / sell_short / buy_to_cover sides.
        """
        max_positions = self._max_positions(len(symbols))
        # Equal weight per leg. N is the universe size (capped at max_positions).
        n_legs = max(min(len(symbols), max_positions), 1)
        per_leg_notional = float(total_equity) / float(n_legs)
        signal_map = {str(s.symbol): s for s in signals}
        decisions: list[dict[str, Any]] = []
        for symbol in symbols:
            sig = signal_map.get(symbol)
            direction = 0
            reason = "no_signal"
            if sig is not None:
                reason = sig.reason
                if sig.signal == "BUY":
                    direction = 1
                elif sig.signal == "SELL":
                    direction = -1
            ref_price = float(open_map.get(symbol) or 0.0)
            if ref_price <= 0.0:
                decisions.append({
                    "symbol": symbol,
                    "action": "hold",
                    "target_cash_amount": 0.0,
                    "cash_change": 0.0,
                    "confidence": 0.6,
                    "reason": f"{reason}:no_ref_price",
                    "signal": sig.signal if sig else "HOLD",
                })
                continue
            pos = portfolio.positions.get(symbol) if portfolio else None
            cur_shares = float(getattr(pos, "shares", 0.0) or 0.0)
            cur_notional = cur_shares * ref_price
            target_notional = direction * per_leg_notional
            delta_notional = target_notional - cur_notional
            delta_shares = round(delta_notional / ref_price, 2)
            if delta_shares == 0.0:
                decisions.append({
                    "symbol": symbol,
                    "action": "hold",
                    "target_cash_amount": round(abs(target_notional), 2),
                    "cash_change": 0.0,
                    "confidence": 0.6,
                    "reason": reason,
                    "signal": sig.signal if sig else "HOLD",
                })
                continue
            # Decompose net delta into legal legs across the long/short seam.
            legs: list[tuple[str, float]] = []
            if cur_shares > 0 and cur_shares + delta_shares < 0:
                # Cross from long into short: close the long first, then open a short.
                legs.append(("sell", cur_shares))
                legs.append(("sell_short", -(cur_shares + delta_shares)))
            elif cur_shares < 0 and cur_shares + delta_shares > 0:
                # Cross from short back into long: cover then buy fresh.
                legs.append(("buy_to_cover", -cur_shares))
                legs.append(("buy", cur_shares + delta_shares))
            elif delta_shares > 0:
                # Adding exposure on the long side (or closing a short).
                if cur_shares < 0:
                    legs.append(("buy_to_cover", min(delta_shares, -cur_shares)))
                    remainder = delta_shares - min(delta_shares, -cur_shares)
                    if remainder > 0:
                        legs.append(("buy", remainder))
                else:
                    legs.append(("buy", delta_shares))
            else:  # delta_shares < 0
                abs_delta = -delta_shares
                if cur_shares > 0:
                    legs.append(("sell", min(abs_delta, cur_shares)))
                    remainder = abs_delta - min(abs_delta, cur_shares)
                    if remainder > 0:
                        legs.append(("sell_short", remainder))
                else:
                    legs.append(("sell_short", abs_delta))
            for leg_side, leg_qty in legs:
                if leg_qty <= 0.0:
                    continue
                sim_order = {
                    "symbol": symbol,
                    "side": leg_side,
                    "quantity": round(float(leg_qty), 2),
                    "reference_price": ref_price,
                    "schedule": {"slices": 1},
                    "constraints": {},
                }
                decisions.append({
                    "symbol": symbol,
                    "action": "sim_order",
                    "target_cash_amount": round(abs(target_notional), 2),
                    "cash_change": round(delta_notional, 2),
                    "confidence": 0.6,
                    "reason": reason,
                    "signal": sig.signal if sig else "HOLD",
                    "sim_order": sim_order,
                })
        return decisions

    def on_bar(self, ctx: Dict[str, Any]) -> List[Dict]:
        open_map = dict(ctx.get("open_map") or ctx.get("open_price_map") or {})
        if not open_map:
            return []
        portfolio = ctx.get("portfolio")
        total_equity = float(
            ctx.get("equity_for_sizing")
            or getattr(portfolio, "total_equity", 0.0)
            or self.total_equity
        )
        symbols = [
            symbol
            for symbol in (ctx.get("symbols") or [])
            if float(open_map.get(symbol) or 0.0) > 0
        ]
        if not symbols:
            return []
        current_date = ctx["date"].strftime("%Y-%m-%d")
        run_id = self._run_id(ctx)
        cache_key = (run_id, current_date)
        cached_signals = self._signal_cache.get(cache_key)
        if cached_signals is None:
            raw_signals = self._signals_for_day(ctx, symbols)
            self._signal_cache[cache_key] = [
                _Signal(symbol=item.symbol, signal=item.signal, reason=item.reason)
                for item in raw_signals
            ]
        else:
            signal_map = {str(item.symbol): item for item in cached_signals}
            raw_signals = [signal_map[symbol] for symbol in symbols if symbol in signal_map]
        if self.allow_short:
            decisions = self._long_short_decisions(
                total_equity=total_equity,
                symbols=symbols,
                signals=raw_signals,
                open_map=open_map,
                portfolio=portfolio,
            )
        else:
            decisions = self._target_cash_amounts(
                total_equity=total_equity,
                symbols=symbols,
                signals=raw_signals,
                open_map=open_map,
                portfolio=portfolio,
            )
        orders: list[dict[str, Any]] = []
        for decision in decisions:
            planned = plan_orders(
                decision=decision,
                snapshot_price=float(open_map.get(decision["symbol"]) or 0.0),
                cfg=self.cfg,
                portfolio={
                    "equity": total_equity,
                    "positions": {
                        ticker: {
                            "shares": float(getattr(pos, "shares", 0.0) or 0.0),
                            "position_value": float(getattr(pos, "shares", 0.0) or 0.0)
                            * float(open_map.get(ticker) or getattr(pos, "avg_price", 0.0) or 0.0),
                        }
                        for ticker, pos in (portfolio.positions.items() if portfolio else [])
                    },
                },
            )
            orders.extend(planned)
            record = {"date": current_date, **decision}
            record_key = (run_id, current_date, str(decision["symbol"]))
            existing_index = self._record_index.get(record_key)
            if existing_index is None:
                self._record_index[record_key] = len(self.records)
                self.records.append(record)
            else:
                self.records[existing_index] = record
        if self.daily_output_dir is not None:
            self._write_jsonl_rows(
                self.daily_output_dir / f"{self.__class__.__name__.lower()}_decisions.daily.jsonl",
                self.records,
            )
        return orders

    def _signals_for_day(self, ctx: Dict[str, Any], symbols: list[str]) -> list[_Signal]:
        raise NotImplementedError


class FinAgentStrategy(_BaseExternalStrategy):
    # FinAgent's env.action_map = {"SELL": -1, "HOLD": 0, "BUY": 1} → run L/S.
    allow_short: bool = True

    def _parallel_workers(self) -> int:
        # bs4 deepcopy + XML parsing are protected by their own module-level
        # locks. Monkey-patches use idempotent `_stockbench_*_guard` gates
        # and only assign identical closures (CPython setattr is GIL-atomic
        # → benign race). Uncapped concurrency is safe. Override via
        # FINAGENT_PARALLEL_WORKERS env var.
        env_override = os.getenv("FINAGENT_PARALLEL_WORKERS")
        if env_override is not None:
            try:
                return max(int(env_override), 1)
            except (TypeError, ValueError):
                pass
        return super()._parallel_workers()

    def _dependency_issues(self) -> list[str]:
        issues: list[str] = []
        if importlib.util.find_spec("colorlog") is None:
            issues.append("missing_python_package:colorlog")
        if importlib.util.find_spec("backtrader") is None:
            issues.append("missing_python_package:backtrader")
        if importlib.util.find_spec("datasets") is None:
            issues.append("missing_python_package:datasets")
        if importlib.util.find_spec("polars") is None:
            issues.append("missing_python_package:polars")
        if importlib.util.find_spec("faiss") is None:
            issues.append("missing_python_package:faiss")
        return issues

    def _patch_finagent_runtime(self) -> None:
        with _push_import_path(FINAGENT_ROOT), _push_cwd(FINAGENT_ROOT):
            from finagent.provider.provider import OpenAIProvider as _FinAgentProvider
            from finagent.prompt.custom import Prompt as _FinAgentPrompt
            from finagent import utils as _finagent_utils
            from finagent.asset import ASSET as _FINAGENT_ASSET
            from finagent.prompt.trading import decision as _finagent_decision_module
            from finagent.prompt.trading import high_level_reflection as _finagent_high_module
            from finagent.prompt.trading import low_level_reflection as _finagent_low_module
            from finagent.prompt.trading import latest_market_intelligence_summary as _finagent_latest_module
            from finagent.prompt.trading import past_market_intelligence_summary as _finagent_past_module
            from finagent.prompt.trading.decision import DecisionTrading as _DecisionTrading
            from finagent.prompt.trading.high_level_reflection import HighLevelReflectionTrading as _HighLevelReflectionTrading
            from finagent.prompt.trading.low_level_reflection import LowLevelReflectionTrading as _LowLevelReflectionTrading
            from tools import main_strategy as _finagent_main_strategy
            from finagent.utils import parse_semi_formatted_xml as _parse_finagent_xml

            # bs4's deepcopy is not thread-safe when 20 worker threads
            # simultaneously deep-copy shared BeautifulSoup module objects
            # stored in finagent.asset.base.AssetPool. Patch get_module to
            # serialize deepcopy under the process-level lock we defined
            # at module load time.
            from finagent.asset.base import Asset as _AssetPool
            if not getattr(_AssetPool, "_stockbench_deepcopy_guard", False):
                _orig_get_module = _AssetPool.get_module
                _lock_ref = _FINAGENT_BS4_DEEPCOPY_LOCK

                def _threadsafe_get_module(self, name=None):
                    with _lock_ref:
                        return _orig_get_module(self, name)

                _AssetPool.get_module = _threadsafe_get_module
                _AssetPool._stockbench_deepcopy_guard = True

            if not getattr(_FINAGENT_ASSET, "_stockbench_asset_info_fallback", False):
                _original_get_asset_info = _FINAGENT_ASSET.get_asset_info

                def _fallback_get_asset_info(symbol=None):
                    symbol_text = str(symbol or "").strip().upper()
                    if symbol_text in _FINAGENT_ASSET.assets.get("asset_infos", {}):
                        return _original_get_asset_info(symbol_text)
                    return {
                        "symbol": symbol_text,
                        "companyName": symbol_text or "UNKNOWN",
                        "exchange": "UNKNOWN",
                        "sector": "Unknown",
                        "industry": "Unknown",
                        "description": f"{symbol_text or 'UNKNOWN'} equity from the stockbench universe.",
                    }

                _FINAGENT_ASSET.get_asset_info = _fallback_get_asset_info
                _FINAGENT_ASSET._stockbench_asset_info_fallback = True

            if not getattr(_finagent_main_strategy, "_stockbench_utf8_file_guard", False):
                def _safe_save_html(html, path):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(html))

                _finagent_utils.save_html = _safe_save_html
                _finagent_main_strategy.save_html = _safe_save_html
                _finagent_decision_module.save_html = _safe_save_html
                _finagent_high_module.save_html = _safe_save_html
                _finagent_low_module.save_html = _safe_save_html
                _finagent_latest_module.save_html = _safe_save_html
                _finagent_past_module.save_html = _safe_save_html
                _finagent_main_strategy._stockbench_utf8_file_guard = True

            if not getattr(_FinAgentProvider, "_stockbench_embedding_fallback", False):
                def _fallback_embed_documents(self, texts):
                    vectors = []
                    for text in texts:
                        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
                        seed = int.from_bytes(digest[:8], "big", signed=False)
                        rng = random.Random(seed)
                        vectors.append([rng.uniform(-1.0, 1.0) for _ in range(1536)])
                    return vectors

                def _fallback_embed_query(self, text):
                    return _fallback_embed_documents(self, [text])[0]

                def _fallback_get_embedding_dim(self):
                    return 1536

                _FinAgentProvider.embed_documents = _fallback_embed_documents
                _FinAgentProvider.embed_query = _fallback_embed_query
                _FinAgentProvider.get_embedding_dim = _fallback_get_embedding_dim
                _FinAgentProvider._stockbench_embedding_fallback = True

            if not getattr(_FinAgentProvider, "_stockbench_completion_guard", False):
                _original_create_completion = _FinAgentProvider.create_completion

                def _guarded_create_completion(self, *args, **kwargs):
                    def _do_call():
                        return _original_create_completion(self, *args, **kwargs)

                    response_text, info = _call_with_retry(
                        _do_call,
                        label=f"finagent:create_completion:{getattr(self, 'llm_model', '')}",
                    )
                    if not response_text:
                        return "<output></output>", info
                    if not isinstance(response_text, str):
                        response_text = str(response_text)
                    if "<output" not in response_text:
                        escaped = response_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        response_text = f"<output><string name=\"reasoning\">{escaped}</string></output>"
                    return response_text, info

                _FinAgentProvider.create_completion = _guarded_create_completion
                _FinAgentProvider._stockbench_completion_guard = True

            # Also raise the OpenAIProvider's inner backoff retries so
            # single-attempt failures inside its @backoff loop don't burn
            # our outer budget instantly.
            try:
                if hasattr(_FinAgentProvider, "retries"):
                    _FinAgentProvider.retries = EXTERNAL_LLM_MAX_ATTEMPTS
            except Exception:
                pass

            if not getattr(_FinAgentPrompt, "_stockbench_prompt_guard", False):
                def _guarded_get_response_dict(self, provider, model, messages, check_keys=["decision", "reasoning"]):
                    response, info = provider.create_completion(messages, model=model)
                    try:
                        response_dict, soup = _parse_finagent_xml(response)
                    except Exception:
                        from bs4 import BeautifulSoup
                        fallback_soup = BeautifulSoup("<output></output>", "html.parser")
                        return _finagent_prompt_fallback_payload(list(check_keys or []), response), fallback_soup
                    for key in check_keys:
                        if key not in response_dict:
                            response_dict[key] = f"No {key} in response"
                    return response_dict, soup

                _FinAgentPrompt.get_response_dict = _guarded_get_response_dict
                _FinAgentPrompt._stockbench_prompt_guard = True

            if not getattr(_LowLevelReflectionTrading, "_stockbench_low_level_params_guard", False):
                _original_convert_low = _LowLevelReflectionTrading.convert_to_params

                def _guarded_convert_low(self, state, info, params, memory=None, provider=None, diverse_query=None):
                    try:
                        return _original_convert_low(self, state, info, params, memory=memory, provider=provider, diverse_query=diverse_query)
                    except Exception:
                        fallback = dict(params or {})
                        fallback.update({
                            "short_term_past_price_movement": "unknown",
                            "medium_term_past_price_movement": "unknown",
                            "long_term_past_price_movement": "unknown",
                            "short_term_past_date_range": int(getattr(self, "short_term_past_date_range", 1) or 1),
                            "medium_term_past_date_range": int(getattr(self, "medium_term_past_date_range", 7) or 7),
                            "long_term_past_date_range": int(getattr(self, "long_term_past_date_range", 14) or 14),
                            "short_term_next_price_movement": "unknown",
                            "medium_term_next_price_movement": "unknown",
                            "long_term_next_price_movement": "unknown",
                            "short_term_next_date_range": int(getattr(self, "short_term_next_date_range", 1) or 1),
                            "medium_term_next_date_range": int(getattr(self, "medium_term_next_date_range", 7) or 7),
                            "long_term_next_date_range": int(getattr(self, "long_term_next_date_range", 14) or 14),
                        })
                        return fallback

                _LowLevelReflectionTrading.convert_to_params = _guarded_convert_low
                _LowLevelReflectionTrading._stockbench_low_level_params_guard = True

            if not getattr(_HighLevelReflectionTrading, "_stockbench_high_reflection_retry_guard", False):
                def _guarded_high_level_get_response_dict(self, provider, model, messages, check_keys=None):
                    required_keys = list(check_keys or ["reasoning", "improvement", "summary", "query"])
                    last_response = ""
                    for _ in range(FINAGENT_XML_RETRY_ATTEMPTS):
                        try:
                            response, _info = provider.create_completion(messages, model=model)
                            last_response = response or ""
                            response_dict, soup = _parse_finagent_xml(response)
                            if all(key in response_dict and response_dict.get(key) not in (None, "", f"No {key} in response") for key in required_keys):
                                return response_dict, soup
                        except Exception:
                            pass
                    from bs4 import BeautifulSoup
                    fallback_soup = BeautifulSoup("<output></output>", "html.parser")
                    return _finagent_high_level_fallback_payload(last_response), fallback_soup

                _HighLevelReflectionTrading.get_response_dict = _guarded_high_level_get_response_dict
                _HighLevelReflectionTrading._stockbench_high_reflection_retry_guard = True

            if not getattr(_DecisionTrading, "_stockbench_decision_retry_guard", False):
                def _guarded_decision_get_response_dict(self, provider, model, messages, check_keys=None):
                    required_keys = list(check_keys or ["action", "reasoning"])
                    last_response = ""
                    for _ in range(FINAGENT_XML_RETRY_ATTEMPTS):
                        try:
                            response, _info = provider.create_completion(messages, model=model)
                            last_response = response or ""
                            response_dict, soup = _parse_finagent_xml(response)
                            if all(key in response_dict and response_dict.get(key) not in (None, "", f"No {key} in response") for key in required_keys):
                                normalized_action = _canonical_strategy_signal(response_dict.get("action")) or "HOLD"
                                response_dict["action"] = normalized_action
                                return response_dict, soup
                        except Exception:
                            pass
                    from bs4 import BeautifulSoup
                    fallback_soup = BeautifulSoup("<output></output>", "html.parser")
                    return _finagent_decision_fallback_payload(last_response), fallback_soup

                _DecisionTrading.get_response_dict = _guarded_decision_get_response_dict
                _DecisionTrading._stockbench_decision_retry_guard = True

            if not getattr(_finagent_main_strategy, "_stockbench_tools_params_guard", False):
                _original_prepared_tools_params = _finagent_main_strategy.prepared_tools_params

                def _guarded_prepared_tools_params(state, info, params, memory=None, provider=None, diverse_query=None, strategy_agents=None, cfg=None, mode="train"):
                    try:
                        return _original_prepared_tools_params(state, info, params, memory=memory, provider=provider, diverse_query=diverse_query, strategy_agents=strategy_agents, cfg=cfg, mode=mode)
                    except Exception:
                        fallback = dict(params or {})
                        fallback.setdefault("strategies", "")
                        fallback.setdefault("trading_records", "")
                        return fallback

                _finagent_main_strategy.prepared_tools_params = _guarded_prepared_tools_params
                _finagent_main_strategy._stockbench_tools_params_guard = True

    def _build_finagent_frames(self, *, datasets: Any, symbol: str, start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        bars = datasets.get_day_bars(symbol, start_date.isoformat(), end_date.isoformat()).copy()
        if bars.empty:
            raise RuntimeError(f"Not enough market data to initialize FinAgent for {symbol}")
        if "timestamp" in bars.columns:
            ts = pd.to_datetime(bars["timestamp"])
        elif "date" in bars.columns:
            ts = pd.to_datetime(bars["date"])
        else:
            ts = pd.to_datetime(bars.index)
        # FinAgent's env compares self.news_df.index to a tz-naive Timestamp,
        # so any tz-aware column blows up. Force both bars + news to naive.
        if getattr(ts.dtype, "tz", None) is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        prices = pd.DataFrame({
            "timestamp": ts.dt.normalize(),
            "open": pd.to_numeric(bars.get("open", bars.get("close")), errors="coerce").fillna(0.0),
            "high": pd.to_numeric(bars.get("high", bars.get("close")), errors="coerce").fillna(0.0),
            "low": pd.to_numeric(bars.get("low", bars.get("close")), errors="coerce").fillna(0.0),
            "close": pd.to_numeric(bars.get("close", bars.get("open")), errors="coerce").fillna(0.0),
            "adj_close": pd.to_numeric(bars.get("adj_close", bars.get("close", bars.get("open"))), errors="coerce").fillna(0.0),
            "volume": pd.to_numeric(bars.get("volume", 0.0), errors="coerce").fillna(0.0),
        }).dropna(subset=["timestamp"])
        news_rows = []
        for index, item in enumerate(datasets.get_news(symbol, start_date.isoformat(), end_date.isoformat()) or []):
            raw_ts = item.get("published_utc") or item.get("datetime") or item.get("timestamp") or item.get("date") or item.get("published_at")
            if not raw_ts:
                continue
            try:
                ts_value = pd.to_datetime(raw_ts)
                if getattr(ts_value, "tzinfo", None) is not None:
                    ts_value = ts_value.tz_convert("UTC").tz_localize(None)
                ts_value = ts_value.normalize()
            except Exception:
                continue
            news_rows.append({
                "timestamp": ts_value,
                "id": f"{symbol}_{index:06d}",
                "type": str(item.get("type") or "news"),
                "source": str(item.get("source") or item.get("publisher") or "unknown"),
                "title": str(item.get("title") or ""),
                "text": str(item.get("description") or item.get("summary") or ""),
            })
        news = pd.DataFrame(news_rows, columns=["timestamp", "id", "type", "source", "title", "text"])
        return prices, news

    def _initialize_symbol(self, *, symbol: str, ctx: Dict[str, Any]) -> Any:
        cache = self._symbol_state()
        key = f"{self._run_id(ctx)}::{symbol}"
        if key in cache:
            return cache[key]
        symbol_lock = self._acquire_symbol_lock(key)
        with symbol_lock:
            if key in cache:
                return cache[key]
            return self._do_initialize_finagent_symbol(symbol=symbol, ctx=ctx, cache=cache, key=key)

    def _do_initialize_finagent_symbol(self, *, symbol: str, ctx: Dict[str, Any], cache: dict, key: str) -> Any:
        # Anchor warm-up + valid_env on the original backtest start (stable
        # across --resume), not on the per-bar date. Prevents cache-key
        # drift after crash-and-resume; the bar-loop below advances
        # valid_env forward via env.step() to catch up.
        anchor_raw = ctx.get("backtest_start") or ctx.get("date")
        start_date = pd.to_datetime(anchor_raw).date()
        _raw_train_years = (((self.cfg or {}).get("finagent", {}) or {}).get("training_years", 2))
        train_years = float(_raw_train_years if _raw_train_years is not None else 2)
        if train_years <= 0:
            history_start = start_date - timedelta(days=30)
        else:
            history_start = start_date - timedelta(days=365 * train_years)
        future_end = start_date + timedelta(days=365 * 10)
        prices, news = self._build_finagent_frames(
            datasets=ctx["datasets"],
            symbol=symbol,
            start_date=history_start,
            end_date=future_end,
        )
        available_dates = sorted(pd.to_datetime(prices["timestamp"]).dt.date.unique())
        if len(available_dates) < 3:
            raise RuntimeError(f"Not enough market data to initialize FinAgent for {symbol}")
        train_candidates = [day for day in available_dates if day < start_date]
        if not train_candidates:
            raise RuntimeError(f"Not enough pre-start history to initialize FinAgent for {symbol}")
        look_back_days = int(getattr(getattr(self, "cfg", {}), "get", lambda *_: None)("features", {}).get("history", {}).get("price_series_days", 7) or 7)
        if len(train_candidates) < look_back_days:
            raise RuntimeError(f"Not enough pre-start history to satisfy FinAgent lookback for {symbol}")
        train_start = train_candidates[0]
        train_end = train_candidates[-1]
        test_end = available_dates[-1]
        temp_dir = Path(tempfile.mkdtemp(prefix=f"finagent_{symbol}_", dir=str(self._artifact_root)))
        with _push_import_path(FINAGENT_ROOT), _push_cwd(FINAGENT_ROOT):
            from mmengine.config import Config
            from finagent.environment.trading import EnvironmentTrading
            from finagent.memory.interface import MemoryInterface
            from finagent.provider.provider import OpenAIProvider
            from finagent.query.diverse_query import DiverseQuery
            from finagent.tools import StrategyAgents
            from tools.main_strategy import run as finagent_run, run_step as finagent_run_step

            self._patch_finagent_runtime()
            cfg = Config.fromfile(str(FINAGENT_ROOT / "configs" / "exp" / "trading" / "AAPL.py"))
            cfg.root = str(temp_dir)
            cfg.workdir = "."
            cfg.tag = symbol
            cfg.selected_asset = symbol
            cfg.train_start_date = train_start.isoformat()
            cfg.train_end_date = train_end.isoformat()
            cfg.valid_start_date = start_date.isoformat()
            cfg.valid_end_date = test_end.isoformat()
            cfg.if_load_trading_record = False
            cfg.trading_record_path = None
            cfg.if_load_memory = False
            cfg.memory_path = None
            cfg.checkpoint_start_date = None
            cfg.tool_use_best_params = False
            cfg.train_environment["selected_asset"] = symbol
            cfg.train_environment["start_date"] = train_start.isoformat()
            cfg.train_environment["end_date"] = train_end.isoformat()
            cfg.train_environment["initial_amount"] = self.total_equity
            cfg.valid_environment["selected_asset"] = symbol
            cfg.valid_environment["start_date"] = start_date.isoformat()
            cfg.valid_environment["end_date"] = test_end.isoformat()
            cfg.valid_environment["initial_amount"] = self.total_equity
            fee_rate = self._estimated_buy_fee_rate()
            cfg.train_environment["transaction_cost_pct"] = fee_rate
            cfg.valid_environment["transaction_cost_pct"] = fee_rate
            cfg.provider["provider_cfg_path"] = str(FINAGENT_ROOT / "configs" / "openai_config.json")
            dataset = _FinAgentDatasetShim(prices=prices, news=news, symbol=symbol)
            provider = OpenAIProvider(cfg.provider["provider_cfg_path"])
            env_model = str(os.getenv("MODEL") or os.getenv("LLM_MODEL") or "").strip()
            if env_model:
                provider.llm_model = env_model
                provider.embedding_model = env_model
                cfg.latest_market_intelligence_summary["model"] = env_model
                cfg.past_market_intelligence_summary["model"] = env_model
                cfg.low_level_reflection["model"] = env_model
                cfg.high_level_reflection["model"] = env_model
                cfg.decision["model"] = env_model
            memory = MemoryInterface(root=str(temp_dir), symbols=[symbol], memory_path="memory", embedding_dim=provider.get_embedding_dim(), max_recent_steps=5, workdir=".", tag=symbol)
            diverse_query = DiverseQuery(memory, provider, top_k=cfg.top_k)
            strategy_agents = StrategyAgents()
            plots = _NoopFinAgentPlots()
            train_env = EnvironmentTrading(dataset=dataset, selected_asset=symbol, start_date=train_start.isoformat(), end_date=train_end.isoformat(), look_back_days=cfg.look_back_days, look_forward_days=cfg.look_forward_days, initial_amount=self.total_equity, transaction_cost_pct=fee_rate, discount=1.0)
            # valid_env: force look_forward_days=0 so per-bar get_state()
            # never exposes t+1..t+N price/news to the LLM. FinAgent's
            # default look_forward=14 is a train-time reflection feature;
            # leaving it on during the backtest is a direct look-ahead
            # into the future 14 trading days, which we consider leakage.
            valid_env = EnvironmentTrading(dataset=dataset, selected_asset=symbol, start_date=start_date.isoformat(), end_date=test_end.isoformat(), look_back_days=cfg.look_back_days, look_forward_days=0, initial_amount=self.total_equity, transaction_cost_pct=fee_rate, discount=1.0)
            # Warm-up cache: memory faiss indices + trading_records keyed by
            # (symbol, train_start, train_end, model). Skips the 6-month
            # training loop entirely on re-runs.
            fa_ckpt_root = CACHE_ROOT / "finagent_warmup" / (env_model or "default").replace("/", "_")
            fa_ckpt_root.mkdir(parents=True, exist_ok=True)
            fa_ckpt_dir = fa_ckpt_root / f"{symbol}_{train_start.isoformat()}_{train_end.isoformat()}"
            fa_trading_records_path = fa_ckpt_dir / "trading_records.pkl"
            fa_loaded_from_cache = False
            trading_records = None
            if train_years > 0 and fa_ckpt_dir.exists() and fa_trading_records_path.exists():
                try:
                    memory.load_local(memory_path=str(fa_ckpt_dir / "memory"))
                    with fa_trading_records_path.open("rb") as _fh:
                        trading_records = pickle.load(_fh)
                    fa_loaded_from_cache = True
                except Exception as _load_exc:
                    import sys as _sys
                    _sys.stderr.write(
                        f"[finagent-warmup-cache] {symbol}: load failed "
                        f"({type(_load_exc).__name__}: {_load_exc}), retraining.\n"
                    )
                    trading_records = None
            # Zero-shot mode: skip training run when training_years <= 0.
            if train_years > 0 and not fa_loaded_from_cache:
                trading_records = finagent_run(cfg, train_env, plots, memory, provider, diverse_query, strategy_agents, str(temp_dir), mode="train")
                try:
                    if fa_ckpt_dir.exists():
                        import shutil as _shutil
                        _shutil.rmtree(fa_ckpt_dir, ignore_errors=True)
                    fa_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    memory.save_local(memory_path=str(fa_ckpt_dir / "memory"))
                    with fa_trading_records_path.open("wb") as _fh:
                        pickle.dump(trading_records, _fh)
                except Exception as _save_exc:
                    import sys as _sys
                    _sys.stderr.write(
                        f"[finagent-warmup-cache] {symbol}: save failed "
                        f"({type(_save_exc).__name__}: {_save_exc}), continuing.\n"
                    )
            elif train_years <= 0:
                # FinAgent's run_step reads several keys off trading_records
                # (reasoning, decision, evaluation, etc.). Pre-seed all keys
                # main_strategy.run_step touches so zero-shot skips training
                # cleanly without KeyError on the first bar.
                # Mirror the trading_records schema FinAgent's run_step
                # writes to (see external/FinAgent/tools/main_strategy.py
                # around lines 337-418).
                trading_records = {
                    "symbol": [], "day": [], "value": [], "cash": [],
                    "position": [], "ret": [], "date": [], "price": [],
                    "discount": [], "kline_path": [], "trading_path": [],
                    "total_profit": [], "total_return": [],
                    "action": [], "reasoning": [],
                }
            state, info = valid_env.reset()
            target_ts = pd.Timestamp(start_date)
            info_ts = pd.Timestamp(info["date"])
            while info_ts < target_ts:
                state, _reward, done, _truncated, info = valid_env.step(0)
                info_ts = pd.Timestamp(info["date"])
                if done:
                    raise RuntimeError(f"FinAgent valid environment exhausted before reaching {start_date.isoformat()} for {symbol}")
        cache[key] = {
            "cfg": cfg,
            "env": valid_env,
            "state": state,
            "info": info,
            "plots": plots,
            "memory": memory,
            "provider": provider,
            "diverse_query": diverse_query,
            "strategy_agents": strategy_agents,
            "trading_records": trading_records,
            "run_step": finagent_run_step,
            "exp_path": str(temp_dir),
            "done": False,
        }
        return cache[key]

    def _signals_for_day(self, ctx: Dict[str, Any], symbols: list[str]) -> list[_Signal]:
        date_obj = pd.to_datetime(ctx["date"]).date()
        # Enter FinAgent's repo cwd ONCE for the whole day; _push_cwd is a
        # process-level chdir, so we must not toggle it from worker threads.
        with _push_import_path(FINAGENT_ROOT), _push_cwd(FINAGENT_ROOT):
            # All 20 symbols fan out fully concurrently. Monkey-patches are
            # idempotent (each guarded by `_stockbench_*_guard`) and only
            # rebind attributes to identical closures, so the check-then-set
            # race under GIL is benign. bs4 deepcopy has its own module lock.
            worker_count = min(self._parallel_workers(), max(len(symbols), 1))
            if worker_count == 1:
                states = [self._initialize_symbol(symbol=sym, ctx=ctx) for sym in symbols]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    init_futures = [
                        pool.submit(self._initialize_symbol, symbol=sym, ctx=ctx)
                        for sym in symbols
                    ]
                    states = [fut.result() for fut in init_futures]

            def _step_one(symbol: str, state: Any) -> _Signal:
                if state["done"]:
                    return _Signal(symbol=symbol, signal="HOLD", reason="finagent:done")
                info_date = pd.to_datetime(state["info"]["date"]).date()
                while info_date < date_obj and not state["done"]:
                    next_state, _reward, done, _truncated, next_info = state["env"].step(0)
                    state["state"] = next_state
                    state["info"] = next_info
                    state["done"] = bool(done)
                    info_date = pd.to_datetime(next_info["date"]).date()
                if info_date != date_obj:
                    return _Signal(symbol=symbol, signal="HOLD", reason=f"finagent:date_mismatch:{info_date}")
                try:
                    action = state["run_step"](
                        state["cfg"],
                        state["state"],
                        state["info"],
                        state["plots"],
                        state["memory"],
                        state["provider"],
                        state["diverse_query"],
                        state["strategy_agents"],
                        state["exp_path"],
                        state["trading_records"],
                        "valid",
                    )
                except AssertionError as _bs4_err:
                    # bs4 deepcopy race condition under concurrent threads:
                    # AssertionError in BeautifulSoup._feed() when shared
                    # module objects are deep-copied simultaneously. Emit
                    # HOLD for this symbol+bar so the run does not crash.
                    import sys as _sys
                    _sys.stderr.write(
                        f"[finagent-bs4-guard] {symbol} {date_obj}: "
                        f"bs4 deepcopy AssertionError, emitting HOLD: {_bs4_err}\n"
                    )
                    return _Signal(symbol=symbol, signal="HOLD", reason="finagent:bs4_deepcopy_error")
                signal = _canonical_strategy_signal(action) or "HOLD"
                env_action = state["env"].action_map.get(signal, 0)
                next_state, _reward, done, _truncated, next_info = state["env"].step(env_action)
                if state["trading_records"]["action"] and state["trading_records"]["action"][-1] != next_info["action"]:
                    state["trading_records"]["action"][-1] = next_info["action"]
                if done:
                    state["trading_records"]["total_profit"].append(next_info["total_profit"])
                    state["trading_records"]["total_return"].append(next_info["total_return"])
                    state["trading_records"]["date"].append(next_info["date"])
                    state["trading_records"]["price"].append(next_info["price"])
                state["state"] = next_state
                state["info"] = next_info
                state["done"] = bool(done)
                return _Signal(symbol=symbol, signal=signal, reason=f"finagent:{signal}")

            worker_count = min(self._parallel_workers(), max(len(symbols), 1))
            if worker_count == 1:
                return [_step_one(sym, state) for sym, state in zip(symbols, states)]
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(_step_one, sym, state) for sym, state in zip(symbols, states)]
                return [fut.result() for fut in futures]


__all__ = [
    "FinAgentStrategy",
]
