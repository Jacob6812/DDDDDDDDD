from __future__ import annotations

from pathlib import Path
import json
from typing import Callable, Dict, List, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay

from backtest.stockbench.core.executor import plan_orders


DesiredState = Tuple[bool | None, str]


class _BaseRuleStrategy:
    strategy_name = "rule_based"

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.total_equity = float(
            (cfg or {}).get("portfolio", {}).get("total_cash", 100000.0)
        )
        self.daily_output_dir: Path | None = None
        self.records: list[dict] = []
        self._lookback_days = 90

    def set_daily_output_dir(self, output_dir: str | Path) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.daily_output_dir = out_dir

    def _append_jsonl_row(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _strategy_cfg(self) -> dict:
        return dict(
            ((self.cfg or {}).get("rule_strategies", {}) or {}).get(
                self.strategy_name, {}
            )
            or {}
        )

    def _max_positions(self, symbol_count: int) -> int:
        return max(
            int(
                (self.cfg or {}).get("backtest", {}).get(
                    "max_positions",
                    (self.cfg or {}).get("risk", {}).get(
                        "max_positions", symbol_count or 1
                    ),
                )
                or 1
            ),
            1,
        )

    def _target_cash_amount(
        self, *, total_equity: float, active_count: int, symbol_count: int
    ) -> float:
        max_positions = self._max_positions(symbol_count)
        default_cap = 1.0 / max_positions
        weight_cap = float(
            (self.cfg or {})
            .get("portfolio_constraints", {})
            .get("max_single_name_weight", default_cap)
            or default_cap
        )
        target_weight = min(weight_cap, 1.0 / max(active_count, 1))
        return round(total_equity * target_weight, 2)

    def _current_value(self, portfolio, symbol: str, open_map: dict[str, float]) -> float:
        position = portfolio.positions.get(symbol) if portfolio else None
        if position is None:
            return 0.0
        return float(getattr(position, "shares", 0.0) or 0.0) * float(
            open_map.get(symbol) or 0.0
        )

    def _portfolio_state(self, portfolio, open_map: dict[str, float], total_equity: float) -> dict:
        return {
            "equity": total_equity,
            "positions": {
                ticker: {
                    "shares": float(getattr(pos, "shares", 0.0) or 0.0),
                    "position_value": float(getattr(pos, "shares", 0.0) or 0.0)
                    * float(open_map.get(ticker) or getattr(pos, "avg_price", 0.0) or 0.0),
                }
                for ticker, pos in (portfolio.positions.items() if portfolio else [])
            },
        }

    def _record_decision(self, date_str: str, decision: dict) -> None:
        row = {"date": date_str, **decision}
        self.records.append(row)
        if self.daily_output_dir is not None:
            self._append_jsonl_row(
                self.daily_output_dir / f"{self.strategy_name}_portfolio_decisions.daily.jsonl",
                row,
            )

    def _history(self, ctx: Dict, symbol: str, lookback_days: int | None = None) -> pd.DataFrame:
        datasets = ctx.get("datasets")
        if datasets is None:
            return pd.DataFrame()
        current_date = pd.Timestamp(ctx["date"]).normalize()
        window = int(lookback_days or self._lookback_days)
        start = (current_date - pd.Timedelta(days=max(window * 4, window + 10))).strftime(
            "%Y-%m-%d"
        )
        end = current_date.strftime("%Y-%m-%d")
        bars = datasets.get_day_bars(symbol, start, end)
        if bars is None or bars.empty:
            return pd.DataFrame()
        history = bars.copy()
        history["date"] = pd.to_datetime(history["date"]).dt.normalize()
        history = history[history["date"] < current_date]
        return history.sort_values("date").reset_index(drop=True)

    def _desired_state(
        self,
        *,
        symbol: str,
        ctx: Dict,
        current_price: float,
        current_value: float,
    ) -> DesiredState:
        raise NotImplementedError

    def on_bar(self, ctx: Dict) -> List[Dict]:
        date_str = ctx["date"].strftime("%Y-%m-%d")
        open_map = dict(ctx.get("open_map") or ctx.get("open_price_map") or {})
        portfolio = ctx.get("portfolio")
        total_equity = float(
            ctx.get("equity_for_sizing")
            or getattr(portfolio, "total_equity", 0.0)
            or self.total_equity
        )
        symbols = [
            symbol
            for symbol in (ctx.get("symbols") or [])
            if float(open_map.get(symbol) or 0.0) > 0.0
        ]
        desired_states: dict[str, DesiredState] = {}
        current_values: dict[str, float] = {}
        for symbol in symbols:
            current_price = float(open_map.get(symbol) or 0.0)
            current_value = self._current_value(portfolio, symbol, open_map)
            current_values[symbol] = current_value
            desired_states[symbol] = self._desired_state(
                symbol=symbol,
                ctx=ctx,
                current_price=current_price,
                current_value=current_value,
            )
        active_symbols = [symbol for symbol, (state, _) in desired_states.items() if state is True]
        target_cash_amount = (
            self._target_cash_amount(
                total_equity=total_equity,
                active_count=len(active_symbols),
                symbol_count=len(symbols),
            )
            if active_symbols
            else 0.0
        )
        orders: list[dict] = []
        portfolio_state = self._portfolio_state(portfolio, open_map, total_equity)
        for symbol in symbols:
            desired, reason = desired_states[symbol]
            current_value = current_values[symbol]
            if desired is True:
                action = "increase" if current_value <= 1e-8 else "hold"
                target_cash = target_cash_amount if action == "increase" else current_value
            elif desired is False:
                action = "close" if current_value > 1e-8 else "hold"
                target_cash = 0.0 if action == "close" else current_value
            else:
                action = "hold"
                target_cash = current_value
            decision = {
                "symbol": symbol,
                "action": action,
                "target_cash_amount": round(target_cash, 2),
                "cash_change": round(target_cash - current_value, 2),
                "confidence": 1.0,
                "reason": reason,
                "trade_date": date_str,
            }
            if action != "hold":
                orders.extend(
                    plan_orders(
                        decision=decision,
                        snapshot_price=float(open_map.get(symbol) or 0.0),
                        cfg=self.cfg,
                        portfolio=portfolio_state,
                    )
                )
            self._record_decision(date_str, decision)
        return orders


class _BaseLongShortStrategy(_BaseRuleStrategy):
    """Base for textbook L/S strategies.

    `_desired_state` returns:
      True  -> open/extend long
      False -> open/extend short
      None  -> hold; close existing exposure if neutral signal.

    Side budget is split between long and short signals proportionally to
    signal counts so the gross-exposure budget stays consistent with the
    long-only flavor (max_positions interprets as max gross slots).
    """

    strategy_name = "rule_long_short"

    def _build_sim_order(self, *, symbol: str, side: str, qty: float, ref_price: float) -> Dict:
        return {
            "symbol": symbol,
            "side": side,
            "quantity": round(qty, 4),
            "reference_price": float(ref_price),
            "schedule": {"slices": 1},
            "constraints": {},
            "order_type": "market",
        }

    def _shares_held(self, portfolio, symbol: str) -> float:
        position = portfolio.positions.get(symbol) if portfolio else None
        if position is None:
            return 0.0
        return float(getattr(position, "shares", 0.0) or 0.0)

    def on_bar(self, ctx: Dict) -> List[Dict]:
        date_str = ctx["date"].strftime("%Y-%m-%d")
        open_map = dict(ctx.get("open_map") or ctx.get("open_price_map") or {})
        portfolio = ctx.get("portfolio")
        total_equity = float(
            ctx.get("equity_for_sizing")
            or getattr(portfolio, "total_equity", 0.0)
            or self.total_equity
        )
        symbols = [
            symbol
            for symbol in (ctx.get("symbols") or [])
            if float(open_map.get(symbol) or 0.0) > 0.0
        ]
        desired_states: dict[str, DesiredState] = {}
        for symbol in symbols:
            current_price = float(open_map.get(symbol) or 0.0)
            current_value = (
                self._shares_held(portfolio, symbol) * current_price
            )
            desired_states[symbol] = self._desired_state(
                symbol=symbol,
                ctx=ctx,
                current_price=current_price,
                current_value=current_value,
            )

        long_syms = [s for s, (st, _) in desired_states.items() if st is True]
        short_syms = [s for s, (st, _) in desired_states.items() if st is False]
        active_count = len(long_syms) + len(short_syms)
        per_leg_cash = (
            self._target_cash_amount(
                total_equity=total_equity,
                active_count=active_count,
                symbol_count=len(symbols),
            )
            if active_count > 0
            else 0.0
        )

        orders: list[dict] = []
        for symbol in symbols:
            desired, reason = desired_states[symbol]
            current_price = float(open_map.get(symbol) or 0.0)
            current_shares = self._shares_held(portfolio, symbol)
            current_value = current_shares * current_price

            if desired is True:
                target_cash = +per_leg_cash
                action_label = "long_signal"
            elif desired is False:
                target_cash = -per_leg_cash
                action_label = "short_signal"
            else:
                target_cash = 0.0
                action_label = "neutral_close"

            target_shares = (target_cash / current_price) if current_price > 0 else 0.0
            delta_shares = target_shares - current_shares

            decision = {
                "symbol": symbol,
                "action": action_label,
                "target_cash_amount": round(target_cash, 2),
                "cash_change": round(target_cash - current_value, 2),
                "confidence": 1.0,
                "reason": reason,
                "trade_date": date_str,
            }

            crosses_zero = (current_shares > 0 and target_shares < 0) or (
                current_shares < 0 and target_shares > 0
            )

            if abs(delta_shares) < 1e-6:
                self._record_decision(date_str, decision)
                continue

            if crosses_zero:
                # close existing leg first, then open opposite
                if current_shares > 0:
                    orders.append(
                        self._build_sim_order(
                            symbol=symbol,
                            side="sell",
                            qty=abs(current_shares),
                            ref_price=current_price,
                        )
                    )
                    if target_shares < 0:
                        orders.append(
                            self._build_sim_order(
                                symbol=symbol,
                                side="sell_short",
                                qty=abs(target_shares),
                                ref_price=current_price,
                            )
                        )
                else:
                    orders.append(
                        self._build_sim_order(
                            symbol=symbol,
                            side="buy_to_cover",
                            qty=abs(current_shares),
                            ref_price=current_price,
                        )
                    )
                    if target_shares > 0:
                        orders.append(
                            self._build_sim_order(
                                symbol=symbol,
                                side="buy",
                                qty=abs(target_shares),
                                ref_price=current_price,
                            )
                        )
            else:
                # single-leg adjustment within current side (or open from flat)
                if delta_shares > 0:
                    side = "buy" if target_shares >= 0 else "buy_to_cover"
                else:
                    side = "sell_short" if target_shares < 0 else "sell"
                orders.append(
                    self._build_sim_order(
                        symbol=symbol,
                        side=side,
                        qty=abs(delta_shares),
                        ref_price=current_price,
                    )
                )

            self._record_decision(date_str, decision)

        # wrap each sim_order in a decision dict so executor.plan_orders dispatches
        # to plan_orders_from_sim_order (which preserves sell_short/buy_to_cover sides)
        engine_orders: list[dict] = []
        for sim in orders:
            wrapped = {"symbol": sim["symbol"], "sim_order": sim}
            from backtest.stockbench.core.executor import plan_orders_from_sim_order

            engine_orders.extend(
                plan_orders_from_sim_order(
                    sim_order=sim,
                    snapshot_price=float(open_map.get(sim["symbol"]) or 0.0),
                    cfg=self.cfg,
                    portfolio=None,
                )
            )
        return engine_orders


class BuyAndHoldStrategy(_BaseRuleStrategy):
    strategy_name = "buy_and_hold"

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del symbol, ctx, current_price
        return True, "buy_and_hold_regime"


class SMACrossoverStrategy(_BaseLongShortStrategy):
    strategy_name = "sma_crossover"

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        params = self._strategy_cfg()
        self.short_window = int(params.get("short_window", 10) or 10)
        self.long_window = int(params.get("long_window", 20) or 20)
        self._lookback_days = max(self.long_window * 3, 90)

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del current_price, current_value
        history = self._history(ctx, symbol, self._lookback_days)
        closes = pd.to_numeric(history.get("close"), errors="coerce").dropna()
        if len(closes) < self.long_window:
            return None, "sma_crossover_insufficient_history"
        short_ma = float(closes.tail(self.short_window).mean())
        long_ma = float(closes.tail(self.long_window).mean())
        return short_ma > long_ma, f"sma_short={short_ma:.4f}|sma_long={long_ma:.4f}"


class WMACrossoverStrategy(_BaseLongShortStrategy):
    strategy_name = "wma_crossover"

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        params = self._strategy_cfg()
        self.short_window = int(params.get("short_window", 10) or 10)
        self.long_window = int(params.get("long_window", 20) or 20)
        self._lookback_days = max(self.long_window * 3, 90)

    def _wma(self, values: pd.Series, window: int) -> float:
        tail = values.tail(window)
        weights = pd.Series(range(1, window + 1), index=tail.index, dtype=float)
        return float((tail * weights).sum() / weights.sum())

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del current_price, current_value
        history = self._history(ctx, symbol, self._lookback_days)
        closes = pd.to_numeric(history.get("close"), errors="coerce").dropna()
        if len(closes) < self.long_window:
            return None, "wma_crossover_insufficient_history"
        short_wma = self._wma(closes, self.short_window)
        long_wma = self._wma(closes, self.long_window)
        return short_wma > long_wma, f"wma_short={short_wma:.4f}|wma_long={long_wma:.4f}"


class ATRBandsStrategy(_BaseLongShortStrategy):
    strategy_name = "atr_bands"

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        params = self._strategy_cfg()
        self.atr_window = int(params.get("atr_window", 14) or 14)
        self.atr_multiplier = float(params.get("atr_multiplier", 1.0) or 1.0)
        self._lookback_days = max(self.atr_window * 4, 90)

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del current_value
        history = self._history(ctx, symbol, self._lookback_days)
        if len(history) < self.atr_window + 1:
            return None, "atr_bands_insufficient_history"
        highs = pd.to_numeric(history.get("high"), errors="coerce")
        lows = pd.to_numeric(history.get("low"), errors="coerce")
        closes = pd.to_numeric(history.get("close"), errors="coerce")
        prev_closes = closes.shift(1)
        tr = pd.concat(
            [(highs - lows), (highs - prev_closes).abs(), (lows - prev_closes).abs()],
            axis=1,
        ).max(axis=1)
        atr = float(tr.tail(self.atr_window).mean())
        reference_close = float(closes.iloc[-1])
        upper = reference_close + self.atr_multiplier * atr
        lower = reference_close - self.atr_multiplier * atr
        if current_price > upper:
            return True, f"atr_upper_breakout={current_price:.4f}>{upper:.4f}"
        if current_price < lower:
            return False, f"atr_lower_breakout={current_price:.4f}<{lower:.4f}"
        return None, f"atr_band_hold={lower:.4f}..{upper:.4f}"


class BollingerBandsStrategy(_BaseLongShortStrategy):
    strategy_name = "bollinger_bands"

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        params = self._strategy_cfg()
        self.window = int(params.get("window", 20) or 20)
        self.std_multiplier = float(params.get("std_multiplier", 2.0) or 2.0)
        self._lookback_days = max(self.window * 4, 90)

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del current_value
        history = self._history(ctx, symbol, self._lookback_days)
        closes = pd.to_numeric(history.get("close"), errors="coerce").dropna()
        if len(closes) < self.window:
            return None, "bollinger_bands_insufficient_history"
        window = closes.tail(self.window)
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        upper = mean + self.std_multiplier * std
        lower = mean - self.std_multiplier * std
        if current_price < lower:
            return True, f"bollinger_oversold={current_price:.4f}<{lower:.4f}"
        if current_price > upper:
            return False, f"bollinger_overbought={current_price:.4f}>{upper:.4f}"
        return None, f"bollinger_hold={lower:.4f}..{upper:.4f}"


class MonthEndEffectStrategy(_BaseRuleStrategy):
    strategy_name = "month_end_effect"

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del symbol, current_price
        current_date = pd.Timestamp(ctx["date"]).normalize()
        previous_business_day = (current_date - BDay(1)).normalize()
        next_business_day = (current_date + BDay(1)).normalize()
        is_first_business_day = previous_business_day.month != current_date.month
        is_last_business_day = next_business_day.month != current_date.month
        if current_value > 1e-8 and is_first_business_day:
            return False, "month_start_exit"
        if current_value <= 1e-8 and is_last_business_day:
            return True, "month_end_entry"
        return None, "calendar_hold"


RULE_STRATEGIES: dict[str, Callable[[Dict], _BaseRuleStrategy]] = {
    "buy_and_hold": BuyAndHoldStrategy,
    "sma_crossover": SMACrossoverStrategy,
    "wma_crossover": WMACrossoverStrategy,
    "atr_bands": ATRBandsStrategy,
    "bollinger_bands": BollingerBandsStrategy,
    "month_end_effect": MonthEndEffectStrategy,
}


def get_rule_strategy_class(name: str) -> Callable[[Dict], _BaseRuleStrategy] | None:
    return RULE_STRATEGIES.get(str(name or "").strip().lower())
