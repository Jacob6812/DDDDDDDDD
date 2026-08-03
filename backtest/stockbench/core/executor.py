from __future__ import annotations

import json
import os
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)
from backtest.stockbench.core.schemas import Order

# Generate orders based on decisions and price information (refactored: consistent with backtesting engine logic)
def plan_orders(decision: Dict, snapshot_price: float, cfg: Dict, portfolio: Dict | None = None) -> List[Dict]:
    if decision.get("sim_order"):
        return plan_orders_from_sim_order(
            sim_order=dict(decision.get("sim_order") or {}),
            snapshot_price=snapshot_price,
            cfg=cfg,
            portfolio=portfolio,
        )
    twap_slices: int = int(cfg.get("execution", {}).get("twap_slices", 1))
    price_guard_bps: float = float(cfg.get("execution", {}).get("price_guard_bps", 0))
    # Get backtesting fund configuration, prioritize using portfolio.total_cash
    portfolio_cash = float(cfg.get("portfolio", {}).get("total_cash", 1_000_000))
    backtest_cash_default: float = float(cfg.get("backtest", {}).get("cash", portfolio_cash))

    symbol = decision.get("symbol", "UNKNOWN")
    target_cash_amount = float(decision.get("target_cash_amount", 0))
    action = decision.get("action", "hold")

    # Fund/position information
    equity = float((portfolio or {}).get("equity", backtest_cash_default))
    
    # Get trade execution price: prioritize using open price, consistent with backtesting engine logic
    from backtest.stockbench.core.price_utils import get_unified_price
    
    # Get open price as trading reference price (consistent with backtesting engine)
    ref_price = get_unified_price(symbol, {}, portfolio, "open", snapshot_price)
    if not ref_price or ref_price <= 0:
        ref_price = snapshot_price  # Fallback to snapshot price
    
    logger.debug(f"[EXECUTOR] {symbol}: ref_price={ref_price:.4f}, snapshot_price={snapshot_price:.4f}")
    
    # Calculate current position value (using same price reference)
    position_info = (portfolio or {}).get("positions", {}).get(symbol, {})
    shares = float(position_info.get("shares", 0.0))
    
    if shares > 0:
        current_position_value = shares * ref_price
        logger.debug(f"[EXECUTOR] {symbol}: Current position {shares} shares × {ref_price:.4f} = {current_position_value:.2f}")
    else:
        current_position_value = 0.0
        logger.debug(f"[EXECUTOR] {symbol}: No position")
    
    # DarwinTrade portfolio decisions use target_cash_amount as trade delta notional.
    cash_change = float(decision.get("cash_change", 0.0) or 0.0)
    logger.debug(f"[EXECUTOR] {symbol}: target_cash_amount={target_cash_amount:.2f}, current_position_value={current_position_value:.2f}")
    logger.debug(f"[EXECUTOR] {symbol}: decision cash_change = {cash_change:.2f}")

    if action == "increase":
        cash_change = max(0.0, cash_change if cash_change else target_cash_amount)
        logger.debug(f"[EXECUTOR] {symbol}: increase operation, adjusted cash_change = {cash_change:.2f}")
    elif action == "decrease" or action == "close":
        fallback_change = -target_cash_amount if target_cash_amount > 0 else cash_change
        cash_change = min(0.0, cash_change if cash_change else fallback_change)
        logger.debug(f"[EXECUTOR] {symbol}: {action} operation, adjusted cash_change = {cash_change:.2f}")
    
    if abs(cash_change) <= 0 or ref_price <= 0:
        logger.info(f"[EXECUTOR] {symbol}: Skip trade - cash_change={cash_change:.2f}, ref_price={ref_price:.4f}")
        return []

    # Calculate current total position value
    total_current_position_value = sum(
        float(pos.get("position_value", 0.0)) 
        for pos in (portfolio or {}).get("positions", {}).values()
    )
    
    target_value = abs(cash_change)
    
    # Key fix: use ref_price to calculate shares, consistent with backtesting engine logic
    qty_total = round(target_value / ref_price, 2)
    
    logger.debug(f"[EXECUTOR] {symbol}: target_value={target_value:.2f}, ref_price={ref_price:.4f}")
    logger.debug(f"[EXECUTOR] {symbol}: Calculate shares = {target_value:.2f} ÷ {ref_price:.4f} = {qty_total}")
    
    if qty_total <= 0:
        logger.info(f"[EXECUTOR] {symbol}: Shares is 0, skip trade")
        return []

    qty_per_slice = max(round(qty_total / max(twap_slices, 1), 2), 0)
    if qty_per_slice == 0:
        qty_per_slice = qty_total  # Merge small orders into one slice
        twap_slices = 1

    # Final validation: ensure actual trade amount does not exceed target amount
    actual_trade_amount = qty_total * ref_price
    expected_amount = target_value
    
    if action == "increase" and actual_trade_amount > expected_amount * 1.01:  # Allow 1% error
        # If calculated trade amount significantly exceeds expected, readjust share count
        qty_total = round(expected_amount / ref_price, 2)
        actual_trade_amount = qty_total * ref_price
        logger.warning(f"[EXECUTOR] {symbol}: Adjust shares to prevent overspending - new shares={qty_total}, actual amount={actual_trade_amount:.2f}")
    
    logger.info(f"[EXECUTOR] {symbol}: Final trade - shares={qty_total}, unit price={ref_price:.4f}, total amount={actual_trade_amount:.2f}")
    logger.debug(f"[EXECUTOR] {symbol}: Expected vs Actual - target={expected_amount:.2f}, actual={actual_trade_amount:.2f}, difference={actual_trade_amount-expected_amount:.2f}")

    # Set trading price protection (use ref_price instead of snapshot_price)
    px_guard = ref_price * price_guard_bps / 10_000.0
    limit = ref_price + px_guard

    # Determine trade direction
    side = "buy" if cash_change > 0 else "sell"
    
    orders: List[Dict] = []
    for i in range(max(twap_slices, 1)):
        ord_obj = Order(symbol=symbol, side=side, qty=qty_per_slice, limit=round(limit, 4), slice=i + 1, twap_slices=twap_slices)
        orders.append(ord_obj.model_dump())
    
    logger.info(f"[EXECUTOR] {symbol}: Generated {len(orders)} {side} orders, each {qty_per_slice} shares, limit price {limit:.4f}")
    return orders


def plan_orders_from_sim_order(
    sim_order: Dict,
    snapshot_price: float,
    cfg: Dict,
    portfolio: Dict | None = None,
) -> List[Dict]:
    del cfg, portfolio
    symbol = str(sim_order.get("symbol", "UNKNOWN"))
    original_side = str(sim_order.get("side", "buy")).lower()
    side = {
        "buy": "buy",
        "sell": "sell",
        "sell_short": "sell",
        "buy_to_cover": "buy",
    }.get(original_side, original_side)
    quantity = float(sim_order.get("quantity") or 0.0)
    reference_price = float(sim_order.get("reference_price") or snapshot_price or 0.0)
    if quantity <= 0 or reference_price <= 0 or side not in {"buy", "sell"}:
        return []
    schedule = dict(sim_order.get("schedule", {}) or {})
    constraints = dict(sim_order.get("constraints", {}) or {})
    twap_slices = max(int(schedule.get("slices", 1) or 1), 1)
    qty_per_slice = max(round(quantity / twap_slices, 2), 0.0)
    if qty_per_slice <= 0:
        qty_per_slice = round(quantity, 2)
        twap_slices = 1
    limit = sim_order.get("limit_price")
    if limit is None:
        limit = reference_price
    orders: List[Dict] = []
    for i in range(twap_slices):
        ord_obj = Order(
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            qty=qty_per_slice,
            limit=round(float(limit), 4),
            slice=i + 1,
            twap_slices=twap_slices,
            order_id=str(sim_order.get("order_id", "") or "") or None,
            parent_order_id=sim_order.get("parent_order_id"),
            order_type=sim_order.get("order_type"),
            time_in_force=constraints.get("time_in_force"),
            max_participation_rate=constraints.get("max_participation_rate"),
            allowed_slippage_bps=constraints.get("allowed_slippage_bps"),
            expected_impact_bps=sim_order.get("expected_impact_bps"),
            metadata={"sim_order": sim_order},
        )
        orders.append(ord_obj.model_dump())
    return orders
