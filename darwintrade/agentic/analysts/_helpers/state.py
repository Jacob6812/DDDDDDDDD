from __future__ import annotations
from typing import Any
from ....contracts import HoldingState, PortfolioState


def empty_portfolio_state(trade_date: str) -> PortfolioState:
    return PortfolioState(trade_date=trade_date, cash=0.0, total_equity=1.0)


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
