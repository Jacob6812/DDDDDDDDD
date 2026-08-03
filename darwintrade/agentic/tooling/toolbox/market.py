from __future__ import annotations

from typing import Any

from ._mcp import call_tool


def get_stock_data(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_stock_data", payload)


def get_indicators(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_indicators", payload)
