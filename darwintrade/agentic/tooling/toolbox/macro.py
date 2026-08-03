from __future__ import annotations

from typing import Any

from ._mcp import call_tool


def get_market_breadth(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_market_breadth", payload)


def get_volatility_regime(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_volatility_regime", payload)


def get_sector_rotation(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_sector_rotation", payload)


def get_macro_pressure(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_macro_pressure", payload)


def evidence_coverage_check(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("evidence_coverage_check", payload)


def conflict_density_check(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("conflict_density_check", payload)


def trace_consistency_check(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("trace_consistency_check", payload)
