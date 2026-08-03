from __future__ import annotations

from typing import Any

from ._mcp import call_tool


def get_balance_sheet(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_balance_sheet", payload)


def get_cashflow(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_cashflow", payload)


def get_income_statement(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_income_statement", payload)


def get_filing_text(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_filing_text", payload)
