from __future__ import annotations

from typing import Any

from ._mcp import call_tool


def get_news(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_news", payload)


def get_global_news(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_global_news", payload)


def get_insider_transactions(payload: dict[str, Any]) -> dict[str, Any]:
    return call_tool("get_insider_transactions", payload)
