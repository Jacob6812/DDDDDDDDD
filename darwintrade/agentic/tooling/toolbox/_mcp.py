from __future__ import annotations

from typing import Any

from darwintrade.integrations.llm.mcp import build_default_mcp_client


def call_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return build_default_mcp_client().call_tool(tool_name, payload).output
