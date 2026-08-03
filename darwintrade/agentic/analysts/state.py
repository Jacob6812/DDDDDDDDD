from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def merge_dicts(
    current: dict[str, Any], update: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(current)
    merged.update(update)
    return merged


class AnalystState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    remaining_steps: int
    symbol: str
    trade_date: str
    query: str
    current_role: str
    current_request: dict[str, Any]
    current_plan: dict[str, Any]
    current_tool_arguments: Annotated[dict[str, dict[str, Any]], merge_dicts]
    current_outputs: Annotated[dict[str, Any], merge_dicts]
    current_score: float
    snapshots: Annotated[list[dict[str, Any]], operator.add]
    tool_traces: Annotated[list[dict[str, Any]], operator.add]
    ledgers: Annotated[list[dict[str, Any]], operator.add]
    role_outputs: Annotated[dict[str, str], merge_dicts]
    role_payloads: Annotated[dict[str, Any], merge_dicts]
    shared_state: Annotated[dict[str, Any], merge_dicts]
    signal: str
    raw_report: str
    aggregator_summary: dict[str, Any]
