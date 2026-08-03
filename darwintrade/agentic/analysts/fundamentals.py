from __future__ import annotations

from typing import Any

from darwintrade.integrations.llm.real_data_provider import summarize_statement_fundamentals

from ..tooling.contracts import EvidenceRequest


def build_fundamentals_request(
    symbol: str, trade_date: str, query: str
) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=f"fundamentals-{symbol}-{trade_date}",
        symbol=symbol,
        trade_date=trade_date,
        query=query,
        intent_type="fundamental_quality",
        timeliness="quarterly",
        regulatory_domain="equity",
        required_params=["symbol", "trade_date"],
        expected_data_type="financial_statement_bundle",
        max_tools=3,
        max_latency_class="batch",
    )