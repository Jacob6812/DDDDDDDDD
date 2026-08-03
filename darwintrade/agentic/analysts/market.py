from __future__ import annotations

from typing import Any

from ..tooling.contracts import EvidenceRequest


def build_market_request(symbol: str, trade_date: str, query: str) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=f"market-{symbol}-{trade_date}",
        symbol=symbol,
        trade_date=trade_date,
        query=query,
        intent_type="market_signal",
        timeliness="daily",
        regulatory_domain="equity",
        required_params=["symbol", "trade_date"],
        expected_data_type="indicator_bundle",
        max_tools=2,
    )