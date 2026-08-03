from __future__ import annotations

from typing import Any

from ..tooling.contracts import EvidenceRequest


def build_news_request(symbol: str, trade_date: str, query: str) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=f"news-{symbol}-{trade_date}",
        symbol=symbol,
        trade_date=trade_date,
        query=query,
        intent_type="event_risk",
        timeliness="daily",
        regulatory_domain="equity",
        required_params=["symbol", "trade_date"],
        expected_data_type="news_bundle",
        max_tools=2,
    )