from __future__ import annotations

import re

from .contracts import EvidenceRequest, ToolManifestRecord, ToolPlan, ToolPlanItem
from .registry import ToolRegistry

LATENCY_COST = {"realtime": 0.0, "interactive": 0.05, "batch": 0.14, "offline": 0.22}
TIMELINESS_RANK = {"realtime": 4, "daily": 3, "weekly": 2, "monthly": 1, "quarterly": 0}
WEIGHTS = {
    "timeliness_match": 0.24,
    "intent_match": 0.22,
    "regulatory_domain_match": 0.16,
    "keyword_overlap": 0.14,
    "parameter_coverage": 0.1,
    "data_type_match": 0.08,
    "free_tier_bonus": 0.04,
    "budget_cost": -0.08,
}


class ToolPlanner:
    def __init__(self, registry: ToolRegistry, weights: dict[str, float] | None = None) -> None:
        self.registry = registry
        self.weights = weights or WEIGHTS

    def score_and_rank(self, request: EvidenceRequest) -> ToolPlan:
        records = self.registry.list()
        if not records:
            return ToolPlan(
                request_id=request.request_id,
                query=request.query,
                ranked_tools=[],
                selected_tool_ids=[],
                trace={"reason": "empty_manifest", "weights": dict(self.weights)},
            )
        scored = [self._score_record(request, record) for record in records]
        scored.sort(key=lambda item: (-item.score, item.tool_id))
        ranked = [
            ToolPlanItem(
                rank=index + 1,
                tool_id=item.tool_id,
                score=item.score,
                compliant=item.compliant,
                trace=item.trace,
                rationale=item.rationale,
            )
            for index, item in enumerate(scored)
        ]
        selected = [item.tool_id for item in ranked if item.compliant][: request.max_tools]
        return ToolPlan(
            request_id=request.request_id,
            query=request.query,
            ranked_tools=ranked,
            selected_tool_ids=selected,
            trace={"weights": dict(self.weights), "candidate_count": len(records)},
        )

    def _score_record(self, request: EvidenceRequest, record: ToolManifestRecord) -> ToolPlanItem:
        trace = {
            "timeliness_match": self._timeliness_match(request.timeliness, record.timeliness),
            "intent_match": 1.0 if request.intent_type in record.intent_types else 0.0,
            "regulatory_domain_match": 1.0
            if request.regulatory_domain in record.regulatory_domains
            else 0.0,
            "keyword_overlap": self._keyword_overlap(request.query, record.description),
            "parameter_coverage": self._parameter_coverage(request.required_params, record.required_params),
            "data_type_match": 1.0 if request.expected_data_type == record.response_type else 0.0,
            "free_tier_bonus": 1.0 if record.free_tier else 0.0,
            "budget_cost": LATENCY_COST.get(record.latency_class, 0.12),
        }
        score = round(sum(self.weights.get(key, 0.0) * value for key, value in trace.items()), 6)
        compliant = trace["timeliness_match"] > 0 and self._latency_allowed(
            request.max_latency_class, record.latency_class
        )
        rationale = (
            "finance attribute fit"
            if trace["timeliness_match"] and trace["intent_match"]
            else "penalized for weak finance attribute fit"
        )
        return ToolPlanItem(0, record.tool_id, score, compliant, trace, rationale)

    def _timeliness_match(self, requested: str, offered: str) -> float:
        if requested == offered:
            return 1.0
        req_rank = TIMELINESS_RANK.get(requested, 0)
        offered_rank = TIMELINESS_RANK.get(offered, 0)
        if offered_rank > req_rank:
            return 0.85
        if offered_rank + 1 == req_rank:
            return 0.35
        return 0.0

    def _keyword_overlap(self, query: str, description: str) -> float:
        query_terms = self._terms(query)
        description_terms = self._terms(description)
        if not query_terms:
            return 0.0
        return len(query_terms & description_terms) / len(query_terms)

    def _parameter_coverage(self, required: list[str], available: list[str]) -> float:
        if not required:
            return 1.0
        available_set = set(available)
        return len([item for item in required if item in available_set]) / len(required)

    def _latency_allowed(self, requested: str, offered: str) -> bool:
        order = ["realtime", "interactive", "batch", "offline"]
        return order.index(offered) <= order.index(requested)

    def _terms(self, text: str) -> set[str]:
        return {term for term in re.findall(r"[a-z0-9_]+", text.lower()) if len(term) > 2}
