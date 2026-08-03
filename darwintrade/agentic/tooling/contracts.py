from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceRequest:
    request_id: str
    symbol: str
    trade_date: str
    query: str
    intent_type: str
    timeliness: str
    regulatory_domain: str
    required_params: list[str] = field(default_factory=list)
    expected_data_type: str = "generic"
    max_tools: int = 3
    max_latency_class: str = "interactive"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolManifestRecord:
    tool_id: str
    server: str
    tool_name: str
    description: str
    timeliness: str
    intent_types: list[str] = field(default_factory=list)
    regulatory_domains: list[str] = field(default_factory=list)
    latency_class: str = "interactive"
    free_tier: bool = True
    required_params: list[str] = field(default_factory=list)
    response_type: str = "generic"
    implementation_key: str = ""
    finance_tags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolPlanItem:
    rank: int
    tool_id: str
    score: float
    compliant: bool
    trace: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolPlan:
    request_id: str
    query: str
    ranked_tools: list[ToolPlanItem] = field(default_factory=list)
    selected_tool_ids: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ranked_tools"] = [item.to_dict() for item in self.ranked_tools]
        return data


@dataclass(slots=True)
class ToolExecution:
    step: int
    tool_id: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    status: str = "completed"
    error: str = ""
    compressed_output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceLedger:
    request: dict[str, Any]
    plan: dict[str, Any]
    executions: list[ToolExecution] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["executions"] = [item.to_dict() for item in self.executions]
        return data
