from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import ClaimCard


@dataclass(slots=True)
class RoleReport:
    role: str
    summary: str
    provenance: str
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: dict[str, float] = field(default_factory=dict)
    tool_trace_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "summary": self.summary,
            "provenance": self.provenance,
            "evidence_refs": self.evidence_refs,
            "uncertainty": self.uncertainty,
            "tool_trace_refs": self.tool_trace_refs,
        }


@dataclass(slots=True)
class AnalyzerResult:
    backend_name: str
    symbol: str
    trade_date: str
    signal: str
    raw_report: str
    role_reports: list[RoleReport] = field(default_factory=list)
    claims: list[ClaimCard] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AnalyzerBackend(Protocol):
    def analyze(self, symbol: str, trade_date: str) -> AnalyzerResult: ...
