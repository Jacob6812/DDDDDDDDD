from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...contracts import ClaimCard, EvidenceRef
from ...interfaces import RoleReport
from ...integrations.llm.prompting import build_role_prompt_spec


@dataclass(slots=True)
class BackendRoleSnapshot:
    role: str
    summary: str
    provenance: str
    timestamp: str
    support: float = 0.7
    semantic_uncertainty: float = 0.2
    evidence_uncertainty: float = 0.2
    tool_trace_refs: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DarwinTradeRoleExecutionLayer:
    RESEARCH_ROLES = {
        "market_regime_analyst",
        "market_analyst",
        "news_analyst",
        "social_analyst",
        "fundamentals_analyst",
        "bull_research_agent",
        "bear_research_agent",
    }

    def build_role_reports(
        self,
        symbol: str,
        trade_date: str,
        snapshots: list[BackendRoleSnapshot],
    ) -> list[RoleReport]:
        reports: list[RoleReport] = []
        for snap in snapshots:
            spec = build_role_prompt_spec(snap.role)
            summary = snap.summary.strip()
            if spec.objective and spec.objective not in summary:
                summary = f"{summary}\n\nDarwinTrade role objective: {spec.objective}"
            reports.append(
                RoleReport(
                    role=snap.role,
                    summary=summary[:2000],
                    provenance=snap.provenance,
                    evidence_refs=[
                        {
                            "source": snap.role,
                            "timestamp": snap.timestamp or trade_date,
                            "title": spec.title,
                            "symbol": symbol,
                        }
                    ],
                    uncertainty={
                        "semantic": snap.semantic_uncertainty,
                        "evidence": snap.evidence_uncertainty,
                    },
                    tool_trace_refs=list(snap.tool_trace_refs or []),
                )
            )
        return reports

    def build_claims(
        self,
        symbol: str,
        trade_date: str,
        snapshots: list[BackendRoleSnapshot],
    ) -> list[ClaimCard]:
        claims: list[ClaimCard] = []
        for snap in snapshots:
            if snap.role not in self.RESEARCH_ROLES:
                continue
            spec = build_role_prompt_spec(snap.role)
            claims.append(
                ClaimCard(
                    asset=symbol,
                    horizon="1d",
                    stance=snap.role,
                    summary=snap.summary[:1600],
                    evidence_refs=[
                        EvidenceRef(
                            source=snap.role,
                            title=spec.title,
                            timestamp=snap.timestamp or trade_date,
                            support=snap.support,
                        )
                    ],
                    provenance=snap.provenance,
                    time_validity=trade_date,
                    u_sem=snap.semantic_uncertainty,
                    u_evi=snap.evidence_uncertainty,
                    verify_flags=["crosscheck"]
                    if snap.role in self.RESEARCH_ROLES
                    else [],
                )
            )
        return claims
