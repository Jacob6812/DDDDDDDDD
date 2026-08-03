from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _dataclass_dict(instance: Any, **overrides: Any) -> dict[str, Any]:
    data = asdict(instance)
    data.update(overrides)
    return data


@dataclass(slots=True)
class EvidenceRef:
    source: str
    title: str
    timestamp: str
    url: str = ""
    support: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class ClaimCard:
    asset: str
    horizon: str
    stance: str
    summary: str
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    provenance: str = ""
    time_validity: str = ""
    u_sem: float = 0.0
    u_evi: float = 0.0
    verify_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class EvidenceLedgerEntry:
    key: str
    source_count: int
    avg_support: float
    timestamps: list[str] = field(default_factory=list)
    refs: list[EvidenceRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class EvidenceLedger:
    asset: str
    horizon: str
    entries: list[EvidenceLedgerEntry] = field(default_factory=list)
    source_coverage: float = 0.0
    support_mean: float = 0.0
    conflict_density: float = 0.0
    verify_required: bool = False
    verify_flags: list[str] = field(default_factory=list)
    verification_status: str = "not_required"

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class DisagreementProfile:
    action_probs: dict[str, float]
    entropy: float
    top_action: str
    top_action_prob: float
    vote_variance: float
    support_weight_mean: float
    directional_dispersion: float

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class Constraint:
    constraint_id: str
    constraint_type: str
    scope: str
    hardness: str
    expression: dict[str, Any]
    source: str
    severity: str = "medium"
    penalty_weight: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)
    recoverable: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)
@dataclass(slots=True)
class HoldingState:
    symbol: str
    shares: float
    market_value: float
    weight: float
    cost_basis: float | None = None
    unrealized_pnl: float | None = None
    holding_days: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class PortfolioState:
    trade_date: str
    cash: float
    total_equity: float
    holdings: dict[str, HoldingState] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    sector_exposures: dict[str, float] = field(default_factory=dict)
    factor_exposures: dict[str, float] = field(default_factory=dict)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    turnover_used: float = 0.0
    turnover_budget: float = 1.0
    drawdown_state: float | None = None
    benchmark_weights: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class DownsideTailPoint:
    label: str
    level: float
    return_value: float
    probability_mass: float | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class DownsideProfile:
    tail_points: list[DownsideTailPoint] = field(default_factory=list)
    horizon_days: int | None = None
    source: str = "distribution_fallback"
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class AssetView:
    symbol: str
    trade_date: str
    expected_return_view: float | None
    confidence: float
    downside_risk: float
    disagreement: float
    horizon_days: int
    proposed_direction: str
    event_type: str | None = None
    event_half_life_days: int | None = None
    liquidity_risk: float | None = None
    regime_sensitivity: dict[str, float] | None = None
    downside_distribution: dict[str, float] | None = None
    downside_profile: DownsideProfile | None = None
    rationale: str = ""
    source: str = "analyzer"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)


@dataclass(slots=True)
class AssetResearchPacket:
    symbol: str
    trade_date: str
    asset_view: AssetView
    role_reports: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    tool_traces: list["ToolCallTrace"] = field(default_factory=list)
    evidence_ledger: EvidenceLedger | None = None
    disagreement_profile: DisagreementProfile | None = None
    verification_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "asset_view": self.asset_view.to_dict(),
            "role_reports": list(self.role_reports),
            "claims": list(self.claims),
            "tool_traces": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in self.tool_traces
            ],
            "evidence_ledger": self.evidence_ledger.to_dict()
            if self.evidence_ledger is not None
            else None,
            "disagreement_profile": self.disagreement_profile.to_dict()
            if self.disagreement_profile is not None
            else None,
            "verification_summary": dict(self.verification_summary),
            "metadata": dict(self.metadata),
        }
@dataclass(slots=True)
class ToolCallTrace:
    step: int
    tool_name: str
    arguments: dict[str, Any]
    output: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_dict(self)
